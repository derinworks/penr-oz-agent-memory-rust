use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    embedding::{DynEmbeddingProvider, ProviderRegistry},
    error::{EmbeddingError, VectorStoreError},
    memory::{MemoryStore, SearchResult as MemorySearchResult},
    vector_store::{QdrantStore, SearchResult as QdrantSearchResult, RESERVED_TEXT_KEY_ERROR},
};

/// Shared application state passed to every handler.
pub struct AppState {
    pub registry: ProviderRegistry,
    /// Qdrant vector store – present only when `[qdrant]` is configured.
    pub vector_store: Option<Arc<QdrantStore>>,
    pub memory: MemoryStore,
}

impl AppState {
    /// Return the active vector store and the resolved embedding provider.
    ///
    /// `provider_override` comes from the per-request `?provider=` query
    /// parameter; when absent the registry default is used.
    fn resolve_store_and_provider<'a>(
        &'a self,
        provider_override: Option<&'a str>,
    ) -> Result<(&'a Arc<QdrantStore>, &'a str, &'a DynEmbeddingProvider), VectorStoreError> {
        let store = self
            .vector_store
            .as_ref()
            .ok_or(VectorStoreError::NotConfigured)?;
        let provider_key = provider_override.unwrap_or(self.registry.default_provider());
        let provider = self.registry.get(Some(provider_key))?;
        Ok((store, provider_key, provider))
    }
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub providers: Vec<String>,
    pub default_provider: String,
    pub vector_store: &'static str,
}

/// Liveness probe – always returns 200 with available provider information.
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut names: Vec<String> = state
        .registry
        .provider_names()
        .into_iter()
        .map(str::to_string)
        .collect();
    names.sort();

    let vector_store = if state.vector_store.is_some() {
        "configured"
    } else {
        "not configured"
    };

    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok",
            providers: names,
            default_provider: state.registry.default_provider().to_string(),
            vector_store,
        }),
    )
}

// ---------------------------------------------------------------------------
// Shared query parameter
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct EmbedQuery {
    /// Optionally override the configured default embedding provider.
    pub provider: Option<String>,
}

// ---------------------------------------------------------------------------
// POST /api/embed
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct EmbedRequest {
    /// The text to embed.
    pub text: String,
}

#[derive(Serialize)]
pub struct EmbedResponse {
    /// Name of the provider that generated this embedding.
    pub provider: String,
    /// Number of dimensions in the resulting vector.
    pub dimensions: usize,
    /// The embedding vector.
    pub embedding: Vec<f32>,
}

/// Generate an embedding for the provided text using the selected provider.
///
/// The provider can be chosen per-request via the `?provider=<name>` query
/// parameter; if omitted the configured default is used.
pub async fn embed(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EmbedQuery>,
    Json(body): Json<EmbedRequest>,
) -> Result<impl IntoResponse, EmbeddingError> {
    if body.text.is_empty() {
        return Err(EmbeddingError::BadRequest(
            EMPTY_TEXT_ERROR.to_string(),
        ));
    }

    let provider_key = query.provider.as_deref().unwrap_or(state.registry.default_provider());
    let provider = state.registry.get(Some(provider_key))?;

    let embedding = provider.embed(&body.text).await?;
    let dimensions = embedding.len();

    Ok((
        StatusCode::OK,
        Json(EmbedResponse {
            provider: provider_key.to_string(),
            dimensions,
            embedding,
        }),
    ))
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_SEARCH_LIMIT: u32 = 5;
const EMPTY_TEXT_ERROR: &str = "Field 'text' must not be empty";
const EMPTY_SEARCH_QUERY_ERROR: &str = "Query parameter 'q' must not be empty";

// ---------------------------------------------------------------------------
// Shared validation helpers
// ---------------------------------------------------------------------------

fn require_non_empty_text(text: &str) -> Result<(), VectorStoreError> {
    if text.is_empty() {
        Err(VectorStoreError::BadRequest(
            EMPTY_TEXT_ERROR.to_string(),
        ))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// POST /api/memory  – store an embedding in Qdrant
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct StoreMemoryQdrantRequest {
    /// The text to embed and store.
    pub text: String,
    /// Optional caller-supplied UUID for the point. A new UUID v4 is generated
    /// when absent. Serde validates the format on deserialization.
    pub id: Option<Uuid>,
    /// Arbitrary metadata stored alongside the embedding in Qdrant.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Serialize)]
pub struct StoreMemoryQdrantResponse {
    /// The ID of the stored point (UUID string).
    pub id: String,
    /// Dimensionality of the stored vector.
    pub dimensions: usize,
    /// The embedding provider that was used.
    pub provider: String,
}

/// Embed `text` and store it in the Qdrant vector store.
///
/// Use the optional `?provider=<name>` query parameter to choose which
/// embedding provider generates the vector.
pub async fn store_memory_qdrant(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EmbedQuery>,
    Json(body): Json<StoreMemoryQdrantRequest>,
) -> Result<impl IntoResponse, VectorStoreError> {
    require_non_empty_text(&body.text)?;
    if body.metadata.contains_key("text") {
        return Err(VectorStoreError::BadRequest(
            RESERVED_TEXT_KEY_ERROR.to_string(),
        ));
    }

    let (store, provider_key, provider) =
        state.resolve_store_and_provider(query.provider.as_deref())?;

    let embedding = provider.embed(&body.text).await?;
    let dimensions = embedding.len();

    let id = store
        .upsert(body.id, embedding, body.text, body.metadata)
        .await?;

    Ok((
        StatusCode::OK,
        Json(StoreMemoryQdrantResponse {
            id,
            dimensions,
            provider: provider_key.to_string(),
        }),
    ))
}

// ---------------------------------------------------------------------------
// POST /api/search  – query Qdrant by vector similarity
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SearchMemoryQdrantRequest {
    /// The query text to embed for the similarity search.
    pub text: String,
    /// Maximum number of results to return (default: 5).
    pub limit: Option<u32>,
    /// Only return results with a cosine similarity score ≥ this value.
    pub score_threshold: Option<f32>,
}

#[derive(Serialize)]
pub struct SearchMemoryQdrantResponse {
    /// Ordered list of nearest-neighbour results (most similar first).
    pub results: Vec<QdrantSearchResult>,
    /// The embedding provider that was used for the query vector.
    pub provider: String,
}

/// Embed `text` and return the nearest memories from the Qdrant vector store.
///
/// Use the optional `?provider=<name>` query parameter to choose which
/// embedding provider generates the query vector (should match the provider
/// used during storage for best results).
pub async fn search_memory_qdrant(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EmbedQuery>,
    Json(body): Json<SearchMemoryQdrantRequest>,
) -> Result<impl IntoResponse, VectorStoreError> {
    require_non_empty_text(&body.text)?;

    let (store, provider_key, provider) =
        state.resolve_store_and_provider(query.provider.as_deref())?;

    let embedding = provider.embed(&body.text).await?;
    let limit = body.limit.unwrap_or(DEFAULT_SEARCH_LIMIT);

    let results = store.search(embedding, limit, body.score_threshold).await?;

    Ok((
        StatusCode::OK,
        Json(SearchMemoryQdrantResponse {
            results,
            provider: provider_key.to_string(),
        }),
    ))
}

// ---------------------------------------------------------------------------
// POST /memory  – store a memory in the in-memory vector store
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct StoreMemoryQuery {
    /// Optionally override the configured default provider.
    pub provider: Option<String>,
}

#[derive(Deserialize)]
pub struct StoreMemoryRequest {
    /// The text content to memorise.
    pub text: String,
    /// Arbitrary key-value metadata attached to the memory.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Optional session tag for grouping related memories.
    pub session: Option<String>,
}

#[derive(Serialize)]
pub struct StoreMemoryResponse {
    pub id: String,
}

/// Store a new memory entry.
///
/// The text is embedded via the selected provider and stored alongside the
/// supplied metadata.  Returns the generated memory ID.
pub async fn store_memory(
    State(state): State<Arc<AppState>>,
    Query(query): Query<StoreMemoryQuery>,
    Json(body): Json<StoreMemoryRequest>,
) -> Result<impl IntoResponse, EmbeddingError> {
    if body.text.is_empty() {
        return Err(EmbeddingError::BadRequest(
            EMPTY_TEXT_ERROR.to_string(),
        ));
    }

    let provider_key = query
        .provider
        .as_deref()
        .unwrap_or(state.registry.default_provider());
    let provider = state.registry.get(Some(provider_key))?;

    let embedding = provider.embed(&body.text).await?;

    let id = state
        .memory
        .store(body.text, body.metadata, body.session, embedding);

    Ok((StatusCode::CREATED, Json(StoreMemoryResponse { id })))
}

// ---------------------------------------------------------------------------
// GET /memory/search
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SearchMemoryQuery {
    /// The search query text.
    pub q: String,
    /// Maximum number of results to return (default: 10).
    pub limit: Option<usize>,
    /// Filter results to a specific session tag.
    pub session: Option<String>,
    /// Optionally override the configured default provider.
    pub provider: Option<String>,
}

#[derive(Serialize)]
pub struct SearchMemoryResponse {
    pub results: Vec<MemorySearchResult>,
}

/// Semantic search over stored memories.
///
/// The query text is embedded and compared against all stored memory vectors
/// using cosine similarity.  Results are returned in descending order of
/// relevance.
pub async fn search_memory(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SearchMemoryQuery>,
) -> Result<impl IntoResponse, EmbeddingError> {
    if query.q.is_empty() {
        return Err(EmbeddingError::BadRequest(
            EMPTY_SEARCH_QUERY_ERROR.to_string(),
        ));
    }

    let provider_key = query
        .provider
        .as_deref()
        .unwrap_or(state.registry.default_provider());
    let provider = state.registry.get(Some(provider_key))?;

    let query_embedding = provider.embed(&query.q).await?;

    let limit = query.limit.unwrap_or(10);
    let results = state
        .memory
        .search(&query_embedding, limit, query.session.as_deref());

    Ok((StatusCode::OK, Json(SearchMemoryResponse { results })))
}

// ---------------------------------------------------------------------------
// DELETE /memory/:id
// ---------------------------------------------------------------------------

/// Delete a stored memory entry by its ID.
pub async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, EmbeddingError> {
    if state.memory.delete(&id) {
        info!(memory_id = %id, "Memory entry deleted");
        Ok(StatusCode::NO_CONTENT)
    } else {
        warn!(memory_id = %id, "Attempted to delete non-existent memory entry");
        Err(EmbeddingError::MemoryNotFound(id))
    }
}
