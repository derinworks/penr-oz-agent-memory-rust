use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};

use crate::{embedding::ProviderRegistry, error::EmbeddingError};

/// Shared application state passed to every handler.
pub struct AppState {
    pub registry: ProviderRegistry,
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub providers: Vec<String>,
    pub default_provider: String,
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

    (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok",
            providers: names,
            default_provider: state.registry.default_provider().to_string(),
        }),
    )
}

// ---------------------------------------------------------------------------
// POST /api/embed
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct EmbedQuery {
    /// Optionally override the configured default provider.
    pub provider: Option<String>,
}

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
        return Err(EmbeddingError::InvalidResponse(
            "Field 'text' must not be empty".to_string(),
        ));
    }

    let provider_name = query.provider.as_deref();
    let provider = state.registry.get(provider_name)?;

    let embedding = provider.embed(&body.text).await?;
    let dimensions = embedding.len();

    Ok((
        StatusCode::OK,
        Json(EmbedResponse {
            provider: provider.name().to_string(),
            dimensions,
            embedding,
        }),
    ))
}
