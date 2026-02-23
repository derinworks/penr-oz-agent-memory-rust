//! Qdrant vector store integration.
//!
//! This module provides a lightweight REST client for Qdrant that supports:
//! - Automatic collection creation on startup (with exponential-backoff retry)
//! - Upserting embeddings with arbitrary JSON metadata
//! - Querying by vector similarity (configurable distance metric)
//!
//! API reference: <https://qdrant.tech/documentation/interfaces/#api-reference>
//!
//! Key endpoints used:
//! - `GET  /collections/{name}`          – check whether a collection exists
//! - `PUT  /collections/{name}`          – create a collection
//! - `PUT  /collections/{name}/points`   – upsert one or more points
//! - `POST /collections/{name}/points/search` – nearest-neighbour search

use std::{collections::HashMap, time::Duration};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::time::sleep;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{config::QdrantConfig, error::VectorStoreError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A stored memory point returned by search queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The point's unique identifier (UUID string).
    pub id: String,
    /// Similarity score returned by Qdrant; higher means more similar.
    /// The range depends on the collection's distance metric
    /// (`Cosine` → \[-1, 1\], `Dot` → unbounded, `Euclid` → \[0, ∞\] inverted).
    pub score: f32,
    /// The original text that was embedded.
    pub text: String,
    /// Arbitrary metadata stored alongside the embedding.
    pub metadata: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// QdrantStore
// ---------------------------------------------------------------------------

pub struct QdrantStore {
    client: Client,
    base_url: String,
    collection: String,
    api_key: Option<String>,
    dimensions: u32,
    distance: String,
}

impl QdrantStore {
    pub fn new(cfg: &QdrantConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build Qdrant HTTP client");
        Self {
            client,
            base_url: cfg.url.trim_end_matches('/').to_string(),
            collection: cfg.collection.clone(),
            api_key: cfg.api_key.clone(),
            dimensions: cfg.dimensions,
            distance: cfg.distance.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.header("api-key", key);
            }
        }
        req
    }

    // -----------------------------------------------------------------------
    // Collection management
    // -----------------------------------------------------------------------

    /// Ensure the configured collection exists, creating it if necessary.
    ///
    /// Retries up to 4 times with exponential backoff (1 s, 2 s, 4 s, 8 s)
    /// on transient failures: network errors, HTTP 429 (rate-limited), and
    /// HTTP 503 (service temporarily unavailable).
    pub async fn ensure_collection(&self) -> Result<(), VectorStoreError> {
        let mut attempts = 0u32;
        let max_attempts = 5;

        loop {
            attempts += 1;
            match self.try_ensure_collection().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let is_transient = matches!(
                        &e,
                        VectorStoreError::Http(_)
                            | VectorStoreError::Api {
                                status: 429 | 503,
                                ..
                            }
                    );

                    if is_transient && attempts < max_attempts {
                        let wait_secs = 2u64.pow(attempts - 1); // 1, 2, 4, 8 …
                        warn!(
                            attempt = attempts,
                            wait_secs,
                            error = %e,
                            "Qdrant not reachable or experiencing transient error, retrying"
                        );
                        sleep(Duration::from_secs(wait_secs)).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
    }

    async fn try_ensure_collection(&self) -> Result<(), VectorStoreError> {
        let path = format!("/collections/{}", self.collection);

        let resp = self
            .request(reqwest::Method::GET, &path)
            .send()
            .await
            .map_err(VectorStoreError::Http)?;

        match resp.status().as_u16() {
            200 => {
                info!(collection = %self.collection, "Qdrant collection already exists");
                Ok(())
            }
            404 => {
                // Collection is missing – create it.
                let body = json!({
                    "vectors": {
                        "size": self.dimensions,
                        "distance": self.distance
                    }
                });

                let create_resp = self
                    .request(reqwest::Method::PUT, &path)
                    .json(&body)
                    .send()
                    .await
                    .map_err(VectorStoreError::Http)?;

                if !create_resp.status().is_success() {
                    return Err(api_error(create_resp).await);
                }

                info!(
                    collection = %self.collection,
                    dimensions = self.dimensions,
                    "Created Qdrant collection"
                );
                Ok(())
            }
            _ => Err(api_error(resp).await),
        }
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// Upsert a single point into the collection.
    ///
    /// - `id`: optional caller-supplied UUID string; a new UUID v4 is generated
    ///   when absent.
    /// - `vector`: the embedding vector.
    /// - `text`: the original text; stored in the payload under `"text"`.
    /// - `metadata`: arbitrary additional payload fields.
    ///
    /// Returns the point ID that was stored.
    pub async fn upsert(
        &self,
        id: Option<String>,
        vector: Vec<f32>,
        text: String,
        metadata: HashMap<String, Value>,
    ) -> Result<String, VectorStoreError> {
        let point_id = id.unwrap_or_else(|| Uuid::new_v4().to_string());

        let mut payload = metadata;
        match payload.entry("text".to_string()) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(VectorStoreError::BadRequest(
                    "'text' is a reserved metadata key; it is used internally to store the original text of the embedding. Please use a different key for your custom metadata.".to_string(),
                ));
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Value::String(text));
            }
        }

        let body = json!({
            "points": [
                {
                    "id": &point_id,
                    "vector": vector,
                    "payload": payload
                }
            ]
        });

        let path = format!("/collections/{}/points", self.collection);
        let resp = self
            .request(reqwest::Method::PUT, &path)
            .json(&body)
            .send()
            .await
            .map_err(VectorStoreError::Http)?;

        if !resp.status().is_success() {
            return Err(api_error(resp).await);
        }

        Ok(point_id)
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Search for the `limit` nearest neighbours of `vector`.
    ///
    /// - `score_threshold`: when provided, only results with a cosine
    ///   similarity ≥ this value are returned.
    pub async fn search(
        &self,
        vector: Vec<f32>,
        limit: u32,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // `with_payload: true` fetches all payload fields from Qdrant.
        // Qdrant supports server-side filtering via `{"include":[…]}` or
        // `{"exclude":[…]}`, but we intentionally request the full payload
        // because (a) `text` is part of SearchResult and must be returned to
        // callers, and (b) metadata keys are arbitrary and unknown at query
        // time, so we cannot enumerate them for an include filter.
        let mut body = json!({
            "vector": vector,
            "limit": limit,
            "with_payload": true
        });

        if let Some(threshold) = score_threshold {
            body["score_threshold"] = json!(threshold);
        }

        let path = format!("/collections/{}/points/search", self.collection);
        let resp = self
            .request(reqwest::Method::POST, &path)
            .json(&body)
            .send()
            .await
            .map_err(VectorStoreError::Http)?;

        if !resp.status().is_success() {
            return Err(api_error(resp).await);
        }

        let parsed: QdrantSearchResponse = resp.json().await.map_err(|e| {
            VectorStoreError::InvalidResponse(format!("Failed to parse search response: {e}"))
        })?;

        let results = parsed
            .result
            .into_iter()
            .map(SearchResult::from)
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert a non-2xx [`reqwest::Response`] into a [`VectorStoreError::Api`].
///
/// Reads the response body for the error message, falling back to a
/// placeholder string if the body cannot be read.
async fn api_error(resp: reqwest::Response) -> VectorStoreError {
    let status = resp.status().as_u16();
    let message = resp
        .text()
        .await
        .unwrap_or_else(|_| "<failed to read response body>".to_string());
    VectorStoreError::Api { status, message }
}

// ---------------------------------------------------------------------------
// Private Qdrant response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantHit>,
}

#[derive(Deserialize)]
struct QdrantHit {
    /// Qdrant returns the id as either a JSON number or a UUID string.
    id: Value,
    score: f32,
    payload: HashMap<String, Value>,
}

impl From<QdrantHit> for SearchResult {
    fn from(mut hit: QdrantHit) -> Self {
        let id = match hit.id {
            Value::String(s) => s,
            Value::Number(n) => n.to_string(),
            other => {
                warn!("Unexpected Qdrant ID format: {:?}. Converting to string.", other);
                other.to_string()
            }
        };
        // `remove` extracts "text" in a single lookup and transfers ownership,
        // leaving the rest of the payload as metadata without a second pass.
        let text = hit
            .payload
            .remove("text")
            .and_then(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            })
            .unwrap_or_default();
        SearchResult {
            id,
            score: hit.score,
            text,
            metadata: hit.payload,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };

    use super::*;
    use crate::config::QdrantConfig;

    fn make_store(base_url: &str) -> QdrantStore {
        QdrantStore::new(&QdrantConfig {
            url: base_url.to_string(),
            collection: "test_col".to_string(),
            api_key: None,
            dimensions: 3,
            distance: "Cosine".to_string(),
        })
    }

    // -----------------------------------------------------------------------
    // ensure_collection
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn ensure_collection_creates_when_missing() {
        let server = MockServer::start().await;

        // GET → 404 (collection not found)
        Mock::given(method("GET"))
            .and(path("/collections/test_col"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;

        // PUT → 200 (collection created)
        Mock::given(method("PUT"))
            .and(path("/collections/test_col"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result": true,
                "status": "ok",
                "time": 0.001
            })))
            .expect(1)
            .mount(&server)
            .await;

        make_store(&server.uri())
            .ensure_collection()
            .await
            .expect("ensure_collection should succeed");
    }

    #[tokio::test]
    async fn ensure_collection_skips_when_exists() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/collections/test_col"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result": { "status": "green" },
                "status": "ok",
                "time": 0.001
            })))
            .expect(1)
            .mount(&server)
            .await;

        make_store(&server.uri())
            .ensure_collection()
            .await
            .expect("ensure_collection should succeed without creating");

        // PUT must NOT be called – wiremock will assert the expect(1) on GET
        // and panic if an unexpected PUT is received.
    }

    #[tokio::test]
    async fn ensure_collection_returns_error_on_create_failure() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/collections/test_col"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        Mock::given(method("PUT"))
            .and(path("/collections/test_col"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let result = make_store(&server.uri()).ensure_collection().await;
        assert!(matches!(result, Err(VectorStoreError::Api { status: 500, .. })));
    }

    // -----------------------------------------------------------------------
    // upsert
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn upsert_returns_generated_id_on_success() {
        let server = MockServer::start().await;

        Mock::given(method("PUT"))
            .and(path("/collections/test_col/points"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result": { "operation_id": 0, "status": "completed" },
                "status": "ok",
                "time": 0.001
            })))
            .mount(&server)
            .await;

        let store = make_store(&server.uri());
        let id = store
            .upsert(None, vec![0.1, 0.2, 0.3], "hello".to_string(), HashMap::new())
            .await
            .expect("upsert should succeed");

        // A generated UUID should be non-empty
        assert!(!id.is_empty());
        // Should be a valid UUID
        assert!(uuid::Uuid::parse_str(&id).is_ok(), "returned id should be a valid UUID");
    }

    #[tokio::test]
    async fn upsert_uses_supplied_id() {
        let server = MockServer::start().await;
        let custom_id = "00000000-0000-0000-0000-000000000001".to_string();

        Mock::given(method("PUT"))
            .and(path("/collections/test_col/points"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result": { "operation_id": 1, "status": "completed" },
                "status": "ok",
                "time": 0.001
            })))
            .mount(&server)
            .await;

        let store = make_store(&server.uri());
        let returned_id = store
            .upsert(
                Some(custom_id.clone()),
                vec![0.1, 0.2, 0.3],
                "hello".to_string(),
                HashMap::new(),
            )
            .await
            .expect("upsert should succeed");

        assert_eq!(returned_id, custom_id);
    }

    #[tokio::test]
    async fn upsert_rejects_reserved_text_key_in_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("text".to_string(), serde_json::Value::String("oops".to_string()));

        let result = make_store("http://unused")
            .upsert(None, vec![0.1, 0.2, 0.3], "hello".to_string(), metadata)
            .await;

        assert!(matches!(result, Err(VectorStoreError::BadRequest(_))));
    }

    #[tokio::test]
    async fn upsert_returns_error_on_api_failure() {
        let server = MockServer::start().await;

        Mock::given(method("PUT"))
            .and(path("/collections/test_col/points"))
            .respond_with(ResponseTemplate::new(400).set_body_string("wrong dimension"))
            .mount(&server)
            .await;

        let result = make_store(&server.uri())
            .upsert(None, vec![0.1], "hello".to_string(), HashMap::new())
            .await;

        assert!(matches!(result, Err(VectorStoreError::Api { status: 400, .. })));
    }

    // -----------------------------------------------------------------------
    // search
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn search_returns_results() {
        let server = MockServer::start().await;
        let hit_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";

        Mock::given(method("POST"))
            .and(path("/collections/test_col/points/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result": [
                    {
                        "id": hit_id,
                        "version": 0,
                        "score": 0.95,
                        "payload": {
                            "text": "original text",
                            "source": "unit-test"
                        }
                    }
                ],
                "status": "ok",
                "time": 0.001
            })))
            .mount(&server)
            .await;

        let results = make_store(&server.uri())
            .search(vec![0.1, 0.2, 0.3], 5, None)
            .await
            .expect("search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, hit_id);
        assert_eq!(results[0].score, 0.95);
        assert_eq!(results[0].text, "original text");
        assert_eq!(
            results[0].metadata.get("source").and_then(|v| v.as_str()),
            Some("unit-test")
        );
        // "text" must not appear in metadata
        assert!(!results[0].metadata.contains_key("text"));
    }

    #[tokio::test]
    async fn search_returns_empty_results() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/collections/test_col/points/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "result": [],
                "status": "ok",
                "time": 0.001
            })))
            .mount(&server)
            .await;

        let results = make_store(&server.uri())
            .search(vec![0.1, 0.2, 0.3], 5, Some(0.8))
            .await
            .expect("search should succeed with empty results");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_returns_error_on_api_failure() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/collections/test_col/points/search"))
            .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
            .mount(&server)
            .await;

        let result = make_store(&server.uri())
            .search(vec![0.1, 0.2, 0.3], 5, None)
            .await;

        assert!(matches!(result, Err(VectorStoreError::Api { status: 503, .. })));
    }
}
