//! OpenAI-compatible embedding provider.
//!
//! API reference: <https://platform.openai.com/docs/api-reference/embeddings>
//!
//! Any endpoint that follows the OpenAI embeddings contract can be used here,
//! including Azure OpenAI, LocalAI, vLLM, and similar proxies.
//!
//! Request  → POST {base_url}{embeddings_path}
//! Body     → `{"model": "…", "input": "…"}`
//! Response → `{"data": [{"embedding": […]}]}`
//!
//! Auth schemes (configured via `auth_scheme`):
//! - `"bearer"` (default) → `Authorization: Bearer <api_key>`
//! - `"api-key"`          → `api-key: <api_key>` (Azure OpenAI style)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{config::ProviderConfig, error::EmbeddingError};

use super::{Embedding, EmbeddingProvider};

pub struct OpenAIProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
    auth_scheme: String,
    embeddings_path: String,
}

impl OpenAIProvider {
    pub fn new(cfg: &ProviderConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone().unwrap_or_default(),
            auth_scheme: cfg.auth_scheme.clone().unwrap_or_else(|| "bearer".to_string()),
            embeddings_path: cfg.embeddings_path.clone().unwrap_or_else(|| "/v1/embeddings".to_string()),
        }
    }
}

#[derive(Serialize)]
struct OpenAIRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingObject {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    data: Vec<EmbeddingObject>,
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError> {
        let url = format!("{}{}", self.base_url, self.embeddings_path);
        let body = OpenAIRequest {
            model: &self.model,
            input: text,
        };

        let mut request = self.client.post(&url).json(&body);
        if !self.api_key.is_empty() {
            request = match self.auth_scheme.as_str() {
                "api-key" => request.header("api-key", &self.api_key),
                _ => request.bearer_auth(&self.api_key),
            };
        }

        let response = request.send().await?;

        let status = response.status();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(EmbeddingError::AuthenticationError);
        }
        if !status.is_success() {
            let message = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ProviderError {
                status: status.as_u16(),
                message,
            });
        }

        let parsed: OpenAIResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse OpenAI response: {e}"))
        })?;

        parsed
            .data
            .into_iter()
            .next()
            .map(|obj| obj.embedding)
            .ok_or_else(|| EmbeddingError::InvalidResponse("Empty data array".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use wiremock::{
        matchers::{header, method, path},
        Mock, MockServer, ResponseTemplate,
    };

    use super::*;
    use crate::config::ProviderConfig;

    fn make_config(base_url: &str) -> ProviderConfig {
        ProviderConfig {
            provider_type: "openai".to_string(),
            base_url: base_url.to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("test-key".to_string()),
            auth_scheme: None,
            embeddings_path: None,
        }
    }

    #[tokio::test]
    async fn embed_returns_first_embedding_on_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"embedding": [0.4_f32, 0.5_f32, 0.6_f32], "index": 0, "object": "embedding"}],
                "model": "text-embedding-3-small",
                "object": "list"
            })))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.uri()));
        let embedding = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding, vec![0.4, 0.5, 0.6]);
    }

    #[tokio::test]
    async fn embed_returns_auth_error_on_401() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.uri()));
        let result = provider.embed("hello world").await;
        assert!(matches!(result, Err(EmbeddingError::AuthenticationError)));
    }

    #[tokio::test]
    async fn embed_returns_error_on_empty_data() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [],
                "model": "text-embedding-3-small",
                "object": "list"
            })))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::new(&make_config(&server.uri()));
        let result = provider.embed("hello world").await;
        assert!(matches!(result, Err(EmbeddingError::InvalidResponse(_))));
    }

    #[tokio::test]
    async fn embed_uses_api_key_header_scheme() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"embedding": [0.1_f32, 0.2_f32], "index": 0, "object": "embedding"}],
                "model": "text-embedding-3-small",
                "object": "list"
            })))
            .mount(&server)
            .await;

        let cfg = ProviderConfig {
            provider_type: "openai".to_string(),
            base_url: server.uri(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("test-key".to_string()),
            auth_scheme: Some("api-key".to_string()),
            embeddings_path: None,
        };
        let provider = OpenAIProvider::new(&cfg);
        let embedding = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding, vec![0.1, 0.2]);
    }

    #[tokio::test]
    async fn embed_uses_custom_embeddings_path() {
        let server = MockServer::start().await;
        let custom_path = "/openai/deployments/my-model/embeddings";

        Mock::given(method("POST"))
            .and(path(custom_path))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"embedding": [0.3_f32, 0.4_f32], "index": 0, "object": "embedding"}],
                "model": "text-embedding-3-small",
                "object": "list"
            })))
            .mount(&server)
            .await;

        let cfg = ProviderConfig {
            provider_type: "openai".to_string(),
            base_url: server.uri(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("test-key".to_string()),
            auth_scheme: None,
            embeddings_path: Some(custom_path.to_string()),
        };
        let provider = OpenAIProvider::new(&cfg);
        let embedding = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding, vec![0.3, 0.4]);
    }
}
