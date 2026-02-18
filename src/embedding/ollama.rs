//! Ollama-compatible embedding provider.
//!
//! API reference: <https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings>
//!
//! Request  → POST {base_url}/api/embed
//! Body     → `{"model": "…", "input": "…"}`
//! Response → `{"embeddings": [[…]]}`

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{config::ProviderConfig, error::EmbeddingError};

use super::{Embedding, EmbeddingProvider};

pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OllamaProvider {
    pub fn new(cfg: &ProviderConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
        }
    }
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct OllamaResponse {
    embeddings: Vec<Vec<f32>>,
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError> {
        let url = format!("{}/api/embed", self.base_url);
        let body = OllamaRequest {
            model: &self.model,
            input: text,
        };

        let response = self.client.post(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let message = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ProviderError {
                status: status.as_u16(),
                message,
            });
        }

        let parsed: OllamaResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse Ollama response: {e}"))
        })?;

        parsed
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::InvalidResponse("Empty embeddings array".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use wiremock::{
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };

    use super::*;
    use crate::config::ProviderConfig;

    fn make_config(base_url: &str) -> ProviderConfig {
        ProviderConfig {
            provider_type: "ollama".to_string(),
            base_url: base_url.to_string(),
            model: "nomic-embed-text".to_string(),
            api_key: None,
        }
    }

    #[tokio::test]
    async fn embed_returns_first_embedding_on_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[0.1_f32, 0.2_f32, 0.3_f32]]
            })))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new(&make_config(&server.uri()));
        let embedding = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn embed_returns_error_on_provider_failure() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(
                ResponseTemplate::new(500).set_body_string("model not found"),
            )
            .mount(&server)
            .await;

        let provider = OllamaProvider::new(&make_config(&server.uri()));
        let result = provider.embed("hello world").await;
        assert!(matches!(
            result,
            Err(EmbeddingError::ProviderError { status: 500, .. })
        ));
    }

    #[tokio::test]
    async fn embed_returns_error_on_empty_embeddings() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": []
            })))
            .mount(&server)
            .await;

        let provider = OllamaProvider::new(&make_config(&server.uri()));
        let result = provider.embed("hello world").await;
        assert!(matches!(
            result,
            Err(EmbeddingError::InvalidResponse(_))
        ));
    }
}
