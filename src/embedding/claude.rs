//! Claude-compatible (Anthropic) embedding provider.
//!
//! Uses the Anthropic embeddings endpoint, which is powered by Voyage AI models
//! (e.g. `voyage-3`, `voyage-3-lite`).
//!
//! API reference: <https://docs.anthropic.com/en/docs/build-with-claude/embeddings>
//!
//! Request  → POST {base_url}/v1/embeddings
//! Headers  → `x-api-key: …`, `anthropic-version: 2023-06-01`
//! Body     → `{"model": "…", "input": ["…"]}`
//! Response → `{"data": [{"embedding": […]}]}`

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{config::ProviderConfig, error::EmbeddingError};

use super::{Embedding, EmbeddingProvider};

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct ClaudeProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl ClaudeProvider {
    pub fn new(cfg: &ProviderConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone().unwrap_or_default(),
        }
    }
}

#[derive(Serialize)]
struct ClaudeRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbeddingObject {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ClaudeResponse {
    data: Vec<EmbeddingObject>,
}

#[async_trait]
impl EmbeddingProvider for ClaudeProvider {
    async fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError> {
        if self.api_key.is_empty() {
            return Err(EmbeddingError::AuthenticationError);
        }

        let url = format!("{}/v1/embeddings", self.base_url);
        let body = ClaudeRequest {
            model: &self.model,
            input: vec![text],
        };

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await?;

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

        let parsed: ClaudeResponse = response.json().await.map_err(|e| {
            EmbeddingError::InvalidResponse(format!("Failed to parse Claude response: {e}"))
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
        matchers::{method, path},
        Mock, MockServer, ResponseTemplate,
    };

    use super::*;
    use crate::config::ProviderConfig;

    fn make_config(base_url: &str) -> ProviderConfig {
        ProviderConfig {
            provider_type: "claude".to_string(),
            base_url: base_url.to_string(),
            model: "voyage-3".to_string(),
            api_key: Some("test-anthropic-key".to_string()),
        }
    }

    #[tokio::test]
    async fn embed_returns_first_embedding_on_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"embedding": [0.7_f32, 0.8_f32, 0.9_f32], "index": 0}],
                "model": "voyage-3"
            })))
            .mount(&server)
            .await;

        let provider = ClaudeProvider::new(&make_config(&server.uri()));
        let embedding = provider.embed("hello world").await.unwrap();
        assert_eq!(embedding, vec![0.7, 0.8, 0.9]);
    }

    #[tokio::test]
    async fn embed_returns_auth_error_when_api_key_missing() {
        let provider = ClaudeProvider::new(&ProviderConfig {
            provider_type: "claude".to_string(),
            base_url: "http://localhost:11434".to_string(),
            model: "voyage-3".to_string(),
            api_key: None,
        });
        let result = provider.embed("hello world").await;
        assert!(matches!(result, Err(EmbeddingError::AuthenticationError)));
    }

    #[tokio::test]
    async fn embed_returns_auth_error_on_401() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;

        let provider = ClaudeProvider::new(&make_config(&server.uri()));
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
                "model": "voyage-3"
            })))
            .mount(&server)
            .await;

        let provider = ClaudeProvider::new(&make_config(&server.uri()));
        let result = provider.embed("hello world").await;
        assert!(matches!(result, Err(EmbeddingError::InvalidResponse(_))));
    }
}
