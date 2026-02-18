pub mod claude;
pub mod ollama;
pub mod openai;

use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    config::{EmbeddingConfig, ProviderConfig},
    error::EmbeddingError,
};

/// The result of an embedding operation: a vector of floats representing semantic content.
pub type Embedding = Vec<f32>;

/// Core abstraction for any embedding provider.
///
/// Implementors are expected to be `Send + Sync` so they can be shared across
/// async tasks and Axum handlers.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Return a human-readable name for this provider (e.g. "ollama", "openai").
    fn name(&self) -> &str;

    /// Generate an embedding for the given input text.
    async fn embed(&self, text: &str) -> Result<Embedding, EmbeddingError>;
}

/// A type-erased, heap-allocated embedding provider.
pub type DynEmbeddingProvider = Arc<dyn EmbeddingProvider>;

/// Build a concrete [`EmbeddingProvider`] from a [`ProviderConfig`].
pub fn build_provider(
    _name: &str,
    cfg: &ProviderConfig,
) -> Result<DynEmbeddingProvider, EmbeddingError> {
    match cfg.provider_type.as_str() {
        "ollama" => Ok(Arc::new(ollama::OllamaProvider::new(cfg))),
        "openai" => Ok(Arc::new(openai::OpenAIProvider::new(cfg))),
        "claude" => Ok(Arc::new(claude::ClaudeProvider::new(cfg))),
        unknown => Err(EmbeddingError::ConfigError(format!(
            "Unknown provider type: '{unknown}'"
        ))),
    }
}

/// Registry of all configured embedding providers.
pub struct ProviderRegistry {
    providers: std::collections::HashMap<String, DynEmbeddingProvider>,
    default: String,
}

impl ProviderRegistry {
    /// Build a registry from the embedding section of the application config.
    pub fn from_config(cfg: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let mut providers = std::collections::HashMap::new();
        for (name, provider_cfg) in &cfg.providers {
            let provider = build_provider(name, provider_cfg)?;
            providers.insert(name.clone(), provider);
        }
        Ok(Self {
            providers,
            default: cfg.default_provider.clone(),
        })
    }

    /// Look up a provider by name, falling back to the configured default.
    pub fn get(&self, name: Option<&str>) -> Result<&DynEmbeddingProvider, EmbeddingError> {
        let key = name.unwrap_or(&self.default);
        self.providers
            .get(key)
            .ok_or_else(|| EmbeddingError::ProviderNotFound(key.to_string()))
    }

    /// The name of the default provider.
    pub fn default_provider(&self) -> &str {
        &self.default
    }

    /// All registered provider names.
    pub fn provider_names(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }
}
