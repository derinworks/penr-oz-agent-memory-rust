use std::{collections::HashMap, fs};

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub embedding: EmbeddingConfig,
    pub qdrant: Option<QdrantConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub default_provider: String,
    pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: String,
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    /// Auth header scheme used when `api_key` is set.
    /// `"bearer"` (default) → `Authorization: Bearer <key>`
    /// `"api-key"`          → `api-key: <key>` (Azure OpenAI style)
    pub auth_scheme: Option<String>,
    /// Path appended to `base_url` to reach the embeddings endpoint.
    /// Defaults to `"/v1/embeddings"` when absent.
    pub embeddings_path: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct QdrantConfig {
    /// URL of the Qdrant instance, e.g. `http://localhost:6334`.
    /// Can be overridden by the `QDRANT_URL` environment variable.
    pub url: String,
    /// Name of the Qdrant collection to use.
    /// Can be overridden by the `QDRANT_COLLECTION` environment variable.
    pub collection: String,
    /// Optional API key for Qdrant Cloud or secured instances.
    /// Can be overridden by the `QDRANT_API_KEY` environment variable.
    pub api_key: Option<String>,
    /// Dimensionality of the stored vectors. Must match the embedding model output.
    /// Defaults to 768 when not specified.
    #[serde(default = "default_dimensions")]
    pub dimensions: u32,
}

fn default_dimensions() -> u32 {
    768
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            collection: "agent_memory".to_string(),
            api_key: None,
            dimensions: default_dimensions(),
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&contents)?;

        // Override / create Qdrant config from environment variables.
        // Setting QDRANT_URL is sufficient to enable Qdrant even without a
        // [qdrant] section in the config file.
        if let Ok(url) = std::env::var("QDRANT_URL") {
            let qdrant = config.qdrant.get_or_insert_with(QdrantConfig::default);
            qdrant.url = url;
        }
        if let Ok(collection) = std::env::var("QDRANT_COLLECTION") {
            let qdrant = config.qdrant.get_or_insert_with(QdrantConfig::default);
            qdrant.collection = collection;
        }
        if let Ok(api_key) = std::env::var("QDRANT_API_KEY") {
            let qdrant = config.qdrant.get_or_insert_with(QdrantConfig::default);
            qdrant.api_key = Some(api_key);
        }

        Ok(config)
    }
}
