use std::{collections::HashMap, fs};

use serde::Deserialize;
use tracing::warn;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub embedding: EmbeddingConfig,
    pub qdrant: Option<QdrantConfig>,
    pub database: Option<DatabaseConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    /// SQLite connection URL, e.g. `sqlite:./agent_memory.db`.
    /// Can be overridden by the `DATABASE_URL` environment variable.
    pub url: String,
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
    /// URL of the Qdrant instance, e.g. `http://localhost:6333`.
    /// Port 6333 is the HTTP REST API; 6334 is the gRPC port.
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
    /// Distance metric used when creating the collection.
    /// Valid values: `"Cosine"` (default), `"Euclid"`, `"Dot"`.
    /// Must be chosen to match the embedding model's geometry.
    #[serde(default = "default_distance")]
    pub distance: String,
}

fn default_dimensions() -> u32 {
    768
}

fn default_distance() -> String {
    "Cosine".to_string()
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            collection: "agent_memory".to_string(),
            api_key: None,
            dimensions: default_dimensions(),
            distance: default_distance(),
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&contents)?;

        // QDRANT_URL is the sole trigger for enabling Qdrant when no [qdrant]
        // section is present in the config file.  The other two variables only
        // override fields on a config that is already present (either from the
        // TOML or because QDRANT_URL was set above), so they can never
        // accidentally activate Qdrant on their own.
        if let Ok(url) = std::env::var("QDRANT_URL") {
            if !url.is_empty() {
                let qdrant = config.qdrant.get_or_insert_with(QdrantConfig::default);
                qdrant.url = url;
            }
        }
        if let Some(qdrant) = config.qdrant.as_mut() {
            if let Ok(collection) = std::env::var("QDRANT_COLLECTION") {
                if !collection.is_empty() {
                    qdrant.collection = collection;
                }
            }
            if let Ok(api_key) = std::env::var("QDRANT_API_KEY") {
                qdrant.api_key = Some(api_key);
            }
        } else if std::env::var("QDRANT_COLLECTION").is_ok()
            || std::env::var("QDRANT_API_KEY").is_ok()
        {
            warn!(
                env_vars = "QDRANT_COLLECTION, QDRANT_API_KEY",
                "Qdrant env vars set but Qdrant is not configured; they will have no effect"
            );
        }

        // DATABASE_URL enables the session store when no [database] section is
        // present, or overrides the url when one is already configured.
        if let Ok(url) = std::env::var("DATABASE_URL") {
            if !url.is_empty() {
                if let Some(db) = &mut config.database {
                    db.url = url;
                } else {
                    config.database = Some(DatabaseConfig { url });
                }
            }
        }

        Ok(config)
    }
}
