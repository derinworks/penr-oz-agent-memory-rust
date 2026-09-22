use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use tracing::{info, warn};

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

/// Path of the git-ignored local override file that accompanies `path`:
/// `config.toml` → `config.local.toml`. Extensionless paths get `.local`
/// appended (`myconfig` → `myconfig.local`).
fn local_override_path(path: &str) -> PathBuf {
    let path = Path::new(path);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("config");
    let file_name = match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{stem}.local.{ext}"),
        None => format!("{stem}.local"),
    };
    path.with_file_name(file_name)
}

/// Recursively merge `overlay` into `base`. Tables merge key-by-key; any
/// non-table value (including arrays) in the overlay replaces the base value.
fn merge_toml(base: &mut toml::Value, overlay: toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(base_table), toml::Value::Table(overlay_table)) => {
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) if base_value.is_table() && overlay_value.is_table() => {
                        merge_toml(base_value, overlay_value);
                    }
                    _ => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::load_from_files(path)?;
        config.apply_env_overrides(|key| std::env::var(key).ok());
        Ok(config)
    }

    /// Load and merge the config files only, without applying environment
    /// overrides. Kept separate from [`Config::load`] so the file-merging
    /// behaviour can be exercised independently of the ambient environment.
    fn load_from_files(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Every failure below names the file it came from: two files are read
        // here, and the caller only knows the path of the tracked one, so a bare
        // io/toml error would blame `config.toml` for a typo in
        // `config.local.toml`.
        let contents =
            fs::read_to_string(path).map_err(|e| format!("Failed to read '{path}': {e}"))?;
        let mut value: toml::Value =
            toml::from_str(&contents).map_err(|e| format!("Failed to parse '{path}': {e}"))?;

        // Merge the optional git-ignored local override file (e.g.
        // `config.local.toml`) over the tracked config so users never have to
        // edit the tracked file for local customisation. Environment-variable
        // overrides below still take precedence over both files.
        let override_path = local_override_path(path);
        if override_path.is_file() {
            let display = override_path.display();
            let override_contents = fs::read_to_string(&override_path)
                .map_err(|e| format!("Failed to read '{display}': {e}"))?;
            let override_value: toml::Value = toml::from_str(&override_contents)
                .map_err(|e| format!("Failed to parse '{display}': {e}"))?;
            merge_toml(&mut value, override_value);
            info!(
                path = %override_path.display(),
                "Applied local configuration override"
            );
        }

        Ok(value.try_into()?)
    }

    /// Apply the `QDRANT_*` / `DATABASE_URL` overrides on top of the merged
    /// config files. `lookup` resolves a variable name to its value, mirroring
    /// `std::env::var(..).ok()`; taking it as a parameter keeps the precedence
    /// rules testable without touching the process environment.
    fn apply_env_overrides(&mut self, lookup: impl Fn(&str) -> Option<String>) {
        // QDRANT_URL is the sole trigger for enabling Qdrant when no [qdrant]
        // section is present in the config file.  The other two variables only
        // override fields on a config that is already present (either from the
        // TOML or because QDRANT_URL was set above), so they can never
        // accidentally activate Qdrant on their own.
        if let Some(url) = lookup("QDRANT_URL") {
            if !url.is_empty() {
                let qdrant = self.qdrant.get_or_insert_with(QdrantConfig::default);
                qdrant.url = url;
            }
        }
        if let Some(qdrant) = self.qdrant.as_mut() {
            if let Some(collection) = lookup("QDRANT_COLLECTION") {
                if !collection.is_empty() {
                    qdrant.collection = collection;
                }
            }
            if let Some(api_key) = lookup("QDRANT_API_KEY") {
                qdrant.api_key = Some(api_key);
            }
        } else if lookup("QDRANT_COLLECTION").is_some() || lookup("QDRANT_API_KEY").is_some() {
            warn!(
                env_vars = "QDRANT_COLLECTION, QDRANT_API_KEY",
                "Qdrant env vars set but Qdrant is not configured; they will have no effect"
            );
        }

        // DATABASE_URL enables the session store when no [database] section is
        // present, or overrides the url when one is already configured.
        if let Some(url) = lookup("DATABASE_URL") {
            if !url.is_empty() {
                if let Some(db) = &mut self.database {
                    db.url = url;
                } else {
                    self.database = Some(DatabaseConfig { url });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_CONFIG: &str = r#"
        [server]
        host = "127.0.0.1"
        port = 8080

        [embedding]
        default_provider = "ollama"

        [embedding.providers.ollama]
        type = "ollama"
        base_url = "http://localhost:11434"
        model = "nomic-embed-text"
    "#;

    /// Temporary directory that cleans itself up on drop; avoids adding a
    /// `tempfile` dev-dependency for these few tests.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("config-test-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&dir).expect("create temp dir");
            Self(dir)
        }

        fn write(&self, name: &str, contents: &str) -> String {
            let path = self.0.join(name);
            fs::write(&path, contents).expect("write temp file");
            path.to_str().expect("utf-8 path").to_string()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn local_override_path_inserts_local_before_extension() {
        assert_eq!(
            local_override_path("config.toml"),
            PathBuf::from("config.local.toml")
        );
        assert_eq!(
            local_override_path("conf/app.toml"),
            PathBuf::from("conf/app.local.toml")
        );
        assert_eq!(local_override_path("config"), PathBuf::from("config.local"));
    }

    #[test]
    fn load_without_override_file_uses_base_config() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);

        let config = Config::load_from_files(&path).expect("load config");

        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8080);
        assert!(config.qdrant.is_none());
        assert!(config.database.is_none());
    }

    #[test]
    fn override_file_merges_over_base_config() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);
        dir.write(
            "config.local.toml",
            r#"
                [server]
                port = 9090

                [qdrant]
                url = "http://localhost:6333"
                collection = "agent_memory"
                dimensions = 1024

                [database]
                url = "sqlite::memory:"
            "#,
        );

        let config = Config::load_from_files(&path).expect("load config");

        // Overridden key wins; untouched sibling key is preserved.
        assert_eq!(config.server.port, 9090);
        assert_eq!(config.server.host, "127.0.0.1");

        // Whole sections can be enabled from the override file alone.
        let qdrant = config.qdrant.expect("qdrant enabled by override");
        assert_eq!(qdrant.url, "http://localhost:6333");
        assert_eq!(qdrant.dimensions, 1024);
        assert_eq!(qdrant.distance, "Cosine"); // serde default still applies

        let database = config.database.expect("database enabled by override");
        assert_eq!(database.url, "sqlite::memory:");
    }

    #[test]
    fn override_merges_nested_provider_tables_key_by_key() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);
        dir.write(
            "config.local.toml",
            r#"
                [embedding.providers.ollama]
                base_url = "http://ollama.internal:11434"

                [embedding.providers.openai]
                type = "openai"
                base_url = "https://api.openai.com"
                api_key = "sk-local-secret"
                model = "text-embedding-3-small"
            "#,
        );

        let config = Config::load_from_files(&path).expect("load config");

        // Existing provider: only the overridden field changes.
        let ollama = &config.embedding.providers["ollama"];
        assert_eq!(ollama.base_url, "http://ollama.internal:11434");
        assert_eq!(ollama.model, "nomic-embed-text");

        // New provider added purely from the override file.
        let openai = &config.embedding.providers["openai"];
        assert_eq!(openai.api_key.as_deref(), Some("sk-local-secret"));

        assert_eq!(config.embedding.default_provider, "ollama");
    }

    #[test]
    fn invalid_override_file_is_an_error_naming_the_override_file() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);
        dir.write("config.local.toml", "not valid toml [");

        let error = Config::load_from_files(&path)
            .expect_err("malformed override must fail")
            .to_string();

        // The caller only knows the tracked path, so the error has to say which
        // of the two files is actually broken.
        assert!(
            error.contains("config.local.toml"),
            "error should name the override file: {error}"
        );
    }

    #[test]
    fn invalid_base_file_is_an_error_naming_the_base_file() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", "not valid toml [");

        let error = Config::load_from_files(&path)
            .expect_err("malformed base config must fail")
            .to_string();

        assert!(
            error.contains("config.toml") && !error.contains("config.local.toml"),
            "error should name the base file: {error}"
        );
    }

    /// Build an environment lookup from a fixed key/value list, so these tests
    /// neither read nor mutate the process environment.
    fn env(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
            .collect();
        move |key| {
            vars.iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.clone())
        }
    }

    #[test]
    fn qdrant_url_env_enables_qdrant_when_absent_from_files() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);
        let mut config = Config::load_from_files(&path).expect("load config");
        assert!(config.qdrant.is_none());

        config.apply_env_overrides(env(&[
            ("QDRANT_URL", "http://qdrant.internal:6333"),
            ("QDRANT_COLLECTION", "from-env"),
            ("QDRANT_API_KEY", "qdrant-key"),
        ]));

        let qdrant = config.qdrant.expect("qdrant enabled by env");
        assert_eq!(qdrant.url, "http://qdrant.internal:6333");
        assert_eq!(qdrant.collection, "from-env");
        assert_eq!(qdrant.api_key.as_deref(), Some("qdrant-key"));
        assert_eq!(qdrant.dimensions, 768); // default, untouched by env
    }

    #[test]
    fn qdrant_collection_env_alone_does_not_enable_qdrant() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);
        let mut config = Config::load_from_files(&path).expect("load config");

        config.apply_env_overrides(env(&[
            ("QDRANT_COLLECTION", "from-env"),
            ("QDRANT_API_KEY", "qdrant-key"),
        ]));

        assert!(config.qdrant.is_none());
    }

    #[test]
    fn env_overrides_win_over_local_override_file() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);
        dir.write(
            "config.local.toml",
            r#"
                [qdrant]
                url = "http://from-file:6333"
                collection = "from-file"

                [database]
                url = "sqlite://from-file.db"
            "#,
        );

        let mut config = Config::load_from_files(&path).expect("load config");
        config.apply_env_overrides(env(&[
            ("QDRANT_URL", "http://from-env:6333"),
            ("DATABASE_URL", "sqlite://from-env.db"),
        ]));

        let qdrant = config.qdrant.expect("qdrant configured");
        assert_eq!(qdrant.url, "http://from-env:6333");
        assert_eq!(qdrant.collection, "from-file"); // not set in env
        assert_eq!(
            config.database.expect("database").url,
            "sqlite://from-env.db"
        );
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let dir = TempDir::new();
        let path = dir.write("config.toml", BASE_CONFIG);
        let mut config = Config::load_from_files(&path).expect("load config");

        config.apply_env_overrides(env(&[("QDRANT_URL", ""), ("DATABASE_URL", "")]));

        assert!(config.qdrant.is_none());
        assert!(config.database.is_none());
    }
}
