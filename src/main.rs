mod config;
mod embedding;
mod error;
mod memory;
mod routes;
mod vector_store;

use std::{net::{IpAddr, SocketAddr}, sync::Arc};

use axum::{routing::{delete, get, post}, Router};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::{
    config::Config,
    embedding::ProviderRegistry,
    memory::MemoryStore,
    routes::{delete_memory, embed, health, search_memory, store_memory, AppState},
    vector_store::QdrantStore,
};

#[tokio::main]
async fn main() {
    // Initialise structured logging; fall back to INFO if RUST_LOG is unset.
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.toml".to_string());
    let config = Config::load(&config_path)
        .unwrap_or_else(|e| panic!("Failed to load configuration from '{config_path}': {e}"));
    info!("Loaded configuration from '{config_path}'");

    let registry = ProviderRegistry::from_config(&config.embedding)
        .expect("Failed to initialise embedding provider registry");

    info!(
        default = %registry.default_provider(),
        providers = ?registry.provider_names(),
        "Embedding provider registry ready"
    );

    // Initialise the Qdrant vector store if configured.
    let vector_store = if let Some(qdrant_cfg) = &config.qdrant {
        let store = Arc::new(QdrantStore::new(qdrant_cfg));
        store
            .ensure_collection()
            .await
            .unwrap_or_else(|e| panic!("Failed to initialise Qdrant collection: {e}"));
        info!(
            url = %qdrant_cfg.url,
            collection = %qdrant_cfg.collection,
            dimensions = qdrant_cfg.dimensions,
            "Qdrant vector store ready"
        );
        Some(store)
    } else {
        info!("Qdrant not configured – /api/memory and /api/search are disabled");
        None
    };

    let state = Arc::new(AppState {
        registry,
        vector_store,
        memory: MemoryStore::new(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/embed", post(embed))
        .route("/api/memory", post(routes::store_memory_qdrant))
        .route("/api/search", post(routes::search_memory_qdrant))
        .route("/memory", post(store_memory))
        .route("/memory/search", get(search_memory))
        .route("/memory/{id}", delete(delete_memory))
        .with_state(state);

    let host: IpAddr = config.server.host.parse().expect("Invalid server host address");
    let addr = SocketAddr::from((host, config.server.port));

    info!(%addr, "Starting server");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind listener");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
