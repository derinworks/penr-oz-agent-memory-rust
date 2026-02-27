mod config;
mod embedding;
mod error;
mod memory;
mod routes;
mod session_store;
mod vector_store;

use std::{net::{IpAddr, SocketAddr}, sync::Arc};

use axum::{routing::{delete, get, post}, Router};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::{
    config::Config,
    embedding::ProviderRegistry,
    memory::MemoryStore,
    routes::{
        create_session, delete_memory, embed, get_session, health, list_sessions,
        search_memory, search_memory_qdrant, store_memory, store_memory_qdrant, AppState,
    },
    session_store::SessionStore,
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

    // Initialise the SQLite session store if configured.
    let session_store = if let Some(db_cfg) = &config.database {
        let store = Arc::new(
            SessionStore::new(&db_cfg.url)
                .await
                .unwrap_or_else(|e| panic!("Failed to initialise session store: {e}")),
        );
        info!(url = %db_cfg.url, "Session store ready");
        Some(store)
    } else {
        info!("Database not configured – /api/sessions endpoints are disabled");
        None
    };

    let session_api_key = std::env::var("SESSION_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    if session_api_key.is_some() {
        info!("Session API key configured – /api/sessions endpoints require X-Api-Key header");
    } else {
        info!("SESSION_API_KEY not set – /api/sessions endpoints are unauthenticated");
    }

    let state = Arc::new(AppState {
        registry,
        vector_store,
        memory: MemoryStore::new(),
        session_store,
        session_api_key,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/embed", post(embed))
        .route("/api/memory", post(store_memory_qdrant))
        .route("/api/search", post(search_memory_qdrant))
        .route("/memory", post(store_memory))
        .route("/memory/search", get(search_memory))
        .route("/memory/{id}", delete(delete_memory))
        .route("/api/sessions", get(list_sessions).post(create_session))
        .route("/api/sessions/:id", get(get_session))
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
