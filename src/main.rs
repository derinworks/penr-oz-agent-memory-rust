mod config;
mod embedding;
mod error;
mod routes;

use std::{net::SocketAddr, sync::Arc};

use axum::{routing::{get, post}, Router};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::{
    config::Config,
    embedding::ProviderRegistry,
    routes::{embed, health, AppState},
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

    let state = Arc::new(AppState { registry });

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/embed", post(embed))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()
        .expect("Invalid server address");

    info!(%addr, "Starting server");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind listener");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
