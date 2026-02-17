mod config;

use axum::{Router, routing::get, http::StatusCode};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let cfg = config::Config::load("config.toml").expect("Failed to load config.toml");
    info!("Loaded config: {:?}", cfg);

    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let app = Router::new().route("/health", get(health));

    info!("Starting server on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind address");
    axum::serve(listener, app)
        .await
        .expect("Server error");
}

async fn health() -> StatusCode {
    StatusCode::OK
}
