mod config;

use axum::{Router, routing::get, http::StatusCode};
use std::net::{IpAddr, SocketAddr};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.toml".to_string());
    let cfg = config::Config::load(&config_path).expect("Failed to load config");
    info!("Loaded config: {:?}", cfg);

    let host: IpAddr = cfg.server.host.parse().expect("Invalid server host address");
    let addr = SocketAddr::from((host, cfg.server.port));
    let app = Router::new().route("/health", get(health));

    info!("Starting server on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind address");
    axum::serve(listener, app)
        .await
        .expect("Server error");
}

async fn health() -> StatusCode {
    StatusCode::OK
}
