use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;
use tracing::{error, warn};

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Migration error: {0}")]
    Migration(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Session not found: {0}")]
    NotFound(String),

    #[error("Session store not configured")]
    NotConfigured,

    #[error("Unauthorized: {0}")]
    Unauthorized(String),
}

impl From<sqlx::Error> for SessionError {
    fn from(e: sqlx::Error) -> Self {
        SessionError::Database(e.to_string())
    }
}

impl IntoResponse for SessionError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            SessionError::NotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            SessionError::NotConfigured => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            SessionError::Unauthorized(_) => (StatusCode::UNAUTHORIZED, self.to_string()),
            _ => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            ),
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Provider '{0}' is not configured")]
    ProviderNotFound(String),

    #[error("Invalid response from provider: {0}")]
    InvalidResponse(String),

    #[error("Authentication failed: missing or invalid API key")]
    AuthenticationError,

    #[error("Provider returned error: {status} - {message}")]
    ProviderError { status: u16, message: String },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Memory entry '{0}' not found")]
    MemoryNotFound(String),
}

/// Describe a `reqwest` transport failure including its underlying cause.
///
/// Every `EmbeddingError::HttpError` in this crate comes from a `send()` call,
/// so it means the provider could not be reached at all. `reqwest::Error`'s own
/// `Display` stops at "error sending request for url (…)" and hides the actual
/// reason (connection refused, DNS failure, TLS error) in its source chain, so
/// walk the chain to give operators something actionable.
fn describe_transport_error(e: &reqwest::Error) -> String {
    let action = if e.is_timeout() {
        "timed out"
    } else {
        "is unreachable"
    };
    let mut message = match e.url() {
        Some(url) => format!("Embedding provider {action} at {url}"),
        None => format!("Embedding provider {action}"),
    };
    let mut source = std::error::Error::source(e);
    while let Some(cause) = source {
        message.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    message
}

impl IntoResponse for EmbeddingError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            EmbeddingError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            // An unreachable provider is an upstream dependency failure, not an
            // internal fault of this server: report 502 with the real cause
            // instead of an opaque 500.
            EmbeddingError::HttpError(e) => {
                let message = describe_transport_error(e);
                error!(error = %message, "Embedding provider request failed");
                (StatusCode::BAD_GATEWAY, message)
            }
            EmbeddingError::ProviderNotFound(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            EmbeddingError::AuthenticationError => (StatusCode::UNAUTHORIZED, self.to_string()),
            EmbeddingError::MemoryNotFound(id) => {
                warn!(memory_id = %id, "Memory entry not found");
                (StatusCode::NOT_FOUND, self.to_string())
            }
            EmbeddingError::ProviderError { status, .. } => {
                let http_status =
                    StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                (http_status, self.to_string())
            }
            EmbeddingError::ConfigError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        (status, Json(json!({ "error": message }))).into_response()
    }
}

#[derive(Debug, Error)]
pub enum VectorStoreError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Qdrant API error: {status} - {message}")]
    Api { status: u16, message: String },

    #[error("Invalid response from Qdrant: {0}")]
    InvalidResponse(String),

    #[error("Vector store not configured")]
    NotConfigured,

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Internal dependency error: {0}")]
    InternalDependencyError(String),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
}

impl IntoResponse for VectorStoreError {
    fn into_response(self) -> Response {
        match self {
            VectorStoreError::Embedding(e) => e.into_response(),
            other => {
                let status = match &other {
                    VectorStoreError::NotConfigured => StatusCode::SERVICE_UNAVAILABLE,
                    VectorStoreError::BadRequest(_) => StatusCode::BAD_REQUEST,
                    VectorStoreError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
                    VectorStoreError::InternalDependencyError(_) => {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                    VectorStoreError::Api { status, .. } => {
                        StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                    }
                    VectorStoreError::Http(_) | VectorStoreError::InvalidResponse(_) => {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                    // The outer match handles Embedding; this arm is a defensive fallback.
                    VectorStoreError::Embedding(_) => StatusCode::INTERNAL_SERVER_ERROR,
                };

                (status, Json(json!({ "error": other.to_string() }))).into_response()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port 1 is never listening, so this fails at connect time without any
    /// network access — the same failure mode as `cargo run` with no embedding
    /// provider running.
    const UNREACHABLE_URL: &str = "http://127.0.0.1:1/api/embed";

    async fn connect_error() -> reqwest::Error {
        reqwest::Client::new()
            .post(UNREACHABLE_URL)
            .send()
            .await
            .expect_err("a request to a closed port must fail")
    }

    async fn body_error_message(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json body");
        value["error"].as_str().expect("error field").to_string()
    }

    #[tokio::test]
    async fn unreachable_provider_maps_to_bad_gateway_with_cause() {
        let response = EmbeddingError::HttpError(connect_error().await).into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let message = body_error_message(response).await;
        assert!(
            message.contains("Embedding provider is unreachable"),
            "message should name the failing dependency: {message}"
        );
        assert!(
            message.contains(UNREACHABLE_URL),
            "message should include the provider URL: {message}"
        );
        // The cause chain, not just reqwest's opaque top-level Display.
        assert!(
            message.len() > format!("Embedding provider is unreachable at {UNREACHABLE_URL}").len(),
            "message should include the underlying cause: {message}"
        );
    }

    #[tokio::test]
    async fn unreachable_provider_maps_to_bad_gateway_through_vector_store_error() {
        let response =
            VectorStoreError::Embedding(EmbeddingError::HttpError(connect_error().await))
                .into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn provider_reported_status_is_still_forwarded() {
        let response = EmbeddingError::ProviderError {
            status: 404,
            message: "model not found".to_string(),
        }
        .into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn not_configured_errors_keep_their_service_unavailable_status() {
        assert_eq!(
            VectorStoreError::NotConfigured.into_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            SessionError::NotConfigured.into_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
