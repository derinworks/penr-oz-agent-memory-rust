use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;
use tracing::warn;

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

    #[error("Bad request: {0}")]
    BadRequest(String),

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
            SessionError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
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

impl IntoResponse for EmbeddingError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            EmbeddingError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            EmbeddingError::ProviderNotFound(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            EmbeddingError::AuthenticationError => {
                (StatusCode::UNAUTHORIZED, self.to_string())
            }
            EmbeddingError::MemoryNotFound(id) => {
                warn!(memory_id = %id, "Memory entry not found");
                (StatusCode::NOT_FOUND, self.to_string())
            }
            EmbeddingError::ProviderError { status, .. } => {
                let http_status =
                    StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                (http_status, self.to_string())
            }
            EmbeddingError::ConfigError(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
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
