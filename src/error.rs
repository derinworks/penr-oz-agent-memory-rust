use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

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
}

impl IntoResponse for EmbeddingError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            EmbeddingError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            EmbeddingError::ProviderNotFound(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            EmbeddingError::AuthenticationError => {
                (StatusCode::UNAUTHORIZED, self.to_string())
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

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
}

impl IntoResponse for VectorStoreError {
    fn into_response(self) -> Response {
        // Delegate to the embedded EmbeddingError's own IntoResponse impl.
        if let VectorStoreError::Embedding(e) = self {
            return e.into_response();
        }

        let status = match &self {
            VectorStoreError::NotConfigured => StatusCode::SERVICE_UNAVAILABLE,
            VectorStoreError::BadRequest(_) => StatusCode::BAD_REQUEST,
            VectorStoreError::Api { status, .. } => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
            }
            VectorStoreError::Http(_) | VectorStoreError::InvalidResponse(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
            // Embedding variant is handled by the early-return above and is
            // unreachable here, but listed explicitly to keep the match exhaustive.
            VectorStoreError::Embedding(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}
