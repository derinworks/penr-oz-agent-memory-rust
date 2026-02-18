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
