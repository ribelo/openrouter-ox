use thiserror::Error;

use crate::ApiRequestError;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("API request failed: {0}")]
    ApiError(#[from] ApiRequestError),

    #[error("Failed to parse JSON response: {0}")]
    JsonParsingError(#[from] serde_json::Error),

    #[error("Model response missing or has unexpected format: {0}")]
    ResponseParsingError(String),

    #[error("Agent reached maximum iterations ({limit}) without completing task")]
    MaxIterationsReached { limit: u32 },

    #[error("Tool execution failed: {0}")]
    ToolError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("Internal Agent Error: {0}")]
    InternalError(String),
}
