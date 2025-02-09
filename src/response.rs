use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ErrorResponse;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Limit,
    ContentFilter,
    ToolCalls,
}

#[derive(Debug, Deserialize)]
pub struct NonStreamingMessage {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
    pub role: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct NonStreamingChoice {
    pub index: u32,
    pub message: NonStreamingMessage,
    pub finish_reason: Option<FinishReason>,
    pub logprobs: Option<serde_json::Value>,
    pub error: Option<ErrorResponse>,
}

#[derive(Debug, Deserialize)]
pub struct StreamingChoice {
    pub finish_reason: Option<FinishReason>,
    pub delta: Delta,
    pub error: Option<ErrorResponse>,
}

#[derive(Debug, Deserialize)]
pub struct NonChatChoice {
    pub finish_reason: Option<String>,
    pub text: String,
    pub error: Option<ErrorResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    pub accepted_prediction_tokens: u32,
    pub audio_tokens: u32,
    pub reasoning_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub audio_tokens: u32,
    pub cached_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<NonStreamingChoice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: ResponseUsage,
}
