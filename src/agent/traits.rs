use std::pin::Pin;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::Stream;

use super::error::AgentError;
use super::events::AgentEvent;
use super::executor::AgentExecutor;
use crate::{
    message::Messages,
    response::ChatCompletionResponse,
    tool::ToolBox,
    ApiRequestError, OpenRouter,
};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AgentInput {
    #[schemars(
        description = "The question/request to be processed by the agent. As parrent agent you should provide complete context and all necessary information required for task completion, including any relevant background, constraints, or specific requirements. Questions should be clear, detailed and actionable."
    )]
    pub question: String,
}

#[async_trait]
pub trait BaseAgent: Clone + Send + Sync + 'static {
    fn agent_name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap()
    }
    fn description(&self) -> Option<&str>;
    fn instructions(&self) -> Option<String>;
    fn model(&self) -> &str;
    fn max_tokens(&self) -> Option<u32> {
        None
    }
    fn stop_sequences(&self) -> Option<&Vec<String>> {
        None
    }
    fn temperature(&self) -> Option<f64> {
        None
    }
    fn top_p(&self) -> Option<f64> {
        None
    }
    fn top_k(&self) -> Option<usize> {
        None
    }
    fn tools(&self) -> Option<&ToolBox> {
        None
    }
    fn max_iterations(&self) -> usize {
        12
    }

    /// Makes a single, direct call to the chat completion API.
    /// Uses the provided executor to perform the actual API call.
    async fn once(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, ApiRequestError>;

    /// Creates a streaming response for a single call to the chat completion API.
    /// Uses the provided executor to perform the actual API call.
    fn stream_once(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Pin<
        Box<
            dyn Stream<Item = Result<crate::response::ChatCompletionChunk, ApiRequestError>> + Send,
        >,
    >;
}

#[async_trait]
pub trait Agent: BaseAgent {
    /// Runs the agent, potentially multiple times if tools are involved,
    /// until a final ChatCompletionResponse is obtained or max iterations are reached.
    /// Uses the provided executor to perform the execution.
    async fn run(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, AgentError>;

    /// Runs the agent, streaming events like text chunks, tool calls, and results.
    /// Uses the provided executor to perform the execution.
    ///
    /// The stream yields events including:
    /// - AgentStart: When the agent begins execution
    /// - TextChunk: Individual text chunks from the model
    /// - ToolCallRequested: When the model requests a tool call
    /// - ToolResult: When a tool returns a result
    /// - StreamEnd: When a model turn completes
    /// - AgentFinish: When the agent successfully completes its task
    /// - AgentError: When an error occurs during execution
    fn run_events(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>> + Send + 'static>>;
}

#[async_trait]
pub trait TypedAgent: BaseAgent {
    type Output: JsonSchema + Serialize + DeserializeOwned;

    /// Makes a single call to the chat completion API, requesting a structured JSON response
    /// matching the type `Output`. Parses the response and returns an instance of `Output`.
    /// Uses the provided executor to perform the execution.
    async fn once_typed(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError>;

    /// Runs the agent potentially multiple times, handling tool calls, until a structured
    /// response `Output` is obtained or the maximum number of iterations is reached.
    /// Uses the provided executor to perform the execution.
    async fn run_typed(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError>;
}

// Note: The AnyAgent trait and its implementation have potential design issues
// regarding return types (Value vs specific Agent results) and generic constraints.
// The following implementation attempts to satisfy the trait signature using AgentError
// and converting results to Value, but might need refinement based on actual usage.
#[async_trait]
pub trait AnyAgent: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
    async fn once_any(
        &self,
        executor: &AgentExecutor,
        messages: Messages,
    ) -> Result<Value, AgentError>;
    async fn run_any(
        &self,
        executor: &AgentExecutor,
        messages: Messages,
    ) -> Result<Value, AgentError>;
}

#[async_trait]
impl<T> AnyAgent for T
where
    T: BaseAgent + Agent + Sync, // Require Agent trait for run method
{
    fn name(&self) -> &str {
        T::agent_name(self)
    }

    fn description(&self) -> Option<&str> {
        T::description(self)
    }

    async fn once_any(
        &self,
        executor: &AgentExecutor,
        messages: Messages,
    ) -> Result<Value, AgentError> {
        let resp = self.once(executor, messages).await?; // Returns AgentError
        serde_json::to_value(resp).map_err(AgentError::JsonParsingError)
    }

    async fn run_any(
        &self,
        executor: &AgentExecutor,
        messages: Messages,
    ) -> Result<Value, AgentError> {
        // Use the Agent::run method which handles iterations and tools
        let resp = self.run(executor, messages).await?; // Returns AgentError
        serde_json::to_value(resp).map_err(AgentError::JsonParsingError)
    }
}
