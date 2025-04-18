use std::{error::Error, pin::Pin};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::Stream;

use super::error::AgentError;
use super::events::AgentEvent;
use super::executor::AgentExecutor;
use crate::{
    message::Messages, response::{ChatCompletionResponse, ToolCall}, tool::ToolBox, ApiRequestError, OpenRouter,
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
    fn description(&self) -> Option<&str> {
        None
    }
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
    ) -> Result<ChatCompletionResponse, ApiRequestError> {
        executor.execute_once(self, messages).await
    }

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
    > {
        executor.stream_once(self, messages)
    }

    /// Asynchronous hook invoked for each tool call requested by the model,
    /// acting as middleware before potential execution.
    ///
    /// This method allows the agent implementation to intercept a specific `ToolCall`
    /// and decide how to handle it. It can:
    /// 1.  **Inspect:** Examine the tool name and arguments.
    /// 2.  **Modify:** Change the `ToolCall` (e.g., alter arguments) before potentially executing it.
    /// 3.  **Execute:** Call the actual tool implementation (e.g., using `self.tools().invoke()`)
    ///     and return its resulting `Messages`.
    /// 4.  **Handle Directly:** Generate `Messages` without calling the actual tool (e.g.,
    ///     return a cached response, ask the user for confirmation and return their input
    ///     as a message, return a "permission denied" message).
    /// 5.  **Reject/Error:** Return an `Err(AgentError)` if the hook logic fails or if
    ///     the tool call should definitively halt the agent's execution.
    ///
    /// # Arguments
    /// * `tool_call`: The specific `ToolCall` proposed by the LLM for this step.
    ///
    /// # Returns
    /// - `Ok(Messages)`: One or more messages resulting from handling the tool call
    ///   (either by direct handling in the hook or by invoking the underlying tool).
    ///   These messages will be added to the conversation history for the next LLM turn.
    /// - `Err(AgentError)`: An error indicating a failure in the hook or underlying
    ///   tool execution that should stop the agent run. This could be a
    ///   `AgentError::ToolError` if the tool itself failed, or `AgentError::CallbackError`
    ///   if the hook logic failed.
    ///
    /// # Default Implementation
    /// The default implementation directly invokes the corresponding tool from the
    /// agent's `ToolBox` using `self.tools().unwrap().invoke(&tool_call).await`.
    /// It assumes the agent has tools configured if this is called.
    /// Override this method in your agent implementation to add custom middleware logic.
    async fn invoke_tool(
        &self,
        tool_call: ToolCall,
    ) -> Result<Messages, AgentError> {
        // Default: Directly invoke the tool via the ToolBox
        match self.tools() {
            Some(toolbox) => Ok(toolbox.invoke(&tool_call).await),
            None => Err(AgentError::InternalError(format!(
                "Agent '{}' received tool call for '{}' but has no ToolBox configured.",
                self.agent_name(),
                tool_call.function.name.as_deref().unwrap_or("<unknown>")
            ))),
        }
    }
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
    ) -> Result<ChatCompletionResponse, AgentError> {
        executor.execute_run(self, messages).await
    }

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
    fn stream_run(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>> + Send + 'static>> {
        executor.stream_run(self, messages)
    }
}

impl<T: BaseAgent> Agent for T {}

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
    ) -> Result<Self::Output, AgentError> {
        executor.execute_once_typed(self, messages).await
    }

    /// Runs the agent potentially multiple times, handling tool calls, until a structured
    /// response `Output` is obtained or the maximum number of iterations is reached.
    /// Uses the provided executor to perform the execution.
    async fn run_typed(
        &self,
        executor: &AgentExecutor,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError> {
        executor.execute_run_typed(self, messages).await
    }
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
