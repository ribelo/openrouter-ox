use std::{collections::HashMap, marker::PhantomData, pin::Pin, sync::Arc};

use async_stream::try_stream;
use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio_stream::{Stream, StreamExt};

use crate::{
    message::{Message, Messages, SystemMessage, ToolMessage},
    response::{ChatCompletionResponse, FinishReason, ToolCall, Usage},
    tool::{Tool, ToolBox},
    ApiRequestError, OpenRouter,
};

/// Helper struct to accumulate fragmented tool calls across multiple chunks.
#[derive(Debug, Clone, Default)]
struct PartialToolCall {
    index: Option<usize>,
    id: Option<String>,
    type_field: String,
    function_name: String,
    arguments_buffer: String,
}

impl PartialToolCall {
    /// Merges information from a tool call delta into this partial tool call.
    /// We only care about index and merging arguments.
    /// Assumes id, type, name are set initially by the aggregator.
    fn merge(&mut self, delta_tool_call: &ToolCall) {
        // Ensure index is set if it wasn't (e.g., if first delta somehow missed it)
        // Primarily relies on aggregator setting it initially.
        if self.index.is_none() {
            self.index = delta_tool_call.index;
        }

        if self.id.is_none() {
            self.id = delta_tool_call.id.clone();
        }

        if self.type_field.is_empty() && !delta_tool_call.type_field.is_empty() {
            self.type_field = delta_tool_call.type_field.clone();
        }

        if let Some(ref name) = delta_tool_call.function.name {
            if !name.is_empty() {
                self.function_name = name.clone();
            }
        }

        // Append argument chunks
        if !delta_tool_call.function.arguments.is_empty() {
            self.arguments_buffer
                .push_str(&delta_tool_call.function.arguments);
        }
    }

    /// Attempts to convert the accumulated partial data into a final ToolCall.
    /// Returns None if essential information (like id, name) is missing.
    fn finalize(self) -> Option<ToolCall> {
        // Essential fields must be present
        let id = self.id?;
        let function_name = self.function_name; // This unwraps Option<String> to String

        // Type defaults to "function" if not specified but name is present
        let type_field = self.type_field;

        Some(ToolCall {
            index: self.index,
            id: Some(id), // id is String here
            type_field,
            function: crate::response::FunctionCall {
                name: Some(function_name),
                arguments: self.arguments_buffer,
            },
        })
    }
}

/// Manages a collection of partial tool calls during streaming
#[derive(Debug, Default)]
struct PartialToolCallsAggregator {
    // Key: index of the tool call
    // Value: The accumulator for that specific tool call
    calls: HashMap<usize, PartialToolCall>,
}

impl PartialToolCallsAggregator {
    /// Creates a new, empty accumulator
    fn new() -> Self {
        Self::default()
    }

    /// Adds a tool call delta fragment to the appropriate accumulator
    /// Creates a new accumulator if the index hasn't been seen before
    fn add_delta(&mut self, delta_call: &ToolCall) {
        // Use index as the key, defaulting to 0 if missing
        let index = delta_call.index.unwrap_or(0);

        self.calls
            .entry(index)
            .or_default() // Get existing PartialToolCall or create a default one
            .merge(delta_call); // Merge the delta data
    }

    /// Consumes the accumulator and returns a finalized list of ToolCalls
    /// Incomplete calls are filtered out with warnings
    /// The list is sorted by the original index
    fn finalize(self) -> Vec<ToolCall> {
        let mut finalized: Vec<ToolCall> = self
            .calls
            .into_values() // Consume the map and take ownership of PartialToolCalls
            .filter_map(|partial| {
                // Attempt to finalize each partial call
                match partial.finalize() {
                    Some(call) => Some(call),
                    None => {
                        eprintln!(
                            "Warning: Ignoring incomplete tool call data during finalization."
                        );
                        None // Filter out incomplete calls
                    }
                }
            })
            .collect();

        // Sort by index for deterministic order
        finalized.sort_by_key(|call| call.index.unwrap_or(0));

        finalized
    }

    /// Checks if any partial tool calls have been accumulated
    fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

/// Event variants emitted during agent execution
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Emitted once at the beginning of the agent's execution
    AgentStart,

    /// A chunk of text generated by the language model
    TextChunk { delta: String },

    /// The language model has requested a tool call.
    /// This is emitted after the model's response stream finishes for that turn
    /// and the full tool call details are assembled.
    ToolCallRequested { tool_call: ToolCall },

    /// The result of a tool execution. Contains the content returned by the tool.
    /// This is emitted after the agent successfully invokes the tool.
    ToolResult { message: ToolMessage },

    /// Emitted when the final LLM turn finishes, includes usage stats if available
    StreamEnd { usage: Option<Usage> },

    /// Emitted once when the agent successfully completes its task
    AgentFinish,

    /// Emitted when an error occurs during agent execution. The stream terminates after this.
    AgentError { error: AgentErrorSerializable },
}

/// A serializable representation of AgentError for the stream.
/// Needed because the original AgentError might contain non-serializable types.
#[derive(Debug, Clone, Serialize, Error)]
pub enum AgentErrorSerializable {
    #[error("API request failed: {0}")]
    ApiError(String),

    #[error("Failed to parse JSON response: {0}")]
    JsonParsingError(String),

    #[error("Model response missing or has unexpected format: {0}")]
    ResponseParsingError(String),

    #[error("Agent reached maximum iterations ({limit}) without completing task")]
    MaxIterationsReached { limit: usize },

    #[error("Tool execution failed: {0}")]
    ToolError(String),

    #[error("Internal Agent Error: {0}")]
    InternalError(String),
}

impl From<&AgentError> for AgentErrorSerializable {
    fn from(error: &AgentError) -> Self {
        match error {
            AgentError::ApiError(e) => AgentErrorSerializable::ApiError(e.to_string()),
            AgentError::JsonParsingError(e) => {
                AgentErrorSerializable::JsonParsingError(e.to_string())
            }
            AgentError::ResponseParsingError(s) => {
                AgentErrorSerializable::ResponseParsingError(s.clone())
            }
            AgentError::MaxIterationsReached { limit } => {
                AgentErrorSerializable::MaxIterationsReached { limit: *limit }
            }
            AgentError::ToolError(e) => AgentErrorSerializable::ToolError(e.to_string()),
            AgentError::InternalError(e) => AgentErrorSerializable::InternalError(e.to_string()),
        }
    }
}

impl From<AgentError> for AgentErrorSerializable {
    fn from(error: AgentError) -> Self {
        AgentErrorSerializable::from(&error)
    }
}

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("API request failed: {0}")]
    ApiError(#[from] ApiRequestError),

    #[error("Failed to parse JSON response: {0}")]
    JsonParsingError(#[from] serde_json::Error),

    #[error("Model response missing or has unexpected format: {0}")]
    ResponseParsingError(String),

    #[error("Agent reached maximum iterations ({limit}) without completing task")]
    MaxIterationsReached { limit: usize },

    #[error("Tool execution failed: {0}")]
    ToolError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("Internal Agent Error: {0}")]
    InternalError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AgentInput {
    #[schemars(
        description = "The question/request to be processed by the agent. As parrent agent you should provide complete context and all necessary information required for task completion, including any relevant background, constraints, or specific requirements. Questions should be clear, detailed and actionable."
    )]
    pub question: String,
}

#[async_trait]
pub trait BaseAgent: Clone + Send + Sync + 'static {
    fn openrouter(&self) -> &OpenRouter;
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
    /// Returns ApiRequestError on failure.
    async fn once(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, ApiRequestError> {
        let mut combined_messages = Messages::new();

        // Add system instructions if they exist
        if let Some(instructions) = self.instructions() {
            if !instructions.is_empty() {
                combined_messages.push(Message::system(instructions));
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages.into());

        // Build and send the request
        self.openrouter()
            .chat_completion()
            .model(self.model())
            .messages(combined_messages) // Use the combined messages
            .maybe_max_tokens(self.max_tokens())
            .maybe_temperature(self.temperature())
            .maybe_top_p(self.top_p())
            .maybe_stop(self.stop_sequences().cloned())
            .maybe_tools(self.tools().cloned())
            .build()
            .send()
            .await
    }
    fn stream_once(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Pin<
        Box<
            dyn Stream<Item = Result<crate::response::ChatCompletionChunk, ApiRequestError>> + Send,
        >,
    > {
        let mut combined_messages = Messages::new();

        // Add system instructions if they exist
        if let Some(instructions) = self.instructions() {
            if !instructions.is_empty() {
                combined_messages.push(Message::system(instructions));
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages.into());

        // Build and send the stream request
        Box::pin(
            self.openrouter()
                .chat_completion()
                .model(self.model())
                .messages(combined_messages) // Use the combined messages
                .maybe_max_tokens(self.max_tokens())
                .maybe_temperature(self.temperature())
                .maybe_top_p(self.top_p())
                .maybe_stop(self.stop_sequences().cloned())
                .maybe_tools(self.tools().cloned())
                .build()
                .stream(),
        )
    }
}

#[async_trait]
pub trait Agent: BaseAgent {
    /// Runs the agent, potentially multiple times if tools are involved,
    /// until a final ChatCompletionResponse is obtained or max iterations are reached.
    /// Returns AgentError on failure.
    async fn run(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, AgentError> {
        // Prepare initial messages including system instructions
        let mut agent_messages = Messages::default();
        if let Some(instructions) = self.instructions() {
            agent_messages.push(SystemMessage::from(vec![instructions]));
        }
        agent_messages.extend(messages.into());

        // Loop for handling tool interactions or getting a final response
        for _ in 0..self.max_iterations() {
            // Make an API call
            let resp = self.once(agent_messages.clone()).await?; // Returns ApiRequestError, converted by `?` into AgentError
            let agent_message = Message::from(resp.clone());
            agent_messages.push(agent_message);

            // Check if the response contains tool calls
            if let (Some(tools), Some(tool_calls)) = (self.tools(), resp.tool_calls()) {
                // If there are tool calls, invoke them and add results to messages
                for tool_call in tool_calls {
                    let msgs = tools.invoke(&tool_call).await;
                    agent_messages.extend(msgs);
                }
            } else {
                // If no tool calls, the model intends to provide the final answer.
                return Ok(resp);
            }
        }

        // If the loop completes without returning (meaning max_iterations was reached
        // because every iteration resulted in tool calls), return an error.
        Err(AgentError::MaxIterationsReached {
            limit: self.max_iterations(),
        })
    }

    /// Runs the agent, streaming events like text chunks, tool calls, and results.
    ///
    /// This method provides a detailed view into the agent's execution lifecycle.
    /// It streams AgentEvent variants as they occur, including:
    /// - AgentStart: When the agent begins execution
    /// - TextChunk: Individual text chunks from the model
    /// - ToolCallRequested: When the model requests a tool call
    /// - ToolResult: When a tool returns a result
    /// - StreamEnd: When a model turn completes
    /// - AgentFinish: When the agent successfully completes its task
    /// - AgentError: When an error occurs during execution
    ///
    /// Returns a pinned, boxed stream of `Result<AgentEvent, AgentError>`.
    /// The stream yields events until the agent finishes or an error occurs.
    fn run_events(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>> + Send + 'static>> {
        let mut agent_messages = Messages::default();
        if let Some(instructions) = self.instructions() {
            agent_messages.push(SystemMessage::from(vec![instructions]));
        }
        agent_messages.extend(messages.into());

        let max_iter = self.max_iterations();

        let cloned_self = self.clone();
        Box::pin(try_stream! {
            yield AgentEvent::AgentStart;

            let mut current_messages = agent_messages;
            let mut iteration = 0;

            loop {
                if iteration >= max_iter {
                    // Use the serializable error for yielding event
                    let serializable_error = AgentErrorSerializable::from(&AgentError::MaxIterationsReached { limit: max_iter });
                    yield AgentEvent::AgentError { error: serializable_error };
                    // Propagate the original error to stop the stream correctly via try_stream!
                    Err(AgentError::MaxIterationsReached { limit: max_iter })?;
                }
                iteration += 1;

                // --- LLM Interaction ---
                let mut api_stream = cloned_self.stream_once(current_messages.clone());

                let mut accumulated_content = String::new();
                let mut partial_tool_calls = PartialToolCallsAggregator::new();
                let mut final_chunk_usage: Option<Usage> = None;
                let mut finish_reason = None;

                while let Some(chunk_result) = api_stream.next().await {
                    let chunk = chunk_result?; // Propagate API errors
                    // println!("\n\nChunk: \n{:?}\n", chunk);

                    // Process choices in the chunk (usually just one)
                    for choice in &chunk.choices {
                        if let Some(ref delta_content) = choice.delta.content {
                            if !delta_content.is_empty() {
                                accumulated_content.push_str(delta_content);
                                yield AgentEvent::TextChunk { delta: delta_content.clone() };
                            }
                        }
                        if let Some(ref delta_tool_calls) = choice.delta.tool_calls {
                            for delta_call in delta_tool_calls {
                                // Add the delta to our tool calls manager
                                partial_tool_calls.add_delta(delta_call);
                            }
                        }
                        if let Some(ref reason) = choice.finish_reason {
                            finish_reason = Some(reason.clone());
                        }
                    }
                    final_chunk_usage = chunk.usage.clone(); // Capture usage from the last chunk
                }

                yield AgentEvent::StreamEnd { usage: final_chunk_usage };

                // --- Finalize Tool Calls for this turn ---
                let finalized_tool_calls = if finish_reason == Some(FinishReason::ToolCalls) || !partial_tool_calls.is_empty() {
                    // Convert partial calls into final ToolCall objects
                    partial_tool_calls.finalize()
                } else {
                    Vec::new()
                };

                // --- Process Accumulated Response ---
                let content = if accumulated_content.is_empty() {
                    None
                } else {
                    Some(accumulated_content)
                };

                let final_tool_calls_option = if finalized_tool_calls.is_empty() {
                    None
                } else {
                    Some(finalized_tool_calls.clone())
                };

                // println!("\n\nFinal Tool Calls: \n{:?}\n\n", final_tool_calls_option);

                let assistant_message = Message::assistant_with_tool_calls(content, final_tool_calls_option.clone());
                current_messages.push(assistant_message.clone());

                // --- Tool Call Handling ---
                if let (Some(tools), Some(calls_to_invoke)) = (cloned_self.tools(), final_tool_calls_option.as_ref()) {
                    if !calls_to_invoke.is_empty() {
                        let mut tool_results = Messages::default();

                        // Yield ToolCallRequested events for all finalized tool calls
                        for tool_call in calls_to_invoke.iter() {
                            yield AgentEvent::ToolCallRequested { tool_call: tool_call.clone() };
                        }

                        // Process tool calls sequentially for now - simpler
                        for tool_call in calls_to_invoke.iter() {
                            // Invoke the tool
                            let result_messages = tools.invoke(tool_call).await;

                            // Yield each resulting message as a ToolResult event
                            for msg in result_messages.0.iter() {
                                if let Message::Tool(tool_msg) = msg {
                                    yield AgentEvent::ToolResult { message: tool_msg.clone() };
                                }
                            }

                            // Add to tool results for history
                            tool_results.extend(result_messages);
                        }

                        // Add tool results to history for the next iteration
                        current_messages.extend(tool_results);
                        // Continue to the next loop iteration
                        continue;
                    }
                }

                // --- No Tool Calls or Agent decides to finish ---
                yield AgentEvent::AgentFinish;
                break; // Exit the loop and finish the stream
            }
        })
    }
}

// pub trait AgentArcExt: Clone {
//     async fn run_events_shared(
//         &self,
//         messages: impl Into<Messages> + Send,
//     ) -> Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>> + Send + 'static>>;
// }

#[async_trait]
pub trait TypedAgent: BaseAgent {
    type Output: JsonSchema + Serialize + DeserializeOwned;
    /// Makes a single call to the chat completion API, requesting a structured JSON response
    /// matching the type `T`. Parses the response and returns an instance of `T`.
    /// Returns AgentError on failure (API error, JSON parsing error, response format error).
    async fn once_typed(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError> {
        let resp = self
            .openrouter()
            .chat_completion()
            .model(self.model())
            .messages(messages.into())
            .response_format::<Self::Output>() // Request JSON format matching T
            .maybe_max_tokens(self.max_tokens())
            .maybe_temperature(self.temperature())
            .maybe_top_p(self.top_p())
            .maybe_stop(self.stop_sequences().cloned())
            // Note: Tools are generally not recommended with forced JSON output,
            // but we include them if the agent provides them. The model might ignore them.
            .maybe_tools(self.tools().cloned())
            .build()
            .send()
            .await?; // Converts ApiRequestError to AgentError

        // Extract the response content
        let json_str = resp
            .choices
            .first()
            .and_then(|choice| choice.message.content.first())
            .map(|content| content.to_string())
            .ok_or_else(|| {
                AgentError::ResponseParsingError(
                    "Model response missing expected content structure".to_string(),
                )
            })?;
        let json_value: Value = serde_json::from_str(&json_str).unwrap();
        println!(
            "json_value:\n{}",
            serde_json::to_string_pretty(&json_value).unwrap()
        );

        // Parse the JSON string into type T
        serde_json::from_str::<Self::Output>(&json_str).map_err(AgentError::from)
        // Converts serde_json::Error to AgentError
    }

    /// Runs the agent potentially multiple times, handling tool calls, until a structured
    /// response `T` is obtained or the maximum number of iterations is reached.
    /// Returns AgentError on failure.
    async fn run_typed(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError> {
        // Prepare initial messages including system instructions
        let mut combined_messages = Messages::new();
        if let Some(instructions) = self.instructions() {
            // Ensure instructions are not empty before adding
            if !instructions.is_empty() {
                combined_messages.push(SystemMessage::from(vec![instructions]));
            }
        }
        combined_messages.extend(messages.into());

        // If the agent doesn't use tools, make a single structured call using the combined messages.
        if self.tools().is_none() {
            // once_structured expects the full message list including instructions
            return self.once_typed(combined_messages).await;
        }

        // Loop for handling tool interactions
        for i in 0..self.max_iterations() {
            // Make a regular API call (allows for tool calls) using the current combined messages
            let resp = self.once(combined_messages.clone()).await?; // Returns AgentError
            let agent_message = Message::from(resp.clone());
            combined_messages.push(agent_message); // Add assistant's response to history

            // Check if the response contains tool calls
            if let (Some(tools), Some(tool_calls)) = (self.tools(), resp.tool_calls()) {
                // If there are tool calls, invoke them and add results to message history
                for tool_call in tool_calls {
                    let msgs = tools.invoke(&tool_call).await;
                    combined_messages.extend(msgs); // Add tool results to history
                }
                // Continue the loop for the next interaction
            } else {
                // If no tool calls, the model *should* have provided the final structured answer.
                // *However*, the previous call (`self.once`) didn't enforce the structure.
                // We need one final call that *does* enforce the JSON structure, using the
                // complete conversation history accumulated so far (including instructions).
                // Check if this is the *last* iteration. If so, we *must* try the structured call.
                // Otherwise, we could assume the model *intended* a final answer and try the structured call.
                // For simplicity and robustness, always try the structured call when no tool calls are present.
                return self.once_typed(combined_messages).await;
            }

            // Check iteration count *after* potential tool call processing
            // to ensure max_iterations limits the number of LLM calls.
            if i == self.max_iterations() - 1 && resp.tool_calls().is_some() {
                // If we just processed tool calls on the last allowed iteration,
                // we need one final attempt to get the structured response.
                // This requires one more LLM call than max_iterations strictly allows,
                // but it's necessary after tool use on the final iteration.
                // Alternatively, we could error out here, but trying one last time seems better.
                // Re-evaluate this logic if it causes issues. A simpler approach might be
                // to error immediately if max_iterations is reached and the last response had tool calls.
                // Let's stick to the original logic for now: try one last structured call.
                // Update: Changed logic slightly - the final attempt happens *outside* the loop
                // if max_iterations is reached without a non-tool-call response.
            }
        }

        // If the loop completes without returning (meaning max_iterations was reached,
        // and the last response likely involved tool calls), make one final attempt
        // to get the structured output using the full history.
        // This handles the case where the agent uses tools right up to the iteration limit.
        // Note: This final call might exceed the intended number of LLM calls by one.
        // Consider returning MaxIterationsReached if the loop finishes without a structured response attempt.
        // Let's return the error instead of making an extra call.
        Err(AgentError::MaxIterationsReached {
            limit: self.max_iterations(),
        })
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
    async fn once_any(&self, messages: Messages) -> Result<Value, AgentError>;
    async fn run_any(&self, messages: Messages) -> Result<Value, AgentError>;
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

    async fn once_any(&self, messages: Messages) -> Result<Value, AgentError> {
        let resp = self.once(messages).await?; // Returns AgentError
        serde_json::to_value(resp).map_err(AgentError::JsonParsingError)
    }

    async fn run_any(&self, messages: Messages) -> Result<Value, AgentError> {
        // Use the Agent::run method which handles iterations and tools
        let resp = self.run(messages).await?; // Returns AgentError
        serde_json::to_value(resp).map_err(AgentError::JsonParsingError)
    }
}

#[derive(Debug, Clone, Builder)]
pub struct SimpleAgent {
    #[builder(field)]
    pub tools: Option<ToolBox>,
    pub openrouter: OpenRouter,
    #[builder(into)]
    pub description: Option<String>,
    #[builder(into)]
    pub instructions: Option<String>,
    #[builder(into)]
    pub model: String,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_iterations: Option<usize>,
}

impl<S: simple_agent_builder::State> SimpleAgentBuilder<S> {
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        match &mut self.tools {
            Some(toolbox) => {
                toolbox.add(tool);
            }
            None => {
                let toolbox = ToolBox::builder();
                self.tools = Some(toolbox.tool(tool).build());
            }
        }
        self
    }
}

#[async_trait]
impl BaseAgent for SimpleAgent {
    fn openrouter(&self) -> &OpenRouter {
        &self.openrouter
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    fn stop_sequences(&self) -> Option<&Vec<String>> {
        self.stop_sequences.as_ref()
    }

    fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    fn tools(&self) -> Option<&ToolBox> {
        self.tools.as_ref()
    }

    fn max_iterations(&self) -> usize {
        self.max_iterations.unwrap_or(12)
    }
}

// Implement Agent for SimpleAgent
#[async_trait]
impl Agent for SimpleAgent {}

#[derive(Debug, Clone, Builder)]
pub struct SimpleTypedAgent<T: JsonSchema + Serialize + DeserializeOwned> {
    pub openrouter: OpenRouter,
    #[builder(into)]
    pub name: Option<String>,
    #[builder(into)]
    pub description: Option<String>,
    #[builder(into)]
    pub instructions: Option<String>,
    #[builder(into)]
    pub model: String,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub tools: Option<ToolBox>,
    pub max_iterations: Option<usize>,
    #[builder(default)]
    pub output: PhantomData<T>,
}

#[async_trait]
impl<T> BaseAgent for SimpleTypedAgent<T>
where
    T: Clone + JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn openrouter(&self) -> &OpenRouter {
        &self.openrouter
    }

    fn agent_name(&self) -> &str {
        self.name
            .as_deref()
            .unwrap_or_else(|| std::any::type_name::<Self>().split("::").last().unwrap())
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    fn stop_sequences(&self) -> Option<&Vec<String>> {
        self.stop_sequences.as_ref()
    }

    fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    fn tools(&self) -> Option<&ToolBox> {
        self.tools.as_ref()
    }

    fn max_iterations(&self) -> usize {
        self.max_iterations.unwrap_or(12)
    }
}

// Implement TypedAgent for SimpleTypedAgent
#[async_trait]
impl<T> TypedAgent for SimpleTypedAgent<T>
where
    T: Clone + JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    type Output = T;
}

#[cfg(test)]
mod tests {
    use schemars::{generate::SchemaSettings, schema_for};
    use tokio_stream::StreamExt;

    use crate::{
        message::{ToolMessage, UserMessage},
        tool::Tool,
    };

    use super::*;

    #[derive(Debug, JsonSchema, Serialize, Deserialize)]
    struct CalculatorToolInput {
        a: f64,
        b: f64,
    }
    #[derive(Default)]
    struct CalculatorTool {}

    #[async_trait]
    impl Tool for CalculatorTool {
        type Input = CalculatorToolInput;

        type Error = String;

        fn name(&self) -> &str {
            "CalculatorTool"
        }
        async fn invoke(
            &self,
            tool_call_id: &str,
            input: Self::Input,
        ) -> Result<Messages, Self::Error> {
            Ok(ToolMessage::new(tool_call_id, (input.a + input.b).to_string()).into())
        }
    }

    #[derive(Debug, Clone)]
    struct CalculatorAgent {
        openrouter: OpenRouter,
        tools: ToolBox,
    }

    impl BaseAgent for CalculatorAgent {
        fn openrouter(&self) -> &OpenRouter {
            &self.openrouter
        }
        fn agent_name(&self) -> &str {
            "CalculatorAgent"
        }
        fn description(&self) -> Option<&str> {
            Some("An agent that performs calculator operations")
        }
        fn instructions(&self) -> Option<String> {
            Some("You are a calculator. Evaluate the mathematical expression.".to_string())
        }
        fn model(&self) -> &str {
            "google/gemini-2.0-flash-001"
        }
        fn tools(&self) -> Option<&ToolBox> {
            Some(&self.tools)
        }
    }
    // Implement the Agent trait for CalculatorAgent
    impl Agent for CalculatorAgent {}

    #[derive(Debug, JsonSchema, Serialize, Deserialize)]
    pub struct CalculatorResult {
        pub value: f64,
    }

    impl TypedAgent for CalculatorAgent {
        type Output = CalculatorResult;
    }

    #[derive(Debug, Clone)]
    struct OrchestratorAgent {
        openrouter: OpenRouter,
        tools: ToolBox,
    }

    impl BaseAgent for OrchestratorAgent {
        fn agent_name(&self) -> &str {
            "Orchestrator"
        }
        fn description(&self) -> Option<&str> {
            Some("An orchestrator that manages agents, always forwarding questions to the appropriate agent and summarizing responses.")
        }
        fn instructions(&self) -> Option<String> {
            Some("You are an orchestrator. Use the available tools to answer the user's question. If a calculation is needed, use the CalculatorAgent tool.".to_string())
        }
        fn model(&self) -> &str {
            // Use a model known to be good at tool use
            "openai/gpt-4o-mini"
        }
        fn tools(&self) -> Option<&ToolBox> {
            Some(&self.tools)
        }

        fn openrouter(&self) ->  &OpenRouter {
            &self.openrouter
        }
    }
    // Implement the Agent trait for OrchestratorAgent
    impl Agent for OrchestratorAgent {}

    #[derive(JsonSchema, Serialize, Deserialize)]
    pub struct CalculatorAgentQuestion {
        pub question: String,
    }

    #[async_trait]
    impl Tool for CalculatorAgent {
        type Input = CalculatorAgentQuestion;

        type Error = String;

        fn name(&self) -> &str {
            self.agent_name()
        }
        async fn invoke(
            &self,
            tool_call_id: &str,
            input: Self::Input,
        ) -> Result<Messages, Self::Error> {
            let question = input.question;
            let response = self
                .once_typed(UserMessage::new(vec![question]))
                .await
                .unwrap();
            Ok(ToolMessage::new(tool_call_id, serde_json::to_string(&response).unwrap()).into())
        }
    }

    // Helper to create OpenRouter client from env var
    fn create_openrouter() -> OpenRouter {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY environment variable not set");
        OpenRouter::builder().api_key(api_key).build()
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_agent() {
        let openrouter = create_openrouter();
        let tools = ToolBox::builder().tool(CalculatorTool::default()).build();
        let agent = CalculatorAgent { openrouter, tools };
        let message = UserMessage::new(vec!["What is 2 + 2?"]);
        // Use once for a direct API call
        let resp = agent.run(message).await;
        match resp {
            Ok(r) => {
                println!("Simple Agent Response: {:?}", r);
                assert!(!r.choices.is_empty());
            }
            Err(e) => panic!("Simple Agent test failed: {:?}", e),
        }
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_typed_agent_once() {
        let openrouter = create_openrouter();
        let tools = ToolBox::builder().tool(CalculatorTool::default()).build();
        let agent = CalculatorAgent { openrouter, tools };
        let message = UserMessage::new(vec!["What is 2 + 2?"]);
        // Use once_structured for a direct API call expecting a specific JSON structure
        let resp = agent.once_typed(message).await;
        match resp {
            Ok(r) => {
                // 'r' is the parsed CalculatorResult
                println!("Simple Typed Agent Response: {:?}", r);
                // Check the value field of the result
                // Use a tolerance for floating point comparison
                assert!(
                    (r.value - 4.0).abs() < f64::EPSILON,
                    "Expected value to be 4.0, but got {}",
                    r.value
                );
            }
            Err(e) => panic!("Simple Typed Agent test failed: {:?}", e),
        }
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_agent_run() {
        let openrouter = create_openrouter();
        let tools = ToolBox::builder().tool(CalculatorTool::default()).build();
        let agent = CalculatorAgent { openrouter, tools };
        let message = UserMessage::new(vec!["What is 5 * 8?"]);
        // Use run (even though this agent has no tools, it tests the Agent trait path)
        let resp = agent.run(message).await;
        match resp {
            Ok(r) => {
                println!("Simple Agent Run Response: {:?}", r);
                assert!(!r.choices.is_empty());
                // Basic check if the response might contain the answer
                assert!(r.choices[0].message.content[0].to_string().contains("40"));
            }
            Err(e) => panic!("Simple Agent run test failed: {:?}", e),
        }
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_agent_run_events() {
        let openrouter = create_openrouter();
        let tools = ToolBox::builder().tool(CalculatorTool::default()).build();
        let agent = CalculatorAgent { openrouter, tools };
        let message = UserMessage::new(vec!["What is 5 + 8? Use calculator tool"]);
        // Use run (even though this agent has no tools, it tests the Agent trait path)
        let mut stream = agent.run_events(message);
        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => println!("Event: {:?}", event),
                Err(e) => panic!("Event stream error: {:?}", e),
            }
        }
    }

    // Unit tests for tool call handling
    #[test]
    fn test_partial_tool_call_merge_and_finalize() {
        // Create a partial tool call
        let mut partial = PartialToolCall::default();

        // Create some fragments
        let fragment1 = ToolCall {
            index: Some(0),
            id: Some("call_123".to_string()),
            type_field: "function".to_string(),
            function: crate::response::FunctionCall {
                name: Some("TestTool".to_string()),
                arguments: "{".to_string(), // Start of JSON
            },
        };

        let fragment2 = ToolCall {
            index: Some(0),
            id: Some("".to_string()), // Empty as it's a continuation
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"a\":5,".to_string(), // Middle of JSON
            },
        };

        let fragment3 = ToolCall {
            index: Some(0),
            id: Some("".to_string()),
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"b\":7}".to_string(), // End of JSON
            },
        };

        // Merge the fragments
        partial.merge(&fragment1);
        partial.merge(&fragment2);
        partial.merge(&fragment3);

        // Finalize and check the result
        let finalized = partial.finalize().expect("Should finalize successfully");

        assert_eq!(finalized.id, Some("call_123".to_string()));
        assert_eq!(finalized.function.name, Some("TestTool".to_string()));
        assert_eq!(finalized.function.arguments, "{\"a\":5,\"b\":7}");
    }

    #[test]
    fn test_partial_tool_calls_manager() {
        // Create the PartialToolCalls manager
        let mut manager = PartialToolCallsAggregator::new();

        // Create tool call deltas for two different tool calls
        let delta1_first = ToolCall {
            index: Some(0),
            id: Some("call_123".to_string()),
            type_field: "function".to_string(),
            function: crate::response::FunctionCall {
                name: Some("ToolOne".to_string()),
                arguments: "{\"param\":".to_string(),
            },
        };

        let delta1_second = ToolCall {
            index: Some(0),
            id: Some("".to_string()),
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"value\"}".to_string(),
            },
        };

        let delta2_first = ToolCall {
            index: Some(1),
            id: Some("call_456".to_string()),
            type_field: "function".to_string(),
            function: crate::response::FunctionCall {
                name: Some("ToolTwo".to_string()),
                arguments: "{\"x\":10,".to_string(),
            },
        };

        let delta2_second = ToolCall {
            index: Some(1),
            id: Some("".to_string()),
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"y\":20}".to_string(),
            },
        };

        // Add all deltas to the manager
        manager.add_delta(&delta1_first);
        manager.add_delta(&delta2_first);
        manager.add_delta(&delta1_second);
        manager.add_delta(&delta2_second);

        // Finalize and check results
        let finalized = manager.finalize();

        // We should have two tool calls
        assert_eq!(finalized.len(), 2, "Should have two finalized tool calls");

        // They should be sorted by index
        assert_eq!(finalized[0].index, Some(0));
        assert_eq!(finalized[1].index, Some(1));

        // Check first tool call
        assert_eq!(finalized[0].id, Some("call_123".to_string()));
        assert_eq!(finalized[0].function.name, Some("ToolOne".to_string()));
        assert_eq!(finalized[0].function.arguments, "{\"param\":\"value\"}");

        // Check second tool call
        assert_eq!(finalized[1].id, Some("call_456".to_string()));
        assert_eq!(finalized[1].function.name, Some("ToolTwo".to_string()));
        assert_eq!(finalized[1].function.arguments, "{\"x\":10,\"y\":20}");
    }

    #[tokio::test]
    // #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_agent_run_events() {
        let openrouter = create_openrouter();
        let tools = ToolBox::builder().tool(CalculatorTool::default()).build();
        let agent = Arc::new(CalculatorAgent { openrouter, tools });
        let message = UserMessage::new(vec!["What is 5 + 7?"]);

        // Use run_events to get a detailed stream of agent execution events
        let mut event_stream = agent.run_events(message.clone());

        // Collect all events
        let mut events = Vec::new();
        while let Some(event_result) = event_stream.next().await {
            match event_result {
                Ok(event) => {
                    println!("Received event: {:?}", event);
                    events.push(event);
                }
                Err(e) => panic!("Agent event stream error: {:?}", e),
            }
        }

        // Verify that we received the expected event types in the correct sequence
        assert!(!events.is_empty(), "Should receive at least some events");

        // First event should be AgentStart
        assert!(
            matches!(events[0], AgentEvent::AgentStart),
            "First event should be AgentStart"
        );

        // Last event should be AgentFinish
        assert!(
            matches!(events.last().unwrap(), AgentEvent::AgentFinish),
            "Last event should be AgentFinish"
        );

        // Verify we have the expected event types
        let has_text_chunks = events
            .iter()
            .any(|e| matches!(e, AgentEvent::TextChunk { .. }));
        assert!(has_text_chunks, "Should have at least one TextChunk event");

        let has_stream_end = events
            .iter()
            .any(|e| matches!(e, AgentEvent::StreamEnd { .. }));
        assert!(has_stream_end, "Should have at least one StreamEnd event");

        // Tool events are optional depending on whether the agent needed to use tools
        let has_tool_calls = events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolCallRequested { .. }));
        let has_tool_results = events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolResult { .. }));

        // If a tool was called, we should also have its result
        if has_tool_calls {
            assert!(
                has_tool_results,
                "If there are tool calls, there should be tool results"
            );

            // Find ToolResult events to check if they contain the expected result (12)
            let tool_results: Vec<_> = events
                .iter()
                .filter_map(|e| {
                    if let AgentEvent::ToolResult { message } = e {
                        Some(message.content.clone())
                    } else {
                        None
                    }
                })
                .collect();

            // At least one tool result should contain the answer (12)
            let contains_answer = tool_results.iter().any(|content| content.contains("12"));
            assert!(
                contains_answer,
                "Tool results should contain the expected answer (12)"
            );
        }
    }

    // Test requires implementing Tool for Agent first (commented out above)
    // #[tokio::test]
    // #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    // async fn test_orchestrator_with_tool() {
    //     let openrouter = create_openrouter();
    //     let tools = ToolBox::builder().tool(CalculatorTool::default()).build();
    //     let calculator_agent = CalculatorAgent { openrouter, tools };

    //     // Ensure CalculatorAgent implements Tool (requires uncommenting the impl Tool for T: Agent block)
    //     let tools = ToolBox::builder().tool(calculator_agent).build();

    //     let orchestrator = OrchestratorAgent { openrouter, tools };
    //     let message = UserMessage::new(vec!["Please calculate 123 + 456 for me."]);

    //     // Use run for the orchestrator to potentially use tools
    //     let resp = orchestrator.run(message).await;

    //     match resp {
    //         Ok(r) => {
    //             println!("Orchestrator Response: {:?}", r);
    //             assert!(!r.choices.is_empty());
    //             // Check if the final response contains the calculated sum (579)
    //             assert!(r.choices[0]
    //                 .message
    //                 .content_text()
    //                 .unwrap_or_default()
    //                 .contains("579"));
    //         }
    //         Err(e) => panic!("Orchestrator test failed: {:?}", e),
    //     }
    // }
}
