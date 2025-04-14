use std::pin::Pin;

use async_stream::try_stream;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::{Stream, StreamExt};

use super::error::AgentError;
use super::events::{AgentErrorSerializable, AgentEvent};
use super::tool_call_aggregator::PartialToolCallsAggregator;
use crate::{
    message::{AssistantMessage, Message, Messages, SystemMessage, ToolMessage},
    response::{ChatCompletionResponse, FinishReason, ToolCall, Usage},
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
        let mut combined_messages = Messages::default();

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
        let mut combined_messages = Messages::default();

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
            agent_messages.push(SystemMessage::text(instructions));
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
            if !instructions.is_empty() {
                agent_messages.push(SystemMessage::text(instructions));
            }
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
                        if let Some(reason) = choice.finish_reason {
                            finish_reason = Some(reason);
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
                // Create Content object from accumulated text
                let message_content = if accumulated_content.is_empty() {
                    crate::message::Content::default() // Empty Vec<ContentPart>
                } else {
                    crate::message::Content::from(accumulated_content) // Convert String to Content(vec![TextPart])
                };

                let final_tool_calls_option = if finalized_tool_calls.is_empty() {
                    None
                } else {
                    Some(finalized_tool_calls) // Clone here for potential later use
                };

                // Create AssistantMessage using its constructor with the content
                // and then assign the finalized tool calls
                let mut assistant_message = AssistantMessage::new(message_content); // Use the Content object
                assistant_message.tool_calls = final_tool_calls_option.clone();     // Assign finalized tool calls

                current_messages.push(assistant_message); // Push the complete assistant message to history

                // --- Tool Call Handling ---
                // Use the Option directly here without unwrapping
                if let (Some(tools), Some(calls_to_invoke)) = (cloned_self.tools(), final_tool_calls_option.as_ref()) {
                     // Check if the vector inside the Option is non-empty
                    if !calls_to_invoke.is_empty() {
                        let mut tool_results = Messages::default();

                        // Yield ToolCallRequested events for all finalized tool calls
                        for tool_call in calls_to_invoke.iter() {
                            yield AgentEvent::ToolCallRequested { tool_call: tool_call.clone() };
                        }

                        // Process tool calls sequentially for now - simpler
                        for tool_call in calls_to_invoke.iter() {
                            // Invoke the tool - ToolBox::invoke handles internal errors and returns Messages
                            let result_messages = tools.invoke(tool_call).await; // Removed .map_err and ?

                            // Check if the returned message is actually an error message from the tool invocation
                            // This requires checking the content of the ToolMessage within result_messages
                            // Note: ToolBox::invoke wraps errors from Tool::invoke and deserialization errors
                            // into a ToolMessage with the error text as content.
                            // We might want to propagate this as an AgentError instead of just a ToolResult event.
                            // For now, just yield the result as is.

                            // Yield each resulting message as a ToolResult event
                            for msg in result_messages.0.iter() {
                                if let Message::Tool(tool_msg) = msg {
                                    yield AgentEvent::ToolResult { message: tool_msg.clone() };
                                }
                                // Consider adding logic here to detect tool invocation errors and yield AgentEvent::AgentError if needed
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
                // Check finish reason *after* processing potential tool calls
                if finish_reason != Some(FinishReason::ToolCalls) {
                     yield AgentEvent::AgentFinish;
                    break; // Exit the loop and finish the stream
                }
                // If finish_reason was ToolCalls but we didn't invoke any (e.g., bad tool call format), loop again
            }
        })
    }
}

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
        let mut combined_messages = Messages::default();

        // Add system instructions if they exist
        if let Some(instructions) = self.instructions() {
            if !instructions.is_empty() {
                combined_messages.push(Message::system(instructions));
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages.into());

        let resp = self
            .openrouter()
            .chat_completion()
            .model(self.model())
            .messages(combined_messages) // Use the combined messages
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
        let mut combined_messages = Messages::default();
        if let Some(instructions) = self.instructions() {
            // Ensure instructions are not empty before adding
            if !instructions.is_empty() {
                combined_messages.push(Message::system(instructions));
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
