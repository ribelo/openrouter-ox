mod config;
pub mod context;
mod error;
mod events;
mod simple_agent;
mod tool_call_aggregator;
mod typed_agent;

use std::{any::Any, marker::PhantomData, pin::Pin, sync::Arc};

pub use crate::OpenRouter;
use async_stream::try_stream;
use bon::Builder;
use context::AgentContext;
use derive_more::Deref;
pub use error::AgentError;
pub use events::{AgentErrorSerializable, AgentEvent};
use futures::stream::BoxStream;
use futures_util::{Stream, TryStreamExt};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use tokio_stream::StreamExt;
use tool_call_aggregator::PartialToolCallsAggregator;

use crate::{
    message::{AssistantMessage, Message, Messages},
    response::{ChatCompletionChunk, ChatCompletionResponse, FinishReason, Usage},
    tool::{Tool, ToolBox},
};

#[derive(Clone, Builder)]
pub struct Agent<S: Clone + Send + Sync + 'static> {
    #[builder(into, start_fn)]
    pub state: Arc<S>,
    #[builder(field)]
    pub context: AgentContext,
    #[builder(field)]
    pub instruction: Option<Arc<dyn Fn(&AgentContext) -> String + Send + Sync>>,
    pub(crate) openrouter: OpenRouter,
    #[builder(into)]
    pub name: String,
    #[builder(into)]
    pub description: Option<String>,
    #[builder(into)]
    pub model: String,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    #[builder(default = 12)]
    pub max_iterations: u32,
}

impl<S: Clone + Send + Sync + 'static, BS: agent_builder::State> AgentBuilder<S, BS> {
    pub fn tools(mut self, toolbox: ToolBox) -> Self {
        self.context.tools = Some(toolbox);
        self
    }
    pub fn tool<T: Tool>(mut self, tool: T) -> Self {
        self.context.tools.get_or_insert_default().add(tool);
        self
    }
    pub fn dependency<T: Any + Send + Sync>(self, value: T) -> Self {
        self.context.insert(value);
        self
    }
    pub fn instruction<F>(mut self, f: F) -> Self
    where
        F: Fn(&AgentContext) -> String + Send + Sync + 'static,
    {
        self.instruction = Some(Arc::new(f));
        self
    }
}

impl<S: Clone + Send + Sync + 'static, BS: agent_builder::IsComplete> AgentBuilder<S, BS> {
    pub fn typed<O: JsonSchema + DeserializeOwned>(self) -> TypedAgent<S, O> {
        TypedAgent {
            agent: self.build(),
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Deref)]
pub struct TypedAgent<S: Clone + Send + Sync + 'static, O: JsonSchema + DeserializeOwned> {
    #[deref]
    agent: Agent<S>,
    _phantom: PhantomData<O>,
}

impl<S: Clone + Send + Sync> Agent<S> {
    pub async fn generate(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, AgentError> {
        let messages = messages.into();
        let mut combined_messages = Messages::default();

        // Check if combined_messages already contains a system message before potentially adding one.
        let has_system_message = messages.iter().any(|m| matches!(m, Message::System(_)));

        // Add system instructions only if they exist, are not empty, and no system message is present yet.
        if !has_system_message {
            if let Some(instruction) = self.instruction.as_deref().map(|f| f(&self.context)) {
                if !instruction.is_empty() {
                    combined_messages.push(Message::system(instruction));
                }
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages);

        // Build and send the request
        self.openrouter
            .chat_completion()
            .messages(combined_messages) // Use the combined messages
            .model(self.model.clone())
            .maybe_max_tokens(self.max_tokens)
            .maybe_temperature(self.temperature)
            .maybe_top_p(self.top_p)
            .maybe_stop(self.stop_sequences.clone())
            .maybe_tools(self.context.tools.clone())
            .build()
            .send()
            .await
            .map_err(|e| AgentError::ApiError(e))
    }

    /// Streams the parts (chunks) of a single `GenerateContentResponse` from one call to the LLM.
    /// Makes a single, streaming call to the underlying model API.
    /// Yields `Result<GenerateContentResponse, AgentError>` items representing the chunks.
    pub fn stream_response(
        & self,
        messages: impl Into<Messages> + Send,
    // ) -> Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AgentError>> + Send>> {
    ) -> BoxStream<'static, Result<ChatCompletionChunk, AgentError>> {
        let messages = messages.into();
        let mut combined_messages = Messages::default();

        // Check if combined_messages already contains a system message before potentially adding one.
        let has_system_message = messages.iter().any(|m| matches!(m, Message::System(_)));

        // Add system instructions only if they exist, are not empty, and no system message is present yet.
        if !has_system_message {
            if let Some(instruction) = self.instruction.as_deref().map(|f| f(&self.context)) {
                if !instruction.is_empty() {
                    combined_messages.push(Message::system(instruction));
                }
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages);

        // Build and send the stream request
        Box::pin(
            self.openrouter
                .chat_completion()
                .messages(combined_messages) // Use the combined messages
                .model(self.model.clone())
                .maybe_max_tokens(self.max_tokens)
                .maybe_temperature(self.temperature)
                .maybe_top_p(self.top_p)
                .maybe_stop(self.stop_sequences.clone())
                .maybe_tools(self.context.tools.clone())
                .build()
                .stream()
                .map_err(|e| AgentError::ApiError(e)),
        )
    }

    /// Executes the agent process, potentially involving multiple turns (LLM calls) and tool calls,
    /// until a final `GenerateContentResponse` is produced or an error occurs.
    /// Returns the final `Result<GenerateContentResponse, AgentError>`.
    pub async fn execute(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, AgentError> {
        let messages = messages.into();
        let mut combined_messages = Messages::default();

        // Check if combined_messages already contains a system message before potentially adding one.
        let has_system_message = messages.iter().any(|m| matches!(m, Message::System(_)));

        // Add system instructions only if they exist, are not empty, and no system message is present yet.
        if !has_system_message {
            if let Some(instruction) = self.instruction.as_deref().map(|f| f(&self.context)) {
                if !instruction.is_empty() {
                    combined_messages.push(Message::system(instruction));
                }
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages);

        // Loop for handling tool interactions or getting a final response
        for _ in 0..self.max_iterations {
            // Make an API call
            let resp = self.generate(combined_messages.clone()).await?; // Returns ApiRequestError, converted by `?` into AgentError
            let agent_message = Message::from(resp.clone());
            combined_messages.push(agent_message);

            // Check if the response contains tool calls
            if let (Some(tools), Some(tool_calls)) =
                (self.context.tools.as_ref(), resp.tool_calls())
            {
                // If there are tool calls, invoke them and add results to messages
                for tool_call in tool_calls {
                    let msgs = tools.invoke(&tool_call).await;
                    combined_messages.extend(msgs);
                }
            } else {
                // If no tool calls, the model intends to provide the final answer.
                return Ok(resp);
            }
        }

        // If the loop completes without returning (meaning max_iterations was reached
        // because every iteration resulted in tool calls), return an error.
        Err(AgentError::MaxIterationsReached {
            limit: self.max_iterations,
        })
    }

    /// Streams events related to the entire agent execution process.
    /// This may include multiple turns, yielding events like `AgentStart`, `AgentResponse` (LLM chunks),
    /// `StreamEnd` (per LLM call), tool calls, `AgentFinish`, or `AgentError`.
    /// Yields `Result<AgentEvent, AgentError>` items.
    pub fn stream_events<'a>(
        &'a self,
        messages: impl Into<Messages> + Send,
    ) -> Pin<Box<dyn Stream<Item = Result<AgentEvent, AgentError>> + Send + 'static>> {
        let messages = messages.into();
        let mut combined_messages = Messages::default();

        // Check if combined_messages already contains a system message before potentially adding one.
        let has_system_message = messages.iter().any(|m| matches!(m, Message::System(_)));

        // Add system instructions only if they exist, are not empty, and no system message is present yet.
        if !has_system_message {
            if let Some(instruction) = self.instruction.as_deref().map(|f| f(&self.context)) {
                if !instruction.is_empty() {
                    combined_messages.push(Message::system(instruction));
                }
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages);

        let max_iter = self.max_iterations;
        let agent_clone = self.clone();

        Box::pin(try_stream! {
            yield AgentEvent::AgentStart;

            let mut current_messages = combined_messages;
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
                let mut api_stream = agent_clone.stream_response(current_messages.clone());

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
                if let (Some(tools), Some(calls_to_invoke)) = (agent_clone.context.tools.as_ref(), final_tool_calls_option) {
                     // Check if the vector inside the Option is non-empty
                    if !calls_to_invoke.is_empty() {
                        let mut tool_results = Messages::default();

                        // Yield ToolCallRequested events for all finalized tool calls
                        for tool_call in calls_to_invoke.iter() {
                            yield AgentEvent::ToolCallRequested { tool_call: tool_call.clone() };
                        }

                        // Process tool calls sequentially for now - simpler
                        for tool_call in calls_to_invoke {
                            // Invoke the tool - ToolBox::invoke handles internal errors and returns Messages
                            let result_messages = tools.invoke(&tool_call).await;
                            // let result_messages = tools.invoke(&tool_call).await; // Removed .map_err and ?

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

impl<S, O> TypedAgent<S, O>
where
    S: Clone + Send + Sync + 'static,
    O: JsonSchema + DeserializeOwned,
{
    /// Gets a single, structured response of type `Output` from one call to the LLM.
    /// Makes a single, direct call to the underlying model API, configured to return JSON
    /// matching the `Output` type's schema. Parses the response into `Output`.
    /// Returns `Result<Self::Output, AgentError>`.
    pub async fn generate_typed(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<O, AgentError> {
        let messages = messages.into();
        let mut combined_messages = Messages::default();

        // Check if combined_messages already contains a system message before potentially adding one.
        let has_system_message = messages.iter().any(|m| matches!(m, Message::System(_)));

        // Add system instructions only if they exist, are not empty, and no system message is present yet.
        if !has_system_message {
            if let Some(instruction) = self.instruction.as_deref().map(|f| f(&self.context)) {
                if !instruction.is_empty() {
                    combined_messages.push(Message::system(instruction));
                }
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages);

        let resp = self
            .openrouter
            .chat_completion()
            .messages(combined_messages) // Use the combined messages
            .response_format::<O>() // Request JSON format matching Output
            .model(self.model.clone())
            .maybe_max_tokens(self.max_tokens)
            .maybe_temperature(self.temperature)
            .maybe_top_p(self.top_p)
            .maybe_stop(self.stop_sequences.clone())
            // Note: Tools are generally not recommended with forced JSON output,
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

        // Parse the JSON string into the Output type
        serde_json::from_str::<O>(&json_str).map_err(AgentError::from)
        // Converts serde_json::Error to AgentError
    }

    /// Executes the agent process, potentially involving multiple turns (LLM calls) and tool calls,
    /// with the final goal of producing a structured response of type `Output`.
    /// If the agent has tools, it uses the multi-turn `execute` method first, then makes a final
    /// call with `get_typed_response` to get the structured output. If no tools are present,
    /// it directly calls `get_typed_response`.
    /// Returns `Result<Self::Output, AgentError>`.
    pub async fn execute_typed(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<O, AgentError> {
        let messages = messages.into();
        let mut combined_messages = Messages::default();

        // Check if combined_messages already contains a system message before potentially adding one.
        let has_system_message = messages.iter().any(|m| matches!(m, Message::System(_)));

        // Add system instructions only if they exist, are not empty, and no system message is present yet.
        if !has_system_message {
            if let Some(instruction) = self.instruction.as_deref().map(|f| f(&self.context)) {
                if !instruction.is_empty() {
                    combined_messages.push(Message::system(instruction));
                }
            }
        }

        // Add the user-provided messages
        combined_messages.extend(messages);

        if self.context.tools.is_none() {
            // No tools, directly ask for the typed response in a single turn.
            self.generate_typed(combined_messages).await
        } else {
            // Has tools, run the multi-turn execution first.
            let response = self.execute(combined_messages.clone()).await?;
            // Take the final response content from the multi-turn execution...
            combined_messages.push(Message::from(response.clone()));
            dbg!(&combined_messages);

            self.generate_typed(combined_messages).await
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio_stream::StreamExt;

    use crate::{
        message::{Messages, ToolMessage, UserMessage},
        // response::ChatCompletionResponse, // Not directly used in corrected tests
        tool::Tool,
        // ApiRequestError, // Replaced by AgentError
    };

    use super::*;

    #[derive(Debug, schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
    struct CalculatorToolInput {
        a: f64,
        b: f64,
    }
    #[derive(Default, Debug, Clone)] // Added Debug, Clone
    struct CalculatorTool {}

    #[async_trait::async_trait]
    impl Tool for CalculatorTool {
        type Input = CalculatorToolInput;

        type Error = String;

        fn name(&self) -> &str {
            "CalculatorTool"
        }
        fn description(&self) -> Option<&str> {
            Some("A tool that adds two numbers.")
        }
        async fn invoke(
            &self,
            tool_call_id: &str,
            input: Self::Input,
        ) -> Result<Messages, Self::Error> {
            Ok(ToolMessage::new(tool_call_id, (input.a + input.b).to_string()).into())
        }
    }

    #[derive(Debug, schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
    pub struct CalculatorResult {
        pub value: f64,
    }

    // Note: TypedAgent is now just a marker trait. The actual methods are on OpenRouterTypedAgent.
    // We don't need an explicit `impl TypedAgent for CalculatorAgent`.

    // #[derive(Debug, Clone)]
    // struct OrchestratorAgent {
    //     // Store config, not the client or tools directly
    // }

    // impl OrchestratorAgent {
    //     fn new() -> Self {
    //         Self {}
    //     }
    // }

    // #[async_trait]
    // impl AgentConfig for OrchestratorAgent {
    //     fn name(&self) -> &str {
    //         "Orchestrator"
    //     }
    //     fn description(&self, _ctx: &AgentContext) -> Option<&str> {
    //         Some("An orchestrator that manages agents, always forwarding questions to the appropriate agent and summarizing responses.")
    //     }
    //     fn instructions(&self, _ctx: &AgentContext) -> Option<String> {
    //         Some("You are an orchestrator. Use the available tools to answer the user's question. If a calculation is needed, use the CalculatorTool tool.".to_string())
    //     }
    //     fn model(&self, _ctx: &AgentContext) -> &str {
    //         // Use a model known to be good at tool use
    //         "openai/gpt-4o-mini"
    //     }

    //     // Implement other optional methods from Agent trait with defaults or specific values
    //     fn max_tokens(&self, _ctx: &AgentContext) -> Option<u32> {
    //         Some(250)
    //     }
    //     fn temperature(&self, _ctx: &AgentContext) -> Option<f64> {
    //         Some(0.5)
    //     }
    //     fn top_p(&self, _ctx: &AgentContext) -> Option<f64> {
    //         None
    //     }
    //     fn stop_sequences(&self, _ctx: &AgentContext) -> Option<&Vec<String>> {
    //         None
    //     }
    //     fn max_iterations(&self, _ctx: &AgentContext) -> u32 {
    //         5 // Allow tool use
    //     }
    // }

    // --- Agent as a Tool ---
    // We need a struct that holds the necessary context to run the agent when invoked as a tool.
    #[derive(Clone)] // Clone is important for ToolBox
    struct CalculatorAgentToolRunner {
        openrouter: OpenRouter,
        // We don't store the agent state itself here, as the Agent trait is stateless.
        // We just need the client to build the executor.
    }

    #[derive(schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
    pub struct CalculatorAgentQuestion {
        pub question: String,
    }

    // #[async_trait::async_trait]
    // impl Tool for CalculatorAgentToolRunner {
    //     type Input = CalculatorAgentQuestion;
    //     type Error = AgentError; // Tool invocation can fail with agent errors

    //     fn name(&self) -> &str {
    //         // We use a fixed name, could potentially get it from the Agent trait
    //         // but that requires an instance, which we don't necessarily have here.
    //         // If we instantiated CalculatorAgent here, we could use Agent::name(&CalculatorAgent::new())
    //         "CalculatorAgentTool"
    //     }
    //     fn description(&self) -> Option<&str> {
    //         Some("Runs the CalculatorAgent to answer a question, expecting a numeric result.")
    //     }

    //     async fn invoke(
    //         &self,
    //         tool_call_id: &str,
    //         input: Self::Input,
    //     ) -> Result<Messages, Self::Error> {
    //         let question = input.question;
    //         let typed_agent_executor = Agent::builder()
    //             .openrouter(self.openrouter.clone())
    //             // Set builder fields directly using methods from the AgentConfig trait
    //             .name("CalculatorAgent")
    //             .description(agent_config.description(&default_ctx).map(|s| s.to_string()))
    //             .instruction(instruction_fn) // Set the instruction function
    //             .model(agent_config.model(&default_ctx).to_string())
    //             .max_tokens(agent_config.max_tokens(&default_ctx))
    //             .temperature(agent_config.temperature(&default_ctx))
    //             .top_p(agent_config.top_p(&default_ctx))
    //             .stop_sequences(agent_config.stop_sequences(&default_ctx).cloned()) // Clone if Some
    //             .max_iterations(agent_config.max_iterations(&default_ctx))
    //             // Add the *basic* calculator tool, so the agent *could* use it if needed,
    //             // though for simple questions it might directly respond with JSON.
    //             .tool(CalculatorTool::default())
    //             .typed::<CalculatorResult>(); // Make it a typed agent executor

    //         // Execute the agent to get a typed response
    //         let response: CalculatorResult = typed_agent_executor
    //             .execute_typed(UserMessage::text(question)) // Use execute_typed for potential tool use
    //             .await?; // Propagate AgentError

    //         // Serialize the successful result to JSON string for the ToolMessage
    //         let response_json = serde_json::to_string(&response).map_err(|e| {
    //             AgentError::ResponseParsingError(format!("Failed to serialize tool result: {}", e))
    //         })?;

    //         Ok(ToolMessage::new(tool_call_id, response_json).into())
    //     }
    // }

    // Helper to create OpenRouter client from env var
    fn create_openrouter() -> crate::OpenRouter {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY environment variable not set");
        crate::OpenRouter::builder().api_key(api_key).build()
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_agent_execute() {
        let openrouter = create_openrouter();
        let agent = Agent::builder(())
            .openrouter(openrouter)
            .name("CalculatorAgent")
            .description("An agent that performs calculator operations")
            .instruction(|_|"You are a calculator. Evaluate the mathematical expression. Use the available tools if necessary.".to_string())
            .model("openai/gpt-4o-mini")
            .max_tokens(100)
            .temperature(0.0)
            .max_iterations(5)
            .tool(CalculatorTool::default()) // Add the tool
            .build();

        let message = UserMessage::text("What is 2 + 2? Use the tool.");
        // Use execute for potential multi-turn interaction
        let resp = agent.execute(message).await;
        match resp {
            Ok(r) => {
                println!("Simple Agent Execute Response: {:?}", r);
                assert!(!r.choices.is_empty());
                let content = r
                    .choices
                    .first()
                    .unwrap()
                    .message
                    .content
                    .first()
                    .map(|c| c.to_string())
                    .unwrap_or_default();
                // Check if the final response contains the answer (might be text)
                assert!(content.contains('4'), "Expected response to contain '4'");
            }
            Err(e) => panic!("Simple Agent execute test failed: {:?}", e),
        }
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_typed_agent_generate() {
        let openrouter = create_openrouter();
        // Create the typed agent executor
        let agent = Agent::builder(())
                    .openrouter(openrouter)
                    .name("CalculatorAgent")
                    .description("An agent that performs calculator operations")
                    .instruction(|_|"You are a calculator. Evaluate the mathematical expression. Use the available tools if necessary.".to_string())
                    .model("openai/gpt-4o-mini")
                    .max_tokens(100)
                    .temperature(0.0)
                    .max_iterations(5)
                    .typed::<CalculatorResult>(); // Specify the output type

        let message = UserMessage::text("What is 2 + 2? Respond with JSON.");
        // Use generate_typed for a single-turn typed response
        let resp = agent.generate_typed(message).await;
        match resp {
            Ok(r) => {
                // 'r' is the parsed CalculatorResult
                println!("Simple Typed Agent Generate Response: {:?}", r);
                // Check the value field of the result
                // Use a tolerance for floating point comparison
                assert!(
                    (r.value - 4.0).abs() < f64::EPSILON,
                    "Expected value to be 4.0, but got {}",
                    r.value
                );
            }
            Err(e) => panic!("Simple Typed Agent generate test failed: {:?}", e),
        }
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_typed_agent_execute() {
        let openrouter = create_openrouter();
        // Create the typed agent executor
        let agent = Agent::builder(())
                    .openrouter(openrouter)
                    .name("CalculatorAgent")
                    .description("An agent that performs calculator operations")
                    .instruction(|_|"You are a calculator. Evaluate the mathematical expression. Use the available tools if necessary.".to_string())
                    .model("google/gemini-2.0-flash-001")
                    .max_tokens(100)
                    .temperature(0.0)
                    .max_iterations(5)
                    .tool(CalculatorTool::default()) // Add the tool
                    .typed::<CalculatorResult>(); // Specify the output type

        let message = UserMessage::text(
            "What is 5 * 8? Use the tool if needed, then give me the final result as JSON.",
        );
        // Use execute_typed for potential multi-turn interaction ending in a typed response
        let resp = agent.execute_typed(message).await;
        match resp {
            Ok(r) => {
                // 'r' is the parsed CalculatorResult
                println!("Simple Typed Agent Execute Response: {:?}", r);
                assert!(
                    (r.value - 40.0).abs() < f64::EPSILON,
                    "Expected value to be 40.0, but got {}",
                    r.value
                );
            }
            Err(e) => panic!("Simple Typed Agent execute test failed: {:?}", e),
        }
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_agent_stream_events() {
        let openrouter = create_openrouter();
        let agent = Agent::builder(())
                    .openrouter(openrouter)
                    .name("CalculatorAgent")
                    .description("An agent that performs calculator operations")
                    .instruction(|_|"You are a calculator. Evaluate the mathematical expression. Use the available tools if necessary.".to_string())
                    .model("openai/gpt-4o-mini")
                    .max_tokens(100)
                    .temperature(0.0)
                    .max_iterations(5)
                    .tool(CalculatorTool::default()) // Add the tool
                    .build();

        let message = UserMessage::text("What is 5 + 7? Use the calculator tool.");

        // Use stream_events to get a detailed stream of agent execution events
        let mut event_stream = agent.stream_events(message.clone());

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
            matches!(events.first(), Some(AgentEvent::AgentStart)),
            "First event should be AgentStart, got {:?}",
            events.first()
        );

        // Last event should be AgentFinish or AgentError
        assert!(
            matches!(
                events.last(),
                Some(AgentEvent::AgentFinish) | Some(AgentEvent::AgentError { .. })
            ),
            "Last event should be AgentFinish or AgentError, got {:?}",
            events.last()
        );

        // Verify we have the expected event types if successful
        if matches!(events.last(), Some(AgentEvent::AgentFinish)) {
            let has_text_chunks = events
                .iter()
                .any(|e| matches!(e, AgentEvent::TextChunk { .. }));
            // Text chunks might not appear if the response is only tool calls initially
            // assert!(has_text_chunks, "Should have at least one TextChunk event");

            let has_stream_end = events
                .iter()
                .any(|e| matches!(e, AgentEvent::StreamEnd { .. }));
            assert!(has_stream_end, "Should have at least one StreamEnd event");

            // Tool events are expected for this prompt
            let has_tool_calls = events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolCallRequested { .. }));
            assert!(
                has_tool_calls,
                "Should have ToolCallRequested event for '5 + 7? Use the calculator tool.'"
            );

            let has_tool_results = events
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolResult { .. }));
            assert!(
                has_tool_results,
                "Should have ToolResult event for '5 + 7? Use the calculator tool.'"
            );

            // Find ToolResult events to check if they contain the expected result (12)
            let tool_results_content: Vec<_> = events
                .iter()
                .filter_map(|e| {
                    if let AgentEvent::ToolResult { message } = e {
                        // Tool result content might be JSON or plain text
                        Some(message.content.clone())
                    } else {
                        None
                    }
                })
                .collect();

            println!("Tool results content: {:?}", tool_results_content);

            // At least one tool result should contain the answer (12)
            let contains_answer = tool_results_content
                .iter()
                .any(|content| content.contains("12"));
            assert!(
                contains_answer,
                "Tool results should contain the expected answer (12)"
            );
        }
    }

    // #[tokio::test]
    // #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    // async fn test_orchestrator_with_agent_tool() {
    //     let openrouter = create_openrouter();

    //     // The tool that knows how to run the CalculatorAgent
    //     let calculator_agent_tool = CalculatorAgentToolRunner {
    //         openrouter: openrouter.clone(),
    //     };

    //     // Build the orchestrator agent executor, providing the CalculatorAgentToolRunner as a tool
    //     let orchestrator_executor = Agent::builder()
    //         .openrouter(openrouter.clone())
    //         .config(OrchestratorAgent::new()) // Instantiate the orchestrator logic
    //         .tool(calculator_agent_tool) // Add the agent runner tool
    //         .build();

    //     let message =
    //         UserMessage::text("Please calculate 123 + 456 for me using the CalculatorAgentTool.");

    //     // Use execute for the orchestrator to potentially use tools
    //     let resp = orchestrator_executor.execute(message).await;

    //     match resp {
    //         Ok(r) => {
    //             println!("Orchestrator Response: {:?}", r);
    //             assert!(!r.choices.is_empty());
    //             let content = r
    //                 .choices
    //                 .first()
    //                 .unwrap()
    //                 .message
    //                 .content
    //                 .first()
    //                 .map(|c| c.to_string())
    //                 .unwrap_or_default();
    //             // Check if the final response contains the calculated sum (579)
    //             assert!(
    //                 content.contains("579"),
    //                 "Expected orchestrator response to contain '579', got: {}",
    //                 content
    //             );
    //         }
    //         Err(e) => panic!("Orchestrator test failed: {:?}", e),
    //     }
    // }
}
