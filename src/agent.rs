use std::marker::PhantomData;

use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::{
    message::{Message, Messages, SystemMessage},
    response::ChatCompletionResponse,
    tool::{AnyTool, Tool, ToolBox},
    ApiRequestError, OpenRouter,
};

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
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AgentInput {
    #[schemars(
        description = "The question/request to be processed by the agent. As parrent agent you should provide complete context and all necessary information required for task completion, including any relevant background, constraints, or specific requirements. Questions should be clear, detailed and actionable."
    )]
    pub question: String,
}

#[async_trait]
pub trait BaseAgent: Send + Sync {
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
        self.openrouter()
            .chat_completion()
            .model(self.model())
            .messages(messages.into())
            .maybe_max_tokens(self.max_tokens())
            .maybe_temperature(self.temperature())
            .maybe_top_p(self.top_p())
            .maybe_stop(self.stop_sequences().cloned())
            .maybe_tools(self.tools().cloned())
            .build()
            .send()
            .await
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
}

#[async_trait]
pub trait TypedAgent: BaseAgent {
    type Output: JsonSchema + Serialize + DeserializeOwned;
    /// Makes a single call to the chat completion API, requesting a structured JSON response
    /// matching the type `T`. Parses the response and returns an instance of `T`.
    /// Returns AgentError on failure (API error, JSON parsing error, response format error).
    async fn once_structured(
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
            .get(0)
            .and_then(|choice| choice.message.content.get(0))
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
    async fn run_structured(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<Self::Output, AgentError> {
        // Prepare initial messages including system instructions
        let mut agent_messages = Messages::default();
        if let Some(instructions) = self.instructions() {
            agent_messages.push(SystemMessage::from(vec![instructions]));
        }
        agent_messages.extend(messages.into());

        // If the agent doesn't use tools, just make a single structured call.
        if self.tools().is_none() {
            return self.once_structured(agent_messages).await;
        }

        // Loop for handling tool interactions
        for _ in 0..self.max_iterations() {
            // Make a regular API call (allows for tool calls)
            let resp = self.once(agent_messages.clone()).await?; // Returns AgentError
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
                // If no tool calls, the model *should* have provided the final structured answer.
                // *However*, the previous call (`self.once`) didn't enforce the structure.
                // We need one final call that *does* enforce the JSON structure, using the
                // conversation history accumulated so far.
                return self.once_structured(agent_messages).await;
            }
        }

        // This point should ideally not be reached if max_iterations > 0,
        // because the loop either returns Ok(T) after a structured call,
        // or returns the result of the final structured call attempt on the last iteration.
        // If it somehow gets here, it implies failure to get structured output within limits.
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
    T: JsonSchema + Serialize + DeserializeOwned + Send + Sync,
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
    T: JsonSchema + Serialize + DeserializeOwned + Send + Sync,
{
    type Output = T;
}

#[cfg(test)]
mod tests {
    use schemars::{generate::SchemaSettings, schema_for};

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
            "openai/gpt-4o-mini"
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
        fn openrouter(&self) -> &OpenRouter {
            &self.openrouter
        }
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
                .once_structured(UserMessage::new(vec![question]))
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
        let resp = agent.once_structured(message).await;
        dbg!(&resp);
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
    async fn schema_test() {
        // --- New Top-Level Struct ---
        /// Represents the complete extracted information from a plant protection product label document.
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
        #[serde(rename_all = "snake_case")]
        pub struct CompleteProductLabelDto {
            /// Title of the document.
            pub title: String,
            /// A brief summary or description. Two sentences.
            pub description: String,
            /// Keywords associated with the label.
            pub tags: Vec<String>,
            /// Detailed structured data extracted from the product label content.
            pub label_data: ProductLabelDataDto,
        }
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
        #[serde(rename_all = "snake_case")]
        pub struct ProductLabelDataDto {
            pub name: String,
            pub price: f32,
            pub qty: u32,
        }

        // let value = serde_json::to_string_pretty(&CompleteProductLabelDto::default()).unwrap();
        // println!("{}", value);

        // let mut schema_settings =SchemaSettings::draft2020_12();
        // schema_settings.inline_subschemas = true;
        // let generator = schema_settings.into_generator();
        // let schema = generator.into_root_schema_for::<CompleteProductLabelDto>();
        // let schema_json = serde_json::to_string_pretty(&schema).unwrap();
        // println!("{}", schema_json);

        let openrouter = create_openrouter();
        let agent = SimpleTypedAgent::<CompleteProductLabelDto>::builder()
            .name("test")
            .openrouter(openrouter)
            .model("google/gemini-2.0-flash-001").build();

        let response = agent.once_structured(UserMessage::new(vec!["fill the structure with random data"])).await;
        dbg!(response);
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
