mod error;
mod events;
mod simple_agent;
mod tool_call_aggregator;
mod traits;
mod typed_agent;

pub use error::AgentError;
pub use events::{AgentErrorSerializable, AgentEvent};
pub use simple_agent::SimpleAgent;
pub use traits::{Agent, AgentInput, AnyAgent, BaseAgent, TypedAgent};
pub use typed_agent::SimpleTypedAgent;

#[cfg(test)]
mod tests {
    use schemars::{generate::SchemaSettings, schema_for};
    use tokio_stream::StreamExt;

    use crate::{
        message::{ToolMessage, UserMessage},
        tool::Tool,
    };

    use super::*;
    use std::sync::Arc;

    #[derive(Debug, schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
    struct CalculatorToolInput {
        a: f64,
        b: f64,
    }
    #[derive(Default)]
    struct CalculatorTool {}

    #[async_trait::async_trait]
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
        ) -> Result<crate::message::Messages, Self::Error> {
            Ok(ToolMessage::new(tool_call_id, (input.a + input.b).to_string()).into())
        }
    }

    #[derive(Debug, Clone)]
    struct CalculatorAgent {
        openrouter: crate::OpenRouter,
        tools: crate::tool::ToolBox,
    }

    impl BaseAgent for CalculatorAgent {
        fn openrouter(&self) -> &crate::OpenRouter {
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
        fn tools(&self) -> Option<&crate::tool::ToolBox> {
            Some(&self.tools)
        }
    }
    // Implement the Agent trait for CalculatorAgent
    impl Agent for CalculatorAgent {}

    #[derive(Debug, schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
    pub struct CalculatorResult {
        pub value: f64,
    }

    impl TypedAgent for CalculatorAgent {
        type Output = CalculatorResult;
    }

    #[derive(Debug, Clone)]
    struct OrchestratorAgent {
        openrouter: crate::OpenRouter,
        tools: crate::tool::ToolBox,
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
        fn tools(&self) -> Option<&crate::tool::ToolBox> {
            Some(&self.tools)
        }

        fn openrouter(&self) -> &crate::OpenRouter {
            &self.openrouter
        }
    }
    // Implement the Agent trait for OrchestratorAgent
    impl Agent for OrchestratorAgent {}

    #[derive(schemars::JsonSchema, serde::Serialize, serde::Deserialize)]
    pub struct CalculatorAgentQuestion {
        pub question: String,
    }

    #[async_trait::async_trait]
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
        ) -> Result<crate::message::Messages, Self::Error> {
            let question = input.question;
            let response = self
                .once_typed(UserMessage::new(vec![question]))
                .await
                .unwrap();
            Ok(ToolMessage::new(tool_call_id, serde_json::to_string(&response).unwrap()).into())
        }
    }

    // Helper to create OpenRouter client from env var
    fn create_openrouter() -> crate::OpenRouter {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY environment variable not set");
        crate::OpenRouter::builder().api_key(api_key).build()
    }

    #[tokio::test]
    #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_simple_agent() {
        let openrouter = create_openrouter();
        let tools = crate::tool::ToolBox::builder()
            .tool(CalculatorTool::default())
            .build();
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
        let tools = crate::tool::ToolBox::builder()
            .tool(CalculatorTool::default())
            .build();
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
        let tools = crate::tool::ToolBox::builder()
            .tool(CalculatorTool::default())
            .build();
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
        let tools = crate::tool::ToolBox::builder()
            .tool(CalculatorTool::default())
            .build();
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

    #[tokio::test]
    // #[ignore] // Ignored by default to avoid making API calls unless explicitly run
    async fn test_agent_run_events() {
        let openrouter = create_openrouter();
        let tools = crate::tool::ToolBox::builder()
            .tool(CalculatorTool::default())
            .build();
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
