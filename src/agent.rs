use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    message::{AssistantMessage, Message, Messages, SystemMessage, UserMessage},
    response::ChatCompletionResponse,
    tool::{Tool, ToolBox},
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
pub trait Agent: Send + Sync {
    fn openrouter(&self) -> &OpenRouter;
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap()
    }
    fn description(&self) -> &str;
    fn instructions(&self) -> Option<&str>;
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
    async fn send(
        &self,
        messages: impl Into<Messages> + Send,
    ) -> Result<ChatCompletionResponse, ApiRequestError> {
        let system_message = self
            .instructions()
            .map(|instructions| SystemMessage::from(vec![instructions]));
        let mut agent_messages = Messages::default();
        if let Some(system_message) = system_message {
            agent_messages.push(system_message);
        }
        agent_messages.extend(messages.into());
        for i in 0..self.max_iterations() {
            let resp = self.once(agent_messages.clone()).await?;
            let agent_message = Message::from(resp.clone());
            agent_messages.push(agent_message);
            if let (Some(tools), Some(tool_calls)) = (self.tools(), resp.tool_calls()) {
                for tool_call in tool_calls {
                    let msgs = tools.invoke(&tool_call).await;
                    agent_messages.extend(msgs);
                    if tools.return_direct(&tool_call) {
                        break;
                    }
                }
            } else {
                return Ok(resp);
            }
        }
        self.once(agent_messages).await
    }

    fn return_direct(&self) -> bool {
        false
    }
}

#[async_trait]
pub trait AnyAgent: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn once_any(&self, messages: Messages)
        -> Result<ChatCompletionResponse, ApiRequestError>;
    async fn send_any(&self, messages: Messages)
        -> Result<ChatCompletionResponse, ApiRequestError>;
}

#[async_trait]
impl<T: Agent> AnyAgent for T {
    fn name(&self) -> &str {
        self.name()
    }

    fn description(&self) -> &str {
        self.description()
    }

    async fn once_any(
        &self,
        messages: Messages,
    ) -> Result<ChatCompletionResponse, ApiRequestError> {
        self.once(messages).await
    }
    async fn send_any(
        &self,
        messages: Messages,
    ) -> Result<ChatCompletionResponse, ApiRequestError> {
        self.send(messages).await
    }
}

#[async_trait]
impl<T: Agent> Tool for T {
    type Input = AgentInput;

    type Error = String;

    fn name(&self) -> &str {
        self.name()
    }

    async fn invoke(
        &self,
        tool_call_id: &str,
        input: Self::Input,
    ) -> Result<Messages, Self::Error> {
        let message = UserMessage::new(vec![input.question]);
        match self.send(Messages::from(message)).await {
            Ok(res) => {
                let last_content = res.choices[0].message.content[0]
                    .as_text()
                    .unwrap()
                    .to_string();
                Ok(AssistantMessage::new(vec![last_content]).into())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    fn return_direct(&self) -> bool {
        self.return_direct()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct CalculatorAgent {
        openrouter: OpenRouter,
    }

    impl Agent for CalculatorAgent {
        fn openrouter(&self) -> &OpenRouter {
            &self.openrouter
        }
        fn name(&self) -> &str {
            "CalculatorAgent"
        }
        fn description(&self) -> &str {
            "An agent that performs calculator operations"
        }
        fn instructions(&self) -> Option<&str> {
            Some("Provide a mathematical expression to evaluate")
        }
        fn model(&self) -> &str {
            "openai/gpt-4o-2024-11-20"
        }
    }

    #[derive(Debug, Clone)]
    struct OrchestratorAgent {
        openrouter: OpenRouter,
        tools: ToolBox,
    }

    impl Agent for OrchestratorAgent {
        fn openrouter(&self) -> &OpenRouter {
            &self.openrouter
        }
        fn name(&self) -> &str {
            "Orchestrator"
        }
        fn description(&self) -> &str {
            "An orchestrator that manages agents, always forwarding questions to the appropriate agent and summarizing responses."
        }
        fn instructions(&self) -> Option<&str> {
            Some("Forward the question to the correct agent and provide a summary of the aggregated responses.")
        }
        fn model(&self) -> &str {
            "openai/o3-mini"
        }
        fn tools(&self) -> Option<&ToolBox> {
            Some(&self.tools)
        }
    }

    #[tokio::test]
    async fn test_simple_agent() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let openrouter = OpenRouter::builder().api_key(api_key).build();
        let agent = CalculatorAgent { openrouter };
        let message = UserMessage::new(vec!["What is 2 + 2?"]);
        let resp = agent.once(message).await.unwrap();
        dbg!(&resp);
    }
    #[tokio::test]
    async fn test_orchestrator() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let openrouter = OpenRouter::builder().api_key(api_key).build();
        let calculator_agent = CalculatorAgent {
            openrouter: openrouter.clone(),
        };
        let tools = ToolBox::builder().tool(calculator_agent).build();
        let orchestrator = OrchestratorAgent { openrouter, tools };
        let message = UserMessage::new(vec!["What is 2 + 2?"]);
        let resp = orchestrator.send(message).await.unwrap();
        dbg!(&resp);
    }
}
