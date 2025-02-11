use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    message::{AssistantMessage, Message, Messages, SystemMessage, UserMessage},
    response::ChatCompletionResponse,
    tool::{AnyTool, Tool, ToolBox},
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
pub trait Agent: Send + Sync + Clone {
    fn openrouter(&self) -> &OpenRouter;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn instructions(&self) -> &str;
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
    async fn once<T, U>(&self, messages: T) -> Result<ChatCompletionResponse, ApiRequestError>
    where
        T: IntoIterator<Item = U> + Send,
        U: Into<Message> + Send,
    {
        let messages: Messages = messages.into_iter().map(|m| m.into()).collect();
        self.openrouter()
            .chat_completion()
            .model(self.model())
            .messages(messages)
            .maybe_max_tokens(self.max_tokens())
            .maybe_temperature(self.temperature())
            .maybe_top_p(self.top_p())
            .maybe_stop(self.stop_sequences().cloned())
            .maybe_tools(self.tools().cloned())
            .build()
            .send()
            .await
    }
    async fn send<T, U>(&self, messages: T) -> Result<ChatCompletionResponse, ApiRequestError>
    where
        T: IntoIterator<Item = U> + Send,
        U: Into<Message> + Send,
    {
        let system_message = SystemMessage::from(vec![self.instructions()]);
        let mut agent_messages = Messages::default();
        agent_messages.push(system_message);
        agent_messages.extend(messages.into_iter().map(|m| m.into()));
        for i in 0..self.max_iterations() {
            let resp = self.once(agent_messages.clone()).await?;
            let agent_message = Message::from(resp.clone());
            agent_messages.push(agent_message);
            if let (Some(tools), Some(tool_calls)) = (self.tools(), resp.tool_calls()) {
                for tool_call in tool_calls {
                    let msg = tools.invoke(tool_call).await;
                    agent_messages.push(msg);
                }
            } else {
                return Ok(resp);
            }
            let payload = serde_json::to_string_pretty(&agent_messages).unwrap();
            // println!("{i}\n: {payload}\n")
        }
        self.once(agent_messages).await
    }
}

#[async_trait]
impl<T: Agent> Tool for T {
    type Input = AgentInput;

    type Output = String;

    type Error = String;

    fn name(&self) -> &str {
        self.name()
    }

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        let message = UserMessage::new(vec![input.question]);
        match self.send(Messages::from(message)).await {
            Ok(res) => Ok(res.choices[0].message.content[0]
                .as_text()
                .unwrap()
                .to_string()),
            Err(e) => Err(e.to_string()),
        }
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
        fn instructions(&self) -> &str {
            "Provide a mathematical expression to evaluate"
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
        fn instructions(&self) -> &str {
            "Forward the question to the correct agent and provide a summary of the aggregated responses."
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
        let resp = agent.once(vec![message]).await.unwrap();
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
        let resp = orchestrator.send(vec![message]).await.unwrap();
        dbg!(&resp);
    }
}
