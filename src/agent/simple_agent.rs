use async_trait::async_trait;
use bon::Builder;

use super::traits::{Agent, BaseAgent};
use crate::{tool::Tool, tool::ToolBox, OpenRouter};

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
