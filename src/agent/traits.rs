use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::tool::ToolBox;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AgentInput {
    #[schemars(
        description = "The question/request to be processed by the agent. As parrent agent you should provide complete context and all necessary information required for task completion, including any relevant background, constraints, or specific requirements. Questions should be clear, detailed and actionable."
    )]
    pub question: String,
}

pub trait Agent: Clone + Send + Sync + 'static {
    fn name(&self) -> &str {
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
    fn max_iterations(&self) -> u32 {
        12
    }
}

pub trait TypedAgent: Agent {
    type Output: JsonSchema + DeserializeOwned;
}
