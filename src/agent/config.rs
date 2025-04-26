use super::context::AgentContext;

pub trait AgentConfig: Clone + Send + Sync + 'static {
    fn name(&self) -> &str {
        std::any::type_name::<Self>().split("::").last().unwrap()
    }
    fn description(&self, ctx: &AgentContext) -> Option<&str> {
        None
    }
    fn instructions(&self, ctx: &AgentContext) -> Option<String>;
    fn model(&self, ctx: &AgentContext) -> &str;
    fn max_tokens(&self, ctx: &AgentContext) -> Option<u32> {
        None
    }
    fn stop_sequences(&self, ctx: &AgentContext) -> Option<&Vec<String>> {
        None
    }
    fn temperature(&self, ctx: &AgentContext) -> Option<f64> {
        None
    }
    fn top_p(&self, ctx: &AgentContext) -> Option<f64> {
        None
    }
    fn top_k(&self, ctx: &AgentContext) -> Option<usize> {
        None
    }
    fn max_iterations(&self, ctx: &AgentContext) -> u32 {
        12
    }
}

#[derive(Debug, Clone)]
pub struct StaticConfig {
    pub name: String,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_iterations: u32,
}
