use std::pin::Pin;

use async_stream::try_stream;
use bon::Builder;
use futures::stream::BoxStream;
use schemars::{generate::SchemaSettings, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_stream::{Stream, StreamExt};

use crate::{
    message::{Message, Messages},
    provider_preference::ProviderPreferences,
    response::{ChatCompletionChunk, ChatCompletionResponse},
    tool::{ToolBox, ToolChoice},
    ApiRequestError, ErrorResponse, OpenRouter, BASE_URL,
};

const API_URL: &str = "api/v1/chat/completions";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub r#type: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Builder)]
pub struct Request {
    #[builder(field)]
    pub messages: Messages,
    #[builder(field)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolBox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    #[serde(skip)]
    pub open_router: OpenRouter,
}

impl<S: request_builder::State> RequestBuilder<S> {
    pub fn messages(mut self, messages: impl IntoIterator<Item = impl Into<Message>>) -> Self {
        self.messages = messages.into_iter().map(Into::into).collect();
        self
    }
    pub fn message(mut self, message: impl Into<Message>) -> Self {
        self.messages.push(message.into());
        self
    }
    pub fn response_format<T: JsonSchema + DeserializeOwned>(mut self) -> Self {
        let type_name = std::any::type_name::<T>().split("::").last().unwrap();
        let mut schema_settings = SchemaSettings::draft2020_12();
        schema_settings.inline_subschemas = true;
        let schema_generator = schema_settings.into_generator();
        let json_schema = schema_generator.into_root_schema_for::<T>();
        let response_format = json!({
            "type": "json_schema",
            "json_schema": {"name": type_name, "schema": json_schema},
        });
        self.response_format = Some(response_format);
        self
    }
}

impl Request {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.messages.push(message.into());
    }
    pub async fn send(&self) -> Result<ChatCompletionResponse, ApiRequestError> {
        let url = format!("{}/{}", BASE_URL, API_URL);

        // Use bearer_auth instead of manually constructing the Authorization header
        let res = self
            .open_router
            .client
            .post(&url)
            .bearer_auth(&self.open_router.api_key)
            .json(self)
            .send()
            .await?;

        if res.status().is_success() {
            // Parse the response body directly to the target type
            Ok(res.json::<ChatCompletionResponse>().await?)
        } else {
            Err(ApiRequestError::InvalidRequestError(res.json().await?))
        }
    }

    // pub fn stream(&self) -> Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, ApiRequestError>> + Send>> {
    pub fn stream(&self) -> BoxStream<'static, Result<ChatCompletionChunk, ApiRequestError>> {
        let client = self.open_router.client.clone();
        let api_key = self.open_router.api_key.clone();
        let url = format!("{}/{}", BASE_URL, API_URL);
        let request_data = self.clone();

        Box::pin(try_stream! {
            // Prepare request body with streaming enabled
            let mut body = serde_json::to_value(&request_data)?;
            body.as_object_mut()
                .expect("Request body must be a JSON object")
                .insert("stream".to_string(), serde_json::Value::Bool(true));

            // Send request
            let response = client
                .post(&url)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await?;
            let status = response.status();

            if !response.status().is_success() {
                // Handle error response
                match response.json::<ErrorResponse>().await {
                    Ok(error_response) => {
                        Err(ApiRequestError::InvalidRequestError(error_response))?
                    }
                    Err(json_err) => {
                        Err(ApiRequestError::Stream(format!(
                            "API error (status {status}): Failed to parse error response: {json_err}",
                        )))?
                    }
                }
            } else {
                // Process successful streaming response
                let mut byte_stream = response.bytes_stream();

                while let Some(chunk_result) = byte_stream.next().await {
                    let chunk = chunk_result?;
                    let chunk_str = String::from_utf8(chunk.to_vec())
                        .map_err(|e| ApiRequestError::Stream(format!("UTF-8 decode error: {}", e)))?;

                    for parse_result in ChatCompletionChunk::from_streaming_data(&chunk_str) {
                        yield parse_result?;
                    }
                }
            }
        })
    }
}

impl OpenRouter {
    pub fn chat_completion(&self) -> RequestBuilder<request_builder::SetOpenRouter> {
        Request::builder().open_router(self.clone())
    }
}

#[cfg(test)]
mod test {

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use tokio_stream::StreamExt;

    use crate::{
        message::{Messages, UserMessage},
        tool::{SimpleTool, Tool, ToolBox},
        OpenRouter,
    };

    #[tokio::test]
    async fn test_plain_request() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();

        let payload = serde_json::json!({
            "model": "anthropic/claude-3.5-sonnet:beta",
            "temperature": 1,
            "messages": [{
                "role": "user",
                "content": "Hello!"
            }]
        });

        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .unwrap();

        let json = response.json::<serde_json::Value>().await.unwrap();
    }

    #[tokio::test]
    async fn test_chat_no_stream() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openrouter = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let res = openrouter
            .chat_completion()
            .model("google/gemini-2.0-flash-001")
            .message(UserMessage::text("Hi, I'm John."))
            .build()
            .send()
            .await;
        dbg!(&res);
    }

    #[tokio::test]
    async fn test_chat_stream() {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openai = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let mut res = openai
            .chat_completion()
            .model("anthropic/claude-3.5-sonnet:beta")
            .message(UserMessage::text("Hi, I'm John."))
            .build()
            .stream();
        while let Some(res) = res.next().await {
            dbg!(res);
        }
    }

    #[tokio::test]
    async fn test_response_format() {
        #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
        struct MyResult {
            expresion: String,
            result: String,
        }
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openrouter = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let res = openrouter
            .chat_completion()
            .model("openai/gpt-4o-2024-11-20")
            .message(UserMessage::text("ile to jest 2+2"))
            .response_format::<MyResult>()
            .build()
            .send()
            .await;
        dbg!(&res);
    }

    #[tokio::test]
    async fn test_openai() {
        let payload = json!({
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": "This is a test, use adding tool to verify if everything is working."
                }
              ]
            },
            {
              "role": "assistant",
              "content": "",
              "tool_calls": [
                {
                  "id": "call_sJgwcXOX1wXbylui3t53YFU6",
                  "type": "function",
                  "function": {
                    "name": "adding_tool",
                    "arguments": "{\"a\":5,\"b\":7}"
                  }
                }
              ]
            },
            {
              "role": "tool",
              "content": "{\"result\":12.0}",
              "tool_call_id": "call_sJgwcXOX1wXbylui3t53YFU6"
            }
          ],
          "model": "openai/gpt-4o-2024-11-20",
          "tools": [
            {
              "type": "function",
              "function": {
                "name": "adding_tool",
                "parameters": {
                  "$schema": "https://json-schema.org/draft/2020-12/schema",
                  "properties": {
                    "a": {
                      "format": "float",
                      "type": "number"
                    },
                    "b": {
                      "format": "float",
                      "type": "number"
                    }
                  },
                  "required": [
                    "a",
                    "b"
                  ],
                  "type": "object"
                }
              }
            }
          ]
        });

        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();

        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .unwrap();

        let json = response.json::<serde_json::Value>().await.unwrap();
        dbg!(json);
    }

    // #[tokio::test]
    // async fn test_adding_tool() {
    //     #[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
    //     pub struct AddingToolInput {
    //         a: f32,
    //         b: f32,
    //     }
    //     let adding_tool = SimpleTool::builder()
    //         .name("adding_tool")
    //         .handler(|input: AddingToolInput| async move {
    //             let result = input.a + input.b;
    //             Ok::<_, String>(serde_json::json!({ "result": result }))
    //         })
    //         .build();
    //     let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
    //     let client = reqwest::Client::new();
    //     let openrouter = OpenRouter::builder()
    //         .api_key(api_key)
    //         .client(client)
    //         .build();
    //     let tools = ToolBox::builder().tool(adding_tool).build();
    //     let mut messages = Messages::default().user(
    //         "This is a test, use adding tool with any value to verify if everything is working.",
    //     );
    //     let res = openrouter
    //         .chat_completion()
    //         .model("openai/gpt-4o-2024-11-20")
    //         // .model("anthropic/claude-3.7-sonnet:beta")
    //         .tools(tools.clone())
    //         .messages(messages.clone())
    //         .build()
    //         .send()
    //         .await
    //         .unwrap();
    //     messages.push(res.clone());
    //     if let Some(tool_calls) = res.tool_calls() {
    //         for tool_call in tool_calls {
    //             let msg = tools.invoke(&tool_call).await;
    //             messages.extend(msg);
    //         }
    //     }
    //     dbg!(&messages);
    //     let res = openrouter
    //         .chat_completion()
    //         .model("openai/gpt-4o-2024-11-20")
    //         // .model("anthropic/claude-3.7-sonnet:beta")
    //         .tools(tools.clone())
    //         .messages(messages.clone())
    //         .build()
    //         .send()
    //         .await
    //         .unwrap();
    //     dbg!(&res);
    // }

    // #[tokio::test]
    // async fn test_stream_adding_tool() {
    //     #[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
    //     pub struct AddingToolInput {
    //         a: f32,
    //         b: f32,
    //     }
    //     let adding_tool = SimpleTool::builder()
    //         .name("adding_tool")
    //         .handler(|input: AddingToolInput| async move {
    //             let result = input.a + input.b;
    //             Ok::<_, String>(serde_json::json!({ "result": result }))
    //         })
    //         .build();
    //     let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
    //     let client = reqwest::Client::new();
    //     let openrouter = OpenRouter::builder()
    //         .api_key(api_key)
    //         .client(client)
    //         .build();
    //     let tools = ToolBox::builder().tool(adding_tool).build();
    //     let messages = Messages::default().user(
    //         "This is a test, use adding tool with any value to verify if everything is working.",
    //     );
    //     // let mut messages = Messages::from(UserMessage::from(vec![
    //     //     "Hello, how are you?",
    //     // ]));
    //     let mut stream = openrouter
    //         .chat_completion()
    //         // .model("google/gemini-2.0-flash-001")
    //         .model("openai/gpt-4o-mini")
    //         .tools(tools.clone())
    //         .messages(messages.clone())
    //         .build()
    //         .stream();
    //     while let Some(chunk) = stream.next().await {
    //         match chunk {
    //             Ok(chunk) => println!("{:?}", chunk),
    //             Err(_) => todo!(),
    //         }
    //     }
    // }

    // #[tokio::test]
    // async fn test_wikipedia_tool() {
    //     #[derive(Debug, Clone)]
    //     pub struct Wikipedia;

    //     #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
    //     pub struct WikipediaInput {
    //         query: String,
    //     }

    //     #[async_trait::async_trait]
    //     impl Tool for Wikipedia {
    //         type Input = WikipediaInput;
    //         type Error = String;

    //         fn name(&self) -> &str {
    //             "wikipedia"
    //         }

    //         async fn invoke(
    //             &self,
    //             tool_call_id: &str,
    //             input: Self::Input,
    //         ) -> Result<Messages, Self::Error> {
    //             let url = format!(
    //                         "https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}",
    //                         input.query
    //                     );
    //             let res = reqwest::get(&url)
    //                 .await
    //                 .map_err(|e| e.to_string())?
    //                 .json::<serde_json::Value>()
    //                 .await
    //                 .map_err(|e| e.to_string())?;
    //             let msgs = UserMessage::new(vec![res.to_string()]);
    //             Ok(msgs.into())
    //         }
    //     }

    //     let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    //     let client = reqwest::Client::new();
    //     let openrouter = OpenRouter::builder()
    //         .api_key(api_key)
    //         .client(client)
    //         .build();
    //     let tools = ToolBox::builder().tool(Wikipedia).build();
    //     let mut messages = Messages::default().user("Search Apollo project on Wikipedia.");
    //     let res = openrouter
    //         .chat_completion()
    //         .model("openai/gpt-4o-2024-11-20")
    //         .tools(tools.clone())
    //         .messages(messages.clone())
    //         .build()
    //         .send()
    //         .await
    //         .unwrap();
    //     messages.push(res.clone());
    //     if let Some(tool_calls) = res.tool_calls() {
    //         for tool_call in tool_calls {
    //             let msg = tools.invoke(&tool_call).await;
    //             messages.extend(msg);
    //         }
    //     }
    //     let res = openrouter
    //         .chat_completion()
    //         .model("openai/gpt-4o-2024-11-20")
    //         .tools(tools.clone())
    //         .messages(messages)
    //         .build()
    //         .send()
    //         .await
    //         .unwrap();
    //     dbg!(&res);
    // }

    // #[tokio::test]
    // async fn test_json_output() {
    //     #[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
    //     pub struct TestOutput {
    //         probability: f64,
    //         explanation: String,
    //     }

    //     let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
    //     let openrouter = OpenRouter::builder().api_key(api_key).build();
    //     let mut messages = Messages::default().user("Can we live forever in the future?");
    //     let res = openrouter
    //         .chat_completion()
    //         .model("google/gemini-2.0-flash-001")
    //         .messages(messages.clone())
    //         .response_format::<TestOutput>()
    //         .build()
    //         .send()
    //         .await
    //         .unwrap();
    //     dbg!(&res);
    // }
}
