use bon::Builder;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    message::{Message, Messages},
    provider_preference::ProviderPreferences,
    tool::ToolChoice,
    ApiRequestError, ErrorResponse, OpenRouter, BASE_URL,
};

const API_URL: &str = "api/v1/chat/completions";

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename = "text")]
pub struct TextType;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename = "json_object")]
pub struct JsonType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFormat {
    Text {
        #[serde(rename = "type")]
        r#type: TextType,
    },
    Json {
        #[serde(rename = "type")]
        r#type: JsonType,
    },
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
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
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
}

impl Request {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.messages.push(message.into());
    }
    pub async fn send(&self) -> Result<Value, ApiRequestError> {
        let url = format!("{}/{}", BASE_URL, API_URL);
        let _body = serde_json::to_string(self)?;
        dbg!(&_body);
        let req = self
            .open_router
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.open_router.api_key),
            )
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "YOUR_SITE_URL")
            .header("X-Title", "YOUR_SITE_NAME")
            .json(self);
        dbg!(&req);
        let res = req.send().await?;
        dbg!(&res);
        let value = res.json().await?;
        Ok(value)
        // if res.status().is_success() {
        //     let data: ChatCompletionResponse = res.json().await?;
        //     Ok(data)
        // } else {
        //     let error_response: ErrorResponse = res.json().await?;
        //     Err(ApiRequestError::InvalidRequestError {
        //         message: error_response.error.message,
        //         param: error_response.error.param,
        //         code: error_response.error.code,
        //     })
        // }
    }

    // pub async fn stream(
    //     &self,
    // ) -> impl Stream<Item = Result<ChatCompletionChunkResponse, ApiRequestError>> {
    //     let url = format!("{}/{}", BASE_URL, API_URL);
    //     let mut body = serde_json::to_value(self).unwrap();
    //     body["stream"] = serde_json::Value::Bool(true);

    //     let stream = self
    //         .openai
    //         .client
    //         .post(url)
    //         .bearer_auth(&self.openai.api_key)
    //         .json(&body)
    //         .send()
    //         .await
    //         .unwrap()
    //         .bytes_stream();

    //     let filtered_stream = stream.flat_map(|chunk| {
    //         let chunk = match chunk {
    //             Ok(bytes) => String::from_utf8(bytes.to_vec())
    //                 .map_err(|e| ApiRequestError::Stream(e.to_string())),
    //             Err(e) => Err(ApiRequestError::Stream(e.to_string())),
    //         };

    //         let responses = chunk
    //             .map(|data| match data.as_str() {
    //                 "" => vec![],
    //                 s if s.starts_with("data: ") => s
    //                     .split("\n\n")
    //                     .filter(|chunk| !chunk.is_empty() && chunk != &"data: [DONE]")
    //                     .filter_map(|chunk| chunk.strip_prefix("data: "))
    //                     .map(|json_str| {
    //                         serde_json::from_str::<ChatCompletionChunkResponse>(json_str)
    //                             .map_err(ApiRequestError::SerdeError)
    //                     })
    //                     .filter(|res| {
    //                         res.as_ref().is_ok_and(|res| {
    //                             !res.choices.iter().any(|choice| {
    //                                 choice.delta.content.as_ref().is_some_and(|s| {
    //                                     dbg!(s);
    //                                     dbg!(s.is_empty())
    //                                 })
    //                             })
    //                         })
    //                     })
    //                     .collect(),
    //                 _ => vec![Err(ApiRequestError::Stream(format!(
    //                     "Invalid event data: {}",
    //                     data
    //                 )))],
    //             })
    //             .unwrap_or_else(|e| vec![Err(e)]);

    //         futures::stream::iter(responses)
    //     });

    //     Box::pin(filtered_stream)
    // }
}

// impl TokenCount for Message {
//     fn token_count(&self) -> usize {
//         match self {
//             Message::System(message) => message.content.token_count(),
//             Message::User(message) => message.content.token_count(),
//             Message::Assistant(message) => message.content.token_count(),
//             Message::Tool(message) => message.content.token_count(),
//         }
//     }
// }

// impl TokenCount for Messages {
//     fn token_count(&self) -> usize {
//         self.0.iter().map(|m| m.token_count()).sum()
//     }
// }

impl OpenRouter {
    pub fn chat_completion(&self) -> RequestBuilder<request_builder::SetOpenRouter> {
        Request::builder().open_router(self.clone())
    }
}

#[cfg(test)]
mod test {

    use futures::StreamExt;

    use crate::{
        message::{Message, UserMessage},
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
        dbg!(&payload);

        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .unwrap();

        dbg!(&response);
        let json = response.json::<serde_json::Value>().await.unwrap();
        dbg!(json);
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
            .model("anthropic/claude-3.5-sonnet:beta")
            .message(UserMessage::from(vec!["Hi, I'm John."]))
            .build()
            .send()
            .await;
        // dbg!(&res);
    }

    // #[tokio::test]
    // async fn test_chat_stream() {
    //     let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    //     let client = reqwest::Client::new();
    //     let openai = OpenAi::builder().api_key(api_key).client(client).build();
    //     let mut res = openai
    //         .chat_completion()
    //         .model("gpt-4o")
    //         .stream(true)
    //         .messages(Message::user("Hi, I'm John."))
    //         .build()
    //         .stream()
    //         .await;
    //     while let Some(res) = res.next().await {
    //         dbg!(String::from(res.unwrap()));
    //     }
    // }

    // // #[cfg(feature = "tools")]
    // // #[tokio::test]
    // // async fn test_wikipedia_tool() {
    // //     #[derive(Debug)]
    // //     pub struct Wikipedia;

    // //     #[async_trait::async_trait]
    // //     impl ToTool for Wikipedia {
    // //         fn to_tool(&self) -> Tool {
    // //             ToolBuilder::default()
    // //                 .name("wikipedia")
    // //                 .description("Search in wikipedia")
    // //                 .add_parameter::<String>("query", "Query")
    // //                 .build()
    // //                 .unwrap()
    // //         }
    // //         async fn call_tool(
    // //             &self,
    // //             tool_call_id: &str,
    // //             input: serde_json::Value,
    // //         ) -> ToolCallResult {
    // //             dbg!(&input);
    // //             let query = input["query"].as_str().unwrap();
    // //             let url = format!("https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}", query);
    // //             let res = reqwest::get(&url)
    // //                 .await
    // //                 .unwrap()
    // //                 .json::<serde_json::Value>()
    // //                 .await
    // //                 .unwrap();

    // //             ToolCallResult {
    // //                 tool_call_id: tool_call_id.to_string(),
    // //                 content: res.to_string(),
    // //             }
    // //         }
    // //     }

    // //     let api_key = std::env::var("OPENAI_API_KEY").unwrap();
    // //     let client = reqwest::Client::new();
    // //     let openai = OpenAiBuilder::default()
    // //         .api_key(api_key)
    // //         .client(&client)
    // //         .build()
    // //         .unwrap();
    // //     let tools = Tools::default().add_tool(Wikipedia);
    // //     let mut messages = Messages::from(Message::user("Search Apollo project on Wikipedia."));
    // //     let res = openai
    // //         .chat_completion()
    // //         .model("gpt-4-1106-preview")
    // //         .tools(tools.clone())
    // //         .messages(messages.clone())
    // //         .build()
    // //         .unwrap()
    // //         .send()
    // //         .await
    // //         .unwrap();

    // //     match &res.choices[0].message {
    // //         Message::Assistant(msg) => {
    // //             if let Some(tool_calls) = &msg.tool_calls {
    // //                 let results = tools.call_tools(tool_calls).await;
    // //                 let tool_msgs = Messages::from(results.clone());
    // //                 messages.push_message(msg.clone());
    // //                 messages.extend(tool_msgs.into_iter());
    // //                 dbg!(&messages);
    // //                 let res = openai
    // //                     .chat_completion()
    // //                     .model("gpt-4-1106-preview")
    // //                     .tools(tools.clone())
    // //                     .messages(messages.clone())
    // //                     .build()
    // //                     .unwrap()
    // //                     .send()
    // //                     .await
    // //                     .unwrap();

    // //                 dbg!(&res);
    // //             }
    // //         }
    // //         _ => panic!("Not a tool call"),
    // //     }
    // // }
}
