use std::fmt;

use bon::{builder, Builder};
use derive_more::Deref;
use serde::{Deserialize, Serialize};

use crate::response::ToolCall;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Builder, Deref)]
pub struct TextContent {
    pub text: String,
}

impl TextContent {
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl<T: Into<String>> From<T> for TextContent {
    fn from(text: T) -> Self {
        TextContent::new(text)
    }
}

impl fmt::Display for TextContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.text)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Builder, derive_more::Deref)]
pub struct ImageContent {
    #[deref]
    pub image_url: ImageUrl,
}

impl fmt::Display for ImageContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.image_url)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Builder, derive_more::Deref)]
pub struct ImageUrl {
    #[deref]
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl fmt::Display for ImageUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.url)
    }
}

impl ImageContent {
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, derive_more::Display, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text(TextContent),
    Image(ImageContent),
}

impl ContentPart {
    pub fn as_text(&self) -> Option<&TextContent> {
        match self {
            ContentPart::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageContent> {
        match self {
            ContentPart::Image(image) => Some(image),
            _ => None,
        }
    }
}

impl<T: Into<String>> From<T> for ContentPart {
    fn from(s: T) -> Self {
        ContentPart::Text(TextContent { text: s.into() })
    }
}

#[derive(
    Debug,
    Default,
    Serialize,
    Deserialize,
    Clone,
    PartialEq,
    Eq,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::Into,
    derive_more::IntoIterator,
)]
pub struct Content(pub Vec<ContentPart>);

impl<T> FromIterator<T> for Content
where
    T: Into<ContentPart>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl From<String> for Content {
    fn from(content: String) -> Self {
        Self(vec![ContentPart::Text(TextContent { text: content })])
    }
}

impl From<&str> for Content {
    fn from(content: &str) -> Self {
        Self(vec![ContentPart::Text(TextContent {
            text: content.to_string(),
        })])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub struct SystemMessage {
    content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl SystemMessage {
    pub fn new<T, U>(content: T) -> Self
    where
        T: IntoIterator<Item = U>,
        U: Into<ContentPart>,
    {
        Self {
            content: content.into_iter().map(Into::into).collect(),
            name: None,
        }
    }
    /// Add a text part to the message content
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: Content::from(text.into()),
            name: None,
        }
    }

    /// Create a system message with a single image part
    pub fn image(image: ImageContent) -> Self {
        Self {
            content: Content(vec![ContentPart::Image(image)]),
            name: None,
        }
    }

    /// Create a system message with a single image URL part
    pub fn image_url(url: impl Into<String>) -> Self {
        Self {
            content: Content(vec![ContentPart::Image(ImageContent::new(url))]),
            name: None,
        }
    }

    /// Create a system message with a single content part
    pub fn part(part: impl Into<ContentPart>) -> Self {
        Self {
            content: Content(vec![part.into()]),
            name: None,
        }
    }
    pub fn content(&self) -> &Content {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.content
    }

    pub fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl From<&str> for SystemMessage {
    fn from(text: &str) -> Self {
        Self {
            content: Content::from(text),
            name: None,
        }
    }
}

impl From<String> for SystemMessage {
    fn from(text: String) -> Self {
        Self {
            content: Content::from(text),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl UserMessage {
    /// Create a new user message from an iterator of content parts.
    pub fn new<T, U>(content: T) -> Self
    where
        T: IntoIterator<Item = U>,
        U: Into<ContentPart>,
    {
        Self {
            content: content.into_iter().map(Into::into).collect(),
            name: None,
        }
    }

    /// Create a user message with a single text part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: Content::from(text.into()),
            name: None,
        }
    }

    /// Create a user message with a single image part.
    #[must_use]
    pub fn image(image: ImageContent) -> Self {
        Self {
            content: Content(vec![ContentPart::Image(image)]),
            name: None,
        }
    }

    /// Create a user message with a single image URL part.
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self {
            content: Content(vec![ContentPart::Image(ImageContent::new(url))]),
            name: None,
        }
    }

    /// Create a user message with a single content part.
    #[must_use]
    pub fn part(part: impl Into<ContentPart>) -> Self {
        Self {
            content: Content(vec![part.into()]),
            name: None,
        }
    }

    /// Get a reference to the message content.
    pub fn content(&self) -> &Content {
        &self.content
    }

    /// Get a mutable reference to the message content.
    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.content
    }

    /// Add a content part to the message.
    pub fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    /// Check if the message content is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get the number of content parts in the message.
    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl From<&str> for UserMessage {
    fn from(text: &str) -> Self {
        Self {
            content: Content::from(text),
            name: None,
        }
    }
}

impl From<String> for UserMessage {
    fn from(text: String) -> Self {
        Self {
            content: Content::from(text),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl AssistantMessage {
    /// Create a new assistant message from an iterator of content parts.
    pub fn new<T, U>(content: T) -> Self
    where
        T: IntoIterator<Item = U>,
        U: Into<ContentPart>,
    {
        Self {
            content: content.into_iter().map(Into::into).collect(),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }

    /// Create an assistant message with a single text part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: Content::from(text.into()),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }

    /// Create an assistant message with a single image part.
    #[must_use]
    pub fn image(image: ImageContent) -> Self {
        Self {
            content: Content(vec![ContentPart::Image(image)]),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }

    /// Create an assistant message with a single image URL part.
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self {
            content: Content(vec![ContentPart::Image(ImageContent::new(url))]),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }

    /// Create an assistant message with a single content part.
    #[must_use]
    pub fn part(part: impl Into<ContentPart>) -> Self {
        Self {
            content: Content(vec![part.into()]),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }

    /// Get a reference to the message content.
    pub fn content(&self) -> &Content {
        &self.content
    }

    /// Get a mutable reference to the message content.
    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.content
    }

    /// Add a content part to the message.
    pub fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    /// Check if the message content is empty.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get the number of content parts in the message.
    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl From<&str> for AssistantMessage {
    fn from(text: &str) -> Self {
        Self {
            content: Content::from(text),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }
}

impl From<String> for AssistantMessage {
    fn from(text: String) -> Self {
        Self {
            content: Content::from(text),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMessage {
    pub content: String,
    pub tool_call_id: String,
}

impl ToolMessage {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    pub fn content(&self) -> &String {
        &self.content
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

impl Message {
    /// Create a system message with the given text
    pub fn system(text: impl Into<String>) -> Self {
        Message::System(SystemMessage::from(text.into()))
    }

    /// Create a user message with the given text
    pub fn user(text: impl Into<String>) -> Self {
        Message::User(UserMessage::from(text.into()))
    }

    /// Create an assistant message with the given text
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant(AssistantMessage::from(text.into()))
    }

    /// Create a tool message with the given tool call ID and content
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool(ToolMessage::new(tool_call_id, content))
    }
}

impl From<SystemMessage> for Message {
    fn from(message: SystemMessage) -> Self {
        Message::System(message)
    }
}

impl From<UserMessage> for Message {
    fn from(message: UserMessage) -> Self {
        Message::User(message)
    }
}

impl From<AssistantMessage> for Message {
    fn from(message: AssistantMessage) -> Self {
        Message::Assistant(message)
    }
}

impl From<ToolMessage> for Message {
    fn from(message: ToolMessage) -> Self {
        Message::Tool(message)
    }
}

#[derive(
    Debug,
    Clone,
    Default,
    Serialize,
    Deserialize,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::IntoIterator,
    derive_more::Into,
)]
pub struct Messages(pub Vec<Message>);

impl Messages {
    /// Creates a new `Messages` container from an iterator of messages.
    pub fn new<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        Messages(iter.into_iter().map(Into::into).collect())
    }

    /// Pushes a new message onto the end of the list.
    pub fn push(&mut self, message: impl Into<Message>) {
        self.0.push(message.into());
    }

    /// Inserts a message at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    pub fn insert(&mut self, index: usize, message: impl Into<Message>) {
        self.0.insert(index, message.into());
    }

    /// Adds a system message with the given text content to the list and returns a mutable reference to self.
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.push(Message::System(SystemMessage::from(content.into())));
        self
    }

    /// Adds a user message with the given text content to the list and returns a mutable reference to self.
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.push(Message::User(UserMessage::from(content.into())));
        self
    }

    /// Adds an assistant message with the given text content to the list and returns a mutable reference to self.
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.push(Message::Assistant(AssistantMessage::from(content.into())));
        self
    }
}

impl From<Message> for Messages {
    fn from(value: Message) -> Self {
        Messages(vec![value])
    }
}

impl From<UserMessage> for Messages {
    fn from(value: UserMessage) -> Self {
        Messages(vec![Message::User(value)])
    }
}

impl From<AssistantMessage> for Messages {
    fn from(value: AssistantMessage) -> Self {
        Messages(vec![Message::Assistant(value)])
    }
}

impl From<SystemMessage> for Messages {
    fn from(value: SystemMessage) -> Self {
        Messages(vec![Message::System(value)])
    }
}

impl From<ToolMessage> for Messages {
    fn from(value: ToolMessage) -> Self {
        Messages(vec![Message::Tool(value)])
    }
}

impl FromIterator<Message> for Messages {
    fn from_iter<I: IntoIterator<Item = Message>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::message::UserMessage;
    use crate::response::ToolCall;

    use super::{
        AssistantMessage, Content, ContentPart, ImageContent, ImageUrl, Message, SystemMessage, ToolMessage,
    };

    #[test]
    fn test_assistant_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "Hello John! How can I assist you today?"
            }],
            "refusal": null,
            "role": "assistant"
        });

        let msg: AssistantMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &msg.content[0] {
            assert_eq!(text_content.text, "Hello John! How can I assist you today?");
            assert!(msg.refusal.is_none());
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_system_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant"
            }],
            "role": "system"
        });

        let msg: SystemMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &msg.content[0] {
            assert_eq!(text_content.text, "You are a helpful assistant");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_user_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "What is the weather?"
            }],
            "role": "user"
        });

        let msg: UserMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &msg.content[0] {
            assert_eq!(text_content.text, "What is the weather?");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_tool_message_deserialization() {
        let json = json!({
            "content": "The temperature is 72F",
            "role": "tool",
            "tool_call_id": "weather_123"
        });

        let msg: ToolMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content, "The temperature is 72F");
        assert_eq!(msg.tool_call_id, "weather_123");
    }

    #[test]
    fn test_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "Hello John! How can I assist you today?"
            }],
            "refusal": null,
            "role": "assistant"
        });

        let msg: Message = serde_json::from_value(json).unwrap();
        match msg {
            Message::Assistant(assistant_msg) => {
                if let ContentPart::Text(text_content) = &assistant_msg.content[0] {
                    assert_eq!(text_content.text, "Hello John! How can I assist you today?");
                    assert!(assistant_msg.refusal.is_none());
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected assistant message"),
        }
    }

    #[test]
    fn test_from_str_implementation() {
        // Test SystemMessage From<&str>
        let system_msg = SystemMessage::from("You are a helpful assistant");
        assert_eq!(system_msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &system_msg.content[0] {
            assert_eq!(text_content.text, "You are a helpful assistant");
        } else {
            panic!("Expected text content");
        }

        // Test UserMessage From<&str>
        let user_msg = UserMessage::from("What is the weather?");
        assert_eq!(user_msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &user_msg.content[0] {
            assert_eq!(text_content.text, "What is the weather?");
        } else {
            panic!("Expected text content");
        }

        // Test AssistantMessage From<&str>
        let assistant_msg = AssistantMessage::from("The weather is sunny");
        assert_eq!(assistant_msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &assistant_msg.content[0] {
            assert_eq!(text_content.text, "The weather is sunny");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_from_string_implementation() {
        // Test SystemMessage From<String>
        let system_msg = SystemMessage::from(String::from("You are a helpful assistant"));
        assert_eq!(system_msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &system_msg.content[0] {
            assert_eq!(text_content.text, "You are a helpful assistant");
        } else {
            panic!("Expected text content");
        }

        // Test UserMessage From<String>
        let user_msg = UserMessage::from(String::from("What is the weather?"));
        assert_eq!(user_msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &user_msg.content[0] {
            assert_eq!(text_content.text, "What is the weather?");
        } else {
            panic!("Expected text content");
        }

        // Test AssistantMessage From<String>
        let assistant_msg = AssistantMessage::from(String::from("The weather is sunny"));
        assert_eq!(assistant_msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &assistant_msg.content[0] {
            assert_eq!(text_content.text, "The weather is sunny");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_message_convenience_constructors() {
        // Test Message::system
        let system_msg = Message::system("You are a helpful assistant");
        match system_msg {
            Message::System(msg) => {
                assert_eq!(msg.content.len(), 1);
                if let ContentPart::Text(text_content) = &msg.content[0] {
                    assert_eq!(text_content.text, "You are a helpful assistant");
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected system message"),
        }

        // Test Message::user
        let user_msg = Message::user("What is the weather?");
        match user_msg {
            Message::User(msg) => {
                assert_eq!(msg.content.len(), 1);
                if let ContentPart::Text(text_content) = &msg.content[0] {
                    assert_eq!(text_content.text, "What is the weather?");
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected user message"),
        }

        // Test Message::assistant
        let assistant_msg = Message::assistant("The weather is sunny");
        match assistant_msg {
            Message::Assistant(msg) => {
                assert_eq!(msg.content.len(), 1);
                if let ContentPart::Text(text_content) = &msg.content[0] {
                    assert_eq!(text_content.text, "The weather is sunny");
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected assistant message"),
        }

        // Test Message::tool
        let tool_msg = Message::tool("weather_123", "The temperature is 72F");
        match tool_msg {
            Message::Tool(msg) => {
                assert_eq!(msg.content, "The temperature is 72F");
                assert_eq!(msg.tool_call_id, "weather_123");
            }
            _ => panic!("Expected tool message"),
        }
    }

    #[test]
    fn test_message_serialization() {
        // Test SystemMessage serialization
        let system_msg = SystemMessage::from("You are a helpful assistant");
        let json = serde_json::to_value(&system_msg).unwrap();
        assert_eq!(
            json,
            json!({
                "content": [{
                    "type": "text",
                    "text": "You are a helpful assistant"
                }]
            })
        );

        // Test UserMessage serialization
        let user_msg = UserMessage::from("What is the weather?");
        let json = serde_json::to_value(&user_msg).unwrap();
        assert_eq!(
            json,
            json!({
                "content": [{
                    "type": "text",
                    "text": "What is the weather?"
                }]
            })
        );

        // Test AssistantMessage serialization
        let assistant_msg = AssistantMessage::from("The weather is sunny");
        let json = serde_json::to_value(&assistant_msg).unwrap();
        assert_eq!(
            json,
            json!({
                "content": [{
                    "type": "text",
                    "text": "The weather is sunny"
                }]
            })
        );

        // Test ToolMessage serialization
        let tool_msg = ToolMessage::new("weather_123", "The temperature is 72F");
        let json = serde_json::to_value(&tool_msg).unwrap();
        assert_eq!(
            json,
            json!({
                "content": "The temperature is 72F",
                "tool_call_id": "weather_123"
            })
        );

        // Test Message serialization
        let msg = Message::user("What is the weather?");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(
            json,
            json!({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "What is the weather?"
                }]
            })
        );
    }
}
