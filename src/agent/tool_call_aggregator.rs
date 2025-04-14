use crate::response::ToolCall;
use std::collections::HashMap;

/// Helper struct to accumulate fragmented tool calls across multiple chunks.
#[derive(Debug, Clone, Default)]
pub struct PartialToolCall {
    pub index: Option<usize>,
    pub id: Option<String>,
    pub type_field: String,
    pub function_name: String,
    pub arguments_buffer: String,
}

impl PartialToolCall {
    /// Merges information from a tool call delta into this partial tool call.
    /// We only care about index and merging arguments.
    /// Assumes id, type, name are set initially by the aggregator.
    pub fn merge(&mut self, delta_tool_call: &ToolCall) {
        // Ensure index is set if it wasn't (e.g., if first delta somehow missed it)
        // Primarily relies on aggregator setting it initially.
        if self.index.is_none() {
            self.index = delta_tool_call.index;
        }

        if self.id.is_none() {
            self.id = delta_tool_call.id.clone();
        }

        if self.type_field.is_empty() && !delta_tool_call.type_field.is_empty() {
            self.type_field = delta_tool_call.type_field.clone();
        }

        if let Some(ref name) = delta_tool_call.function.name {
            if !name.is_empty() {
                self.function_name = name.clone();
            }
        }

        // Append argument chunks
        if !delta_tool_call.function.arguments.is_empty() {
            self.arguments_buffer
                .push_str(&delta_tool_call.function.arguments);
        }
    }

    /// Attempts to convert the accumulated partial data into a final ToolCall.
    /// Returns None if essential information (like id, name) is missing.
    pub fn finalize(self) -> Option<ToolCall> {
        // Essential fields must be present
        let id = self.id?;
        let function_name = self.function_name; // This unwraps Option<String> to String

        // Type defaults to "function" if not specified but name is present
        let type_field = self.type_field;

        Some(ToolCall {
            index: self.index,
            id: Some(id), // id is String here
            type_field,
            function: crate::response::FunctionCall {
                name: Some(function_name),
                arguments: self.arguments_buffer,
            },
        })
    }
}

/// Manages a collection of partial tool calls during streaming
#[derive(Debug, Default)]
pub struct PartialToolCallsAggregator {
    // Key: index of the tool call
    // Value: The accumulator for that specific tool call
    calls: HashMap<usize, PartialToolCall>,
}

impl PartialToolCallsAggregator {
    /// Creates a new, empty accumulator
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a tool call delta fragment to the appropriate accumulator
    /// Creates a new accumulator if the index hasn't been seen before
    pub fn add_delta(&mut self, delta_call: &ToolCall) {
        // Use index as the key, defaulting to 0 if missing
        let index = delta_call.index.unwrap_or(0);

        self.calls
            .entry(index)
            .or_default() // Get existing PartialToolCall or create a default one
            .merge(delta_call); // Merge the delta data
    }

    /// Consumes the accumulator and returns a finalized list of ToolCalls
    /// Incomplete calls are filtered out with warnings
    /// The list is sorted by the original index
    pub fn finalize(self) -> Vec<ToolCall> {
        let mut finalized: Vec<ToolCall> = self
            .calls
            .into_values() // Consume the map and take ownership of PartialToolCalls
            .filter_map(|partial| {
                // Attempt to finalize each partial call
                match partial.finalize() {
                    Some(call) => Some(call),
                    None => {
                        eprintln!(
                            "Warning: Ignoring incomplete tool call data during finalization."
                        );
                        None // Filter out incomplete calls
                    }
                }
            })
            .collect();

        // Sort by index for deterministic order
        finalized.sort_by_key(|call| call.index.unwrap_or(0));

        finalized
    }

    /// Checks if any partial tool calls have been accumulated
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_tool_call_merge_and_finalize() {
        // Create a partial tool call
        let mut partial = PartialToolCall::default();

        // Create some fragments
        let fragment1 = ToolCall {
            index: Some(0),
            id: Some("call_123".to_string()),
            type_field: "function".to_string(),
            function: crate::response::FunctionCall {
                name: Some("TestTool".to_string()),
                arguments: "{".to_string(), // Start of JSON
            },
        };

        let fragment2 = ToolCall {
            index: Some(0),
            id: Some("".to_string()), // Empty as it's a continuation
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"a\":5,".to_string(), // Middle of JSON
            },
        };

        let fragment3 = ToolCall {
            index: Some(0),
            id: Some("".to_string()),
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"b\":7}".to_string(), // End of JSON
            },
        };

        // Merge the fragments
        partial.merge(&fragment1);
        partial.merge(&fragment2);
        partial.merge(&fragment3);

        // Finalize and check the result
        let finalized = partial.finalize().expect("Should finalize successfully");

        assert_eq!(finalized.id, Some("call_123".to_string()));
        assert_eq!(finalized.function.name, Some("TestTool".to_string()));
        assert_eq!(finalized.function.arguments, "{\"a\":5,\"b\":7}");
    }

    #[test]
    fn test_partial_tool_calls_manager() {
        // Create the PartialToolCalls manager
        let mut manager = PartialToolCallsAggregator::new();

        // Create tool call deltas for two different tool calls
        let delta1_first = ToolCall {
            index: Some(0),
            id: Some("call_123".to_string()),
            type_field: "function".to_string(),
            function: crate::response::FunctionCall {
                name: Some("ToolOne".to_string()),
                arguments: "{\"param\":".to_string(),
            },
        };

        let delta1_second = ToolCall {
            index: Some(0),
            id: Some("".to_string()),
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"value\"}".to_string(),
            },
        };

        let delta2_first = ToolCall {
            index: Some(1),
            id: Some("call_456".to_string()),
            type_field: "function".to_string(),
            function: crate::response::FunctionCall {
                name: Some("ToolTwo".to_string()),
                arguments: "{\"x\":10,".to_string(),
            },
        };

        let delta2_second = ToolCall {
            index: Some(1),
            id: Some("".to_string()),
            type_field: "".to_string(),
            function: crate::response::FunctionCall {
                name: None,
                arguments: "\"y\":20}".to_string(),
            },
        };

        // Add all deltas to the manager
        manager.add_delta(&delta1_first);
        manager.add_delta(&delta2_first);
        manager.add_delta(&delta1_second);
        manager.add_delta(&delta2_second);

        // Finalize and check results
        let finalized = manager.finalize();

        // We should have two tool calls
        assert_eq!(finalized.len(), 2, "Should have two finalized tool calls");

        // They should be sorted by index
        assert_eq!(finalized[0].index, Some(0));
        assert_eq!(finalized[1].index, Some(1));

        // Check first tool call
        assert_eq!(finalized[0].id, Some("call_123".to_string()));
        assert_eq!(finalized[0].function.name, Some("ToolOne".to_string()));
        assert_eq!(finalized[0].function.arguments, "{\"param\":\"value\"}");

        // Check second tool call
        assert_eq!(finalized[1].id, Some("call_456".to_string()));
        assert_eq!(finalized[1].function.name, Some("ToolTwo".to_string()));
        assert_eq!(finalized[1].function.arguments, "{\"x\":10,\"y\":20}");
    }
}
