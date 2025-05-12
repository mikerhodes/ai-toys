Based on the code analysis, here's my architectural plan for adding tool use to the Rapport chatbot:

# Architectural Plan for Adding Tool Use to Rapport

## Overview

To add tool use capabilities to the Rapport chatbot, we need to implement several components:

1. A tool registry system
2. Configuration settings to enable/disable tools
3. Tool definitions, including their schemas
4. Integration with the chat gateway
5. UI modifications to control tool use

Let's use a simple `def add(a:int, b:int)` tool as our example.

## Implementation Steps

### 1. Create a Tool Data Model

First, we need to create a data model for tools in `chatmodel.py`:

```python
class Tool(BaseModel):
    name: str
    description: str
    enabled: bool = True
    schema: dict  # JSON Schema describing the function
    type: Literal["Tool"] = "Tool"

class ToolCall(BaseModel):
    tool_name: str
    arguments: dict
    result: Optional[str] = None
    role: Literal["assistant"] = "assistant"
    type: Literal["ToolCall"] = "ToolCall"

class ToolResult(BaseModel):
    tool_name: str
    result: str
    role: Literal["tool"] = "tool"
    type: Literal["ToolResult"] = "ToolResult"
```

Then update the `Message` type to include these new message types:

```python
Message = Annotated[
    Union[
        SystemMessage,
        UserMessage,
        AssistantMessage,
        IncludedFile,
        IncludedImage,
        ToolCall,
        ToolResult,
    ],
    Field(discriminator="type"),
]
```

### 2. Create a Tool Registry

Create a new file `src/rapport/tools.py` to define and register tools:

```python
from typing import Dict, Callable, List, Any
from pydantic import BaseModel

class ToolRegistry:
    """Registry for all available tools"""
    
    _tools: Dict[str, 'Tool'] = {}
    
    @classmethod
    def register(cls, tool: 'Tool'):
        cls._tools[tool.name] = tool
        
    @classmethod
    def get_tool(cls, name: str) -> 'Tool':
        return cls._tools.get(name)
        
    @classmethod
    def list_tools(cls) -> List['Tool']:
        return list(cls._tools.values())
        
    @classmethod
    def get_enabled_tools(cls, config) -> List['Tool']:
        """Return only tools that are enabled in the config"""
        enabled_tools = config.enabled_tools or []
        return [tool for tool in cls._tools.values() 
                if tool.name in enabled_tools]
    
    @classmethod
    def get_tool_schemas(cls, config) -> List[dict]:
        """Return schemas for enabled tools"""
        return [tool.get_schema() for tool in cls.get_enabled_tools(config)]

class Tool:
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str, fn: Callable):
        self.name = name
        self.description = description
        self.fn = fn
        # Register the tool automatically
        ToolRegistry.register(self)
        
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments"""
        return self.fn(**kwargs)
        
    def get_schema(self) -> dict:
        """Return the JSON schema for this tool"""
        # This would be implemented by subclasses
        raise NotImplementedError()
        
# Example add tool implementation
class AddTool(Tool):
    def __init__(self):
        super().__init__(
            name="add",
            description="Add two integers together",
            fn=self.add_function
        )
        
    def add_function(self, a: int, b: int) -> int:
        return a + b
        
    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"]
            }
        }

# Initialize the add tool
add_tool = AddTool()
```

### 3. Update the Config Model

Modify `appconfig.py` to include tool configuration:

```python
@dataclass
class Config:
    preferred_model: Optional[str] = field(default=None)
    obsidian_directory: Optional[str] = field(default=None)
    last_used_model: Optional[str] = field(default=None)
    custom_system_prompt: Optional[str] = field(default=None)
    enabled_tools: List[str] = field(default_factory=list)  # List of enabled tool names
```

### 4. Modify the Settings Screen

Update `view_config.py` to include tool settings:

```python
# After the existing form fields, add:
st.subheader("Tool Settings")

from rapport.tools import ToolRegistry

tools = ToolRegistry.list_tools()
enabled_tools = config.enabled_tools or []

tool_settings = {}
for tool in tools:
    tool_settings[tool.name] = st.checkbox(
        f"Enable {tool.name} tool",
        value=tool.name in enabled_tools,
        help=tool.description
    )

# Update the save settings logic
if submitted:
    # Update config with new values
    new_config = Config(
        preferred_model=preferred_model if preferred_model else None,
        obsidian_directory=obsidian_directory if obsidian_directory else None,
        last_used_model=config.last_used_model,
        custom_system_prompt=custom_system_prompt if custom_system_prompt else None,
        enabled_tools=[name for name, enabled in tool_settings.items() if enabled]
    )
```

### 5. Update Chat Gateway to Support Tool Use

Extend the `ChatAdaptor` protocol in `chatgateway.py` to support tools:

```python
class ChatAdaptor(Protocol):
    """ChatAdaptor adapts an LLM interface to ChatGateway's expectations."""

    def list(self) -> List[str]: ...

    def supports_images(self, model: str) -> bool: ...
    
    def supports_tools(self, model: str) -> bool: ...

    def chat(
        self,
        model: str,
        messages: MessageList,
        tools: Optional[List[dict]] = None,
    ) -> Generator[MessageChunk, None, None]: ...
```

Then update each implementation class (OpenAIAdaptor, AnthropicAdaptor, etc.) to handle tools. For example, in the OpenAIAdaptor class:

```python
def supports_tools(self, model: str) -> bool:
    # Only certain models support tools
    return model in ["gpt-4o", "gpt-4.1"]

def chat(
    self,
    model: str,
    messages: MessageList,
    tools: Optional[List[dict]] = None,
) -> Generator[MessageChunk, None, None]:
    # cast until we make an openai specific message prep function
    messages_content = self._prepare_messages_for_model(messages)
    
    kwargs = {
        "model": model,
        "store": False,
        "stream": True,
        "max_tokens": 8192,
        "messages": messages_content,
    }
    
    if tools and self.supports_tools(model):
        kwargs["tools"] = tools
    
    completion = self.c.chat.completions.create(**kwargs)
    
    # Rest of the method remains the same...
```

### 6. Update ChatGateway to Pass Tool Information

Modify the `ChatGateway` class to pass tool information:

```python
def chat(
    self,
    model: str,
    messages: MessageList,
    config: Optional[Config] = None,
) -> Generator[MessageChunk, None, None]:
    c = self.model_to_client[model]
    
    tools = None
    if config and c.supports_tools(model):
        from rapport.tools import ToolRegistry
        tools = ToolRegistry.get_tool_schemas(config)
    
    response = c.chat(
        model=model,
        messages=messages,
        tools=tools,
    )
    for chunk in response:
        yield chunk
```

### 7. Update Chat View to Handle Tool Calls

Modify `view_chat.py` to handle tool calls and results:

```python
def stream_model_response():
    """Returns a generator that yields chunks of the models respose"""
    # cg = cast(ChatGateway, st.session_state["chat_gateway"])
    response = _s.chat_gateway.chat(
        model=_s.chat.model,
        messages=_s.chat.messages,
        config=_s.config_store.load_config(),
    )
    
    # Rest remains the same...
```

Add functions to handle tool calls:

```python
def _handle_tool_call(tool_call):
    """Process a tool call and generate a tool result"""
    from rapport.tools import ToolRegistry
    
    tool_name = tool_call.tool_name
    arguments = tool_call.arguments
    
    tool = ToolRegistry.get_tool(tool_name)
    if not tool:
        result = f"Error: Tool '{tool_name}' not found"
    else:
        try:
            result = tool.execute(**arguments)
            result = str(result)
        except Exception as e:
            result = f"Error executing tool: {str(e)}"
    
    # Add the tool result to the chat
    tool_result = ToolResult(
        tool_name=tool_name,
        result=result
    )
    _s.chat.messages.append(tool_result)
    save_current_chat()
    return tool_result
```

### 8. Update Message Rendering

Update the `render_chat_messages` function to handle tool calls and results:

```python
def render_chat_messages():
    # Display chat messages from history on app rerun
    for message in _s.chat.messages:
        # Use the type discriminator field to determine the message type
        match message.type:
            case "SystemMessage":
                # Existing code...
            case "IncludedFile":
                # Existing code...
            case "IncludedImage":
                # Existing code...
            case "AssistantMessage" | "UserMessage":
                # Existing code...
            case "ToolCall":
                with st.chat_message(message.role, avatar=":material/build:"):
                    st.markdown(f"**Tool Call**: `{message.tool_name}`")
                    with st.expander("View arguments"):
                        st.json(message.arguments)
            case "ToolResult":
                with st.chat_message(message.role, avatar=":material/api:"):
                    st.markdown(f"**Tool Result**: `{message.tool_name}`")
                    st.code(message.result)
```

### 9. Handle Tool Calls in Message Processing

Update the message processing to detect and handle tool calls from the model's response:

```python
def generate_assistant_message():
    # Using the .empty() container ensures that once the
    # model starts returning content, we replace the spinner
    # with the streamed content. We then also need to write
    # out the full message at the end (for some reason
    # the message otherwise disappears).
    with st.chat_message("assistant"), st.empty():
        try:
            with st.spinner("Thinking...", show_time=False):
                g = wait_n_and_chain(2, stream_model_response())
            m = st.write_stream(g)
            
            # Check if the message contains a tool call
            # This would require parsing the model's response for tool call format
            tool_call_match = re.search(r'\{\{tool:(\w+)\}\}(.*?)\{\{/tool\}\}', m, re.DOTALL)
            if tool_call_match:
                tool_name = tool_call_match.group(1)
                tool_args_str = tool_call_match.group(2).strip()
                try:
                    tool_args = json.loads(tool_args_str)
                    tool_call = ToolCall(tool_name=tool_name, arguments=tool_args)
                    _s.chat.messages.append(tool_call)
                    _handle_tool_call(tool_call)
                    
                    # Continue the conversation with the tool result
                    _s.generate_assistant = True
                except json.JSONDecodeError:
                    _s.chat.messages.append(AssistantMessage(message=m))
            else:
                _s.chat.messages.append(AssistantMessage(message=m))
                
            save_current_chat()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("The server could not be reached")
            st.error(e)
```

## Summary

To add tool use to the Rapport chatbot:

1. **Data Model Extensions**:
   - Create Tool, ToolCall, and ToolResult message types
   - Update the Message union type

2. **Tool Registry System**:
   - Create a registry for tools
   - Implement the basic `add(a, b)` tool
   - Make tools registrable and discoverable

3. **Configuration Updates**:
   - Add tool configuration to the Config class
   - Create UI controls in the Settings screen

4. **Chat Gateway Integration**:
   - Extend the ChatAdaptor protocol
   - Add tool support to each model adapter
   - Pass tool schemas to models that support them

5. **UI and UX Enhancements**:
   - Handle tool calls in the chat flow
   - Render tool calls and results
   - Process and execute tool calls

This architecture allows you to easily add more tools in the future by simply implementing new Tool classes and registering them with the ToolRegistry. The settings UI will automatically discover and display them for enabling/disabling.