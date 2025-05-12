Let me systematically think through what steps would be needed to add tool use to this chatbot, specifically implementing a simple `def add(a:int, b:int)` tool with settings to enable/disable it.

Based on my analysis of the codebase, here's my architectural plan:

## Steps to Add Tool Use to the Rapport Chatbot

### 1. Define Tool Structure and Interface

First, we need to define a structure for tools and their interfaces:

1. Create a new file `src/rapport/tools.py` to define the tool interface, implementation, and registry.
2. Define a `Tool` class or protocol to represent the tools.
3. Implement the specific `add` tool as an example.

### 2. Update Configuration to Support Tool Settings

1. Modify the `Config` class in `appconfig.py` to include tool settings.
2. Update the `view_config.py` file to add UI elements for enabling/disabling tools.

### 3. Integrate Tools with Chat Gateway

1. Update the `ChatGateway` class in `chatgateway.py` to support tool invocation.
2. Add tool-specific message types to `chatmodel.py`.
3. Modify the chat message formatting in the adaptors to handle tool calls in various API formats.

### 4. Update UI to Display Tool Usage

1. Modify `view_chat.py` to render tool calls and results in the chat interface.

Let's detail each of these steps:

## 1. Define Tool Structure and Interface

Create `src/rapport/tools.py`:

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Type, Callable, get_type_hints
import inspect

class Tool(Protocol):
    """Protocol for tools that can be used by the assistant."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters."""
        pass

@dataclass
class SimpleTool:
    """Base implementation of a Tool."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            # Auto-generate parameters from function signature
            self.parameters = {}
            type_hints = get_type_hints(self.function)
            sig = inspect.signature(self.function)
            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, Any).__name__
                self.parameters[param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name} of type {param_type}"
                }
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters."""
        return self.function(**kwargs)

# Example tool implementations
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

# Tool registry
class ToolRegistry:
    """Registry for available tools."""
    _tools: Dict[str, Tool] = {}
    
    @classmethod
    def register(cls, tool: Tool):
        """Register a tool."""
        cls._tools[tool.name] = tool
        
    @classmethod
    def get_tool(cls, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return cls._tools.get(name)
        
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tools."""
        return list(cls._tools.keys())
        
    @classmethod
    def get_enabled_tools(cls, config) -> List[Tool]:
        """Get all enabled tools based on configuration."""
        enabled_tools = []
        if not hasattr(config, 'enabled_tools'):
            return []
            
        for tool_name in config.enabled_tools:
            if tool in cls._tools:
                enabled_tools.append(cls._tools[tool_name])
        return enabled_tools

# Register the add tool
add_tool = SimpleTool(
    name="add",
    description="Add two integers and return the result.",
    function=add
)
ToolRegistry.register(add_tool)
```

## 2. Update Configuration to Support Tool Settings

Update `appconfig.py`:

```python
@dataclass
class Config:
    preferred_model: Optional[str] = field(default=None)
    obsidian_directory: Optional[str] = field(default=None)
    last_used_model: Optional[str] = field(default=None)
    custom_system_prompt: Optional[str] = field(default=None)
    enabled_tools: List[str] = field(default_factory=list)  # Add this line
```

Update `view_config.py` to include tool settings:

```python
# Add this after the existing configuration options
st.subheader("Tool Configuration")

# Import tool registry
from rapport.tools import ToolRegistry

# Get available tools
available_tools = ToolRegistry.list_tools()

# Show checkboxes for each available tool
enabled_tools = []
for tool_name in available_tools:
    tool = ToolRegistry.get_tool(tool_name)
    if st.checkbox(
        f"{tool.name} - {tool.description}", 
        value=tool.name in config.enabled_tools,
        key=f"tool_{tool.name}"
    ):
        enabled_tools.append(tool.name)

# Update the config object to include enabled tools
new_config = Config(
    preferred_model=preferred_model if preferred_model else None,
    obsidian_directory=obsidian_directory if obsidian_directory else None,
    last_used_model=config.last_used_model,
    custom_system_prompt=custom_system_prompt if custom_system_prompt else None,
    enabled_tools=enabled_tools  # Add this line
)
```

## 3. Integrate Tools with Chat Gateway

Update `chatmodel.py` to add tool-related message types:

```python
class ToolCallMessage(BaseModel):
    tool: str
    parameters: Dict[str, Any]
    role: Literal["assistant"] = "assistant"
    type: Literal["ToolCallMessage"] = "ToolCallMessage"

class ToolResultMessage(BaseModel):
    tool: str
    result: Any
    role: Literal["function"] = "function"
    type: Literal["ToolResultMessage"] = "ToolResultMessage"

# Update the Message type annotation to include the new message types
Message = Annotated[
    Union[
        SystemMessage,
        UserMessage,
        AssistantMessage,
        IncludedFile,
        IncludedImage,
        ToolCallMessage,
        ToolResultMessage,
    ],
    Field(discriminator="type"),
]
```

Modify the system prompt in `systemprompt.md` to include information about available tools:

```markdown
You can use tools to help you solve tasks. Here are the available tools:

{tools_description}

To use a tool, format your response like this:
<tool name="tool_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</tool>
```

Update `chatgateway.py` to support tool invocation and message format:

```python
# Add imports
from rapport.tools import ToolRegistry

# Modify the chat method in ChatGateway to process tool calls
def chat(
    self,
    model: str,
    messages: MessageList,
    config=None,  # Add config parameter
) -> Generator[MessageChunk, None, None]:
    # Augment system message with tool descriptions if tools are enabled
    if config and hasattr(config, 'enabled_tools') and config.enabled_tools:
        enabled_tools = ToolRegistry.get_enabled_tools(config)
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in enabled_tools
        ])
        
        # Find and update system message with tools information
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                msg.message = msg.message.format(
                    tools_description=tools_description
                )
                messages[i] = msg
                break
    
    # Continue with existing chat logic
    c = self.model_to_client[model]
    response = c.chat(
        model=model,
        messages=messages,
    )
    for chunk in response:
        yield chunk

# Add a method to process tool calls in the response
def process_tool_call(self, message_text, messages, config):
    """Process tool calls in message text and append results to messages."""
    # Basic regex-based tool call detection (can be improved)
    import re
    tool_pattern = r'<tool name="([^"]+)">(.*?)</tool>'
    param_pattern = r'<parameter name="([^"]+)">([^<]+)</parameter>'
    
    # Find tool calls
    for tool_match in re.finditer(tool_pattern, message_text, re.DOTALL):
        tool_name = tool_match.group(1)
        params_text = tool_match.group(2)
        
        # Find parameters
        params = {}
        for param_match in re.finditer(param_pattern, params_text):
            param_name = param_match.group(1)
            param_value = param_match.group(2)
            params[param_name] = param_value
        
        # Add tool call message
        tool_call_message = ToolCallMessage(
            tool=tool_name,
            parameters=params
        )
        messages.append(tool_call_message)
        
        # Execute tool if enabled
        if config and tool_name in config.enabled_tools:
            tool = ToolRegistry.get_tool(tool_name)
            if tool:
                try:
                    # Type conversion for parameters
                    converted_params = {}
                    for param_name, param_value in params.items():
                        param_type = tool.parameters.get(param_name, {}).get("type")
                        if param_type == "int":
                            converted_params[param_name] = int(param_value)
                        elif param_type == "float":
                            converted_params[param_name] = float(param_value)
                        else:
                            converted_params[param_name] = param_value
                            
                    result = tool.execute(**converted_params)
                    
                    # Add tool result message
                    tool_result_message = ToolResultMessage(
                        tool=tool_name,
                        result=result
                    )
                    messages.append(tool_result_message)
                except Exception as e:
                    tool_result_message = ToolResultMessage(
                        tool=tool_name,
                        result=f"Error: {str(e)}"
                    )
                    messages.append(tool_result_message)
```

## 4. Update UI to Display Tool Usage

Update `view_chat.py` to render tool calls and results:

```python
# Update the render_chat_messages function to handle tool calls and results
def render_chat_messages():
    # Display chat messages from history on app rerun
    for message in _s.chat.messages:
        # Use the type discriminator field to determine the message type
        match message.type:
            case "SystemMessage":
                with st.expander("View system prompt"):
                    st.markdown(message.message)
            case "IncludedFile":
                with st.chat_message(
                    message.role, avatar=":material/upload_file:"
                ):
                    st.markdown(f"Included `{message.name}` in chat.")
                    with st.expander("View file content"):
                        st.markdown(f"```{message.ext}\n{message.data}\n```")
            case "IncludedImage":
                with st.chat_message(
                    message.role, avatar=":material/image:"
                ):
                    st.markdown(f"Included image `{message.name}` in chat.")
                    if _s.chat_gateway.supports_images(_s.model):
                        # make the image a bit smaller
                        a, _ = st.columns([1, 2])
                        with a:
                            st.image(str(message.path))
                    else:
                        st.warning("Change model to use images.")
            case "ToolCallMessage":
                with st.chat_message(
                    message.role, avatar=":material/tools:"
                ):
                    st.markdown(f"**Tool Call**: {message.tool}")
                    st.json(message.parameters)
            case "ToolResultMessage":
                with st.chat_message(
                    message.role, avatar=":material/function:"
                ):
                    st.markdown(f"**Tool Result**: {message.tool}")
                    st.code(str(message.result))
            case "AssistantMessage" | "UserMessage":
                with st.chat_message(message.role):
                    st.markdown(message.message)

# Update the generate_assistant_message function to process tool calls
def generate_assistant_message():
    with st.chat_message("assistant"), st.empty():
        try:
            with st.spinner("Thinking...", show_time=False):
                g = wait_n_and_chain(2, stream_model_response())
            m = st.write_stream(g)
            
            if isinstance(m, str):  # should always be
                # Process any tool calls in the message
                config = _s.config_store.load_config()
                _s.chat_gateway.process_tool_call(m, _s.chat.messages, config)
                
                # Add the original message
                _s.chat.messages.append(AssistantMessage(message=m))
            else:
                st.error("Bad chat return type; not added to chat.")
                
            save_current_chat()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("The server could not be reached")
            st.error(e)
```

## 5. Initialization Changes

Update the `entrypoint.py` to initialize the tool registry:

```python
# Add import
from rapport.tools import ToolRegistry

# Make sure the tools are registered during app startup
```

With these changes, the chatbot will be able to use the `add` tool when enabled in the settings, and users will be able to enable/disable tools through the Settings screen.