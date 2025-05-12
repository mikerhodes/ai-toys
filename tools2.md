Based on my examination of the codebase, let's outline the steps needed to add tool use to the chatbot, using a simple `def add(a:int, b:int)` tool as an example.

# Architecture for Adding Tool Use to Rapport

## Overview of Changes Needed

1. **Data Model Changes**
   - Create a `Tool` class to represent available tools
   - Update the `Config` dataclass in `appconfig.py` to store tool settings
   - Add tool registration mechanism

2. **Tool Framework**
   - Implement a tool registry system
   - Create function handler mechanism 
   - Design a tool response parser

3. **UI Changes**
   - Add tool configuration in `view_config.py`
   - Update chat interface to display tool use

4. **Integration with LLM**
   - Modify system prompt to inform LLM about tool capabilities
   - Update chat message handling to detect and process tool calls
   - Add handling for tool responses

## Detailed Implementation Steps

### 1. Data Model Updates

First, create a new file `tools.py` to define the tool framework:

```python
from typing import Callable, Dict, List, Any, Optional, TypeVar, Generic, Protocol, Type
from pydantic import BaseModel, create_model
import inspect
from dataclasses import dataclass

class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by the AI"""
    name: str
    description: str
    function: Callable
    enabled: bool = False
    
    def get_signature(self) -> Dict[str, Any]:
        """Extract parameter information from the function signature"""
        sig = inspect.signature(self.function)
        params = {}
        for name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            params[name] = (param_type, ... if param.default == inspect.Parameter.empty else param.default)
        return params
    
    def get_schema(self) -> Dict[str, Any]:
        """Generate a schema for this tool in OpenAI function calling format"""
        sig = inspect.signature(self.function)
        params = self.get_signature()
        
        # Create parameter schema
        properties = {}
        required = []
        
        for name, (param_type, default) in params.items():
            if default is ...:
                required.append(name)
            
            # Determine type for schema
            properties[name] = {
                "type": self._get_type_string(param_type),
                "description": f"Parameter {name}"
            }
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def _get_type_string(self, param_type: Type) -> str:
        """Convert Python type to JSON schema type string"""
        if param_type == int:
            return "integer"
        elif param_type == float:
            return "number"
        elif param_type == str:
            return "string"
        elif param_type == bool:
            return "boolean"
        else:
            return "object"

class ToolRegistry:
    """Registry for all available tools"""
    _tools: Dict[str, ToolDefinition] = {}
    
    @classmethod
    def register(cls, name: str, description: str, function: Callable) -> ToolDefinition:
        """Register a new tool"""
        tool = ToolDefinition(name=name, description=description, function=function)
        cls._tools[name] = tool
        return tool
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name"""
        return cls._tools.get(name)
    
    @classmethod
    def get_all_tools(cls) -> Dict[str, ToolDefinition]:
        """Get all registered tools"""
        return cls._tools
    
    @classmethod
    def get_enabled_tools(cls) -> List[ToolDefinition]:
        """Get all enabled tools"""
        return [tool for tool in cls._tools.values() if tool.enabled]
    
    @classmethod
    def get_tool_schemas(cls) -> List[Dict[str, Any]]:
        """Get schemas for all enabled tools"""
        return [tool.get_schema() for tool in cls.get_enabled_tools()]
    
    @classmethod
    def execute_tool(cls, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool with given arguments"""
        tool = cls.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        if not tool.enabled:
            raise ValueError(f"Tool {name} is not enabled")
            
        return tool.function(**arguments)

# Example of registering our add tool
def register_default_tools():
    def add(a: int, b: int) -> int:
        """Add two integers together"""
        return a + b
    
    ToolRegistry.register(
        name="add",
        description="Add two integers together and return the result",
        function=add
    )
```

### 2. Update `appconfig.py` to Support Tool Configuration

Modify the Config class in `appconfig.py` to include tool settings:

```python
@dataclass
class Config:
    preferred_model: Optional[str] = field(default=None)
    obsidian_directory: Optional[str] = field(default=None)
    last_used_model: Optional[str] = field(default=None)
    custom_system_prompt: Optional[str] = field(default=None)
    enabled_tools: List[str] = field(default_factory=list)  # List of enabled tool names
```

### 3. Update `view_config.py` to Add Tool Configuration UI

Modify the settings page to include tool configuration:

```python
# Add this section to the settings form
st.subheader("AI Tools")
    
# Import from our tools module
from rapport.tools import ToolRegistry

# Get all registered tools
all_tools = ToolRegistry.get_all_tools()
    
# Create a checkbox for each tool
tool_enablement = {}
for name, tool in all_tools.items():
    is_enabled = name in config.enabled_tools
    tool_enablement[name] = st.checkbox(
        f"{name}: {tool.description}", 
        value=is_enabled,
        key=f"tool_{name}"
    )
    
# When saving the form, update the enabled_tools list
if submitted:
    # Update tool configuration
    enabled_tools = [name for name, enabled in tool_enablement.items() if enabled]
    
    # Update config with new values including tools
    new_config = Config(
        preferred_model=preferred_model if preferred_model else None,
        obsidian_directory=obsidian_directory if obsidian_directory else None,
        last_used_model=config.last_used_model,
        custom_system_prompt=custom_system_prompt if custom_system_prompt else None,
        enabled_tools=enabled_tools
    )
    
    # Save to disk
    config_store.save_config(new_config)
    st.success("Settings saved successfully!")
```

### 4. Update ChatModel to Support Tool Messages

Add a new message type in `chatmodel.py` for tool calls and results:

```python
class ToolCallMessage(BaseModel):
    """Represents a tool call from the assistant"""
    name: str
    arguments: Dict[str, Any]
    role: Literal["assistant"] = "assistant" 
    type: Literal["ToolCallMessage"] = "ToolCallMessage"
    
class ToolResultMessage(BaseModel):
    """Represents a result from a tool call"""
    name: str
    result: Any
    role: Literal["tool"] = "tool"
    type: Literal["ToolResultMessage"] = "ToolResultMessage"
```

Update the `Message` type annotation to include these new types:

```python
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

### 5. Update the Chat Gateway to Handle Tool Calls

Modify `chatgateway.py` to include tool functionality:

```python
# Add to imports
from rapport.tools import ToolRegistry

# Add to ChatGateway parameters in chat() method
def chat(
    self,
    model: str,
    messages: MessageList,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Generator[MessageChunk, None, None]:
    c = self.model_to_client[model]
    response = c.chat(
        model=model,
        messages=messages,
        tools=tools
    )
    for chunk in response:
        yield chunk
```

### 6. Update Chat Adaptors for Tool Use Support

Modify the OpenAI and potentially other adaptors to support tool use. For example, in the OpenAI adaptor:

```python
def chat(
    self,
    model: str,
    messages: MessageList,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Generator[MessageChunk, None, None]:
    messages_content = self._prepare_messages_for_model(messages)
    
    params = {
        "model": model,
        "store": False,
        "stream": True,
        "max_tokens": 8192,
        "messages": messages_content,
    }
    
    # Add tools if provided
    if tools:
        params["tools"] = tools
    
    completion = self.c.chat.completions.create(**params)
    
    # Rest of method remains the same
```

### 7. Modify View_Chat to Handle Tool Calls and Results

Update `view_chat.py` to handle tool calls and execute them:

```python
def render_chat_messages():
    # Display chat messages from history on app rerun
    for message in _s.chat.messages:
        # Use the type discriminator field to determine the message type
        match message.type:
            # Existing cases...
            case "ToolCallMessage":
                with st.chat_message(message.role, avatar=":material/tools:"):
                    st.markdown(f"**Tool Call:** `{message.name}`")
                    st.json(message.arguments)
            case "ToolResultMessage":
                with st.chat_message(message.role, avatar=":material/lightbulb:"):
                    st.markdown(f"**Tool Result:** `{message.name}`")
                    st.code(str(message.result))
            # Other cases...
```

Add a function to handle tool calls:

```python
def handle_tool_call(tool_call):
    """Execute a tool call and add result to chat"""
    try:
        result = ToolRegistry.execute_tool(tool_call.name, tool_call.arguments)
        _s.chat.messages.append(
            ToolResultMessage(name=tool_call.name, result=result)
        )
        save_current_chat()
        return True
    except Exception as e:
        st.error(f"Error executing tool {tool_call.name}: {str(e)}")
        return False
```

### 8. Update Stream Model Response in View_Chat

Modify the response handling to detect tool calls:

```python
def stream_model_response():
    """Returns a generator that yields chunks of the models respose"""
    # Get enabled tools from config
    config = _s.config_store.load_config()
    tool_schemas = []
    
    # Only send tools if they're enabled in settings
    if config.enabled_tools:
        for tool_name in config.enabled_tools:
            tool = ToolRegistry.get_tool(tool_name)
            if tool:
                tool_schemas.append(tool.get_schema())
    
    response = _s.chat_gateway.chat(
        model=_s.chat.model,
        messages=_s.chat.messages,
        tools=tool_schemas if tool_schemas else None
    )

    # Rest of method remains the same with handling for tool calls...
```

### 9. Update System Prompt to Support Tool Use

Add tool information to the system prompt in `systemprompt.md`:

```markdown
You are Rapport, a helpful, but irreverent, assistant. You have a wide range of knowledge.

The current date is {current_date}.

{tool_instructions}

When asked to explain topics, Rapport uses examples and analogies to help explain the subject.
...
```

Then update the `get_system_message` function:

```python
def get_system_message(extra_prompt=None, config_store=None):
    """Get system message, using custom prompt if provided"""
    
    # Create tool instructions if tools are enabled
    tool_instructions = ""
    if config_store:
        config = config_store.load_config()
        if config.enabled_tools:
            tool_instructions = "You have access to the following tools:\n\n"
            for tool_name in config.enabled_tools:
                tool = ToolRegistry.get_tool(tool_name)
                if tool:
                    tool_instructions += f"- {tool.name}: {tool.description}\n"
            tool_instructions += "\nWhen a user asks you to perform calculations or tasks that match these tools, use them by calling them directly."
    
    system_prompt = default_system_prompt.format(
        extra_prompt=extra_prompt,
        current_date=datetime.now().strftime("%Y-%m-%d"),
        tool_instructions=tool_instructions
    )
    return SystemMessage(message=system_prompt)
```

### 10. Initialization in Entrypoint

Update `entrypoint.py` to initialize the tool registry:

```python
# Add import
from rapport.tools import ToolRegistry, register_default_tools

# At application startup
register_default_tools()  # Register our default tools like add
```

## Summary of Implementation

This architectural approach integrates tool use into the Rapport chatbot with a flexible, extensible framework. Key aspects:

1. **Modular Tool Registry:** Tools are defined and registered centrally, making it easy to add new ones.

2. **Configurable Tools:** Users can enable/disable specific tools via the Settings screen.

3. **Tool Execution Flow:**
   - System prompt informs the LLM about available tools
   - LLM generates tool call messages in response to user queries
   - Application executes the tool and returns results
   - Results are displayed to the user and added to the context for the LLM

4. **Tool Schema Generation:** Tools automatically generate their schema based on function signatures, making them compatible with LLM function calling formats.

5. **UI Integration:** Tools and their outputs are nicely integrated into the chat interface.

This architecture allows for easy expansion with more complex tools in the future, while providing a simple interface for users to control which tools are available to the AI.