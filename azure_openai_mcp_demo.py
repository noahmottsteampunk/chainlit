import os
import asyncio
import json
from typing import Any, Dict, List, Optional
import chainlit as cl
from openai import AsyncAzureOpenAI
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Load API key from environment variable
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

# Azure OpenAI Configuration
client = AsyncAzureOpenAI(
    api_key=api_key,
    api_version="2024-10-21",
    azure_endpoint="https://nass-chainlit-testing.openai.azure.com/"
)

# MCP Server Configuration
MCP_SERVER_CONFIG = {
    "mcpServers": {
        "nass-quickstats": {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "noahmacca/nass-quickstats-mcp-server:latest"
            ]
        }
    }
}

SYSTEM_MESSAGE = """You are an expert agricultural data assistant with access to the USDA NASS QuickStats database through specialized tools. Your role is to help users query, analyze, and visualize agricultural statistics.

# Available Tools

You have access to the following NASS QuickStats MCP tools:

1. **get_param_values**: Retrieve valid values for specific parameters
   - Use this to discover available options for states, commodities, years, etc.
   - Example: Get all available states, or valid commodities for a specific query

2. **get_data**: Query agricultural statistics data
   - Primary tool for retrieving actual statistical data
   - Supports filtering by state, commodity, year, statisticcat, and other parameters
   - Returns detailed records with values and metadata

3. **describe_database**: Get schema and structure information
   - Use to understand available tables and fields
   - Helpful for explaining what data is available

# Query Strategy

When users ask questions:

1. **Understand the request**: Identify what agricultural data they need
2. **Discover parameters**: Use get_param_values to find valid options for:
   - States (state_name parameter)
   - Commodities (commodity_desc parameter)
   - Years (year parameter)
   - Statistical categories (statisticcat_desc parameter)
3. **Fetch data**: Use get_data with appropriate filters
4. **Analyze results**: Process and interpret the data
5. **Visualize when appropriate**: Create charts for trends, comparisons, or distributions

# Data Visualization Guidelines

When data would benefit from visualization:
- Use bar charts for comparing values across categories
- Use line charts for showing trends over time
- Use pie charts for showing proportions or distributions
- Always include clear titles, labels, and legends
- Mention units and data sources in chart titles

# Response Style

- Be conversational and helpful
- Explain your tool usage when relevant
- Provide context about the data (sources, limitations, definitions)
- Offer to drill deeper or show related statistics
- When showing numbers, format them clearly (use commas for thousands)

# Example Interactions

User: "What are the top corn producing states?"
1. Use get_param_values to find states with corn data
2. Use get_data to fetch corn production by state for recent year
3. Present top states with production values
4. Optionally create a bar chart visualization

User: "Show me Iowa corn yield trends"
1. Use get_data to fetch Iowa corn yield data across multiple years
2. Present the trend data
3. Create a line chart showing yield over time

# Important Notes

- Always validate parameters before querying (use get_param_values)
- Handle cases where no data is found gracefully
- Explain data limitations or gaps when encountered
- Be specific about time periods, geographic scope, and measurement units
- If a query is ambiguous, ask clarifying questions

Remember: Your goal is to make USDA agricultural statistics accessible and understandable through intelligent querying and clear presentation."""


async def connect_mcp():
    """Connect to MCP server and return the session."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=MCP_SERVER_CONFIG["mcpServers"]["nass-quickstats"]["command"],
            args=MCP_SERVER_CONFIG["mcpServers"]["nass-quickstats"]["args"],
        )

        stdio_transport = await stdio_client(server_params)
        stdio, write = stdio_transport
        session = ClientSession(stdio, write)

        await session.initialize()

        return session

    except Exception as e:
        print(f"Error connecting to MCP server: {e}")
        raise


async def get_all_mcp_tools(session):
    """Get all available tools from the MCP server."""
    try:
        response = await session.list_tools()
        return response.tools
    except Exception as e:
        print(f"Error listing MCP tools: {e}")
        return []


async def call_mcp_tool(session, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call an MCP tool and return the result."""
    try:
        result = await session.call_tool(tool_name, arguments)

        # Handle the result which contains content array
        if hasattr(result, 'content') and result.content:
            # Get the first content item
            content_item = result.content[0]

            # Extract text from TextContent
            if hasattr(content_item, 'text'):
                result_text = content_item.text

                # Try to parse as JSON if it looks like JSON
                if result_text.strip().startswith('{') or result_text.strip().startswith('['):
                    try:
                        return json.loads(result_text)
                    except json.JSONDecodeError:
                        return result_text
                return result_text

        return str(result)
    except Exception as e:
        print(f"Error calling MCP tool {tool_name}: {e}")
        raise


def create_chart_from_data(data: List[Dict], chart_type: str = "bar",
                          title: str = "", x_field: str = "", y_field: str = "",
                          color_field: str = None) -> go.Figure:
    """Create a Plotly chart from data."""

    if not data:
        return None

    if chart_type == "bar":
        fig = go.Figure(data=[
            go.Bar(
                x=[item.get(x_field, "") for item in data],
                y=[float(item.get(y_field, 0)) if item.get(y_field) else 0 for item in data],
                marker_color='indianred' if not color_field else None,
                text=[item.get(color_field, "") for item in data] if color_field else None
            )
        ])
    elif chart_type == "line":
        fig = go.Figure(data=[
            go.Scatter(
                x=[item.get(x_field, "") for item in data],
                y=[float(item.get(y_field, 0)) if item.get(y_field) else 0 for item in data],
                mode='lines+markers',
                line=dict(color='royalblue', width=2),
                marker=dict(size=8)
            )
        ])
    elif chart_type == "pie":
        fig = go.Figure(data=[
            go.Pie(
                labels=[item.get(x_field, "") for item in data],
                values=[float(item.get(y_field, 0)) if item.get(y_field) else 0 for item in data]
            )
        ])
    else:
        # Default to bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=[item.get(x_field, "") for item in data],
                y=[float(item.get(y_field, 0)) if item.get(y_field) else 0 for item in data]
            )
        ])

    fig.update_layout(
        title=title,
        xaxis_title=x_field.replace("_", " ").title(),
        yaxis_title=y_field.replace("_", " ").title(),
        template="plotly_white",
        showlegend=True if chart_type == "pie" else False,
        height=500
    )

    return fig


@cl.on_chat_start
async def start():
    """Initialize the chat session with MCP connection."""

    # Send welcome message
    await cl.Message(
        content="Welcome to the USDA NASS Agricultural Data Assistant! I can help you query and analyze agricultural statistics from the USDA NASS QuickStats database.\n\nI have access to comprehensive data on crops, livestock, demographics, economics, and environmental factors across the United States.\n\nTry asking questions like:\n- 'What are the top corn producing states?'\n- 'Show me Iowa soybean yield trends over the last 10 years'\n- 'Compare wheat production between Kansas and North Dakota'\n- 'What commodities are available for California?'\n\nWhat would you like to know?"
    ).send()

    try:
        # Connect to MCP server
        session = await connect_mcp()

        # Store session in user session
        cl.user_session.set("mcp_session", session)

        # Get available tools
        tools = await get_all_mcp_tools(session)
        cl.user_session.set("mcp_tools", tools)

        # Initialize message history with system message
        cl.user_session.set("message_history", [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ])

        print(f"MCP session initialized with {len(tools)} tools available")

    except Exception as e:
        await cl.Message(
            content=f"Warning: Could not connect to MCP server. Some features may be unavailable. Error: {str(e)}"
        ).send()
        cl.user_session.set("mcp_session", None)
        cl.user_session.set("mcp_tools", [])


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and process with Azure OpenAI + MCP tools."""

    # Get session data
    mcp_session = cl.user_session.get("mcp_session")
    mcp_tools = cl.user_session.get("mcp_tools", [])
    message_history = cl.user_session.get("message_history", [])

    # Add user message to history
    message_history.append({
        "role": "user",
        "content": message.content
    })

    # Convert MCP tools to OpenAI tool format
    openai_tools = []
    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}}
            }
        }
        openai_tools.append(openai_tool)

    # Create initial response message
    response_message = cl.Message(content="")
    await response_message.send()

    # Track if we need to continue the conversation loop
    continue_loop = True
    max_iterations = 10
    iteration = 0

    while continue_loop and iteration < max_iterations:
        iteration += 1

        # Call Azure OpenAI with tools
        try:
            completion = await client.chat.completions.create(
                model="gpt-4o",
                messages=message_history,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                stream=False
            )

            assistant_message = completion.choices[0].message

            # Check if the model wants to call tools
            if assistant_message.tool_calls:
                # Add assistant message with tool calls to history
                message_history.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Create a step for this tool call
                    async with cl.Step(name=function_name, type="tool") as step:
                        step.input = function_args

                        try:
                            # Call the MCP tool
                            if mcp_session:
                                tool_result = await call_mcp_tool(
                                    mcp_session,
                                    function_name,
                                    function_args
                                )

                                # Convert result to string for OpenAI
                                if isinstance(tool_result, (dict, list)):
                                    result_str = json.dumps(tool_result)
                                else:
                                    result_str = str(tool_result)

                                step.output = result_str

                                # Add tool result to message history
                                message_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": result_str
                                })

                            else:
                                error_msg = "MCP session not available"
                                step.output = error_msg
                                message_history.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": error_msg
                                })

                        except Exception as e:
                            error_msg = f"Error calling tool: {str(e)}"
                            step.output = error_msg
                            message_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg
                            })

                # Continue the loop to get the next response
                continue_loop = True

            else:
                # No more tool calls, we have the final response
                final_content = assistant_message.content or ""

                # Add to message history
                message_history.append({
                    "role": "assistant",
                    "content": final_content
                })

                # Update the response message
                response_message.content = final_content
                await response_message.update()

                # Exit the loop
                continue_loop = False

        except Exception as e:
            error_content = f"Error communicating with Azure OpenAI: {str(e)}"
            response_message.content = error_content
            await response_message.update()

            # Add error to history
            message_history.append({
                "role": "assistant",
                "content": error_content
            })

            continue_loop = False

    # Check if we hit max iterations
    if iteration >= max_iterations:
        warning_msg = "\n\n(Reached maximum tool call iterations)"
        response_message.content += warning_msg
        await response_message.update()

    # Save updated message history
    cl.user_session.set("message_history", message_history)


@cl.on_chat_end
async def end():
    """Clean up MCP session when chat ends."""
    mcp_session = cl.user_session.get("mcp_session")
    if mcp_session:
        try:
            # Close the MCP session if it has a close method
            if hasattr(mcp_session, '__aexit__'):
                await mcp_session.__aexit__(None, None, None)
            print("MCP session closed")
        except Exception as e:
            print(f"Error closing MCP session: {e}")


@cl.on_stop
async def on_stop():
    """Handle stop events."""
    print("Chat stopped by user")


if __name__ == "__main__":
    # This allows running the file directly for testing
    print("Azure OpenAI MCP Demo")
    print("=" * 50)
    print("Configuration:")
    print(f"- Azure Endpoint: https://nass-chainlit-testing.openai.azure.com/")
    print(f"- API Version: 2024-10-21")
    print(f"- Model: gpt-4o")
    print(f"- MCP Server: {MCP_SERVER_CONFIG['mcpServers']['nass-quickstats']['command']} {' '.join(MCP_SERVER_CONFIG['mcpServers']['nass-quickstats']['args'])}")
    print("=" * 50)
    print("\nTo run this application, use: chainlit run azure_openai_mcp_demo.py")
