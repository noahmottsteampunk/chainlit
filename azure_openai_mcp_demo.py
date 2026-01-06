import os
import json
import asyncio
from contextlib import AsyncExitStack
import chainlit as cl
from chainlit.logger import logger
from openai import AsyncAzureOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import plotly.graph_objects as go
import pandas as pd


endpoint = "https://noah-mjw1vza6-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-5.2-chat"
deployment = "gpt-5.2-chat"
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
api_version = "2024-12-01-preview"

client = AsyncAzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

# Hardcoded authorized MCP servers
AUTHORIZED_MCP_SERVERS = [
    {
        "name": "NASS QuickStats",
        "url": "https://nass-mcp-server.thankfulmushroom-b8a37300.eastus.azurecontainerapps.io/mcp",
        "clientType": "streamable-http",
        "description": "USDA National Agricultural Statistics Service data"
    }
]


async def connect_mcp(mcp_config):
    """Connect to a single MCP server."""
    try:
        async with cl.Step(name=f"Connecting to {mcp_config['name']}", type="tool") as step:
            exit_stack = AsyncExitStack()

            # Connect using streamable-http
            transport = await exit_stack.enter_async_context(
                streamablehttp_client(url=mcp_config['url'])
            )
            read, write = transport[:2]

            # Create MCP session
            mcp_session = await exit_stack.enter_async_context(
                ClientSession(
                    read_stream=read,
                    write_stream=write,
                    sampling_callback=None
                )
            )

            # Initialize the session
            await mcp_session.initialize()

            # List available tools
            tool_list = await mcp_session.list_tools()

            # Store the session
            if not hasattr(cl.context.session, 'mcp_sessions'):
                cl.context.session.mcp_sessions = {}
            if not hasattr(cl.context.session, 'mcp_exit_stacks'):
                cl.context.session.mcp_exit_stacks = {}

            cl.context.session.mcp_sessions[mcp_config['name']] = (mcp_session, exit_stack)
            cl.context.session.mcp_exit_stacks[mcp_config['name']] = exit_stack

            step.output = f"Connected! Found {len(tool_list.tools)} tools:\n" + "\n".join([
                f"- {t.name}: {t.description[:80]}..." if len(t.description) > 80 else f"- {t.name}: {t.description}"
                for t in tool_list.tools[:5]
            ])
            if len(tool_list.tools) > 5:
                step.output += f"\n... and {len(tool_list.tools) - 5} more"

            return True

    except Exception as e:
        logger.error(f"Failed to connect to {mcp_config['name']}: {e}")
        await cl.Message(content=f"Failed to connect to {mcp_config['name']}: {str(e)}").send()
        return False




@cl.on_chat_start
async def start():
    cl.user_session.set("message_history", [])

    # Initialize MCP sessions storage
    if not hasattr(cl.context.session, 'mcp_sessions'):
        cl.context.session.mcp_sessions = {}
    if not hasattr(cl.context.session, 'mcp_exit_stacks'):
        cl.context.session.mcp_exit_stacks = {}

    # Send authorized MCP servers to frontend
    from chainlit.context import context
    await context.emitter.emit("authorized_mcps", {
        "mcps": [
            {
                "name": mcp["name"],
                "url": mcp["url"],
                "clientType": mcp["clientType"],
                "description": mcp.get("description", ""),
                "status": "disconnected",
                "tools": []
            }
            for mcp in AUTHORIZED_MCP_SERVERS
        ]
    })

    # Don't auto-connect - let user manually connect via plug icon
    # Don't send a greeting message - let user see the landing page
    # User will see the logo and can start chatting when ready


@cl.on_chat_end
async def end():
    """Clean up MCP connections when chat ends."""
    if hasattr(cl.context.session, 'mcp_exit_stacks'):
        for exit_stack in cl.context.session.mcp_exit_stacks.values():
            try:
                await exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error closing MCP connection: {e}")


async def get_all_mcp_tools():
    """Retrieve all tools from all connected MCP servers."""
    all_tools = []

    if not hasattr(cl.context.session, 'mcp_sessions'):
        return all_tools

    for mcp_name, (mcp_session, _) in cl.context.session.mcp_sessions.items():
        try:
            result = await mcp_session.list_tools()
            logger.info(f"MCP server '{mcp_name}' provides {len(result.tools)} tools")
            for tool in result.tools:
                logger.info(f"  - Tool: {tool.name} - {tool.description}")
                all_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                    "_mcp_connection": mcp_name
                })
        except Exception as e:
            logger.warning(f"Could not list tools from {mcp_name}: {e}")

    logger.info(f"Total MCP tools available: {len(all_tools)}")
    return all_tools


async def call_mcp_tool(tool_name: str, arguments: dict, max_retries: int = 2):
    """Execute an MCP tool from any connected MCP server with retry logic."""
    if not hasattr(cl.context.session, 'mcp_sessions'):
        return [{"type": "text", "text": "No MCP servers connected"}]

    for mcp_name, (mcp_session, _) in cl.context.session.mcp_sessions.items():
        try:
            result = await mcp_session.list_tools()
            tool_names = [t.name for t in result.tools]

            if tool_name in tool_names:
                async with cl.Step(name=tool_name, type="tool") as step:
                    step.input = json.dumps(arguments, indent=2)
                    logger.info(f"Calling MCP tool '{tool_name}' with arguments: {json.dumps(arguments, indent=2)}")

                    # Retry logic for connection failures
                    for attempt in range(max_retries + 1):
                        try:
                            result = await mcp_session.call_tool(tool_name, arguments)
                            logger.info(f"MCP tool '{tool_name}' returned result with {len(result.content) if hasattr(result, 'content') else 0} content items")

                            output_content = []
                            if hasattr(result, 'content'):
                                for content_item in result.content:
                                    if hasattr(content_item, 'text'):
                                        output_content.append(content_item.text)
                                    else:
                                        output_content.append(str(content_item))
                            else:
                                output_content.append(str(result))

                            step.output = "\n".join(output_content)
                            return result.content if hasattr(result, 'content') else [{"type": "text", "text": str(result)}]
                        except Exception as e:
                            if attempt < max_retries and "Connection closed" in str(e):
                                logger.warning(f"MCP tool '{tool_name}' failed (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                                await asyncio.sleep(1.0)
                                continue
                            else:
                                logger.error(f"Error calling MCP tool '{tool_name}': {str(e)}")
                                step.output = f"Error: {str(e)}"
                                return [{"type": "text", "text": f"Error calling tool: {str(e)}"}]
        except Exception:
            continue

    return [{"type": "text", "text": f"Tool {tool_name} not found in any connected MCP server"}]


def create_chart_from_data(data: dict, chart_type: str = "bar", title: str = "Agricultural Data"):
    """
    Create a Plotly chart from agricultural data.

    Args:
        data: Dictionary with 'labels' and 'values' keys, or a list of records
        chart_type: Type of chart - 'bar', 'line', 'scatter', 'pie'
        title: Chart title

    Returns:
        Plotly figure object
    """
    try:
        # Handle different data formats
        if isinstance(data, list):
            # Convert list of records to DataFrame
            df = pd.DataFrame(data)
            if len(df) == 0:
                return None

            # Try to identify x and y columns
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

            if chart_type == "bar":
                fig = go.Figure(data=[go.Bar(x=df[x_col], y=df[y_col])])
            elif chart_type == "line":
                fig = go.Figure(data=[go.Scatter(x=df[x_col], y=df[y_col], mode='lines+markers')])
            elif chart_type == "scatter":
                fig = go.Figure(data=[go.Scatter(x=df[x_col], y=df[y_col], mode='markers')])
            elif chart_type == "pie":
                fig = go.Figure(data=[go.Pie(labels=df[x_col], values=df[y_col])])
            else:
                fig = go.Figure(data=[go.Bar(x=df[x_col], y=df[y_col])])

        elif isinstance(data, dict):
            # Handle dict format with 'labels' and 'values'
            labels = data.get('labels', data.get('x', []))
            values = data.get('values', data.get('y', []))

            if chart_type == "bar":
                fig = go.Figure(data=[go.Bar(x=labels, y=values)])
            elif chart_type == "line":
                fig = go.Figure(data=[go.Scatter(x=labels, y=values, mode='lines+markers')])
            elif chart_type == "scatter":
                fig = go.Figure(data=[go.Scatter(x=labels, y=values, mode='markers')])
            elif chart_type == "pie":
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            else:
                fig = go.Figure(data=[go.Bar(x=labels, y=values)])
        else:
            return None

        # Update layout for better appearance
        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="",
            template="plotly_white",
            height=400
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return None


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])

    all_tools = await get_all_mcp_tools()

    # Check if MCP servers are connected
    if not all_tools:
        await cl.Message(
            content="I don't have access to agricultural data sources right now. Please connect to the NASS QuickStats MCP server using the plug icon (top right) to access agricultural statistics."
        ).send()
        return

    message_history.append({
        "role": "user",
        "content": message.content
    })

    system_message = {
        "role": "system",
        "content": """You are an agricultural data assistant providing direct access to USDA National Agricultural Statistics Service (NASS) QuickStats data.

PRIMARY FUNCTION:
- Help users query, download, and view agricultural statistics from NASS QuickStats
- Focus exclusively on agricultural topics: crop production, livestock, prices, yields, acreage, surveys, and farm statistics
- Provide factual, data-driven responses grounded ONLY in information from connected MCP servers

CRITICAL RULES:
1. ONLY answer questions about agriculture, farming, and agricultural statistics
2. For non-agricultural topics, politely decline and redirect: "I'm specialized in agricultural data from NASS QuickStats. I can only assist with questions about crop production, livestock, farm statistics, agricultural prices, and related agricultural data."
3. NEVER provide information from general knowledge - ALL agricultural facts must come from MCP tool calls
4. If no MCP servers are connected, inform the user: "I don't have access to agricultural data sources right now. Please connect to the NASS QuickStats MCP server using the plug icon to access agricultural statistics."
5. If the requested information is not available from the MCP tools, clearly state: "I don't have access to that specific data in the NASS QuickStats database. The available data may not include [specific request]."
6. Reject harmful, inappropriate, or off-topic requests immediately

RESPONSE GUIDELINES:
- Be concise and data-focused
- PROACTIVELY execute queries - don't ask for confirmation unless the request is genuinely ambiguous
- USE MULTIPLE TOOLS when needed to fully answer a question - you can call multiple tools in sequence
- When users ask for comparisons or multi-year data, immediately call the appropriate tools with sensible defaults
- After retrieving data, AUTOMATICALLY create visualizations when appropriate - don't ask permission
- Present data clearly with relevant context (units, time periods, locations)
- If you retrieve the wrong data type (e.g., production instead of yield), IMMEDIATELY make new queries with the correct parameters - don't ask permission
- Only ask clarifying questions if the request is truly ambiguous (e.g., user says "corn" but doesn't specify grain vs silage vs sweet corn)

CRITICAL - OUTPUT FORMATTING:
- ALWAYS format your responses using proper Markdown syntax
- For data comparisons, ALWAYS create markdown tables using | pipes and dashes
- Use **bold** for emphasis and key terms
- Use ## headers to organize sections
- Example table format:
  | Year | Iowa | Arizona |
  |------|------|---------|
  | 2020 | 2,283,300,000 | 5,858,000 |
- Number formatting: Use commas for readability (e.g., 2,283,300,000 not 2283300000)
- Include units in headers or values (e.g., "Bushels", "BU/ACRE")
- Add footnotes with * for clarifications
- Use bullet points for key observations
- Structure output with clear sections: comparison table, then key observations

UNDERSTANDING USER QUESTIONS:
- "Compare yields" or "yield comparison" = Query for YIELD data (BU/ACRE), not production
- "Compare production" or "total output" = Query for PRODUCTION data (BU total)
- If user doesn't specify, default to YIELD for crop comparisons (it's more meaningful)
- Common metrics: YIELD (BU/ACRE), PRODUCTION (BU), PRICE RECEIVED ($/BU), ACRES HARVESTED

IMPORTANT - HOW TO QUERY FOR SPECIFIC METRICS:
- get_nass_data returns ALL available data - it does NOT have a metric parameter
- The NASS API returns production, yield, price, acreage all mixed together
- You CANNOT filter by metric type with get_nass_data
- The tool returns whatever data is available for that commodity/year/state
- When you call get_nass_data, examine the returned data to find the metric you need
- Look for yield data in the results (usually labeled with "YIELD" or "BU / ACRE")
- If results don't contain the metric you need, the data may not be available

TOOL SELECTION FOR COMPARISONS:
- STATE comparison ("Compare Iowa vs Arizona corn yield"):
  → Call compare_nass_commodities with commodities=["CORN"], metric="YIELD", geography_level="state", and specify both states if possible
  → OR call get_nass_data separately for each state/year and parse results for yield data
- COMMODITY comparison ("Compare corn vs wheat in Iowa"):
  → Call compare_nass_commodities with both commodities, the desired metric, and the state

QUERYING MULTI-YEAR DATA:
- CRITICAL: The NASS API does NOT accept comma-separated years like "2020,2021,2022,2023"
- You MUST make SEPARATE tool calls for each year when querying multiple years
- For example, to compare Iowa and Arizona corn from 2020-2023, you must make 8 calls total:
  * get_nass_data(commodity="CORN", state="Iowa", year="2020")
  * get_nass_data(commodity="CORN", state="Iowa", year="2021")
  * get_nass_data(commodity="CORN", state="Iowa", year="2022")
  * get_nass_data(commodity="CORN", state="Iowa", year="2023")
  * get_nass_data(commodity="CORN", state="Arizona", year="2020")
  * get_nass_data(commodity="CORN", state="Arizona", year="2021")
  * get_nass_data(commodity="CORN", state="Arizona", year="2022")
  * get_nass_data(commodity="CORN", state="Arizona", year="2023")
- After getting all results, aggregate the data and create visualizations
- If data is missing for some years, inform the user which years are unavailable

VISUALIZATION CAPABILITIES:
- You can create charts to visualize agricultural data
- Use the create_chart tool when users ask for trends, comparisons, or visual representations
- Chart types available: bar (comparisons), line (trends over time), scatter (relationships), pie (proportions)
- Always provide both the data and a visualization when appropriate

Remember: You are a specialized agricultural data tool with visualization capabilities, not a general-purpose assistant."""
    }

    messages = [system_message] + message_history

    # Add custom chart creation tool
    chart_tool = {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "Create a visual chart from agricultural data. Use this when users ask to visualize, graph, plot, or chart data. Supports bar charts (comparisons), line charts (trends), scatter plots (relationships), and pie charts (proportions).",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Data to visualize with 'labels' (x-axis) and 'values' (y-axis) arrays",
                        "properties": {
                            "labels": {"type": "array", "items": {"type": "string"}},
                            "values": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["labels", "values"]
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "pie"],
                        "description": "Type of chart: bar (comparisons), line (trends), scatter (relationships), pie (proportions)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Descriptive title for the chart"
                    }
                },
                "required": ["data", "chart_type", "title"]
            }
        }
    }

    # Combine MCP tools with custom chart tool
    combined_tools = all_tools + [chart_tool] if all_tools else [chart_tool]

    try:
        # First call to check for tool calls (non-streaming)
        response = await client.chat.completions.create(
            messages=messages,
            max_completion_tokens=16384,
            model=deployment,
            tools=combined_tools,
            tool_choice="auto",
            stream=False
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            # Handle tool calls
            assistant_message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]
            }
            # Only add content if it's not None/empty
            if response_message.content:
                assistant_message["content"] = response_message.content
            else:
                assistant_message["content"] = None

            message_history.append(assistant_message)

            for i, tool_call in enumerate(tool_calls):
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Add delay between MCP tool calls to avoid overwhelming the server
                if i > 0 and function_name != "create_chart":
                    await asyncio.sleep(0.5)

                # Handle custom chart creation tool
                if function_name == "create_chart":
                    async with cl.Step(name="Creating Chart", type="tool") as step:
                        step.input = json.dumps(function_args, indent=2)

                        fig = create_chart_from_data(
                            data=function_args.get("data"),
                            chart_type=function_args.get("chart_type", "bar"),
                            title=function_args.get("title", "Agricultural Data")
                        )

                        if fig:
                            # Send the chart as an element
                            chart_element = cl.Plotly(
                                name=function_args.get("title", "chart"),
                                figure=fig,
                                display="inline",
                                size="large"
                            )
                            await cl.Message(content="", elements=[chart_element]).send()

                            step.output = "Chart created successfully"
                            tool_content = json.dumps({"status": "success", "message": "Chart displayed to user"})
                        else:
                            step.output = "Failed to create chart"
                            tool_content = json.dumps({"status": "error", "message": "Could not create chart from provided data"})
                else:
                    # Handle MCP tool calls
                    tool_result = await call_mcp_tool(function_name, function_args)

                    # Convert tool result to JSON-serializable format
                    if isinstance(tool_result, list):
                        serialized_result = []
                        for item in tool_result:
                            if hasattr(item, 'text'):
                                serialized_result.append({"type": "text", "text": item.text})
                            elif isinstance(item, dict):
                                serialized_result.append(item)
                            else:
                                serialized_result.append({"type": "text", "text": str(item)})
                        tool_content = json.dumps(serialized_result)
                    else:
                        tool_content = str(tool_result)

                # Always add tool response to message history
                message_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content
                })

            # Stream the second response after tool calls
            msg = cl.Message(content="")
            await msg.send()

            stream = await client.chat.completions.create(
                messages=[system_message] + message_history,
                max_completion_tokens=16384,
                model=deployment,
                stream=True
            )

            final_content = ""
            buffer = ""
            last_update_time = asyncio.get_event_loop().time()
            update_interval = 0.1  # Update every 100ms for smoother rendering

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    final_content += token
                    buffer += token

                    current_time = asyncio.get_event_loop().time()
                    # Stream buffer every 100ms or when buffer reaches 50 chars
                    if (current_time - last_update_time >= update_interval) or len(buffer) >= 50:
                        await msg.stream_token(buffer)
                        buffer = ""
                        last_update_time = current_time

            # Stream any remaining tokens in buffer
            if buffer:
                await msg.stream_token(buffer)

            await msg.update()
        else:
            # Stream the response if no tool calls
            msg = cl.Message(content="")
            await msg.send()

            stream = await client.chat.completions.create(
                messages=messages,
                max_completion_tokens=16384,
                model=deployment,
                stream=True
            )

            final_content = ""
            buffer = ""
            last_update_time = asyncio.get_event_loop().time()
            update_interval = 0.1  # Update every 100ms for smoother rendering

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    final_content += token
                    buffer += token

                    current_time = asyncio.get_event_loop().time()
                    # Stream buffer every 100ms or when buffer reaches 50 chars
                    if (current_time - last_update_time >= update_interval) or len(buffer) >= 50:
                        await msg.stream_token(buffer)
                        buffer = ""
                        last_update_time = current_time

            # Stream any remaining tokens in buffer
            if buffer:
                await msg.stream_token(buffer)

            await msg.update()

        message_history.append({
            "role": "assistant",
            "content": final_content
        })

        # Keep last 20 messages to ensure we don't break tool call sequences
        # (assistant with tool_calls + tool responses must stay together)
        cl.user_session.set("message_history", message_history[-20:])

    except Exception as e:
        error_message = f"Error: {str(e)}"
        await cl.Message(content=error_message).send()
