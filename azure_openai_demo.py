import os
import chainlit as cl
from openai import AzureOpenAI


endpoint = "https://noah-mjw1vza6-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-5.2-chat"
deployment = "gpt-5.2-chat"
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello! I'm an AI assistant powered by Azure OpenAI GPT-5.2. How can I help you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": message.content,
                }
            ],
            max_completion_tokens=16384,
            model=deployment,
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                await msg.stream_token(chunk.choices[0].delta.content)

        await msg.send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
