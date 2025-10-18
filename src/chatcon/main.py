import os
import chainlit as cl
from dotenv import load_dotenv,find_dotenv
from agents import Agent,Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from agents.tool import function_tool


load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")


external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",

) 


model = OpenAIChatCompletionsModel(
    model ="gemini-2.0-flash",
    openai_client=external_client,
)


config = RunConfig(
    model = model,
    model_provider=external_client,
    tracing_disabled=True)

@function_tool("get_weather")
def get_weather(location: str) ->str:
    """
    fetch the weather for a given location.
    """

    return f"the weather in {location} is 22 degrees C."



agent = Agent(name="Assistant",instructions="you are a helpful assistant,that tell weather",tools=[get_weather])


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello! i am assistant.How can i help you").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()


    history.append({"role":"user","content":message.content})
    result = Runner.run_streamed(
        agent,
        input=history,  
        run_config=config)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data,ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)


    history.append({"role":"assistant","content": result.final_output}) 
    cl.user_session.set("history",history)

   # await cl.Message(content=result.final_output).send()

