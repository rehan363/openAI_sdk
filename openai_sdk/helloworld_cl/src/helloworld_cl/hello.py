import chainlit as cl
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os

gemini_api_key= os.getenv("GEMINI_API_KEY")

#STEP 1:  PROVIDER
provider= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

#STEP 2:  MODEL
model= OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

#STEP 3:  CONFIG
run_config= RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

#STEP 4:  AGENT
agent1= Agent(
    name="panaversity support agent",
    instructions="you are here to respond user questions related panaversity .",
)
    

@cl.on_chat_start
async def handel_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello! i am panavesity support agent. how can I help you?").send()


@cl.on_message
async def handel_message(message: cl.Message):
    history= cl.user_session.get("history")
    history.append({"role":"user", "content" : message.content})

    result= await Runner.run(
        agent1,
        input=history,
        run_config=run_config,
    )
    history.append({"role": "assistant", "content": result.final_output} )
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()