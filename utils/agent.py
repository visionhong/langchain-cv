from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType

from utils.custom_tools import (
    # ZeroShotObjectDetectoonTool,
    ImageTransformTool,
    ObjectEraseTool,
    ImageGenerationTool,
    MaleAnimeGenertorTool,
    FemaleAnimeGenertorTool
    )

def image_editor_agent():
    tools = [ImageTransformTool(), ObjectEraseTool()]
    
    return initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", request_timeout=120),
        max_iterations=1,
        verbose=True
    )
    
def image_generator_agent():
    tools = [ImageGenerationTool(), FemaleAnimeGenertorTool(), MaleAnimeGenertorTool()]
    
    return initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", request_timeout=120),
        max_iterations=1,
        verbose=True,
    )
    
