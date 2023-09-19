
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType

from utils.multi_modal_tools import (
    ImageCaptionTool,
    ZeroShotImageClassificationTool, 
    ZeroShotObjectDetectoonTool,
    ImageTransformTool,
    ObjectEraseTool,
    ImageGenerationTool
    ) 

def multi_modal_agent():
    tools = [ImageGenerationTool(), ImageTransformTool(), ObjectEraseTool()]
    
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        input_key='input', output_key="output",
        return_messages=True
    )
    
    return initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=tools,
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", request_timeout=120),
        max_iterations=1,
        verbose=True,
        memory=conversational_memory,
        early_stopping_method='generate',
        return_intermediate_steps=True
    )
