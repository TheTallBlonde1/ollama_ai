import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import ollama
from langchain.schema import (SystemMessage)

load_dotenv()

def test_chat_conversation():

    # model = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    model = ollama.Ollama(model="llama2",base_url="http://localhost:11434")
    template = """You are a chatbot called DB10, you will always respond in a short and concise manner.
            Your only task is to know about Arsenal FC. You should never to answer questions that are irrelevant to Arsenal FC."""

    memory = ConversationBufferMemory()
    memory.chat_memory.add_message(SystemMessage(content=template))

    conversation = ConversationChain(
        llm=model, 
        verbose=False,
        memory=memory
    )

    while True:

        user_input = input("\nEnter your command or prompt: ").strip().lower()

        if user_input in ['/bye', '/exit']:
            break
        elif user_input in ['/?', '/help']:
            display_help()
            continue
        elif len(user_input.strip()) == 0:
            print("Even I know you need to give me an input...")
            continue
        
        response = conversation.predict(input=user_input)
        PrintAI.print_response(value=response)



def display_help():
    print("\nAvailable Commands:")
    print("  /set          Set session variables")
    print("  /show         Show model information")
    print("  /bye          Exit")
    print("  /?, /help     Help for a command")
    print("  /? shortcuts  Help for keyboard shortcuts\n")


class PrintAI:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

    def print_prompt(value: str):
        print(PrintAI.WHITE + f"{value}" + PrintAI.RESET)

    def print_response(value: str):
        print(PrintAI.RED + f"{value}" + PrintAI.RESET)

if  __name__ == '__main__':
    test_chat_conversation()


# response = conversation.predict(input="How many league titles has Arsenal FC won?")
# print(response)

# response  = conversation.predict(input="What is the stadium for home games?")
# print(response)

# response = conversation.predict(input="Tell me a limerick with your name and the team name in it")
# print(response)
