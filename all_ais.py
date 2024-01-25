import argparse
import os

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import chroma
from langchain_community.llms import ollama
from langchain_openai import ChatOpenAI

load_dotenv()

def test_ais(prompt : str):
    # print("\nTesting OpenAI 3.5 Turbo...")
    # PrintAI.print_prompt(prompt)
    # response = get_ai_response_openai_gpt35turbo(prompt)
    # PrintAI.print_response(response)

    # print("\nTesting OpenAI GPT4...")
    # PrintAI.print_prompt(prompt)
    # response = get_ai_response_openai_gpt4(prompt)
    # PrintAI.print_response(response)
    
    # print("\n\nTesting Mistral...\n")
    # PrintAI.print_prompt(prompt)
    # response = get_ai_response_mistral(prompt)
    # PrintAI.print_response(response)

    print("\nTesting Llama2...")
    PrintAI.print_prompt(prompt)
    response = get_ai_response_llama2(prompt)
    #PrintAI.print_response(response)

    # print("\nTesting Phi...")
    # PrintAI.print_prompt(prompt)
    # response = get_ai_response_phi(prompt)
    # PrintAI.print_response(response)

    # print("\nTesting Orca2...")
    # PrintAI.print_prompt(prompt)
    # response = get_ai_response_orca2(prompt)
    # PrintAI.print_response(response)


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
        print(PrintAI.BLUE + f"{value}" + PrintAI.RESET)

    def print_response(value: str):
        print(PrintAI.MAGENTA + f"{value}" + PrintAI.RESET)


def get_ai_response_ollama(model : str, prompt : str) -> str:
    llm = ollama.Ollama(model=model,base_url="http://localhost:11434", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    return llm.invoke(prompt)

def get_ai_response_mistral(prompt : str) -> str:
    response = get_ai_response_ollama("mistral", prompt)
    return response

def get_ai_response_llama2(prompt : str) -> str:
    response = get_ai_response_ollama("llama2:13b", prompt)
    return response

def get_ai_response_orca2(prompt : str) -> str:
    response = get_ai_response_ollama("orca2", prompt)
    return response

def get_ai_response_phi(prompt : str) -> str:
    response = get_ai_response_ollama("phi", prompt)
    return response

def get_ai_response_openai_gpt4(prompt : str) -> str:
    llm = ChatOpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    return response

def get_ai_response_openai_gpt35turbo(prompt : str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    prompt_test = "What is the sport of cricket, what is the ball made of. Tell me what Happens in a test match between two teams (step by step). Do this like you would for a layman"
    # prompt_test = "What is the weather like in London today?"

    parser = argparse.ArgumentParser(description="Process a prompt input.")
    parser.add_argument("--prompt", type=str, help="Input prompt for processing", default=prompt_test)
    args = parser.parse_args()
    test_ais(args.prompt)