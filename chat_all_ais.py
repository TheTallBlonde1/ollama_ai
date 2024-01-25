
import os

from dotenv import load_dotenv
from langchain_community.llms import ollama
from langchain_openai import ChatOpenAI

from all_ais import (get_ai_response_llama2, get_ai_response_mistral,
                     get_ai_response_openai_gpt4,
                     get_ai_response_openai_gpt35turbo, get_ai_response_orca2,
                     get_ai_response_phi)

load_dotenv()

def display_help():
    print("\nAvailable Commands:")
    print("  /set          Set session variables")
    print("  /show         Show model information")
    print("  /bye          Exit")
    print("  /?, /help     Help for a command")
    print("  /? shortcuts  Help for keyboard shortcuts\n")

def print_model_options():
    print("Available AI Models (print /<model-number> or /<model-name> to change model") 
    print("  1. GPT-3.5 Turbo")
    print("  2. GPT-4")
    print("  3. Mistral")
    print("  4. Llama2")
    print("  5. Orca2")
    print("  6. Phi")


def chat_with_ai():
    # ai_models = {"1": get_ai_response_openai_gpt35turbo, "2": get_ai_response_openai_gpt4,
                 
    #              "gpt-3.5 turbo": get_ai_response_openai_gpt35turbo, "gpt-4": get_ai_response_openai_gpt4}

    ai_models = {"1" : get_ai_response_openai_gpt35turbo, "2" : get_ai_response_openai_gpt4,
                 "3" : get_ai_response_mistral, "4" : get_ai_response_llama2, "5" : get_ai_response_orca2, "6": get_ai_response_phi,
                 "gpt-3.5 turbo" : get_ai_response_openai_gpt35turbo, "gpt-4" : get_ai_response_openai_gpt4,
                 "mistral"      : get_ai_response_mistral ,  "llama2":   get_ai_response_llama2,    "orca2":  get_ai_response_orca2 ,   "phi": get_ai_response_phi
                 }
    print("Welcome to AI Chat!")
    print("Type '/? or /help' for available commands.")

    conversation_history = []

    choice = "mistral"
    ai_response_function = ai_models.get(choice)

    while True:

        user_input = input("\nEnter your command or prompt: ").strip().lower()

        if user_input in ['/bye', '/exit']:
            break
        elif user_input in ['/?', '/help']:
            display_help()
            continue
        elif f"/{user_input.lower()}" in ai_models.keys(): # check if user input is a valid key in
            choice = f"{user_input}"  # set choice to user input
            ai_response_function = ai_models.get(choice)  # get function from dictionary using choice as key
        elif len(user_input.strip()) == 0:
            print("Even I know you need to give me an input...")
            continue

        if ai_response_function:
            conversation_history.append(user_input)
            response = ai_response_function(conversation_history)
            print(f"\nAI Response (using {choice}):")
            print(response)
            conversation_history.append(response)
        else:
            print("Invalid AI model selection. Please try again.")

if __name__ == "__main__":
    chat_with_ai()