import argparse
import os

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

def main(file_path : str, question : str):

    #model = "llama2:13b" #mistral , zephyr
    model = "mistral"
    base_url = os.getenv("OLLAMA_ENDPOINT")

    vector_store = get_vector_store(file_path=file_path, model=model,base_url=base_url)
    prompt  = get_prompt()
    llm  = Ollama(model=model,base_url=base_url, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

    chain = (
        { "context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"Using {model} to answer question: {question} on url: {file_path}")
    chain.invoke(question)    

def get_vector_store(file_path : str, model : str, base_url : str) -> Chroma:
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    all_splits = text_splitter.split_documents(data)
    return Chroma.from_documents(documents=all_splits, embedding=OllamaEmbeddings(model=model, base_url=base_url), persist_directory="chromadb_pdf")

def get_prompt() -> ChatPromptTemplate:
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

if __name__ == "__main__":
    question = "Summarize the information in this document"
    file_path = r".\data\Artificial intelligence.pdf"
    main(file_path,question)

