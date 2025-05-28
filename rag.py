import getpass
import os
import yaml
import argparse
from PIL import Image
from typing_extensions import Annotated, TypedDict
from typing import Literal, List
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import StateGraph, START



class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    

def get_atlas_data(atlas_data_filepath: str):
    """_summary_

    Args:
        atlas_data_filepath (str): _description_

    Returns:
        _type_: _description_
    """
    with open(atlas_data_filepath) as f:
        # Parse YAML
        data = yaml.safe_load(f)

        first_matrix = data['matrices'][0]
        tactics = first_matrix['tactics']
        techniques = first_matrix['techniques']

        studies = data['case-studies']
        
        documents = [
            Document(
                page_content=doc["description"],
                metadata={k: v for k, v in doc.items() if k != "description"}
            )
            for doc in (tactics + techniques)
        ]
        documents += [
            Document(
                page_content=doc["summary"],
                metadata={k: v for k, v in doc.items() if k != "summary"}
            )
            for doc in studies
        ]
        return documents
    
    
def setup_environment():
    if not os.environ.get("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"

    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key (optional): ")

    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")
    
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
    
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("Enter your Hugging Face API key: ")
        
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter your Tavily API key: ")


def answer_questions(user_question: str, debug: bool = False):
    if os.environ["GROQ_API_KEY"]:
        model = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)        
    else:
        model = init_chat_model("mistralai:mistral-large-latest", temperature=0)
    
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = InMemoryVectorStore(embeddings)
    
    # Load documents
    documents = get_atlas_data("atlas-data/dist/ATLAS.yaml")
    if debug: print("Number of documents: ", len(documents), "\n\n")

    # Embed documents
    vector_store.add_documents(documents)

    # Get RAG prompt
    prompt = hub.pull("rlm/rag-prompt")
    if debug: print(f"The RAG prompt is : ", prompt.messages[0].prompt.template, "\n\n")
    
    # RAG tools
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(
            state["question"]
        )
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompted_messages = prompt.invoke(
            {"question": state["question"],
            "context": docs_content}
        )
        response = model.invoke(prompted_messages)
        return {"answer": response.content}

    # LangGraph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    if debug:
        with open("graph.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        Image.open("graph.png").show()
    
    result = graph.invoke({"question": user_question})
    print(result['answer'])
    return result['answer']
    
answer_questions("Comment une attaque par data poisoning fonctionne selon ATLAS ?", debug=True)

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", '-q', required=True, help="Input the question you want to ask.")
    parser.add_argument("--debug", '-d', action="store_true", help="To debug or not.")
    
    args = parser.parse_args()
    question: str = args.question
    debug: bool = args.debug
    
    answer_questions(question, debug)