import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import requests
from bs4 import BeautifulSoup

# Function to download and save a Wikipedia article
def download_and_save_article(url, directory="data"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main content of the article
    content_div = soup.find(id="mw-content-text")
    if content_div:
        text = ""
        for p in content_div.find_all('p'):
            text += p.get_text() + "\n"
        
        # Use the last part of the URL as the filename
        filename = url.split('/')[-1] + ".txt"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Article downloaded and saved to {filepath}")
    else:
        print("Could not find the main content of the article.")

# --- Main RAG Application ---

# 1. Download data
article_url = "https://en.wikipedia.org/wiki/Generative_pre-trained_transformer"
download_and_save_article(article_url)

# 2. Load documents
loader = DirectoryLoader('data', glob="**/*.txt")
documents = loader.load()

# 3. Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# 4. Create embeddings and store in FAISS
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embedding_model)
vector_store.save_local("faiss_index")

# 5. Set up the retriever
retriever = vector_store.as_retriever()

# 6. Set up the LLM (using a placeholder for demonstration)
# In a real application, you would use a proper LLM from HuggingFace, OpenAI, etc.
# For this example, we'll simulate the LLM part.

def get_llm_response(prompt):
    # This is a placeholder. Replace with actual LLM call.
    print("\n--- LLM Prompt ---")
    print(prompt)
    print("--- End of LLM Prompt ---\n")
    return "This is a simulated response based on the retrieved context. To get a real answer, integrate a Large Language Model."

# 7. Create the RAG chain
def rag_query(question):
    retrieved_docs = retriever.invoke(question)
    
    context = ""
    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    prompt_template = f"""
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    
    return get_llm_response(prompt_template)

# --- Example Usage ---
if __name__ == "__main__":
    # Example query
    question = "What are the key features of generative pre-trained transformers?"
    
    print(f"Query: {question}")
    answer = rag_query(question)
    print(f"Answer: {answer}")

    # Example 2
    question = "What are the applications of GPT?"
    print(f"Query: {question}")
    answer = rag_query(question)
    print(f"Answer: {answer}")
