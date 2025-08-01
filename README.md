# Simple RAG Application with Local Vector Store

This project is a demonstration of a Retrieval-Augmented Generation (RAG) application built using Python and the LangChain framework. It uses a local FAISS vector store to perform similarity searches on a document and generates answers to questions based on the retrieved context.

## Overview

The application performs the following steps:
1.  **Downloads Data**: Fetches a Wikipedia article and saves it as a text file.
2.  **Loads Documents**: Reads the text file from the `data` directory.
3.  **Splits Text**: Breaks the document into smaller, manageable chunks.
4.  **Creates Embeddings**: Converts the text chunks into numerical vectors using a sentence-transformer model.
5.  **Stores in Vector DB**: Stores these vectors in a local FAISS index for efficient similarity searching.
6.  **Retrieves and Generates**: Takes a user's question, retrieves the most relevant text chunks from the FAISS index, and uses them as context to generate an answer.

**Note**: The final answer generation step is currently simulated. To get real answers, you must integrate a Large Language Model (LLM).

## Features

-   **Local Vector Store**: Uses FAISS for local, on-disk vector storage. No external database is required.
-   **Modular Pipeline**: Built with LangChain, making it easy to swap out components (e.g., different embedding models, loaders, or vector stores).
-   **Extensible**: The code is structured to be easily extended with a real LLM, a web API, or more advanced features.

## Project Structure

```
rag_app/
├── data/
│   └── Generative_pre-trained_transformer.txt  # Downloaded data
├── faiss_index/
│   ├── index.faiss                             # FAISS index file
│   └── index.pkl                               # FAISS mapping file
├── venv/                                       # Python virtual environment
├── app.py                                      # Main application script
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

## How It Works

1.  **Ingestion**: The `app.py` script first downloads a Wikipedia article using `requests` and `BeautifulSoup`.
2.  **Loading**: `DirectoryLoader` loads the `.txt` file.
3.  **Chunking**: `RecursiveCharacterTextSplitter` splits the document to fit into the context window of a language model.
4.  **Embedding**: `HuggingFaceEmbeddings` (using the `all-MiniLM-L6-v2` model) converts each chunk into a vector.
5.  **Storing**: `FAISS.from_documents` creates a FAISS index from the embedded chunks and saves it locally to the `faiss_index` directory.
6.  **Querying**:
    - When a question is asked, it is also converted into a vector.
    - The FAISS index is searched to find the most semantically similar document chunks.
    - These chunks are inserted into a prompt template along with the original question.
    - This complete prompt is then ready to be sent to an LLM.

## Setup and Installation

1.  **Clone the repository or download the files.**

2.  **Create and activate a Python virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
    *On Windows, use `venv\Scripts\activate`*

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To run the script, execute the following command from within the `rag_app` directory:

```sh
python app.py
```

The script will:
- Download the article if it doesn't exist.
- Create and save the FAISS vector store in the `faiss_index` directory.
- Run two example queries and print the context-rich prompts that would be sent to an LLM.

## Next Steps and Improvements

This is a foundational example. To turn it into a full-fledged application, consider the following:

1.  **Integrate a Real LLM**:
    - Replace the `get_llm_response` function with a call to an actual LLM. You can use a local model via `llama-cpp-python` or an API service like OpenAI, Anthropic, or Google Gemini.

2.  **Separate Ingestion and Querying**:
    - Create a separate `ingest.py` script to handle data loading and vector store creation.
    - Modify `app.py` to only load the existing `faiss_index` for querying, making it faster to start.

3.  **Build a User Interface**:
    - Use a library like **Streamlit** or **Gradio** for a quick and easy web interface.
    - For a more robust solution, create a **FastAPI** backend and a **React** or **Vue.js** frontend.

4.  **Expand Data Sources**:
    - Use other LangChain loaders to ingest different types of files like PDFs (`PyPDFLoader`), Word documents, or even data from Notion or Slack.
