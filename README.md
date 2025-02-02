# ChatBot-Ollama

This project provides a FastAPI-based web application for scraping websites, storing the content in a FAISS index, and using a chatbot model (via Ollama) to answer queries based on the scraped data. The app handles pagination of websites and ensures that irrelevant content (such as ads, navigation, footers, etc.) is excluded from indexing.

## Features

- **Pagination Handling**: Scrapes up to 3 pages of a website (configurable).
- **Improved Text Extraction**: Filters out irrelevant content like "Read more" links, scripts, navigation bars, and footers.
- **Optimized Response Handling**: Uses FAISS to store and retrieve website content for accurate responses to user queries.
- **FastAPI Backend**: A fast and efficient backend built using FastAPI.
- **Ollama Integration**: Uses Ollama's Llama2 model to generate answers based on the scraped context.

## Requirements

Before running the application, make sure you have the following installed:

- Python 3.7 or higher
- pip (Python package manager)

Install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
## How It Works

### 1. Website Scraping
The scraping function (`scrape_and_index`) does the following:

- Requests the website and parses the HTML using BeautifulSoup.
- Extracts relevant paragraphs (ignores ads, footers, etc.).
- Follows pagination links to scrape multiple pages (up to a configurable limit of 3 pages).
- Converts the extracted text into chunks and generates embeddings using Ollama's `nomic-embed-text` model.
- Adds the chunks and embeddings into a FAISS index for fast retrieval.

### 2. Retrieving Context
When a user sends a query, the chatbot retrieves relevant chunks from the FAISS index that are most similar to the query by comparing the embedding of the query against the stored embeddings.

The relevant context is then combined with the user's query to create a prompt for the chatbot model (Llama2). The chatbot uses this combined prompt to generate an accurate response.

### 3. Chatbot Responses
The chatbot uses the Llama2 model (via Ollama) to generate responses based on the context retrieved from the website content. It ensures that the answer is based on the scraped data, allowing for more contextually accurate responses.

### Save and Load FAISS Index
The FAISS index and chunks are saved as `faiss_index.pkl` and `chunks.json`, respectively.

When the application starts, it attempts to load the saved index and chunks, allowing you to continue from where you left off.

## Important Notes
- The FAISS index and scraped chunks are stored locally. If you remove or lose these files, the system will start fresh and reindex websites.
- The `nomic-embed-text` embedding model is used for generating text embeddings. Make sure you have access to the Ollama API.
