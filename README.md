---
title: Medical Product Recommender AI
emoji: 💊
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
port: 8000
---

# Symptom-Based Over-The-Counter Medical Product Recommendation System

## Project Overview

This project implements an AI-powered system that recommends appropriate over-the-counter (OTC) medical products based on user-provided symptoms. It leverages a **Retrieval-Augmented Generation (RAG)** architecture, combining a custom product knowledge base with a powerful open-source Large Language Model (LLM) to provide relevant and informative recommendations.

## Objective

The primary objective was to build and deploy an AI system capable of:
* Taking natural language symptoms as input.
* Recommending suitable over-the-counter medical products.
* Providing essential details for each recommendation: Product Name, Use-Case, and Side Effects.

## Scope of Work & Implementation Details

### 1. Model Selection

* **Open-Source LLM:** `microsoft/phi-2` was chosen for its balance of performance and relatively compact size for local inference.
* **Frameworks:**
    * `Hugging Face Transformers`: For loading and interacting with the LLM.
    * `Sentence-Transformers`: For generating embeddings (numerical representations) of text data.
    * `FAISS (Facebook AI Similarity Search)`: For efficient similarity search in the vector database.
    * `FastAPI`: For building a robust and fast inference API.
    * `Uvicorn`: An ASGI server to run the FastAPI application.
    * `Pydantic`: For data validation and serialization within the API.

### 2. Dataset Preparation

* **Dataset Size:** A custom dataset of **~1779** (or your exact count, based on `rag_system.py` output) over-the-counter medical products was used.
* **Fields:** Each product entry includes:
    * `Product Name`
    * `Use For` (use-case)
    * `Side Effects`
* **Format:** The dataset is stored in `medical_products.json`, a JSON array of objects.
* **Generation Method:** The dataset was primarily generated synthetically using generative AI models (like Gemini/ChatGPT) through an iterative process of prompt engineering and data compilation, followed by manual validation to ensure correctness and adherence to the specified format.

### 3. Model Training / Integration (RAG Pipeline)

Instead of full LLM fine-tuning, a **Retrieval-Augmented Generation (RAG)** pipeline was implemented for efficiency and effectiveness:
* **Embedding Model:** `all-MiniLM-L6-v2` (from `sentence-transformers`) was used to convert product descriptions (combination of `Product Name` and `Use For`) into dense vector embeddings.
* **Vector Database:** `FAISS` was used to create a highly efficient index of these product embeddings.
* **Retrieval Process:** When a user query (symptoms) is received, the RAG system:
    1.  Converts the query into an embedding.
    2.  Performs a similarity search in the FAISS index to retrieve the top `k` (default 3) most relevant medical products from the `medical_products.json` dataset.
* **Augmentation & Generation:** The retrieved product information is then dynamically inserted as `context` into a specially crafted prompt for the `Phi-2` LLM. The LLM then generates a natural language recommendation, strictly based on the provided context, ensuring relevance and factual accuracy from the curated dataset.

### 4. Inference API

* **Framework:** `FastAPI`
* **Endpoint:** A `POST` endpoint `/recommend` is exposed:
    * **Input:** `JSON` object with a `symptoms` field (string).
    * **Output:** `JSON` object containing:
        * `recommendation` (string: AI-generated product recommendation)
        * `confidence_score` (float: a heuristic score based on retrieved product relevance)
        * `retrieved_products` (list: details of products retrieved by the RAG system, for transparency/debugging)
* **Model Loading:** The LLM and RAG system are initialized only once during the FastAPI application's startup (`@app.on_event("startup")`) to ensure efficient processing of subsequent requests.
* **Hardware Consideration:** The LLM runs on **CPU** due to GPU VRAM limitations on the development machine. This results in longer inference times per query.

### 5. Deployment

* **Local Deployment:** The solution is designed for local deployment using `uvicorn` to serve the FastAPI application.
* **Future/Cloud Deployment (Bonus consideration):** The API is built with `FastAPI` and can be easily containerized with Docker for deployment on cloud platforms like GCP, AWS, Render, Railway, HuggingFace Spaces, or Fly.io (backend-only).

### 6. Frontend (Optional for Demo)

* **Technology:** Simple HTML, CSS, and JavaScript.
* **Functionality:** A basic web interface (`index.html`) is provided where users can:
    * Input their symptoms into a text area.
    * Receive and display the recommended medicine details and side effects via the `/recommend` API.

## Deliverables Checklist

* [ ] **Clean and commented codebase:** All project files (`.py`, `.json`, `.html`, `requirements.txt`, `README.md`) organized in a single repository.
    * **GitHub Link:** [YOUR_GITHUB_REPOSITORY_LINK_HERE]
* [ ] **API for inference:** Implemented with `FastAPI` (`app.py`).
* [ ] **Model code for fine-tuning or RAG:** Implemented with RAG (`rag_system.py`, `medical_ai_system.py`).
* [ ] **Deployment link or Docker instructions:** Provided for local setup.
    * (Optional: Add notes on cloud deployment if you attempt it.)
* [ ] **Sample Q&A test cases:** Provided within `medical_ai_system.py`'s `if __name__ == "__main__":` block.
* [ ] **ReadMe:** This document.

## How to Run Locally

### Prerequisites

* Python 3.8+ installed.
* (Optional but recommended for Windows): Enable [Developer Mode](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development) for optimal Hugging Face caching performance (resolves symlink warnings).

### Setup Instructions

1.  **Clone the repository** (if hosted on GitHub) or ensure all project files are in a single directory (e.g., `D:\medical_ai_project`).
2.  **Ensure `medical_products.json` is in the root directory.** This file must contain your 1000+ medical product entries.
3.  **Create and Activate a Python Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows (Command Prompt):
    venv\Scripts\activate
    # On Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    ```
4.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    * **Note:** The first time `medical_ai_system.py` or `app.py` runs, the `microsoft/phi-2` LLM (approx. 7GB) will be downloaded. This requires an internet connection and will take significant time.

### Running the System

1.  **Start the FastAPI Server:**
    * Open a **new terminal/command prompt window.**
    * Activate your virtual environment (`(venv)`).
    * Navigate to your project directory (`cd D:\medical_ai_project`).
    * Run:
        ```bash
        uvicorn app:app --reload --host 0.0.0.0 --port 8000
        ```
    * Wait for the server to fully start and display `INFO: Application startup complete.`. This step involves loading the LLM into memory and can take several minutes.

2.  **Access the System in Your Browser:**
    * **Frontend Demo:** Open your web browser and go to `http://127.0.0.1:8000/`. You can input symptoms and get recommendations.
    * **API Documentation (Swagger UI):** Open your web browser and go to `http://127.0.0.1:8000/docs`. You can interact directly with the `/recommend` API endpoint here.

## Sample Q&A Test Cases (from project prompt)

You can test the system with these example symptoms:

* "I have a fever and body pain."
* "I'm sneezing a lot and have a runny nose."
* "I feel chest congestion and sore throat."
* "My stomach hurts and I have heartburn."
* "I have a minor cut on my finger and it's bleeding a little."
* "I feel very tired and dehydrated because of diarrhea."
* "My skin is itchy and red from an insect bite."
* "I have a dry cough and my throat is scratchy."
* "My nose is completely blocked and I can't breathe."

## How to Extend the System (Future Improvements)

* **Expand Dataset:** Continuously add more diverse medical products and their detailed information to `medical_products.json` for broader coverage and more nuanced recommendations.
* **Advanced Prompt Engineering:** Further refine the LLM's prompt to handle more complex queries, ensure stricter adherence to output formats, or incorporate multi-turn conversations.
* **LLM Fine-tuning:** For more domain-specific and accurate responses, fine-tuning a smaller LLM (e.g., Llama 3 8B, Mistral 7B) on a carefully curated Q&A dataset would significantly improve quality. This would require more computational resources (e.g., a GPU with 16GB+ VRAM or cloud services).
* **Robust Confidence Scoring:** Develop a more sophisticated method for calculating confidence scores, potentially incorporating LLM's internal probabilities or RAG re-ranking techniques.
* **Error Handling and Edge Cases:** Enhance error handling for unusual user inputs, ambiguous symptoms, or cases where no relevant products are found.
* **User Interface (UI) / User Experience (UX) Improvement:** Develop a more interactive and visually appealing frontend using frameworks like React, Next.js, or Vue.js.
* **Dockerization:** Containerize the FastAPI application using Docker for easier and more consistent deployment across different environments.
* **Cloud Deployment:** Deploy the Dockerized application to cloud platforms (e.g., AWS EC2/ECS, Google Cloud Run/App Engine, Render, Railway, HuggingFace Spaces) for accessibility and scalability.
* **Re-ranking in RAG:** Implement a re-ranking step in the RAG pipeline using a cross-encoder model to further improve the relevance of retrieved documents before passing them to the LLM.

## Acknowledgements

* Hugging Face `transformers` library and the `microsoft/phi-2` model.
* `sentence-transformers` library and `all-MiniLM-L6-v2` model.
* `FAISS` for efficient similarity search.
* `FastAPI` and `Uvicorn` for API development.
* Google Gemini for guiding the project development and dataset generation.

---
**Date of Completion:** [August 1, 2025]
**Author:** [YARRAPUREDDY LAKSHMI REDDY]