# 📚 Semantic Book Recommender

This project implements a sophisticated **semantic book recommendation system** powered by vector search and state-of-the-art Natural Language Processing (NLP) models. By leveraging **sentence embeddings**, **zero-shot classification**, and **emotion detection**, the system enables intelligent book retrieval and categorization that goes far beyond traditional keyword matching.

---

## 🚀 Features

* **Vector Search with ChromaDB** – Store and query high-density book embeddings for fast, scalable **semantic similarity** retrieval.
* **Sentence Embeddings** – Transform book text into dense vectors using the efficient `all-MiniLM-L6-v2` model for capturing deep context.
* **Zero-Shot Categorization** – Automatically classify book content into customizable genres or topics **without task-specific training data**.
* **Emotion Detection** – Analyze text for emotional tone, enriching recommendation personalization and offering a novel signal for filtering.
* **Interactive UI (Gradio)** – A user-friendly, web-based interface for easy interaction and demonstration of the recommender's capabilities.


---
## 📚 Data Source

The book recommender is trained and built upon a rich corpus of textual data:

* **Dataset:** **7k Books**
* **Source:** **Kaggle** (Link: *https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata*)
* **Content:** The dataset provides close to seven thousand books containing identifiers, title, subtitle, authors, categories, thumbnail url, description, published year, average rating, and number of ratings. The dataset is provided as comma-delimited CSV.
---
## ⚙️ Tech Stack

### 📦 Core Libraries

| Library | Role |
| :--- | :--- |
| **LangChain** | Orchestrates document loading, text splitting, embedding generation, and component chaining. |
| **ChromaDB** | High-performance, open-source vector database for semantic indexing and retrieval. |
| **Transformers (HuggingFace)** | Provides seamless access to powerful pre-trained NLP pipelines and models. |
| **Gradio** | Facilitates the creation of a simple, shareable web-based user interface. |

### 🔑 Models Used

| Component | Model | Highlights & Purpose |
| :--- | :--- | :--- |
| **Sentence Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | **Lightweight** (22M params) and efficient; converts book text into vectors for semantic search. |
| **Zero-Shot Classification** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Fine-tuned on NLI datasets; **excels in multi-domain classification** for genre/topic labeling. |
| **Emotion Analysis** | `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa backbone; provides **interpretable emotion predictions** (e.g., *anger, joy, sadness*). |

---

## 🛠️ Setup & Installation

To set up and run the Semantic Book Recommender locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ThivankaD/Book_Recommender.git](https://github.com/ThivankaD/Book_Recommender.git)
    cd Book_Recommender
    ```

2.  **Run the application:**
    Execute the main script (which initializes the vector store and launches the Gradio UI).
    ```bash
    python gradio-dashboard.py 
    ```

---

## 📂 Project Workflow

The recommendation system follows a structured pipeline from data ingestion to enriched search results:

1.  **Document Loading** – Book texts are ingested using LangChain's `TextLoader`.
2.  **Chunking** – Large texts are split into manageable, contextually-rich pieces using the `CharacterTextSplitter`.
3.  **Embedding Generation** – Each text chunk is vectorized using the HuggingFace sentence embedding model (`all-MiniLM-L6-v2`).
4.  **Vector Storage** – The vectorized chunks are indexed and persisted in the **ChromaDB** vector store.
5.  **Semantic Search** – User queries are embedded and compared against stored vectors to efficiently retrieve the most relevant book passages.
6.  **Categorization & Emotion Analysis** – The retrieved book context is further analyzed using the zero-shot and emotion models to provide **enhanced recommendations** with genre labels and emotional signals.
7.  **UI Display** – Results are presented through the **Gradio** web interface.



**Note:** **All initial data preprocessing (Chunking and Embedding Generation) was performed on Google Colab leveraging the high-performance NVIDIA T4 GPU.**
