# 📚 Semantic Book Recommender AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Gradio](https://img.shields.io/badge/Gradio-UI-ff69b4)](https://www.gradio.app/)

An intelligent book recommendation engine that provides nuanced suggestions by combining a user's text query, desired genre, and emotional tone.

## ✨ About The Project

This project leverages modern NLP techniques to create a multi-faceted recommendation system that understands user intent with high precision. Unlike traditional recommenders, it uses a three-stage refinement process to deliver results that match a user's specific mood and interests.

The system's workflow is as follows:

1.  **Semantic Search:** A user's text query is converted into a vector embedding using the `all-MiniLM-L6-v2` model. This vector is then used to perform a high-speed similarity search against a pre-indexed **Chroma** vector database of book descriptions.

2.  **Categorical Filtering:** The initial list of semantically similar books is then filtered based on the user's selected genre (e.g., 'Fiction', 'Nonfiction').

3.  **Emotional Tone Sorting:** Finally, the filtered results are sorted based on the user's desired emotional tone (e.g., 'Happy', 'Sad', 'Thriller/Fearful'). This unique final step ensures that the most emotionally relevant books appear first.

The entire application is wrapped in a clean, interactive **Gradio** dashboard for a seamless user experience.

---

## 📊 Dataset

This project uses the **7k Books with Metadata** dataset, which is publicly available on Kaggle.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

The necessary data files, such as `books_sentiment.csv` and `tagged_description.txt` etc, are generated from this source dataset by running the preprocessing notebooks included in this repository.

---

## 🛠️ Technologies Used

* **Core Libraries:** Pandas, NumPy, Matplotlib, Seaborn
* **NLP & Embeddings:** LangChain, Transformers, Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** Chroma
* **Web Dashboard:** Gradio
* **Notebooks:** Jupyter for data processing and experimentation.

---

## 🏁 Getting Started

To get a local copy up and running, follow these simple steps.

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/vishwamohansinghnegi/books-recommender.git](https://github.com/vishwamohansinghnegi/books-recommender.git)
    cd books-recommender
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    * Download the `7k-books-with-metadata` dataset from [this Kaggle link](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata).
    * Unzip the file and place the `books.csv` file inside a `data/` directory in the project root.

---

## 🏃‍♀️ Usage

Before launching the app, you must first process the raw data.

1.  **Run the Preprocessing Notebooks:**
    * Open and run the Jupyter Notebooks (`vector_search.ipynb` and `sentiment_analysis.ipynb`) in order.
    * These notebooks will clean the data, perform sentiment analysis, and generate the essential `books_sentiment.csv` and `tagged_description.txt` files required by the application.

2.  **Launch the Gradio Dashboard:**
    * Once the data files are generated, run the dashboard script from your terminal:
    ```sh
    python gradio_dashboard.py
    ```
    * Open your web browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

3.  **Get Recommendations:**
    * Input your query, select a category and tone, and click the "Find recommendations" button!

---

## 📜 License

Distributed under the MIT License. See the `LICENSE` file for more information.

---

## 📧 Contact

Vishwamohan Singh Negi - [LinkedIn](https://www.linkedin.com/in/vishwamohan-singh-negi-001b8a257/)