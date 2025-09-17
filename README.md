# pregnancy-chat-bot

# 🍼 Preggos Chatbot

A **pregnancy companion chatbot** built with **Streamlit**, **Sentence Transformers**, and **Flan-T5**.
It allows you to upload your own pregnancy-related PDFs, embed them into vectors, and then ask natural questions.
The bot retrieves the most relevant context and generates concise answers.

---

## ✨ Features

* 📄 **PDF-based QA** – chatbot answers questions using your uploaded pregnancy PDFs
* 🧠 **Embeddings** – powered by [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html) for semantic search
* 🤖 **LLM** – [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) for lightweight answer generation
* 🔍 **Context Retrieval** – fetches the most relevant text chunk using cosine similarity
* 🎨 **Streamlit UI** – simple and interactive web interface

---

## 🛠️ Tech Stack

* [Python 3.10+](https://www.python.org/)
* [Streamlit](https://streamlit.io/)
* [Sentence-Transformers](https://www.sbert.net/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [scikit-learn](https://scikit-learn.org/)
* [NumPy](https://numpy.org/)

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/preggos-chatbot.git
cd preggos-chatbot
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. First, generate embeddings from your pregnancy PDF (script not included in this snippet).
   Make sure you have a file `pdf_embeddings.json` in the project folder.

2. Run the Streamlit app:

```bash
streamlit run ragg.py
```

3. Open your browser at [http://localhost:8501](http://localhost:8501).

4. Type your question and get an answer with supporting context.

---

## 📂 Project Structure

```
preggos-chatbot/
│── ragg.py                # Main Streamlit app
│── pdf_embeddings.json    # Pre-generated embeddings file
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

---

## 🧪 Example

**Question:**

```
Should I calculate my pregnancy by weeks or months?
```

**Answer:**

```
Calculating pregnancy by weeks is more accurate. A full pregnancy is around 40 weeks (280 days).
```

**Retrieved Context:**
*(Expandable in UI)*

---

## 🚀 Future Improvements

* ✅ Support for **multiple retrieved chunks** (not just one)
* ✅ Option to **upload a PDF directly in the app**
* ✅ Integration with **larger LLMs** for better answers
* ✅ Mobile-friendly UI



Would you like me to also **write the `requirements.txt`** file for this project so everything installs smoothly on GitHub setup?
