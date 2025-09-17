from sentence_transformers import SentenceTransformer
import json
import os
import fitz  # PyMuPDF

PDF_PATH = r"C:\Users\Blue_Emotions\OneDrive\websites\pregnancy chatbot\PregFAQ.pdf"
MODEL_NAME = 'all-MiniLM-L6-v2'
OUTPUT_JSON_FILE = 'pdf_embeddings.json'

def extract_chunks_from_pdf(pdf_path: str) -> list[str]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"error: the file '{pdf_path}' wasn't found")
    
    print(f"extracting and chunking text from: '{pdf_path}'...")
    doc = fitz.open(pdf_path)

    full_text = ""
    for page in doc:
        full_text += page.get_text("text")
    doc.close()

    chunks = full_text.split('\n\n')
    cleaned_chunks = [chunk.strip().replace('\n', ' ') for chunk in chunks]

    return [chunk for chunk in cleaned_chunks if len(chunk) > 100]

def embed_and_store_texts(texts: list[str], model: SentenceTransformer, output_path: str):
    print(f"generating embeddings for {len(texts)} text chunks...")

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings_as_list = embeddings.tolist()

    data_to_store = [
        {"text_chunk": text, "embedding": emb}
        for text, emb in zip(texts, embeddings_as_list)
    ]

    print(f"saving embeddings to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_store, f, indent=2)

if __name__ == "__main__":
    
    try:
        text_chunks = extract_chunks_from_pdf(PDF_PATH)

        if not text_chunks:
            print("warning: no suitable text chunks found")
        else:
            print(f"loading model: '{MODEL_NAME}'...")
            embedding_model = SentenceTransformer(MODEL_NAME)

            embed_and_store_texts(text_chunks, embedding_model, OUTPUT_JSON_FILE)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"an unexpected error occurred: {e}")
