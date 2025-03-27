# bookxpert-assignment


# Matching names 

This task demonstrates a simple application that performs name matching using open-source machine learning tools. It uses an embedding model to convert names into vector embeddings, and FAISS for fast similarity search. Additionally, it persists the FAISS index and the names list to disk and automatically rebuilds the index if the names list changes.

## Features

- **Embedding Model:**  
  Uses the free, open-source model `all-MiniLM-L6-v2` from [sentence-transformers](https://www.sbert.net/) to generate vector embeddings for names.

- **Vector Database:**  
  Utilizes [FAISS](https://github.com/facebookresearch/faiss) to perform efficient similarity searches.

- **Persistence:**  
  Saves the FAISS index (`names.index`) and names list (`names.pkl`) to disk along with a hash (`names_hash.txt`) of the current names list. This avoids reinitializing on every run.

- **Automatic Change Detection:**  
  The application computes a hash of the names list and compares it to the stored hash. If the list has changed, it automatically rebuilds the index.

- **Output:** The best matching name and a list of top similar names (with their similarity scores) are returned.

# Finetune chef

This is a **Streamlit app** that compares responses from:

-  **LLaMA Base Model** (`meta-llama/Llama-2-7b-chat-hf`)
- **Fine-Tuned LoRA Adapter Model** trained on recipe data ()

---

##  What It Does

Enter a dish name (e.g., `Spaghetti Carbonara`), and the app:

1. Generates a recipe using the **base model**
2. Generates the same recipe using the **fine-tuned model**
3. Displays both responses **side by side**


# Speech-to-Text with Voice Feedback (Streamlit + Python)

This is a simple **Streamlit app** that lets you:

- Speak into your microphone
-  Convert speech to text using Google's speech recognition
-  Hear your own words repeated back using Text-to-Speech (TTS)

Perfect for testing voice interfaces or building the foundation for a voice assistant!

---

## Features

- ✅ Real-time speech recognition (English)
- ✅ Converts microphone input into text
- ✅ Speaks the result back using `pyttsx3` (offline TTS)
- ✅ Clean Streamlit UI with status messages

---

## 📦 Requirements

```bash
pip install streamlit speechrecognition pyttsx3 pyaudio

