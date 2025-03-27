import os
import pickle
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "names.index"
NAMES_FILE = "names.pkl"
HASH_FILE = "names_hash.txt"

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_hash(names):
    """Compute a hash of the names list."""
    names_str = ",".join(sorted(names))
    return hashlib.md5(names_str.encode('utf-8')).hexdigest()

def initialize_index(names):
    """Compute embeddings, create FAISS index, and save to disk."""
    embeddings = model.encode(names, convert_to_numpy=True)
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    dimension = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_normalized)
    
    faiss.write_index(index, INDEX_FILE)
    with open(NAMES_FILE, "wb") as f:
        pickle.dump(names, f)
    
    with open(HASH_FILE, "w") as f:
        f.write(compute_hash(names))
    
    return index

def load_index(current_names):
    """Load or rebuild the index based on changes in the names list."""
    rebuild = False
    current_hash = compute_hash(current_names)
    
    if os.path.exists(INDEX_FILE) and os.path.exists(NAMES_FILE) and os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            stored_hash = f.read().strip()
        if stored_hash != current_hash:
            print("Names list has changed. Rebuilding index...")
            rebuild = True
        else:
            print("Loading index from disk...")
    else:
        rebuild = True

    if rebuild:
        index = initialize_index(current_names)
    else:
        index = faiss.read_index(INDEX_FILE)
    
    return index

names = [
    "Geetha", "Gita", "Gitu", "Geethu", "Gitanjali", "Anjali", "Anuj",
    "Jonathan", "Jon", "Johnny", "Joanna", "Alice", "Bob", "Robert", "Rob", "Bobby"
]

index = load_index(names)

def search_name(input_name, top_k=3):
    """Search for the most similar names to the input name."""
    input_embedding = model.encode([input_name], convert_to_numpy=True)
    input_embedding_normalized = input_embedding / np.linalg.norm(input_embedding, axis=1, keepdims=True)
    
    scores, indices = index.search(input_embedding_normalized, top_k)
    matching_names = [(names[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
    best_match = matching_names[0]
    
    return best_match, matching_names

if __name__ == "__main__":
    user_input = input("Enter a name to search: ")
    best_match, matches = search_name(user_input, top_k=5)
    print("\nBest Match:")
    print(f"Name: {best_match[0]}, Similarity Score: {best_match[1]:.4f}")
    
    print("\nTop Matches:")
    for name, score in matches:
        print(f"Name: {name}, Similarity Score: {score:.4f}")
