import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Config
base_model_id = "meta-llama/Llama-2-7b-chat-hf"
adapter_path = "finetuned-llama-recipe-qlora"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device)
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    return tokenizer, adapter_model.eval()

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
st.set_page_config(layout="centered")
st.title("🍽️ Recipe Generator (Fine-Tuned LLaMA)")

dish_name = st.text_input("Enter a Dish Name", "Spaghetti Carbonara")

if st.button("Generate Recipe"):
    tokenizer, model = load_finetuned_model()

    prompt = f"""You are a master chef. Given the name of the dish, provide the ingredients and directions.

### Dish Name:
{dish_name}

### Ingredients:"""

    output = generate(model, tokenizer, prompt)

    st.subheader("🍳 Recipe:")
    st.code(output)
