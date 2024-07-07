import streamlit as st
from src.config import Config
from src.model import load_finetuned_model, load_model_and_tokenizer
from src.utils import generate_text

@st.cache_resource
def load_model():
    model, tokenizer = load_model_and_tokenizer(Config)
    finetuned_model = load_finetuned_model(Config.MODEL_NAME, Config.NEW_MODEL_NAME, Config.DEVICE_MAP)
    return finetuned_model, tokenizer

def main():
    st.title("LLaMA Math Fine-tuned Model")

    model, tokenizer = load_model()

    prompt = st.text_area("Enter your math question:", height=100)
    
    if st.button("Generate Answer"):
        if prompt:
            with st.spinner("Generating answer..."):
                answer = generate_text(model, tokenizer, prompt)
            st.write("Generated Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()