import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class url_pattern_recognition:
    def __init__(self, url_pipe):
        self.url_pipeline = url_pipe

    @classmethod
    def load_model(cls, url_pattern_model="JeswinMS4/URL_DETECTION"):
        url_model = AutoModelForSequenceClassification.from_pretrained(url_pattern_model)
        url_tokenizer = AutoTokenizer.from_pretrained(url_pattern_model)
        url_pipe = transformers.pipeline("text-classification", model=url_model, tokenizer=url_tokenizer)
        return cls(url_pipe)

    def __call__(self, input_text):
        url_detect = self.url_pipeline(input_text)
        return url_detect


def main():
    st.title("Malign/Benign URL Detection")

    # Load the URL detection model
    model = url_pattern_recognition.load_model()

    # Input box for the URL
    input_url = st.text_input("Enter URL:", "")

    if st.button("Detect"):
        if input_url:
            # Detect URL pattern
            predictions = model(input_url)
            for prediction in predictions:
                label = prediction["label"]
                score = prediction["score"]
                st.success(f"Label: {label}, Score: {score}")
        else:
            st.warning("Please enter a valid URL")


if __name__ == "__main__":
    main()
