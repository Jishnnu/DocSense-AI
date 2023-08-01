import base64
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS

checkpoint = "LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",   # Check device configurations for better computation
    torch_dtype=torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipeline = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )

    local_llm = HuggingFacePipeline(pipeline=pipeline)
    return local_llm
