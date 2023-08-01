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
    device_map="auto",
    offload_folder="offload",
    offload_state_dict = True,
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

@st.cache_resource
def llm_qa():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="Database", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    
    # Infer embeddings
    retriever = db.as_retriever()
    qa_retriever=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_retriever

def process_answers(instructions):
    response = ''        
    generated_text = llm_qa(instructions)
    answer = generated_text['result']
    return answer, generated_text

def main():
    st.title('DocSense')
    with st.expander("About"):
        st.markdown(
            """
            Welcome to AI DocSense, your go-to web app for expertly answering AI-related questions. Ask us anything, and we'll provide expert answers to your AI-related queries, making complex concepts simple. Empowering knowledge through intuitive interactions with Generative AI.
            """
        )
    
    question = st.text_area("Enter your Queries")
    if st.button("Search"):
        st.info("QUESTION: " + question)
        st.info("ANSWER: ")
        answer, metadata = process_answers(question)
        st.write(answer)
        st.write(metadata)

if __name__ == "__main__":
    main()