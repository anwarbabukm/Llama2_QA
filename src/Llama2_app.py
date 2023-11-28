import transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from transformers import pipeline
import os 
import chromadb
from chromadb.config import Settings 
from torch import cuda, bfloat16

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Access the secret
hf_auth = os.environ.get('hf_key')

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
model_config = transformers.AutoConfig.from_pretrained(
    model_id, use_auth_token=hf_auth
)

@st.cache_resource
def load_model_and_tokenizer():
    model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    
    return model, tokenizer

#create embeddings here
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
device = torch.device('cpu')
persist_directory = "db"
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False)

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

@st.cache_resource
def extract_text_with_langchain_pdf(pdf_file):
    loader = UnstructuredFileLoader(pdf_file)
    documents = loader.load()
    #print(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text-generation',
        model = model,
        tokenizer = tokenizer,
        max_length = 2000,
        do_sample = True,
        temperature = 1,
        top_k= 5
        #device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    st.title("PDF Text Extractor")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write(uploaded_file.name)
        file_details = {
            "Filename": uploaded_file.name
        }
        with open(uploaded_file.name, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
            #st.write(extracts)
        
        with st.spinner('Embeddings are in process...'):
            extracts=extract_text_with_langchain_pdf(uploaded_file.name)
            db = Chroma.from_documents(extracts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
            db.persist()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)
        user_input = st.text_input("", key="input")

        # Initialize session state for generated responses and past messages
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]

        # Search the database for a response based on user input and update session state
        if user_input:
            global model, tokenizer
            model, tokenizer = load_model_and_tokenizer()

            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use three sentences maximum and keep the answer as concise as possible. 
            Always say "thanks for asking!" at the end of the answer. 
            {context}
            Question: {question}
            Helpful Answer:"""
            rag_prompt_custom = PromptTemplate.from_template(template)

            llm = llm_pipeline()
            db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
            retriever = db.as_retriever()  

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt_custom
                | llm
                | StrOutputParser())

            answer=rag_chain.invoke(user_input)

            st.session_state["past"].append(user_input)
            response = answer
            st.session_state["generated"].append(response)

        # Display conversation history using Streamlit messages
        if st.session_state["generated"]:
            display_conversation(st.session_state)

if __name__ == "__main__":
    main()
