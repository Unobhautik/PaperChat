import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class DocumentQA:
    def __init__(self):
        # Initialize Gemini components
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

    def load_document(self, file_path, file_type):
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return loader.load()

    def process_documents(self, documents, chunk_size=1000, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, chunks, save_path=None):
        self.vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        if save_path:
            self.vector_store.save_local(save_path)
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        return self.vector_store

    def load_vector_store(self, save_path):
        self.vector_store = FAISS.load_local(save_path, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        return self.vector_store

    def setup_qa_chain(self):
        template = """
        You are a helpful assistant that answers questions based on provided documents.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        Provide a comprehensive answer to the question based only on the provided context.
        If the answer cannot be determined from the context, say "I don't have enough information to answer this question."
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return self.qa_chain

    def ask_question(self, question):
        if not self.qa_chain:
            self.setup_qa_chain()
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }


# Streamlit UI
st.set_page_config(page_title="Document QA with Gemini", layout="wide")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ Google API Key not found. Please add your GOOGLE_API_KEY to the .env file.")
    st.markdown("""
    ### How to set up your Google API Key:

    1. Go to https://makersuite.google.com/app/apikey
    2. Sign in with your Google account
    3. Create an API key
    4. Create a `.env` file in your project directory with the following content:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```
    5. Replace `your_google_api_key_here` with the API key you generated
    6. Restart the Streamlit app
    """)
    st.stop()

st.title("🔍 Document QA with Gemini")
st.markdown("Upload your documents and ask questions about them using Google's Gemini AI!")

if 'doc_qa' not in st.session_state:
    st.session_state.doc_qa = DocumentQA()
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    process_button = st.button("Process Documents")

    if process_button and uploaded_files:
        with st.spinner("Processing documents..."):
            documents = []
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.name.split('.')[-1].lower()
                temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                try:
                    docs = st.session_state.doc_qa.load_document(temp_file_path, file_type)
                    documents.extend(docs)
                    st.success(f"Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")

            if documents:
                chunks = st.session_state.doc_qa.process_documents(documents)
                st.info(f"Created {len(chunks)} document chunks")
                vector_store_path = os.path.join(st.session_state.temp_dir, "faiss_index")
                st.session_state.doc_qa.create_vector_store(chunks, save_path=vector_store_path)
                st.session_state.doc_qa.setup_qa_chain()
                st.session_state.documents_processed = True
                st.success("Documents processed successfully! You can now ask questions.")

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Add your Google API key to the .env file
    2. Upload one or more PDF or TXT files
    3. Click "Process Documents"
    4. Ask questions in the main panel
    """)

if st.session_state.documents_processed:
    st.header("🤔 Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.doc_qa.ask_question(question)
                st.markdown("### Answer")
                st.write(result["answer"])
                st.markdown("### Sources")
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')}"):
                        st.markdown(doc.page_content)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
else:
    st.info("👈 Please upload and process documents using the sidebar first.")

st.markdown("---")
st.markdown("### About")
st.markdown("""
This application uses LangChain with Google's Gemini AI to create a document question answering system.
It processes your documents, creates embeddings, and uses retrieval-augmented generation to answer your questions.

Powered by:
- Google Gemini AI for text generation
- Google Embedding API for document embeddings
- FAISS for vector similarity search
- LangChain for the RAG pipeline
""")

if st.session_state.documents_processed:
    st.markdown("### Sample Questions")
    sample_questions = [
        "What are the key components of LangChain?",
        "How does document question answering work?",
        "What are some use cases for LangChain?"
    ]
    selected_question = st.selectbox("Try a sample question:", [""] + sample_questions)
    if selected_question and st.button("Ask Sample Question"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.doc_qa.ask_question(selected_question)
                st.markdown("### Answer")
                st.write(result["answer"])
                st.markdown("### Sources")
                for i, doc in enumerate(result["source_documents"]):
                    with st.expander(f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')}"):
                        st.markdown(doc.page_content)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
