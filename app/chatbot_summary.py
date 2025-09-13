import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# === Streamlit UI ===
st.set_page_config(page_title="📘 Ask Atomic Habits", layout="wide")
st.title("💬 Ask Atomic Habits")

query = st.text_input("Ask a question about the book:")
api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):

        # 1. Load the PDF
        loader = PyMuPDFLoader("atomic_habits.pdf")
        documents = loader.load_and_split()

        # 2. Embed & Index
        os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(documents, embeddings)

        # 3. Create Retriever Chain
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # 4. Ask question
        answer = qa.run(query)

        # 5. Show result
        st.markdown("### 📘 Summary Answer:")
        st.write(answer)
