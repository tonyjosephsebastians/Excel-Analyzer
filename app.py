import streamlit as st
import pandas as pd
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Sidebar for API key input and model selection
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Microsoft_Office_Excel_%282019%E2%80%93present%29.svg/1200px-Microsoft_Office_Excel_%282019%E2%80%93present%29.svg.png", width=50)
    st.title("Excel Analyzer")
    api_key = st.text_input("Enter your API key:", type="password")
    model_name = st.selectbox("Select Model:", ["gemini-2.0-flash", "gemini-1.0-pro"]) # Add more models as needed

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.session_state["api_key_entered"] = True
    else:
        st.warning("Please enter your API key to continue.")
        st.stop()

if "api_key_entered" in st.session_state:

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("Data loaded successfully:")
            st.dataframe(df)

            # Convert DataFrame to text
            text = df.to_string()

            # Split into chunks
            text_splitter = CharacterTextSplitter(
                separator=",",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # Chatbot avatars
            user_avatar = "👤"
            bot_avatar = "🤖"

            # Chat container style
            chat_container_style = """
                <style>
                .chat-container {
                    border: 1px solid #ddd;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                }
                .user-message {
                    text-align: right;
                    color: blue;
                }
                .bot-message {
                    text-align: left;
                    color: green;
                }
                </style>
            """
            st.markdown(chat_container_style, unsafe_allow_html=True)

            # Ask a question
            question = st.text_input("Ask a question about the data:", value="", placeholder="Enter your question here...")
            if question:
                st.markdown(f"""
                    <div class="chat-container user-message">
                        {user_avatar} You: {question}
                    </div>
                """, unsafe_allow_html=True)

                docs = knowledge_base.similarity_search(question)

                model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
                chain = load_qa_chain(llm=model, chain_type="stuff")
                response = chain.run(input_documents=docs, question=question)

                st.markdown(f"""
                    <div class="chat-container bot-message">
                        {bot_avatar} Bot: {response}
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading data: {e}")
