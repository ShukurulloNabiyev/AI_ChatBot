import os
import streamlit as st
from datetime import datetime
import random
import time

from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict


class RAGChatbot:
    def __init__(self, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key

        # Oldindan belgilangan PDF fayl yo'li
        pdf_path = "sariqdevniminib.pdf"  # PDF fayl yo'li shu yerda o'zgartirilishi mumkin

        self.client = OpenAI()
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 10,
                "score_threshold": 0.5
            }
        )

        self.system_prompt = """
        Siz PDF hujjat bo'yicha mutaxassis bo'lgan AI yordamchisisiz. 
        Quyidagi qoidalarga rioya qiling:
        1. Faqat berilgan kontekst asosida javob bering
        2. Kontekstda to'liq javob bo'lmasa, qisman ma'lumot bering
        3. Hech qanday aloqador ma'lumot bo'lmasa, "Kerakli ma'lumot topilmadi" deb javob bering
        4. Javoblaringiz qisqa va tushunarli bo'lsin
        """

        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def retrieve_context(self, query: str) -> str:
        try:
            docs = self.retriever.get_relevant_documents(query)
            context_parts = [f"--- Kontekst Qismi ---\n{doc.page_content}" for doc in docs]
            return "\n".join(context_parts)
        except Exception as e:
            print(f"Kontekst olishda xato: {e}")
            return "Kontekst olishda muammo yuz berdi."

    def get_response(self, query: str) -> str:
        context = self.retrieve_context(query)
        augmented_query = f"""
        Kontekst: 
        {context}

        Savolga kontekst asosida aniq va qisqa javob bering:
        {query}

        Javob berish qoidalari:
        - Faqat kontekstdagi ma'lumotlarga tayanish
        - Kontekstda to'liq javob bo'lmasa, mavjud qismini tushuntirib bering
        - Hech qanday aloqador ma'lumot bo'lmasa, "Ma'lumot topilmadi" deb javob bering
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": augmented_query}],
                temperature=0.3,
                max_tokens=500,
                top_p=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Javob olishda xato: {e}"


import time  # Animatsion chiqish uchun kerak

def main():
    st.set_page_config(page_title="RAG Chatbot", layout="centered")

    # Styling qo'shildi
    st.markdown("""
        <style>
            body {
                background-color: #121212;
                color: white;
                font-family: Arial, sans-serif;
            }
            .chat-bubble {
                padding: 10px 15px;
                border-radius: 20px;
                margin: 10px 0;
                display: inline-block;
                word-wrap: break-word;
            }
            .ai-message {
                background-color: #1F1F1F;
                color: white;
                text-align: left;
                margin-left: 10px;
            }
            .user-message {
                background-color: #FF4500;
                color: white;
                text-align: right;
                margin-left: auto;
                margin-right: 10px;
            }
            .time-stamp {
                font-size: 10px;
                color: #888;
                display: block;
                margin-top: 5px;
                text-align: right;
            }
            textarea {
                background-color: #333333;
                color: white;
                border: none;
                border-radius: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Sariq Devni minib asari asosida ishlaydigan Chatbot")

    with st.sidebar:
        st.header("Sozlamalar")
        api_key = st.text_input("OpenAI API Kaliti", type="password")
        if st.button("Chatbotni Ishga Tushirish"):
            if api_key:
                try:
                    chatbot = RAGChatbot(api_key)
                    st.session_state.chatbot = chatbot
                    st.success("Chatbot muvaffaqiyatli ishga tushdi!")
                except Exception as e:
                    st.error(f"Xatolik: {e}")
            else:
                st.warning("Iltimos, API kalitini kiriting.")

    # Agar xabarlar mavjud bo'lmasa, boshlang'ich xabarlar ro'yxatini yaratamiz
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Avvalgi xabarlarni ko'rsatish
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end;">
                    <div class="chat-bubble user-message">
                        {msg["text"]}
                        <span class="time-stamp">{msg["time"]}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start;">
                    <div class="chat-bubble ai-message">
                        {msg["text"]}
                        <span class="time-stamp">{msg["time"]}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Foydalanuvchi xabari qo'shiladi
    if prompt := st.chat_input("Savolni kiriting"):
        current_time = datetime.now().strftime("%H:%M:%S")  # Vaqtni formatlash
        user_message = {"role": "user", "text": prompt, "time": current_time}
        st.session_state.messages.append(user_message)

        # Foydalanuvchi xabarini ko'rsatish
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <div class="chat-bubble user-message">
                    {prompt}
                    <span class="time-stamp">{current_time}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI javobini olish va sekin ko'rsatish
        if hasattr(st.session_state, "chatbot"):
            full_response = st.session_state.chatbot.get_response(prompt)
            ai_time = datetime.now().strftime("%H:%M:%S")
            ai_message = {"role": "assistant", "text": full_response, "time": ai_time}
            st.session_state.messages.append(ai_message)

            # Sekin yozish uchun bo'sh joy
            response_placeholder = st.empty()
            animated_response = ""
            for char in full_response:
                animated_response += char
                response_placeholder.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-start;">
                        <div class="chat-bubble ai-message">
                            {animated_response}▌
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                time.sleep(0.03)  # Harflar o'rtasidagi kechikish (0.03 soniya)

            # Javobni to'liq ko'rsatish
            response_placeholder.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start;">
                    <div class="chat-bubble ai-message">
                        {full_response}
                        <span class="time-stamp">{ai_time}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Iltimos, avval chatbotni ishga tushiring.")


if __name__ == "__main__":
    main()
