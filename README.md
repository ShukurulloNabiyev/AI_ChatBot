# Sariq Devni Minib RAG Chatbot 🤖📘

## Overview
This is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, powered by OpenAI's API, specifically designed to provide interactive Q&A based on Hududberdi Toxtaboyev's book "Sariq Devni Minib".

## Features
- 📚 Context-based responses from the book
- 🤖 AI-powered question answering
- 💬 Interactive Streamlit interface
- 🔍 Advanced context retrieval using FAISS vector store
- 📝 Customizable system prompts

## Prerequisites
- Python 3.8+
- OpenAI API Key
- Required Python libraries:
  - streamlit
  - openai
  - langchain
  - faiss-cpu

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sariq-devni-minib-chatbot.git
cd sariq-devni-minib-chatbot
```

2. Install required dependencies:
```bash
pip install streamlit openai langchain faiss-cpu
```

3. Prepare the PDF
- Place your "sariqdevniminib.pdf" in the project root directory
- Ensure the PDF is the correct version of the book

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. In the sidebar:
   - Enter your OpenAI API Key
   - Click "Chatbotni Ishga Tushurish" (Launch Chatbot)
   - Start asking questions about the book!

## Key Components
- `RAGChatbot` Class: Handles PDF processing, context retrieval, and response generation
- Custom system prompt for focused, context-aware responses
- FAISS vector store for efficient document retrieval
- Streamlit interface with animated responses

## Configuration
- Modify `pdf_path` to use a different PDF
- Adjust `chunk_size` and `chunk_overlap` in text splitting
- Customize system prompt in `self.system_prompt`

## Prompt Engineering
The chatbot uses a carefully crafted system prompt to:
- Provide context-based answers
- Handle partial information scenarios
- Maintain concise and clear responses

## Limitations
- Requires an active OpenAI API key
- Response quality depends on PDF content and chunk quality
- Limited to the knowledge within the provided PDF

## Contributing
Contributions, issues, and feature requests are welcome!

## License
[Add your license information]

## Acknowledgments
- Hududberdi Toxtaboyev for the original book
- OpenAI for the powerful language models
- Streamlit for the interactive web app framework