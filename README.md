
# Multipdf_chatbot
## Stack:
1. app framework = fastapi
2. llm framework = langchain
3. embedding = HuggingFace
4. vector db = chroma
5. pdfloader = PyPDF Loader
6. llm = mGroq(llama-3.1-70b-versatile)

# Project Demo

Check out the demo of the Multi PDF Documents FastAPI RAG Chatbot for Custom Datasets:
[![Watch the video](https://img.youtube.com/vi/-Kov818J2d4/0.jpg)](https://youtu.be/-Kov818J2d4)

In this demo, I demonstrate how the chatbot uses FastAPI and advanced LLM frameworks to process and respond to queries based on multiple PDF documents. The system integrates various technologies to provide an intelligent Q&A service for custom datasets. Watch the demo to see how it works in action!

# Create Files:
create these files
1. Create folder named "data"
2. Place pdf files in data folder which you wanna use as knowledge base
3. Create ".env" file for api key
5. create "vector_db_dir" keep it empty, vector db data will store in it automatically when code runs.
6. run "pip install -r requireemnts.txt"
7. run "uvicorn main:app --reload"
