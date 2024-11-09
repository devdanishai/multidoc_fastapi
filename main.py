import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vd import embeddings

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Initialize vectorstore and chat chain
working_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore = None
conversation_chain = None

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=persist_directory,
                 embedding_function=embeddings)

def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile",
                   temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True
    )

@app.on_event("startup")
async def startup_event():
    global vectorstore, conversation_chain
    vectorstore = setup_vectorstore()
    conversation_chain = chat_chain(vectorstore)

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        response = conversation_chain({"question": message.message})
        return ChatResponse(response=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)