"""
STEP 1: Simple LLM Chatbot
Just Vertex AI - No Datadog yet
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

# Vertex AI imports
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

print(f"Initializing Vertex AI with project: {PROJECT_ID}")
vertexai.init(project=PROJECT_ID, location=LOCATION)

app = FastAPI(title="Simple LLM Chatbot")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversations in memory
conversations = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = "gemini-1.5-flash"

class ChatResponse(BaseModel):
    session_id: str
    response: str

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "simple-chatbot",
        "project": PROJECT_ID
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Simple chat endpoint
    """
    try:
        print(f"Received message: {request.message[:50]}...")
        
        # Get or create chat session
        if request.session_id not in conversations:
            model = GenerativeModel(request.model)
            conversations[request.session_id] = model.start_chat()
            print(f"Created new session: {request.session_id}")
        
        chat_session = conversations[request.session_id]
        
        # Call Vertex AI
        print("Calling Vertex AI...")
        response = chat_session.send_message(request.message)
        print(f"Got response: {response.text[:50]}...")
        
        return ChatResponse(
            session_id=request.session_id,
            response=response.text
        )
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    """Reset a conversation"""
    if session_id in conversations:
        del conversations[session_id]
        return {"status": "reset", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}

if __name__ == "__main__":
    uvicorn.run(
        "simple_chatbot:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
