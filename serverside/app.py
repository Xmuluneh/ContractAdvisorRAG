import os
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_app import configure_retrieval_chain
from script.utill import MEMORY, load_data, load_qna_data
from evaluate import evaluate_chain, evaluate_metrics, create_dataframe

app = FastAPI()
origins = [
    "http://localhost:5050",
    "http://localhost",
    "http://localhost:8000",  # Assuming this is where your frontend is served
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post('/api/chat')
async def chat(user_question: str = Form(...), files: List[UploadFile] = File(...)):
    print("inside chat api")
    print("Received user question:", user_question)
    for uploaded_file in files:
        contents = await uploaded_file.read()
        # Do something with the file contents if needed
    
    if not user_question:
        raise HTTPException(status_code=400, detail="User question is required")

    CONV_CHAIN = configure_retrieval_chain(files)

    response = CONV_CHAIN.run({
        "question": user_question,
        "chat_history": MEMORY.chat_memory.messages
    })

    return {"response": response}

@app.get('/api/messages')
async def messages():
    messages = [msg for msg in MEMORY.chat_memory.messages]
    return {"messages": messages}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
