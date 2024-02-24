from fastapi import FastAPI
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from scripts.utils import MEMORY
from scripts.doc_loader import load_document
from lanarky import LangchainRouter
from starlette.requests import Request
from starlette.templating import Jinja2Templates

from config import set_environment

set_environment()

app = FastAPI()

def create_chain():
    return ConversationChain(
        llm=ChatOpenAI(
            temperature=0,
            streaming=True,
        ),
        verbose=True,
    )


templates = Jinja2Templates(directory="serverside/templates")
chain = create_chain()


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


langchain_router = LangchainRouter(
    langchain_url="/chat", langchain_object=chain, streaming_mode=1
)
langchain_router.add_langchain_api_route(
    "/chat_json", langchain_object=chain, streaming_mode=2
)
langchain_router.add_langchain_api_websocket_route("/ws", langchain_object=chain)

app.include_router(langchain_router)

@app.post('/')
async def post(user_question: str, files: List[UploadFile]):
    for uploaded_file in files:
        contents = await uploaded_file.read()
        
    if not user_question:
        raise HTTPException(status_code=400, detail="User question is required")

    CONV_CHAIN = configure_retrieval_chain(files)

    response = CONV_CHAIN.run({
        "question": user_question,
        "chat_history": MEMORY.chat_memory.messages
    })

    return {"response": response}

