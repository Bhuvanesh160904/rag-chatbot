"""
FastAPI web server exposing the RAG chatbot as an API.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer_question

app = FastAPI(title="RAG Chatbot API")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    answer, sources = answer_question(request.question)
    return AnswerResponse(answer=answer, sources=list(set(sources)))