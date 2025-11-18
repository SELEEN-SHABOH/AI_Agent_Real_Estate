# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI
import os
import difflib
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import datetime

from database import SessionLocal, FAQ, UnansweredQuestion, Conversation, Message

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Real Estate Assistant")

# Allow origins (add local dev + your production frontend domain(s))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://syrialistings.com:5174",
        "*" 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request / Response models
class Question(BaseModel):
    question: str
    conversation_id: Optional[int] = None  # يدعم ارسال id من الفلاتر لاضافة الى محادثة قائمة

class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

class MessageOut(BaseModel):
    id: int
    conversation_id: int
    role: str
    text: str
    created_at: datetime

# Helper: find best FAQ answer (as before)
def find_relevant_answer(question: str, db: Session):
    faqs = db.query(FAQ).all()
    if not faqs:
        return None
    all_entries = {f.id: f.question for f in faqs}
    best_match_id = max(all_entries, key=lambda id: difflib.SequenceMatcher(None, question, all_entries[id]).ratio())
    similarity = difflib.SequenceMatcher(None, question, all_entries[best_match_id]).ratio()
    if similarity < 0.5:
        return None
    return db.query(FAQ).filter(FAQ.id == best_match_id).first().answer

# Utility: create conversation with auto-title (use first user text as title, truncated)
def create_conversation_from_first_message(db: Session, first_text: str) -> Conversation:
    title = (first_text.strip()[:60]) or "محادثة جديدة"
    conv = Conversation(title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

# Endpoint: list conversations (history)
@app.get("/conversations/", response_model=List[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    rows = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return rows

# Endpoint: get messages for a conversation
@app.get("/conversations/{conv_id}/messages", response_model=List[MessageOut])
def get_conversation_messages(conv_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv.messages

# Main endpoint: ask — now saves messages and conversation
@app.post("/ask/")
async def ask_real_estate_agent(q: Question, db: Session = Depends(get_db)):
    try:
        # ensure conversation exists or create new
        conv = None
        if q.conversation_id:
            conv = db.query(Conversation).filter(Conversation.id == q.conversation_id).first()
        if not conv:
            conv = create_conversation_from_first_message(db, q.question)

        # save user message
        user_msg = Message(conversation_id=conv.id, role="user", text=q.question)
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)

        # find answer from FAQs
        relevant_answer = find_relevant_answer(q.question, db)
        if not relevant_answer:
            # store unanswered
            new_q = UnansweredQuestion(question=q.question)
            db.add(new_q)
            db.commit()
            # Update conversation updated_at
            conv.updated_at = datetime.utcnow()
            db.commit()
            return {"answer": "لا يوجد جواب حاليًا لهذا السؤال. الرجاء التواصل مع فريق الدعم على الرقم 0999999999.", "conversation_id": conv.id}

        # ask OpenAI to rephrase using the relevant_answer
        prompt = f"""
السؤال: {q.question}
الإجابة التالية من قاعدة النظام العقاري:
{relevant_answer}

أعد صياغة الإجابة بشكل واضح ومباشر دون إضافة معلومات جديدة.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أنت مساعد متخصص في النظام العقاري، تجيب فقط بناءً على النص المقدم"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content.strip()

        # save assistant message
        ass_msg = Message(conversation_id=conv.id, role="assistant", text=answer)
        db.add(ass_msg)
        # update conv title updated_at
        conv.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(ass_msg)

        return {"answer": answer, "conversation_id": conv.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
