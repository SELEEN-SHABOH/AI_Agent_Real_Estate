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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://syrialistings.com",
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

# ------------------------------
# Request / Response Models
# ------------------------------
class Question(BaseModel):
    question: str
    conversation_id: Optional[int] = None

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

class ConversationRename(BaseModel):
    title: str

# ------------------------------
# Helper Functions
# ------------------------------
def find_relevant_answer(question: str, db: Session):
    faqs = db.query(FAQ).all()
    if not faqs:
        return None

    all_entries = {f.id: f.question for f in faqs}

    best_match_id = max(all_entries, key=lambda id:
                        difflib.SequenceMatcher(None, question, all_entries[id]).ratio())

    similarity = difflib.SequenceMatcher(
        None, question, all_entries[best_match_id]).ratio()

    if similarity < 0.5:
        return None

    return db.query(FAQ).filter(FAQ.id == best_match_id).first().answer


def create_conversation_from_first_message(db: Session, first_text: str) -> Conversation:
    title = (first_text.strip()[:60]) or "محادثة جديدة"
    conv = Conversation(title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

# ------------------------------
# Routes
# ------------------------------

# List all conversations
@app.get("/conversations/", response_model=List[ConversationOut])
def list_conversations(db: Session = Depends(get_db)):
    rows = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return rows


# List messages in 1 conversation
@app.get("/conversations/{conv_id}/messages", response_model=List[MessageOut])
def get_conversation_messages(conv_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # رجّع الرسائل مرتبة من الأقدم للأحدث
    return db.query(Message).filter(
        Message.conversation_id == conv_id
    ).order_by(Message.created_at.asc()).all()
    

@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: int, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()
    return {"status": "deleted"}


@app.patch("/conversations/{conv_id}")
def rename_conversation(conv_id: int, data: ConversationRename, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.title = data.title
    conv.updated_at = datetime.utcnow()
    db.commit()
    return {"status": "updated"}
    

# MAIN endpoint
@app.post("/ask/")
async def ask_real_estate_agent(q: Question, db: Session = Depends(get_db)):
    try:
        # 1. conversation exist? or create new one
        conv = None
        if q.conversation_id:
            conv = db.query(Conversation).filter(Conversation.id == q.conversation_id).first()

        if not conv:
            conv = create_conversation_from_first_message(db, q.question)

        # 2. Save USER message
        user_msg = Message(conversation_id=conv.id, role="user", text=q.question)
        db.add(user_msg)
        db.commit()

        # 3. check FAQ answer
        relevant_answer = find_relevant_answer(q.question, db)

        # If no answer found — save unresolved
        if not relevant_answer:
            new_q = UnansweredQuestion(question=q.question)
            db.add(new_q)

            conv.updated_at = datetime.utcnow()
            db.commit()

            return {
                "answer": "لا يوجد جواب حاليًا لهذا السؤال. الرجاء التواصل مع فريق الدعم على الرقم 0999999999.",
                "conversation_id": conv.id
            }

        # 4. Ask OpenAI to rephrase
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

        # 5. Save assistant message
        ass_msg = Message(conversation_id=conv.id, role="assistant", text=answer)
        db.add(ass_msg)

        conv.updated_at = datetime.utcnow()
        db.commit()

        return {"answer": answer, "conversation_id": conv.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

