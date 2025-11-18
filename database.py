# database.py
from sqlalchemy import create_engine, Column, Integer, Text, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if "localhost" in DATABASE_URL or "127.0.0.1" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL
    )
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"sslmode": "require"}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FAQ(Base):
    __tablename__ = "faqs"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

class UnansweredQuestion(Base):
    __tablename__ = "unanswered"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(32), nullable=False)  # 'user' or 'assistant'
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

# Create tables if they do not exist
Base.metadata.create_all(bind=engine)
