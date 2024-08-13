from pydantic import BaseModel, Field 
from fastapi import UploadFile, File, UploadFile

class Question(BaseModel):
    question: str = Field(examples=["What is your name?"])

class File(BaseModel):
    file: UploadFile = File(...)
   