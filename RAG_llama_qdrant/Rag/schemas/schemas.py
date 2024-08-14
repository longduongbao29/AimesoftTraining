from pydantic import BaseModel, Field 
from fastapi import UploadFile, File, UploadFile
from Rag.retriever.query_translation import Retriever, MultiQuery, RAGFusion, QueryDecompostion, StepBack, HyDE
class Question(BaseModel):
    question: str = Field(examples=["What is your name?"])

class File(BaseModel):
    file: UploadFile = File(...)
   
class RetrieverSchema(BaseModel):
    question: str
    mode: str
    decomposition_mode : str = Field(default=None)
