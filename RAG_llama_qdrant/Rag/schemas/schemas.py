from pydantic import BaseModel, Field
from fastapi import UploadFile, File, UploadFile
from typing import List
from enum import Enum


class ModeEnum(str, Enum):
    multi_query = "multi-query"
    rag_fusion = "rag-fusion"
    recursive_decomposition = "recursive-decompostion"
    individual_decomposition = "individual-decomposition"
    step_back = "step-back"
    hyde = "hyde"


class Question(BaseModel):
    question: str = Field(examples=["What is your name?"])


class File(BaseModel):
    file: UploadFile = File(...)


class RetrieverSchema(BaseModel):
    mode: ModeEnum
