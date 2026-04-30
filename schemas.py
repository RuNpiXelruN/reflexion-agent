from typing import List

from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing from the answer")
    superfluous: str = Field(description="Critique of what is superfluous in the answer")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word answer to the question")
    reflection: Reflection = Field(description="A reflection on the answer")
    search_queries: List[str] = Field(
        default_factory=list,
        description="1-3 search queries to research information and improve the answer",
    )

class ReviseQuestion(AnswerQuestion):
    references: List[str] = Field(
        default_factory=list,
        description="Citations to the information used in the answer",
    )