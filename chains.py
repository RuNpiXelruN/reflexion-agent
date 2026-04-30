from dotenv import load_dotenv
import datetime
import sys
from pathlib import Path

# Allow `import schemas` when the runner's cwd is not the project root (e.g. IDE "Run file").
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv()

from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser, PydanticToolsParser)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer


llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher. Current time: {time}

            1. {first_instruction},
            2. Reflect and critique your answer. Be severe to maximise improvement.
            3. Recommend search queries to research information and improve your answer.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="""
    Provide ~250 word detailed answer to the user's question.
    """
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You MUST use numerical citations in your answer to ensure it can be verified.
    - Add a references section to the bottom of your answer (which does not count towards the word limit).
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer to make sure it doesn't exceed the word limit.
    - Your answer should be ~250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)

revise_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instructions
)

revise = (
    revise_prompt_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    | parser_pydantic
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="What is the best way to diy a sandstone wall that's about 600mm high using many sized stones, the largest being around 400mm wide and 300mm high?"
    )