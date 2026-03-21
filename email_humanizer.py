"""
===========================================================================
 EMAIL HUMANIZER -- A Beginner's LangChain Single-Agent Project
===========================================================================

 WHAT THIS PROJECT TEACHES YOU:
   1. How LangChain works (chains, prompts, LLMs, tools, agents)
   2. How to build a SINGLE AGENT that uses tools
   3. How to connect LangChain to OpenAI
   4. How prompt templates shape LLM output
   5. How an agent "thinks" using a tool-calling loop

 HOW LANGCHAIN WORKS (the big picture):
   LangChain is a framework that makes it easy to build LLM-powered apps.

     [User Input] --> [Prompt Template] --> [LLM (GPT)] --> [Output]

   - Prompt Template : A reusable template with placeholders (like a form)
   - LLM            : The AI model that generates text (OpenAI GPT)
   - Output         : The generated response

 WHAT IS AN AGENT?
   An agent is an LLM that can USE TOOLS and DECIDE what to do next.
   Unlike a simple chain (input -> LLM -> output), an agent can:
     1. Think about what it needs to do
     2. Pick a tool to use
     3. Use the tool and see the result
     4. Decide if it needs more steps or if it's done

   This is the tool-calling loop:
     THINK -> ACT -> OBSERVE -> THINK -> ... -> FINAL ANSWER

 HOW THIS PROJECT FLOWS:
   1. User provides an email idea (e.g., "thank my team for Q4 results")
   2. Agent calls draft_email tool   -> creates a formal email draft
   3. Agent calls humanize_email tool -> rewrites it to sound natural
   4. Agent returns the final humanized email to the user

 KEY LANGCHAIN COMPONENTS USED:
   - ChatOpenAI      : LLM wrapper that sends prompts to OpenAI's GPT API
   - PromptTemplate  : Template with {placeholders} filled before sending to LLM
   - @tool decorator : Turns a Python function into a tool the agent can call
   - create_agent    : Wires LLM + tools + system prompt into a runnable agent

 SETUP:
   1. pip install -r requirements.txt
   2. Copy .env.example to .env and add your OpenAI API key
   3. python email_humanizer.py

 See langchain_tutorial.md for a full beginner's guide to LangChain.
 See architecture_diagram.drawio for a visual diagram of this project.
===========================================================================
"""

import logging
import sys
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EmailHumanizer")

logger.info("Starting Email Humanizer Agent...")

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    logger.error("OPENAI_API_KEY not set! Copy .env.example to .env and add your key.")
    sys.exit(1)

logger.info("API key loaded successfully")
logger.info("All LangChain components imported")
logger.info("Initializing the LLM (OpenAI GPT)...")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    verbose=True,
)

logger.info("LLM initialized: model=gpt-4.1-mini, temperature=0.7")
logger.info("Defining agent tools...")


@tool
def draft_email(idea: str) -> str:
    """
    Creates a structured email draft from a brief idea or topic.
    Use this tool FIRST when the user provides an email idea.
    Input should be the user's email idea or topic.
    Returns a formal email draft with subject, greeting, body, and closing.
    """
    logger.info(f"[Tool: draft_email] Received idea: '{idea}'")

    draft_prompt = PromptTemplate(
        input_variables=["idea"],
        template="""You are a professional email writer.
Given the following idea, write a structured email draft.

Idea: {idea}

Write the email with:
- A clear subject line
- Professional greeting
- Well-organized body (2-3 short paragraphs)
- Professional closing

Return ONLY the email, nothing else.""",
    )

    formatted_prompt = draft_prompt.format(idea=idea)
    logger.info("[Tool: draft_email] Sending prompt to LLM...")

    response = llm.invoke(formatted_prompt)

    logger.info("[Tool: draft_email] Draft created successfully!")
    return response.content


@tool
def humanize_email(draft: str) -> str:
    """
    Takes a formal email draft and rewrites it to sound more human,
    warm, and natural while keeping the core message intact.
    Use this tool AFTER draft_email to make the email sound natural.
    Input should be the full email draft text.
    Returns a humanized version of the email.
    """
    logger.info("[Tool: humanize_email] Humanizing the email draft...")

    humanize_prompt = PromptTemplate(
        input_variables=["draft"],
        template="""You are an expert at making emails sound human and natural.

Take this email draft and rewrite it to sound like a real person wrote it.

Rules:
- Use contractions (I'm, we're, don't, can't)
- Vary sentence length (mix short and long sentences)
- Add a touch of warmth and personality
- Remove corporate jargon and stiff phrases
- Keep it professional but approachable
- Keep the same core message and structure
- Make it sound like something you'd actually send to a colleague

Email draft:
{draft}

Return ONLY the humanized email, nothing else.""",
    )

    formatted_prompt = humanize_prompt.format(draft=draft)
    logger.info("[Tool: humanize_email] Sending to LLM for humanization...")

    response = llm.invoke(formatted_prompt)

    logger.info("[Tool: humanize_email] Email humanized successfully!")
    return response.content


tools = [draft_email, humanize_email]
logger.info(f"Tools registered: {[t.name for t in tools]}")
logger.info("Creating the agent...")

SYSTEM_PROMPT = """You are an Email Humanizer assistant. Your job is to help users
write natural, human-sounding emails.

When the user gives you an email idea, follow these steps:
1. First, use the draft_email tool to create a structured email draft.
2. Then, use the humanize_email tool to make the draft sound natural and warm.
3. Return the final humanized email to the user.

Always use both tools in order: draft first, then humanize."""

agent_graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    debug=True,
)

logger.info("Agent created and ready to run!")


def run_email_humanizer(email_idea: str) -> str:
    """
    Main function to run the email humanizer agent.

    Args:
        email_idea: A brief description of the email you want to write.
                    Example: "thank my team for hitting Q4 targets"

    Returns:
        A humanized, natural-sounding email.
    """
    logger.info("=" * 60)
    logger.info(f"USER'S EMAIL IDEA: {email_idea}")
    logger.info("=" * 60)
    logger.info("Agent is now thinking... watch the tool-calling loop below!")
    logger.info("-" * 60)

    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=email_idea)]}
    )

    final_email = result["messages"][-1].content

    logger.info("-" * 60)
    logger.info("Agent finished! Here's your humanized email:")
    logger.info("=" * 60)

    return final_email


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EMAIL HUMANIZER AGENT")
    print("  Powered by LangChain + OpenAI")
    print("=" * 60)
    print("\nDescribe the email you want to write, and the agent will")
    print("create a natural, human-sounding email for you.\n")
    print("Type 'quit' to exit.\n")

    while True:
        email_idea = input("Your email idea: ").strip()

        if not email_idea:
            print("Please enter an email idea.\n")
            continue

        if email_idea.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Happy emailing!")
            break

        try:
            humanized_email = run_email_humanizer(email_idea)

            print("\n" + "=" * 60)
            print("YOUR HUMANIZED EMAIL:")
            print("=" * 60)
            print(humanized_email)
            print("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Something went wrong: {e}")
            print(f"\nError: {e}")
            print("Please check your API key and try again.\n")
