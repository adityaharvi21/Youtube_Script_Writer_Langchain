"""
===========================================================================
 EMAIL HUMANIZER — A Beginner's LangChain Single-Agent Project
===========================================================================

 WHAT THIS PROJECT TEACHES YOU:
 ─────────────────────────────
 1. How LangChain works (chains, prompts, LLMs, tools, agents)
 2. How to build a SINGLE AGENT that uses tools
 3. How to connect LangChain to OpenAI
 4. How prompt templates shape LLM output
 5. How an agent "thinks" using the ReAct loop

 HOW LANGCHAIN WORKS (the big picture):
 ───────────────────────────────────────
 LangChain is a framework that makes it easy to build apps powered by LLMs.
 Think of it like LEGO blocks:

   ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │  PROMPT   │───>│   LLM (GPT)  │───>│  OUTPUT  │
   │ TEMPLATE  │    │              │    │  PARSER  │
   └──────────┘    └──────────────┘    └──────────┘

 - Prompt Template : A reusable template with placeholders (like a form)
 - LLM            : The AI model that generates text (OpenAI GPT in our case)
 - Output Parser   : Cleans up the LLM's response into a usable format

 WHAT IS AN AGENT?
 ─────────────────
 An agent is an LLM that can USE TOOLS and DECIDE what to do next.
 Unlike a simple chain (input → LLM → output), an agent can:
   1. Think about what it needs to do
   2. Pick a tool to use
   3. Use the tool and see the result
   4. Decide if it needs more steps or if it's done

 This is called the "ReAct" loop (Reason + Act):

   THINK  →  ACT  →  OBSERVE  →  THINK  →  ... →  FINAL ANSWER

 In our project, the agent:
   - Receives your email idea
   - Uses a "draft_email" tool to create a formal draft
   - Uses a "humanize_email" tool to make it sound natural
   - Returns the final humanized email

===========================================================================
"""

import logging
import sys
import os

# ─────────────────────────────────────────────────────────────────────────
# STEP 1: SET UP LOGGING
# ─────────────────────────────────────────────────────────────────────────
# Logging lets us see exactly what's happening inside our program.
# It's like having a "behind the scenes" camera on the agent's thinking.
# Levels: DEBUG (most detail) → INFO → WARNING → ERROR → CRITICAL (least)
# ─────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,                          # Show INFO and above
    format="%(asctime)s [%(levelname)s] %(message)s",  # Timestamp + level + message
    handlers=[
        logging.StreamHandler(sys.stdout),       # Print to console
    ],
)
logger = logging.getLogger("EmailHumanizer")
logger.info("Starting Email Humanizer Agent...")


# ─────────────────────────────────────────────────────────────────────────
# STEP 2: LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────────────────
# We keep secrets (like API keys) in a .env file so they don't end up
# in our code or git history. python-dotenv reads that file for us.
# ─────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv

load_dotenv()  # Reads .env file and sets environment variables

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("sk-your"):
    logger.error("OPENAI_API_KEY not set! Copy .env.example to .env and add your key.")
    sys.exit(1)

logger.info("API key loaded successfully")


# ─────────────────────────────────────────────────────────────────────────
# STEP 3: IMPORT LANGCHAIN COMPONENTS
# ─────────────────────────────────────────────────────────────────────────
# Let's import everything we need from LangChain.
# Each import is explained below.
# ─────────────────────────────────────────────────────────────────────────

# ChatOpenAI — This is the LLM wrapper. It sends our prompts to OpenAI's
# GPT model and returns the response. Think of it as a phone line to GPT.
from langchain_openai import ChatOpenAI

# PromptTemplate — A template with placeholders like {email_idea}.
# LangChain fills in the blanks before sending to the LLM.
from langchain_core.prompts import PromptTemplate

# @tool decorator — Turns a regular Python function into a "tool" that
# an agent can discover and use. The function's docstring becomes the
# tool's description (the agent reads this to decide when to use it).
from langchain_core.tools import tool

# create_react_agent — Creates an agent that follows the ReAct pattern:
#   Reason → Act → Observe → Repeat
# This is the simplest and most common agent type in LangChain.
from langchain.agents import create_react_agent, AgentExecutor

# hub — LangChain Hub has pre-built prompt templates. We'll pull the
# standard ReAct prompt that tells the agent HOW to think and use tools.
from langchain import hub

logger.info("All LangChain components imported")


# ─────────────────────────────────────────────────────────────────────────
# STEP 4: INITIALIZE THE LLM (Large Language Model)
# ─────────────────────────────────────────────────────────────────────────
# We create a ChatOpenAI instance — this is our connection to GPT.
#
# Parameters explained:
#   model       : Which GPT model to use. "gpt-4o-mini" is fast & cheap.
#   temperature : Controls randomness (0 = predictable, 1 = creative).
#                 We use 0.7 for natural-sounding emails.
#   verbose     : When True, LangChain prints internal details.
# ─────────────────────────────────────────────────────────────────────────

logger.info("Initializing the LLM (OpenAI GPT)...")

llm = ChatOpenAI(
    model="gpt-4o-mini",   # Fast, affordable, and capable
    temperature=0.7,        # Slightly creative for natural writing
    verbose=True,           # Show what's being sent/received
)

logger.info("LLM initialized: model=gpt-4o-mini, temperature=0.7")


# ─────────────────────────────────────────────────────────────────────────
# STEP 5: DEFINE TOOLS
# ─────────────────────────────────────────────────────────────────────────
# Tools are Python functions that the agent can call.
# The @tool decorator registers them with LangChain.
#
# IMPORTANT: The docstring of each function IS the tool's description.
# The agent reads these descriptions to decide WHICH tool to use.
# Write clear, specific docstrings so the agent knows when to use each.
#
# Our agent has 2 tools:
#   1. draft_email    — Creates a structured email draft from an idea
#   2. humanize_email — Rewrites a draft to sound warm and natural
# ─────────────────────────────────────────────────────────────────────────

logger.info("Defining agent tools...")


@tool
def draft_email(idea: str) -> str:
    """
    Creates a structured email draft from a brief idea or topic.
    Use this tool FIRST when the user provides an email idea.
    Input should be the user's email idea or topic.
    Returns a formal email draft with subject, greeting, body, and closing.
    """
    # ------------------------------------------------------------------
    # HOW THIS TOOL WORKS:
    # 1. We create a PromptTemplate with a placeholder {idea}
    # 2. We fill in the placeholder with the user's idea
    # 3. We send the filled prompt to the LLM
    # 4. We return the LLM's response as the tool's output
    #
    # The agent will see this output and can pass it to the next tool.
    # ------------------------------------------------------------------

    logger.info(f"[Tool: draft_email] Received idea: '{idea}'")

    # Create a prompt template for drafting emails
    # The {idea} placeholder will be replaced with the actual idea
    draft_prompt = PromptTemplate(
        input_variables=["idea"],  # List of placeholders in the template
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

    # PromptTemplate.format() fills in the placeholders
    formatted_prompt = draft_prompt.format(idea=idea)
    logger.info(f"[Tool: draft_email] Sending prompt to LLM...")

    # llm.invoke() sends the prompt to OpenAI and returns the response
    response = llm.invoke(formatted_prompt)

    # response.content contains the actual text from GPT
    logger.info(f"[Tool: draft_email] Draft created successfully!")
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
    # ------------------------------------------------------------------
    # This tool takes the output from draft_email and makes it sound
    # like a real person wrote it — not a robot. The prompt instructs
    # the LLM to add warmth, use contractions, vary sentence length, etc.
    # ------------------------------------------------------------------

    logger.info(f"[Tool: humanize_email] Humanizing the email draft...")

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
    logger.info(f"[Tool: humanize_email] Sending to LLM for humanization...")

    response = llm.invoke(formatted_prompt)

    logger.info(f"[Tool: humanize_email] Email humanized successfully!")
    return response.content


# Collect tools into a list — we'll pass this to the agent
tools = [draft_email, humanize_email]
logger.info(f"Tools registered: {[t.name for t in tools]}")


# ─────────────────────────────────────────────────────────────────────────
# STEP 6: CREATE THE AGENT
# ─────────────────────────────────────────────────────────────────────────
# Now we put it all together!
#
# An agent needs 3 things:
#   1. An LLM        — to do the thinking (GPT)
#   2. Tools          — to take actions (our 2 email tools)
#   3. A Prompt       — to tell the agent HOW to think and use tools
#
# We use the standard "ReAct" prompt from LangChain Hub. This prompt
# tells the agent to follow the Think → Act → Observe loop.
#
# The AgentExecutor is the "runtime" that actually runs the agent loop:
#   - It sends the prompt to the LLM
#   - If the LLM wants to use a tool, it runs that tool
#   - It feeds the tool's output back to the LLM
#   - It repeats until the LLM gives a Final Answer
# ─────────────────────────────────────────────────────────────────────────

logger.info("Creating the ReAct agent...")

# Pull the standard ReAct prompt from LangChain Hub
# This prompt contains instructions like:
#   "You have access to the following tools: ..."
#   "Use the following format: Thought: ... Action: ... Observation: ..."
react_prompt = hub.pull("hwchase17/react")

logger.info("ReAct prompt loaded from LangChain Hub")

# Create the agent by combining: LLM + Tools + Prompt
# This doesn't RUN the agent yet — it just wires everything together
agent = create_react_agent(
    llm=llm,            # The brain (GPT)
    tools=tools,         # The hands (our email tools)
    prompt=react_prompt, # The instructions (ReAct format)
)

logger.info("Agent created")

# Wrap the agent in an AgentExecutor — this is what actually RUNS it
# Parameters:
#   agent          : The agent we just created
#   tools          : Same tools list (executor needs to know how to call them)
#   verbose        : True = print the agent's full thought process
#   handle_parsing_errors : True = gracefully handle any output format issues
#   max_iterations : Safety limit — stop after N loops to prevent infinite loops
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                # <-- This prints the full ReAct loop!
    handle_parsing_errors=True,  # Gracefully handle format errors
    max_iterations=10,           # Safety limit
)

logger.info("AgentExecutor ready -- the agent can now run!")


# ─────────────────────────────────────────────────────────────────────────
# STEP 7: RUN THE AGENT
# ─────────────────────────────────────────────────────────────────────────
# Finally! We invoke the agent with the user's email idea.
# The agent will:
#   1. THINK: "I need to draft an email first"
#   2. ACT:   Call draft_email tool
#   3. OBSERVE: Read the draft
#   4. THINK: "Now I should humanize this draft"
#   5. ACT:   Call humanize_email tool
#   6. OBSERVE: Read the humanized version
#   7. THINK: "This looks good, I'll return it"
#   8. FINAL ANSWER: Return the humanized email
# ─────────────────────────────────────────────────────────────────────────


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
    logger.info("Agent is now thinking... watch the ReAct loop below!")
    logger.info("-" * 60)

    # agent_executor.invoke() starts the ReAct loop
    # We pass a dict with "input" — this is what the agent sees as the task
    result = agent_executor.invoke({"input": email_idea})

    # The result is a dict with "input" (what we sent) and "output" (the answer)
    final_email = result["output"]

    logger.info("-" * 60)
    logger.info("Agent finished! Here's your humanized email:")
    logger.info("=" * 60)

    return final_email


# ─────────────────────────────────────────────────────────────────────────
# STEP 8: INTERACTIVE LOOP — Let the user try multiple emails
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  EMAIL HUMANIZER AGENT")
    print("  Powered by LangChain + OpenAI")
    print("=" * 60)
    print("\nDescribe the email you want to write, and the agent will")
    print("create a natural, human-sounding email for you.\n")
    print("Type 'quit' to exit.\n")

    while True:
        # Get the user's email idea
        email_idea = input("Your email idea: ").strip()

        if not email_idea:
            print("Please enter an email idea.\n")
            continue

        if email_idea.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Happy emailing!")
            break

        try:
            # Run the agent
            humanized_email = run_email_humanizer(email_idea)

            # Display the result
            print("\n" + "=" * 60)
            print("YOUR HUMANIZED EMAIL:")
            print("=" * 60)
            print(humanized_email)
            print("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Something went wrong: {e}")
            print(f"\nError: {e}")
            print("Please check your API key and try again.\n")
