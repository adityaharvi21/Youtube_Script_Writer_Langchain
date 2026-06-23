"""
===========================================================================
 Youtube Script Writer Agent
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
   1. User provides an video idea (e.g., "Best AI tools for students in 3 minutes")
   2. Agent calls create_video_outline tool   -> structured video outline
   3. Agent calls write_video_script tool -> writes a full video script based on the outline
   4. Agent returns the final YouTube video script to the user

 KEY LANGCHAIN COMPONENTS USED:
   - ChatOpenAI      : LLM wrapper that sends prompts to OpenAI's GPT API
   - PromptTemplate  : Template with {placeholders} filled before sending to LLM
   - @tool decorator : Turns a Python function into a tool the agent can call
   - create_agent    : Wires LLM + tools + system prompt into a runnable agent

 SETUP:
   1. pip install -r requirements.txt
   2. Copy .env.example to .env and add your OpenAI API key
   3. python youtube_script_writer.py

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
logger = logging.getLogger("YouTubeScriptWriter")

logger.info("Starting YouTube Script Writer Agent...")

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
def create_video_outline(topic: str, audience: str, length: str) -> str:
    """
    Creates a structured video outline based on the provided topic, audience, and length.
    Use this tool FIRST when the user provides a video idea.
    Input should be the user's video idea or topic.
    Returns a structured video outline with hook, key sections, and a closing call-to-action, with a one-line note per section.

    """
    logger.info(f"[Tool: create_video_outline] Received topic: '{topic}', audience: '{audience}', length: '{length}'")

    draft_prompt = PromptTemplate(
        input_variables=["topic", "audience", "length"],
        template="""You are a YouTube content strategist.

Create a high-retention video outline.

Topic: {topic}
Target Audience: {audience}
Video Length: {length}

Requirements:
- Start with a powerful hook (first 5 seconds idea)
- 3 to 5 key sections (each with a one-line explanation)
- Logical flow for viewer engagement
- End with a strong Call-To-Action (CTA)

Format:
Hook:
Sections:
1.
2.
3.
...
CTA:

Return ONLY the outline, nothing else."""
    )

    formatted_prompt = draft_prompt.format(topic=topic, audience=audience, length=length)
    logger.info("[Tool: create_video_outline] Sending prompt to LLM...")

    response = llm.invoke(formatted_prompt)

    logger.info("[Tool: create_video_outline] Outline created successfully!")
    return response.content


@tool
def write_video_script(outline: str) -> str:
    """
    Takes a video outline and expands it into a full natural sounding spoken script with an intro hook, transitions, and an outro/CTA.
    Use this tool AFTER create_video_outline to create the actual video script.
    Input should be the full video outline text.
    Returns a humanized version of the video script.
    """
    logger.info("[Tool: write_video_script] Writing the video script...")

    write_prompt = PromptTemplate(
        input_variables=["outline"],
        template="""You are an expert at writing engaging YouTube scripts.

Take this video outline and expand it into a full, detailed script.

Rules:
- Keep the tone conversational and engaging
- Include specific details and examples
- Maintain a good pace and rhythm
- Ensure smooth transitions between sections

Video outline:
{outline}

Return ONLY the video script, nothing else."""
    )

    formatted_prompt = write_prompt.format(outline=outline)
    logger.info("[Tool: write_video_script] Sending prompt to LLM...")

    response = llm.invoke(formatted_prompt)

    logger.info("[Tool: write_video_script] Video script written successfully!")
    return response.content




tools = [create_video_outline, write_video_script]
logger.info(f"Tools registered: {[t.name for t in tools]}")
logger.info("Creating the agent...")

SYSTEM_PROMPT = """You are a YouTube Script Writer assistant. Your job is to help users
create engaging, high-retention and well-paced video scripts which are short.

When the user provides a video topic and target length, follow these steps:

1. Extract:
   - Video Topic
   - Target Length

2. If target length is missing:
   - Assume default length = 1 minute

3. Find the target Audience for this topic. The target audience should be in 1 or 2 words (e.g., "beginners", "gamers", "parents", "students", "developers", etc.)

4. First, use the create_video_outline tool to generate a structured video outline.
   - Include a strong hook
   - 3–5 key sections
   - A closing call-to-action

5. Then, use the write_video_script tool to expand the outline into a full script.
   - Use a natural, conversational tone
   - Add smooth transitions
   - Keep viewer engagement high

6. Return only the final YouTube script to the user, do not add any sentences or words at the end.

Always use both tools in order:
create_video_outline → write_video_script.

Do NOT skip steps. Do NOT generate the final script without first creating the outline."""

agent_graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    debug=False,
)

logger.info("Agent created and ready to run!")


def run_youtube_script_writer(video_topic: str) -> str:
    """
    Main function to run the YouTube script writer agent.

    Args:
        video_topic: A brief description of the video topic.

    Returns:
        A structured YouTube script.
    """
    logger.info("=" * 60)
    logger.info(f"USER'S VIDEO TOPIC: {video_topic}")
    logger.info("=" * 60)
    #logger.info("Agent is now thinking... watch the tool-calling loop below!")
    #logger.info("-" * 60)

    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=video_topic)]}
    )

    final_script = result["messages"][-1].content

    logger.info("-" * 60)
    logger.info("Agent finished! Here's your YouTube script:")
    logger.info("=" * 60)

    return final_script


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  YOUTUBE SCRIPT WRITER AGENT")
    print("  Powered by LangChain + OpenAI")
    print("=" * 60)
    print("\nDescribe the video topic you want to create a script for, and the agent will")
    print("create a structured YouTube script for you.\n")
    print("Type 'quit' to exit.\n")

    while True:
        video_topic = input("Your video topic with the length of Video (in minutes): ").strip()

        if not video_topic:
            print("Please enter a video topic.\n")
            continue

        if video_topic.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Happy creating YouTube scripts!")
            break

        try:
            final_script = run_youtube_script_writer(video_topic)

            print("\n" + "=" * 60)
            print("YOUR YOUTUBE SCRIPT:")
            print("=" * 60)
            print(final_script)
            print("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"Something went wrong: {e}")
            print(f"\nError: {e}")
            print("Please check your API key and try again.\n")
