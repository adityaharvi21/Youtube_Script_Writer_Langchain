# LangChain Beginner's Tutorial Guide

A step-by-step guide to understanding LangChain by building an Email Humanizer agent.

---

## Table of Contents

1. [What is LangChain?](#1-what-is-langchain)
2. [Core Concepts](#2-core-concepts)
3. [Setting Up Your Environment](#3-setting-up-your-environment)
4. [Your First LLM Call](#4-your-first-llm-call)
5. [Prompt Templates](#5-prompt-templates)
6. [Tools -- Giving the LLM Superpowers](#6-tools----giving-the-llm-superpowers)
7. [Agents -- LLMs That Think and Act](#7-agents----llms-that-think-and-act)
8. [Putting It All Together -- Email Humanizer](#8-putting-it-all-together----email-humanizer)
9. [How the Agent Runs Step by Step](#9-how-the-agent-runs-step-by-step)
10. [Common Mistakes and Tips](#10-common-mistakes-and-tips)
11. [Next Steps](#11-next-steps)

---

## 1. What is LangChain?

LangChain is a Python framework for building applications powered by Large Language Models (LLMs) like OpenAI's GPT.

Think of it this way:
- **Without LangChain**: You write raw API calls to OpenAI, manually manage prompts, parse responses, handle tool calling yourself.
- **With LangChain**: You use pre-built components that snap together like building blocks.

```
[Your App] --> [LangChain] --> [OpenAI API] --> [GPT Model]
```

LangChain handles the plumbing so you focus on the logic.

---

## 2. Core Concepts

### The Basic Pipeline

Every LangChain app follows this pattern:

```
[Input] --> [Prompt Template] --> [LLM] --> [Output]
```

| Component        | What It Does                                    | Analogy                |
|------------------|-------------------------------------------------|------------------------|
| **LLM**          | The AI model that generates text                | The brain              |
| **Prompt Template** | A reusable template with placeholders        | A form with blanks     |
| **Tools**        | Functions the LLM can call                      | The hands              |
| **Agent**        | An LLM that decides which tools to use          | An employee with tools |
| **Messages**     | The conversation format (Human, AI, System)     | A chat thread          |

### Chain vs Agent

| Feature     | Chain                        | Agent                              |
|-------------|------------------------------|-------------------------------------|
| Flow        | Fixed: A -> B -> C           | Dynamic: LLM decides the path      |
| Tools       | Not used                     | LLM picks and calls tools           |
| Flexibility | Does the same thing every time| Can adapt based on input           |
| Use case    | Simple, predictable tasks    | Complex tasks needing decisions     |

---

## 3. Setting Up Your Environment

### Step 1: Install Python dependencies

```bash
pip install langchain langchain-openai openai python-dotenv
```

Or use the requirements.txt from this project:

```bash
pip install -r requirements.txt
```

### Step 2: Get an OpenAI API key

1. Go to https://platform.openai.com/api-keys
2. Create a new key
3. Copy it

### Step 3: Create a .env file

```bash
# .env
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 4: Load the key in Python

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

**Why .env?** Keeps your secret key out of your code and git history.

---

## 4. Your First LLM Call

The simplest thing you can do with LangChain -- send a message to GPT and get a response.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

response = llm.invoke("What is LangChain in one sentence?")
print(response.content)
```

**What's happening:**
1. `ChatOpenAI(...)` creates a connection to OpenAI's API
2. `llm.invoke(...)` sends your text to GPT and waits for a response
3. `response.content` is the actual text GPT returned

**Key parameters:**
- `model`: Which GPT model to use (`gpt-4.1-mini` is fast and cheap, `gpt-4o` is more capable)
- `temperature`: Randomness. 0 = deterministic, 1 = creative. Use 0.7 for natural writing.

---

## 5. Prompt Templates

Instead of hardcoding prompts, use templates with placeholders.

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short paragraph about {topic}."
)

filled_prompt = template.format(topic="artificial intelligence")
print(filled_prompt)
# Output: "Write a short paragraph about artificial intelligence."

response = llm.invoke(filled_prompt)
print(response.content)
```

**Why use templates?**
- Reusable -- same template, different inputs
- Cleaner code -- separates prompt logic from application logic
- Easier to test and modify

### Multiple Placeholders

```python
template = PromptTemplate(
    input_variables=["sender", "recipient", "topic"],
    template="Write an email from {sender} to {recipient} about {topic}."
)

prompt = template.format(sender="Alice", recipient="Bob", topic="project deadline")
```

---

## 6. Tools -- Giving the LLM Superpowers

A tool is a Python function that the agent can call. The `@tool` decorator registers it with LangChain.

```python
from langchain_core.tools import tool

@tool
def calculate_discount(price: float, percent: float) -> float:
    """Calculate the discounted price given a price and discount percentage."""
    return price * (1 - percent / 100)
```

**Important rules for tools:**
1. The **docstring IS the tool description** -- the agent reads it to decide when to use the tool
2. Use **type hints** on parameters -- LangChain uses these to validate inputs
3. Write **clear, specific docstrings** -- vague descriptions confuse the agent

### Good vs Bad Docstrings

```python
# BAD -- too vague, agent won't know when to use this
@tool
def process(text: str) -> str:
    """Process the text."""

# GOOD -- specific, agent knows exactly when to use this
@tool
def draft_email(idea: str) -> str:
    """Creates a structured email draft from a brief idea or topic.
    Use this tool FIRST when the user provides an email idea.
    Input should be the user's email idea or topic.
    Returns a formal email draft with subject, greeting, body, and closing."""
```

---

## 7. Agents -- LLMs That Think and Act

An agent is an LLM that can **decide** which tools to use and in what order.

### Creating an Agent

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

agent = create_agent(
    model=llm,
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant. Use tools when needed.",
    debug=True,
)
```

### Three ingredients:
1. **model** -- The LLM (GPT) that does the thinking
2. **tools** -- The functions the agent can call
3. **system_prompt** -- Instructions that define the agent's role and behavior

### Running the Agent

```python
from langchain_core.messages import HumanMessage

result = agent.invoke(
    {"messages": [HumanMessage(content="Your request here")]}
)

answer = result["messages"][-1].content
print(answer)
```

### The Tool-Calling Loop

When you call `agent.invoke()`, this happens internally:

```
1. RECEIVE   -- Agent gets the user's message
2. THINK     -- LLM reads messages + tool descriptions, decides what to do
3. ACT       -- LLM calls a tool with specific arguments
4. OBSERVE   -- Tool runs, result is added to message history
   (Steps 2-4 repeat until the LLM decides it's done)
5. RESPOND   -- LLM returns a final text answer
```

### System Prompt -- The Agent's Job Description

The system prompt is critical. It tells the agent:
- What role it plays
- What tools to use and in what order
- Any rules or constraints

```python
SYSTEM_PROMPT = """You are an Email Humanizer assistant.
When the user gives you an email idea:
1. First, use the draft_email tool to create a draft.
2. Then, use the humanize_email tool to make it natural.
3. Return the final email."""
```

---

## 8. Putting It All Together -- Email Humanizer

Here's the full flow of our Email Humanizer project:

```
User: "thank my team for finishing the project on time"
  |
  v
Agent receives the message
  |
  v
Agent THINKS: "I should draft this email first"
  |
  v
Agent CALLS: draft_email("thank my team for finishing the project on time")
  |
  v
draft_email tool:
  - Creates a PromptTemplate for email drafting
  - Fills in the user's idea
  - Sends to LLM
  - Returns formal email draft
  |
  v
Agent OBSERVES the draft
  |
  v
Agent THINKS: "Now I should humanize this"
  |
  v
Agent CALLS: humanize_email(the_draft_text)
  |
  v
humanize_email tool:
  - Creates a PromptTemplate for humanization
  - Fills in the draft
  - Sends to LLM
  - Returns warm, natural email
  |
  v
Agent OBSERVES the humanized email
  |
  v
Agent THINKS: "This looks good, I'm done"
  |
  v
Agent RESPONDS with the final humanized email
```

---

## 9. How the Agent Runs Step by Step

When you run `python email_humanizer.py`, here's what happens in order:

| Step | What Happens | Code |
|------|-------------|------|
| 1 | Logging is set up | `logging.basicConfig(...)` |
| 2 | API key is loaded from .env | `load_dotenv()` + `os.getenv(...)` |
| 3 | LLM connection is created | `ChatOpenAI(model="gpt-4.1-mini")` |
| 4 | Two tools are defined | `@tool def draft_email(...)` and `@tool def humanize_email(...)` |
| 5 | Agent is created with LLM + tools + prompt | `create_agent(model=llm, tools=tools, ...)` |
| 6 | Interactive loop starts | `input("Your email idea: ")` |
| 7 | User enters an idea | e.g., "apologize for missing a meeting" |
| 8 | Agent runs the tool-calling loop | `agent_graph.invoke({"messages": [...]})` |
| 9 | Final email is printed | `print(humanized_email)` |

---

## 10. Common Mistakes and Tips

### Mistake 1: Missing API key
```
Error: OPENAI_API_KEY not set
```
**Fix:** Create a `.env` file with your key. Make sure `load_dotenv()` is called before `os.getenv()`.

### Mistake 2: Bad tool docstrings
If the agent doesn't use your tools correctly, check the docstring. The agent reads it to decide *when* and *how* to use the tool.

### Mistake 3: Forgetting type hints on tool parameters
```python
# WRONG -- no type hint, LangChain can't validate
@tool
def my_tool(text):
    """Does something."""

# RIGHT
@tool
def my_tool(text: str) -> str:
    """Does something."""
```

### Mistake 4: Temperature too high or too low
- `temperature=0` -- Deterministic, always same output. Good for factual tasks.
- `temperature=0.7` -- Some creativity. Good for writing.
- `temperature=1.0` -- Very random. Usually too unpredictable.

### Tip: Use logging
Add `logger.info(...)` calls to see what your agent is doing. Set `debug=True` on `create_agent` to see the full reasoning.

### Tip: Start simple
Build a chain first (no agent), test it works, then upgrade to an agent with tools.

---

## 11. Next Steps

Once you're comfortable with this project, try:

1. **Add more tools** -- e.g., a `translate_email` tool or a `summarize_email` tool
2. **Try different models** -- swap `gpt-4.1-mini` for `gpt-4o` and compare quality
3. **Add memory** -- let the agent remember previous emails in the conversation
4. **Build a web UI** -- use Streamlit or Gradio to make it interactive
5. **Explore LangSmith** -- LangChain's tracing tool to debug and monitor agent runs

### Useful Links

- LangChain Docs: https://docs.langchain.com
- OpenAI API Docs: https://platform.openai.com/docs
- LangChain GitHub: https://github.com/langchain-ai/langchain
