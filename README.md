# Email Humanizer - LangChain Single Agent Project

A beginner-friendly project that teaches you how to build a **single agent** using **LangChain + OpenAI**. The agent takes a brief email idea and generates a natural, human-sounding email.

## What You'll Learn

- How LangChain works (LLMs, prompts, tools, agents)
- How to create tools using the `@tool` decorator
- How an agent decides which tools to call and in what order
- How `PromptTemplate` shapes LLM output
- How the agent's tool-calling loop works (think -> act -> observe -> repeat)

## How It Works

```
User's email idea
       |
       v
  [Agent thinks: "I need to draft an email first"]
       |
       v
  [Tool: draft_email] --> creates a formal, structured email
       |
       v
  [Agent thinks: "Now I should humanize this draft"]
       |
       v
  [Tool: humanize_email] --> rewrites it to sound natural and warm
       |
       v
  Final humanized email returned to user
```

## Prerequisites

- Python 3.10 or higher
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/NisargKadam/Langchain_sample_project.git
cd Langchain_sample_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example env file and add your real key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

## Run

```bash
python email_humanizer.py
```

You'll see an interactive prompt:

```
============================================================
  EMAIL HUMANIZER AGENT
  Powered by LangChain + OpenAI
============================================================

Describe the email you want to write, and the agent will
create a natural, human-sounding email for you.

Type 'quit' to exit.

Your email idea:
```

Type your email idea (e.g., `thank my team for finishing the project on time`) and the agent will generate a humanized email. You'll also see detailed logs showing the agent's reasoning and tool calls.

## Example

**Input:**
```
thank my team for finishing the project on time
```

**Output:**
```
Subject: Huge Thanks for Your Amazing Work on the Project!

Hey Team,

I hope you're all doing well! I just wanted to take a minute to say a big
thank you for all the hard work you put into getting the project done on time.
Your dedication and teamwork really made a difference, and I can't tell you
how much I appreciate it.

Each of you brought something special to the table, and I'm so proud to be
part of such a talented group. Let's keep this momentum going and continue
to achieve great things together!

Thanks again for everything!

Best,
[Your Name]
```

## Project Structure

```
.
├── email_humanizer.py   # Main agent code (fully commented)
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── .gitignore           # Keeps secrets and venv out of git
└── README.md            # This file
```

## Tech Stack

- [LangChain](https://python.langchain.com/) - Framework for building LLM applications
- [OpenAI GPT-4o-mini](https://platform.openai.com/) - The LLM powering the agent
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management
