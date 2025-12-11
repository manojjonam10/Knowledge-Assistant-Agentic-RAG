from typing import List, Dict, Tuple, Optional
from .config import client, CHAT_MODEL
from .vector_store import VectorStore


class Context:
    def __init__(self, role: str, task: str):
        self.role = role
        self.task = task

    def as_prompt(self) -> str:
        role_desc = {
            "developer": "You are answering for a software developer who wants implementation-level detail.",
            "manager": "You are answering for a project manager who wants high-level summaries and risks.",
            "support": "You are answering for a support specialist who needs step-by-step instructions.",
        }.get(self.role, "You are answering for a general user.")

        task_desc = {
            "debugging": "The user is troubleshooting or debugging a problem.",
            "planning": "The user is planning or designing a system or process.",
            "onboarding": "The user is new and wants to understand the basics.",
        }.get(self.task, "The user is performing a general information task.")

        return f"{role_desc} {task_desc}"


class RetrievalAgent:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        return self.vs.search(query, k=top_k)


class AnswerAgent:
    def generate_answer(
        self,
        query: str,
        context: Context,
        retrieved_docs: List[Tuple[Dict, float]],
    ) -> str:
        system_context = context.as_prompt()
        sources_text = "\n\n---\n\n".join(
            [f"[Source: {d['metadata']['source_file']}]\n{d['text']}" for d, _ in retrieved_docs]
        )

        prompt = f"""
You are an AI assistant embedded in an internal team knowledge system.

{system_context}

The user asked:
\"\"\"{query}\"\"\"

You have access to the following knowledge snippets (they may be partial or overlapping):

{sources_text}

Instructions:
- Answer the question using ONLY the information from the snippets.
- If relevant snippets are missing, say you are not fully sure and specify what is missing.
- Tailor the language and depth to the user's role and task.
- At the end, list the sources you used in a short "Sources:" section.
"""

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful, honest knowledge assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


class CriticAgent:
    def critique(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Tuple[Dict, float]],
    ) -> str:
        """
        Checks if the answer seems grounded in the provided sources.
        """
        sources_text = "\n\n---\n\n".join(
            [f"[Source: {d['metadata']['source_file']}]\n{d['text']}" for d, _ in retrieved_docs]
        )

        prompt = f"""
You are an AI assistant that reviews another AI's answer for factual grounding.

User question:
\"\"\"{query}\"\"\"

Proposed answer:
\"\"\"{answer}\"\"\"

Available sources:
{sources_text}

Task:
- Check whether the answer is reasonably supported by the sources.
- If some parts seem speculative or unsupported, point them out.
- Return a short critique with one of the labels: [OK], [PARTIAL], or [RISKY].
"""

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict factual critic."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


class FeedbackAgent:
    def collect_feedback(self, answer: str) -> Optional[Dict]:
        print("\n---")
        print("Answer:\n")
        print(answer)
        print("\nWas this helpful? [y]es / [n]o / [enter=skip]: ", end="")
        choice = input().strip().lower()

        if choice not in ["y", "n"]:
            return None

        fb = {"helpful": choice == "y"}
        if choice == "n":
            print("Please briefly describe what was wrong or missing (or leave blank): ")
            comment = input("> ").strip()
            fb["comment"] = comment
        return fb


class PlannerAgent:
    """
    Break a high-level user query into 2–5 focused sub-tasks.
    """

    def plan(self, query: str, context: Context) -> List[str]:
        planning_prompt = f"""
You are part of a multi-agent knowledge assistant.

User role and task context:
{context.as_prompt()}

User's main question:
\"\"\"{query}\"\"\"

Your job:
- Break this into 2–5 smaller, concrete sub-questions or steps
  that can be answered using an internal knowledge base.
- Focus on *information needs*, not general advice.
- Each sub-question should be on its own line and start with "- ".

Example output:
- Clarify the goal of the onboarding process.
- Identify the key steps developers must complete in the first week.
- Determine any relevant policies that affect the process.

Now produce only the list of sub-questions, nothing else.
"""

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise planning agent."},
                {"role": "user", "content": planning_prompt},
            ],
            temperature=0.2,
        )

        raw = resp.choices[0].message.content.strip()

        # Parsing: lines that start with "- "
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- "):
                lines.append(line[2:].strip())
            elif line:  # fallback in case the model didn't use dashes
                lines.append(line)

        # fallback
        if not lines:
            lines = [query]

        return lines
