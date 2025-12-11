import os
from .ingestion import load_documents
from .vector_store import VectorStore
from .agents import Context, RetrievalAgent, AnswerAgent, CriticAgent, FeedbackAgent, PlannerAgent

DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs")

def prompt_choice(prompt: str, valid_options, max_attempts: int = 2) -> str:
    """
    Ask the user to choose one of valid_options.
    - Only accept valid options (case-insensitive).
    - Allow 'exit' / 'quit' to terminate.
    - Exit the program after max_attempts invalid tries which is 2.
    """
    valid = {opt.lower() for opt in valid_options}

    for attempt in range(max_attempts):
        print(prompt)
        choice = input("> ").strip().lower()

        if choice in ["exit", "quit"]:
            print("Exiting...")
            raise SystemExit

        if choice in valid:
            return choice

        remaining = max_attempts - attempt - 1
        if remaining > 0:
            print(f"Invalid choice. Please select one of: {', '.join(valid_options)} "
                  f"(attempts left: {remaining})\n")
        else:
            print("Too many invalid attempts. Exiting.")
            raise SystemExit

def main():
    print("=== Knowledge Assistant (Agentic RAG POC) ===")

    # Build index
    docs = load_documents(DOC_DIR)
    if not docs:
        print(f"No documents found in {DOC_DIR}. Add .md or .txt files and try again.")
        return

    vs = VectorStore()
    vs.build(docs)

    retrieval_agent = RetrievalAgent(vs)
    answer_agent = AnswerAgent()
    critic_agent = CriticAgent()
    feedback_agent = FeedbackAgent()
    planner_agent = PlannerAgent()

    # Choose context
    role = prompt_choice(
        "\nChoose your role: [developer / manager / support]",
        ["developer", "manager", "support"],
    )

    task = prompt_choice(
        "Choose your task: [debugging / planning / onboarding]",
        ["debugging", "planning", "onboarding"],
    )

    context = Context(role=role, task=task)

    print("\nType your question (or 'exit' to quit):")
    while True:
        query = input("\nQ> ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            print("Bye.")
            break

        # Break query into sub-tasks
        subtasks = planner_agent.plan(query, context)
        print("\n[PlannerAgent] Decomposed query into the following sub-tasks:")
        for i, st in enumerate(subtasks, 1):
            print(f"  {i}. {st}")

        # Retrieval on the original query
        retrieved = retrieval_agent.retrieve(query, top_k=5)
        print(f"\n[RetrievalAgent] Retrieved {len(retrieved)} chunks.")

        # Answer generation
        answer = answer_agent.generate_answer(query, context, retrieved)
        print("\n[AnswerAgent] Generated answer.")

        # Critique
        critique = critic_agent.critique(query, answer, retrieved)
        print("\n[CriticAgent] Critique:\n")
        print(critique)

        # Feedback
        feedback = feedback_agent.collect_feedback(answer)
        if feedback:
            print("\n[FeedbackAgent] Feedback recorded:")
            print(feedback)

        print("\nYou can ask another question or type 'exit' to quit.")


if __name__ == "__main__":
    main()
