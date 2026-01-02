from app.services.llm import get_llm


MAX_TURNS = 6  

def add_message(memory: dict, role: str, content: str):
    memory["history"].append({
        "role": role,
        "content": content
    })


def should_summarize(memory: dict) -> bool:
    return len(memory["history"]) > MAX_TURNS


def summarize_memory(memory: dict) -> dict:
    llm = get_llm(temperature=0)

    conversation = "\n".join(
        f'{m["role"]}: {m["content"]}'
        for m in memory["history"]
    )

    prompt = f"""
Summarize the following medical conversation.
Keep facts, symptoms, age group, and intent.
Remove small talk.

Conversation:
{conversation}

Summary:
"""

    summary = llm.invoke(prompt)
    try:
        summary = llm.invoke(prompt)
        summary_text = summary.content if hasattr(summary, "content") else summary
    except Exception:
        summary_text = ""

    return {
        "history": [],
        "summary": summary_text
    }
