from typing import List, Dict
import json
from openai import OpenAI
from backend_config import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


def evaluate_pipelines(question: str, pipeline_outputs: List[Dict]) -> Dict:
    """
    Use GPT-4o as a judge to rate each pipeline answer.
    Returns structured JSON with scores + winner.
    """
    # Build a compact description for the judge
    answers_block = []
    for out in pipeline_outputs:
        answers_block.append(
            f"Pipeline {out['pipeline_id']} ({out['description']}):\nAnswer: {out['answer']}\n"
        )

    answers_text = "\n\n".join(answers_block)

    prompt = f"""
You are an evaluator for Retrieval-Augmented Generation systems.

Question: {question}

You will see answers from 4 different pipelines. For each pipeline, score:
- accuracy (1-10)
- relevance (1-10)
- cost_efficiency (1-10; shorter but still accurate answers are better).

Then pick a single winner: A, B, C, or D.

Return STRICT JSON only, with this structure:

{{
  "A": {{"accuracy": int, "relevance": int, "cost_efficiency": int}},
  "B": {{"accuracy": int, "relevance": int, "cost_efficiency": int}},
  "C": {{"accuracy": int, "relevance": int, "cost_efficiency": int}},
  "D": {{"accuracy": int, "relevance": int, "cost_efficiency": int}},
  "winner": "A" | "B" | "C" | "D"
}}

Answers:
{answers_text}
"""

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: wrap raw response
        data = {"raw": content}

    return data
