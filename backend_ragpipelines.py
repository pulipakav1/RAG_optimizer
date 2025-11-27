from typing import List
from backend_vectorscore import RAGPipeline

# We use different combinations of chunk size & embedding model
PIPELINES: List[RAGPipeline] = [
    RAGPipeline(
        pipeline_id="A",
        description="Chunk=256, Embedding=text-embedding-3-small",
        chunk_size=256,
        embedding_model="text-embedding-3-small",
    ),
    RAGPipeline(
        pipeline_id="B",
        description="Chunk=512, Embedding=text-embedding-3-large",
        chunk_size=512,
        embedding_model="text-embedding-3-large",
    ),
    RAGPipeline(
        pipeline_id="C",
        description="Chunk=1024, Embedding=text-embedding-3-small",
        chunk_size=1024,
        embedding_model="text-embedding-3-small",
    ),
    RAGPipeline(
        pipeline_id="D",
        description="Chunk=512, Embedding=text-embedding-3-large (alt)",
        chunk_size=512,
        embedding_model="text-embedding-3-large",
    ),
]


def index_all_pipelines(raw_texts: List[str]):
    """
    Index the same documents into each pipeline with its own strategy.
    """
    for p in PIPELINES:
        p.index_documents(raw_texts)


def run_all_pipelines(question: str):
    """
    Run all pipelines and collect their answers.
    """
    results = []
    for p in PIPELINES:
        res = p.answer(question)
        results.append(res)
    return results
