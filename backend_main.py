from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from backend_ingestion import pdf_bytes_to_text, merge_texts
from backend_ragpipelines import index_all_pipelines, run_all_pipelines
from backend_evaluator import evaluate_pipelines

app = FastAPI(
    title="RAG Pipeline Optimizer",
    description="Compare multiple RAG configurations on your documents.",
    version="0.1.0",
)

# Enable CORS so Streamlit frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files, extract text, and index into all pipelines.
    """
    try:
        texts = []
        for f in files:
            content = await f.read()
            if f.filename.lower().endswith(".pdf"):
                text = pdf_bytes_to_text(content)
            else:
                text = content.decode("utf-8", errors="ignore")
            texts.append(text)

        merged = merge_texts(texts)
        
        # Validate that we have actual text content
        if not merged or len(merged.strip()) < 10:
            return {
                "status": "error",
                "message": "No text content extracted from documents. Please check if the PDFs contain readable text."
            }
        
        print(f"Indexing {len(merged)} characters of text into all pipelines...")
        index_all_pipelines([merged])
        
        # Verify indexing worked
        from backend_ragpipelines import PIPELINES
        total_chunks = 0
        for p in PIPELINES:
            try:
                count = p.collection.count()
                total_chunks += count
                print(f"Pipeline {p.pipeline_id}: {count} chunks indexed")
            except:
                pass

        return {
            "status": "ok",
            "message": f"Documents indexed into all pipelines ({total_chunks} total chunks)"
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in upload_docs: {error_details}")
        return {"status": "error", "message": str(e), "details": error_details}


@app.post("/ask")
async def ask_question(payload: dict):
    """
    User provides: {"question": "..."}
    We run all pipelines and evaluate them.
    """
    question = payload.get("question", "")
    if not question:
        return {"error": "question is required"}

    pipeline_outputs = run_all_pipelines(question)
    evaluation = evaluate_pipelines(question, pipeline_outputs)

    return {
        "question": question,
        "pipelines": pipeline_outputs,
        "evaluation": evaluation,
    }
