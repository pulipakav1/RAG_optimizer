# RAG Pipeline Optimizer

A comprehensive MLOps platform for benchmarking and optimizing RAG (Retrieval-Augmented Generation) pipelines. Compare multiple RAG configurations with different chunking strategies, embedding models, and reranking techniques to find the best-performing setup for your use case.

## Features

- **Multi-Pipeline Benchmarking**: Test 4 different RAG pipeline configurations simultaneously
- **LLM-as-a-Judge Evaluation**: Uses GPT-4o to evaluate accuracy, relevance, and cost-efficiency
- **Interactive Dashboard**: Streamlit-based UI with bar charts, comparison tables, and detailed results
- **Document Support**: Upload PDF and text files for testing
- **Performance Metrics**: Compare pipelines on accuracy, relevance, and cost-efficiency scores


## Usage

1. **Upload Documents**: Use the Streamlit UI to upload PDF or text files
2. **Index Documents**: Click "Upload & Index" to process and index documents across all pipelines
3. **Ask Questions**: Enter a question and click "Run Evaluation"
4. **View Results**: 
   - See bar charts comparing all pipelines
   - Review the comparison table with detailed scores
   - Read answers from each pipeline


## Pipeline Configurations

The system tests 4 different RAG pipeline configurations:

- **Pipeline A**: Chunk=256, Embedding=text-embedding-3-small
- **Pipeline B**: Chunk=512, Embedding=text-embedding-3-large
- **Pipeline C**: Chunk=1024, Embedding=text-embedding-3-small
- **Pipeline D**: Chunk=512, Embedding=text-embedding-3-large (alternative)

Each pipeline uses:
- ChromaDB for vector storage
- OpenAI embeddings
- GPT-4o-mini for answer generation
- Custom chunking with 20% overlap

## Evaluation Metrics

- **Accuracy**: How correct and factual the answer is
- **Relevance**: How well the answer addresses the question
- **Cost Efficiency**: Balance between performance and API costs

Scores range from 1-10, with GPT-4o as the judge.

##  API Endpoints

- `POST /upload` - Upload and index documents
- `POST /ask` - Run evaluation on a question
- `GET /status` - Check document upload status
- `POST /reset` - Clear all indexed documents

## Technologies

- **Backend**: FastAPI, ChromaDB, OpenAI API
- **Frontend**: Streamlit, Plotly
- **ML/AI**: LangChain, OpenAI GPT-4o, Embeddings



