from typing import List, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI
from backend_chunking import chunk_text
from backend_config import OPENAI_API_KEY


class OpenAIEmbeddingFn(embedding_functions.EmbeddingFunction):
    """
    Custom embedding function for Chroma that uses OpenAI.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]


class RAGPipeline:
    """
    Represents a single RAG pipeline:
    - specific chunk size
    - specific embedding model
    - its own Chroma collection
    """

    def __init__(
        self,
        pipeline_id: str,
        description: str,
        chunk_size: int,
        embedding_model: str,
        persist_directory: str = "./chroma_db",
    ):
        self.pipeline_id = pipeline_id
        self.description = description
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model

        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                persist_directory=persist_directory,
            )
        )

        embedding_fn = OpenAIEmbeddingFn(model_name=self.embedding_model)

        self.collection = self.client.get_or_create_collection(
            name=f"pipeline_{self.pipeline_id}",
            embedding_function=embedding_fn,
        )

        self.llm_client = OpenAI(api_key=OPENAI_API_KEY)

    def clear(self):
        # Get all IDs and delete them, or delete all by getting all results
        try:
            # Get all items in the collection
            results = self.collection.get()
            if results and results.get("ids"):
                # Delete all existing documents by their IDs
                self.collection.delete(ids=results["ids"])
        except Exception as e:
            # If collection is empty or doesn't exist, that's fine
            pass

    def index_documents(self, raw_texts: List[str]):
        """
        Given list of raw texts (from PDFs, etc.), chunk and store into Chroma.
        """
        self.clear()
        doc_id_counter = 0
        ids = []
        docs = []
        metadatas = []

        for doc_idx, raw_text in enumerate(raw_texts):
            # Filter out empty or very short texts
            if not raw_text or len(raw_text.strip()) < 10:
                print(f"Warning: Skipping empty or very short text for pipeline {self.pipeline_id}")
                continue
                
            chunks = chunk_text(raw_text, chunk_size=self.chunk_size, overlap=int(self.chunk_size * 0.2))
            # Filter out empty chunks
            chunks = [ch for ch in chunks if ch and ch.strip()]
            
            for ch_idx, ch in enumerate(chunks):
                doc_id_counter += 1
                ids.append(f"{self.pipeline_id}_{doc_idx}_{ch_idx}_{doc_id_counter}")
                docs.append(ch)
                metadatas.append({"pipeline": self.pipeline_id})

        if docs:
            try:
                self.collection.add(
                    ids=ids,
                    documents=docs,
                    metadatas=metadatas,
                )
                print(f"Pipeline {self.pipeline_id}: Successfully indexed {len(docs)} chunks")
            except Exception as e:
                print(f"Error indexing documents for pipeline {self.pipeline_id}: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"Warning: No documents to index for pipeline {self.pipeline_id}")

    def answer(self, question: str, top_k: int = 4) -> Dict:
        """
        Retrieve top_k chunks and generate answer using GPT-4o-mini.
        Return answer + retrieved context.
        """
        # Check if collection has documents
        try:
            doc_count = self.collection.count()
            if doc_count == 0:
                return {
                    "pipeline_id": self.pipeline_id,
                    "description": self.description,
                    "answer": "No documents indexed. Please upload documents first.",
                    "context": "",
                }
        except Exception as e:
            print(f"Error checking collection count: {e}")
        
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k,
            )

            docs = results["documents"][0] if results["documents"] and len(results["documents"]) > 0 else []
            context = "\n\n".join(docs) if docs else ""
            
            # If no context retrieved, return early
            if not context or context.strip() == "":
                return {
                    "pipeline_id": self.pipeline_id,
                    "description": self.description,
                    "answer": "No relevant context found in the documents. Please try a different question or ensure documents are properly indexed.",
                    "context": "",
                }

            prompt = (
                "You are a helpful assistant answering questions based on the provided context.\n"
                "Use the context below to answer the question. If the context contains relevant information, provide a detailed answer.\n"
                "If the context doesn't contain enough information to fully answer the question, provide the best answer you can based on what is available.\n"
                "Only say 'I am not sure' if the context is completely irrelevant or empty.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )

            completion = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            answer_text = completion.choices[0].message.content
            return {
                "pipeline_id": self.pipeline_id,
                "description": self.description,
                "answer": answer_text,
                "context": context,
            }
        except Exception as e:
            import traceback
            error_msg = f"Error in pipeline {self.pipeline_id}: {str(e)}"
            print(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "pipeline_id": self.pipeline_id,
                "description": self.description,
                "answer": f"Error: {error_msg}",
                "context": "",
            }
