import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from src.document_ingestion.data_ingestion import (
    DocumentHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

# FastAPI app initialization, this is the main entry point for the API
app = FastAPI(title="Document Bot API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
# app.mount for serving static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# Jinja2Templates for rendering HTML templates, this is used for the UI that interacts with the API
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# add_middleware for CORS, this allows the frontend to access the API
# Note: in production, you should restrict allowed origins to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Below endpoints define the API routes for various functionalities 
# serve_ui serves the main UI page, response_class is set to HTMLResponse to return HTML content 
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request) -> HTMLResponse:
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-bot", "version": "0.1"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    """ Analyze a document and return the analysis result.
    This method calls DocumentHandler to read the document and DocumentAnalyzer to perform analysis.
    Args:
        file (UploadFile): The document file to analyze."""
    try:
        dh = DocumentHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    """ Compare two documents and return the comparison result.
    Args:   
        reference (UploadFile): The reference document file.
        actual (UploadFile): The actual document file to compare against the reference.     
    """
    try:
        dc = DocumentComparator()
        dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    """ Build a FAISS index from uploaded files for chat retrieval.
        This endpoint allows users to upload multiple files, which are then processed and indexed for retrieval in a chat context.
    Args:
        files (List[UploadFile]): List of files to index.
        session_id (Optional[str]): Optional session ID for session-based indexing.
        use_session_dirs (bool): Whether to use session directories for indexing.
        chunk_size (int): Size of text chunks to create from documents.
        chunk_overlap (int): Overlap size between text chunks.
        k (int): Number of top results to retrieve during querying.
    """
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    """ Query the FAISS index with a question and return the answer.
        This endpoint allows users to ask questions based on indexed documents, retrieving relevant information.
        A session ID can be provided to use session-specific directories for the FAISS index.
        It calls ConversationalRAG to handle the retrieval and response generation.
    Args: 
        question (str): The question to ask.
        session_id (Optional[str]): Optional session ID for session-based querying.
        use_session_dirs (bool): Whether to use session directories for querying.
        k (int): Number of top results to retrieve.
    """
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])
        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# uvcorn is ASGI server for running FastAPI applications that supports async features which helps in handling multiple requests efficiently
# To run the FastAPI app, use the command below in your terminal:
# uvicorn api.main:app --port 8080 --reload

# GUnicorn is a WSGI server for running Python web applications,that supports multiple workers for handling concurrent requests.
# Gunicorn is typically used in production environments for better performance and reliability. and Flask is a WSGI application framework for Python.