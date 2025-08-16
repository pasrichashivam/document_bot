from __future__ import annotations
import os
import sys
import json
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz 
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# FAISS Manager (load-or-create)
class FaissManager:
    """ FaissManager is responsible for managing the FAISS index.
        It handles loading existing indices or creating new ones, adding documents to the index,
        and ensuring idempotency by checking for existing documents based on their content and metadata.
        It also provides functionality to save metadata about indexed documents to a JSON file.
    """
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}
        
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}
        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None
        
    def _exists(self)-> bool:
        """ Check if FAISS index exists. """
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """ Deduplicate data based on content and metadata to avoid writing of 
            duplicate data in vector database index.
        Args:
            text (str): The content of the document.
            md (Dict[str, Any]): Metadata associated with the document.
        Returns:
            str: A unique fingerprint for the document based on its content and metadata."""
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        """ Save metadata to JSON file. """
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
    def add_documents(self, docs: List[Document]):
        """ Add documents to FAISS index, ensuring idempotency. 
            Returns number of new documents added.
        """
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")
        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)
        if new_docs:
            # This will add new documents to the FAISS index, which first Embeds the chunks and then adds it to the index.
            # Add / Append new documents over the same existing index
            self.vs.add_documents(new_docs) 
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    
    def load_or_create(self, texts: Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        """ Load existing FAISS index or create a new one if it doesn't exist. 
            If texts are provided, they will be used to create the index.
        """
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])  # create new index
        self.vs.save_local(str(self.index_dir))  # save the new index
        return self.vs

class ChatIngestor:
    """ This class is responsible for ingesting documents and building a retriever using FAISS.
        It initializes with paths for temporary files and FAISS index, and allows for session-based directories
        It provides methods to save uploaded files, load documents, split them into chunks, and build a retriever.
    Args:
        temp_base (str): Base directory for temporary files.
        faiss_base (str): Base directory for FAISS index.
        use_session_dirs (bool): Whether to use session directories for indexing.
        session_id (Optional[str]): Optional session ID for session-based indexing.
    """
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.model_loader = ModelLoader()
            
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)
            
            self.log.info("ChatIngestor initialized",
                          session_id=self.session_id,
                          temp_dir=str(self.temp_dir),
                          faiss_dir=str(self.faiss_dir),
                          sessionized=self.use_session)
        except Exception as e:
            self.log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path):
        """ Resolve the base directory for temporary files or FAISS index.
            If session directories are used, it creates a subdirectory for the current session ID.
            Returns the resolved directory path.
        """
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        """ Split documents into smaller chunks using RecursiveCharacterTextSplitter.
            It allows for specifying chunk size and overlap.
            Returns a list of Document objects representing the chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        self.log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriver(self,
            uploaded_files: Iterable,
            *,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            k: int = 5
        ):
        """ Build a retriever from the uploaded files with below steps:
            1. Save the uploaded files to the temporary directory.
            2. Load the saved files as documents.
            3. Split the documents into chunks.
            4. Create or update the FAISS index with the chunks.
            5. Return a retriever that can be used for similarity search.
        Args:
            uploaded_files (Iterable): Iterable of uploaded file-like objects.
            chunk_size (int): Size of text chunks to create from documents.
            chunk_overlap (int): Overlap size between text chunks.
            k (int): Number of top results to retrieve during querying.
        """
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            self.log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            self.log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e


class DocumentHandler:
    """ This class handles saving, reading, and combining PDFs with session-based versioning.
        It uses a session ID to create a unique directory for each session, allowing for organized storage  
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.log = CustomLogger().get_logger(__name__)
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or _session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        self.log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, uploaded_file) -> str:
        """ This method saves the uploaded PDF file to the session directory.
            It checks the file type and saves it with the original filename.
            Raises DocumentPortalException if the file is not a PDF or if saving fails.
        """
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            self.log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            self.log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save PDF: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        """ This method reads the content of a PDF file and returns it as a string.
            It uses the PyMuPDF library to extract text from each page.
        """
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            self.log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            self.log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process PDF: {pdf_path}", e) from e

class DocumentComparator:
    """
    This class is responsible for comparing documents by saving, reading, and combining PDFs.
    It uses session-based versioning to manage different document versions.
    It allows saving uploaded files, reading PDF content, and combining documents for comparison.
    It also provides functionality to clean up old sessions based on a specified retention policy.
    """
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.session_id = session_id or _session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        """ Save the uploaded files to the session directory.
            It checks if the files are PDFs and saves them with their original names."""
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            self.log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
        except Exception as e:
            self.log.error("Error saving PDF files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", e) from e

    def read_pdf(self, pdf_path: Path) -> str:
        """ Read the content of a PDF file and return it as a string.
            It uses PyMuPDF to extract text from each page.
            Raises DocumentPortalException if the PDF is encrypted or if reading fails.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()  # type: ignore
                    if text.strip():
                        parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
            self.log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            self.log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading PDF", e) from e

    def combine_documents(self) -> str:
        """ Combine all PDF documents in the session directory into a single string.
            It reads each PDF file, extracts its content, and concatenates them.
            and returns the combined text."""
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            self.log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e

    def clean_old_sessions(self, keep_latest: int = 3):
        """ Clean up old session directories, keeping only the latest 'keep_latest' sessions.
            It sorts the session directories by creation time and removes the older ones.
            Raises DocumentPortalException if cleaning fails.
        """
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                self.log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            self.log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", e) from e
