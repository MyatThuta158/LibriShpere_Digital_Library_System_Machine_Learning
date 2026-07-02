"""
RAG Engine for LibriSphere Chatbot — token-efficient build.

Token budget per request (approx):
  - System prompt  : ~60  tokens
  - Retrieved docs : ~600 tokens  (2 chunks x 300 chars)
  - Live DB info   : ~80  tokens  (only membership plans, no book list)
  - Question       : ~30  tokens
  - Answer         : 256  tokens max
  Total            : ~1026 tokens  — well within free tier limits
"""

import os
import logging
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
ABOUT_FILE = (
    BASE_DIR.parent.parent
    / "Library_System_Backend"
    / "storage"
    / "app"
    / "AboutLibrary.txt"
)
CHROMA_DIR = BASE_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_qa_chain    = None
_vectorstore = None
_embeddings  = None   # cached so model loads only once


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            from django.conf import settings as ds
            key = getattr(ds, "GEMINI_API_KEY", "")
        except Exception:
            pass
    return key


def _build_embeddings() -> HuggingFaceEmbeddings:
    """Return cached embeddings — loads the model only on the very first call."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading HuggingFace embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded.")
    return _embeddings


def _load_or_build_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    chroma_str = str(CHROMA_DIR)

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        logger.info("Loading ChromaDB from %s", chroma_str)
        return Chroma(
            persist_directory=chroma_str,
            embedding_function=embeddings,
            collection_name="librisphere_docs",
        )

    logger.info("Building ChromaDB from %s", ABOUT_FILE)
    if not ABOUT_FILE.exists():
        raise FileNotFoundError(
            f"AboutLibrary.txt not found at {ABOUT_FILE}. "
            "Create it at Library_System_Backend/storage/app/AboutLibrary.txt"
        )

    docs = TextLoader(str(ABOUT_FILE), encoding="utf-8").load()

    # Smaller chunks = fewer tokens sent to LLM per retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,       # ← reduced from 800
        chunk_overlap=40,     # ← reduced from 100
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_str,
        collection_name="librisphere_docs",
    )


def _fetch_membership_plans() -> str:
    """
    Fetch ONLY membership plans from DB — short and focused.
    We no longer inject the full book list to save tokens.
    """
    try:
        from api.models import MembershipPlans
        plans = MembershipPlans.objects.all().values("planname", "duration", "price")
        if not plans:
            return ""
        lines = ["Membership plans (live):"]
        for p in plans:
            lines.append(f"- {p['planname']}: {p['duration']} days, ${p['price']}")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("DB fetch failed: %s", exc)
        return ""


def _fetch_books_for_search(keyword: str) -> str:
    """
    Only called for book-search queries. Returns max 5 matching titles.
    """
    try:
        from api.models import ElectronicResources
        results = (
            ElectronicResources.objects
            .filter(name__icontains=keyword)
            .select_related("author")
            .values("name", "author__name", "isbn")[:5]
        )
        if not results:
            # Fallback: return 5 most recent books
            results = (
                ElectronicResources.objects
                .select_related("author")
                .order_by("-id")
                .values("name", "author__name", "isbn")[:5]
            )
        lines = ["Matching books:"]
        for r in results:
            lines.append(
                f"- \"{r['name']}\" by {r['author__name'] or 'Unknown'}"
                f" (ISBN: {r['isbn'] or 'N/A'})"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Book search failed: %s", exc)
        return ""


def _build_prompt() -> PromptTemplate:
    # Short, tight prompt — no padding, no repeating instructions
    template = (
        "You are LibriBot for LibriSphere library. "
        "Answer briefly using only the context below. "
        "Be direct and helpful.\n\n"
        "Context:\n{context}\n\n"
        "Q: {question}\nA:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_rag():
    global _qa_chain, _vectorstore

    if _qa_chain is not None:
        return

    api_key = _get_api_key()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to .env"
        )

    logger.info("Initializing RAG engine...")

    embeddings   = _build_embeddings()
    _vectorstore = _load_or_build_vectorstore(embeddings)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=256,   # ← reduced from 1024
    )

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},     # ← reduced from 5; 2 chunks is enough
        ),
        chain_type_kwargs={"prompt": _build_prompt()},
        return_source_documents=False,
    )

    logger.info("RAG engine ready.")


def get_answer(question: str) -> str:
    global _qa_chain

    if _qa_chain is None:
        initialize_rag()

    question = (question or "").strip()
    if not question:
        return "Please ask me something about LibriSphere!"

    q_lower = question.lower()

    # ── Fast path: pricing/membership questions ──────────────────────────
    # These are answered directly from DB + fallback — no LLM call needed,
    # so response is near-instant and uses zero API tokens.
    if any(w in q_lower for w in ["price", "cost", "fee", "how much",
                                   "membership", "plan", "subscription", "subscribe"]):
        plans = _fetch_membership_plans()
        if plans:
            return (
                "Here are LibriSphere's current membership plans:\n\n"
                + plans.replace("Membership plans (live):\n", "")
                + "\n\nVisit the Membership page to subscribe!"
            )
        return _fallback_answer(question)

    # ── Fast path: book search — inject DB results, skip RAG retrieval ───
    extra_ctx = ""
    if _is_book_search(q_lower):
        keyword   = _extract_search_keyword(question)
        extra_ctx = _fetch_books_for_search(keyword)

    query = question
    if extra_ctx:
        query = f"{question}\n\n[Library data]\n{extra_ctx}"

    # ── LLM path: everything else goes through RAG ───────────────────────
    try:
        result = _qa_chain.invoke({"query": query})
        answer = (result.get("result") or "").strip()
        return answer if answer else _fallback_answer(question)

    except Exception as exc:
        err = str(exc)
        logger.error("RAG error: %s", err)

        if "RESOURCE_EXHAUSTED" in err or "429" in err:
            return (
                "The AI service has reached its rate limit. "
                "Please try again in a few minutes."
            )
        if "API_KEY_INVALID" in err or "400" in err:
            return "AI service configuration error. Please contact the administrator."

        return (
            "Sorry, I could not process your question. "
            "Please try again or contact info@librisphere.com."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_book_search(q: str) -> bool:
    return any(kw in q for kw in [
        "book", "ebook", "find", "search", "recommend",
        "suggest", "do you have", "author", "isbn", "genre",
    ])


def _extract_search_keyword(question: str) -> str:
    """Pull a likely book/topic keyword from the question."""
    stop = {"find", "search", "book", "ebook", "for", "a", "an", "the",
            "do", "you", "have", "can", "i", "looking", "about", "any"}
    words = [w for w in question.lower().split() if w not in stop and len(w) > 2]
    return words[0] if words else ""


def _fallback_answer(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["price", "cost", "fee", "plan", "membership", "how much"]):
        return (
            "LibriSphere membership plans:\n"
            "• Basic Monthly — $9.99/month\n"
            "• Standard Quarterly — $24.99/quarter\n"
            "• Premium Annual — $79.99/year\n"
            "• Student Plan — $19.99 for 6 months"
        )
    return (
        "I'm not sure about that. Visit our Resources page "
        "or contact info@librisphere.com."
    )


def rebuild_vectorstore() -> str:
    global _qa_chain, _vectorstore
    import shutil

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    _qa_chain    = None
    _vectorstore = None
    initialize_rag()
    return "Vector store rebuilt successfully."
