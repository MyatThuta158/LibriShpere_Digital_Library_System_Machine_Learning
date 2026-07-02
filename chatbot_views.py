"""
Chatbot API views for LibriSphere.

Endpoints:
  POST /api/chat          — Send a message, get a RAG-powered answer
  POST /api/chat/rebuild  — Admin utility: rebuild the ChromaDB vector store
  GET  /api/chat/health   — Check that the RAG engine is loaded and ready
"""

import logging
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)


# ── /api/chat ────────────────────────────────────────────────────────────────

@csrf_exempt
@api_view(["POST"])
def chat(request):
    """
    Accept a JSON body  { "message": "..." }
    Return              { "answer": "..." }
    """
    message = (request.data or {}).get("message", "").strip()

    if not message:
        return Response(
            {"error": "Field 'message' is required and cannot be empty."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(message) > 1000:
        return Response(
            {"error": "Message is too long. Please keep it under 1000 characters."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        from api.rag_engine import get_answer
        answer = get_answer(message)
        return Response({"answer": answer}, status=status.HTTP_200_OK)

    except EnvironmentError as exc:
        logger.error("RAG environment error: %s", exc)
        return Response(
            {
                "error": "Chatbot is not configured properly. "
                         "Please contact the administrator.",
                "detail": str(exc),
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    except Exception as exc:
        error_str = str(exc)
        logger.error("Unexpected chatbot error: %s", exc, exc_info=True)

        # Give a clearer message for common API errors
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            return Response(
                {"error": "The AI service is temporarily unavailable due to rate limits. Please try again in a few minutes."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        if "API_KEY" in error_str or "401" in error_str or "403" in error_str:
            return Response(
                {"error": "AI service configuration error. Please contact the administrator."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return Response(
            {"error": "An unexpected error occurred. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── /api/chat/health ─────────────────────────────────────────────────────────

@api_view(["GET"])
def chat_health(request):
    """
    Returns the current status of the RAG engine.
    200 if ready, 503 if not yet initialized or misconfigured.
    """
    try:
        from api import rag_engine
        is_ready = rag_engine._qa_chain is not None
        return Response(
            {
                "status": "ready" if is_ready else "not_initialized",
                "message": (
                    "LibriBot RAG engine is running."
                    if is_ready
                    else "RAG engine has not been initialized yet. "
                         "Send the first chat message to trigger initialization."
                ),
            },
            status=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        return Response(
            {"status": "error", "message": str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ── /api/chat/rebuild ────────────────────────────────────────────────────────

@csrf_exempt
@api_view(["POST"])
def chat_rebuild(request):
    """
    Force-rebuild the ChromaDB vector store from the latest AboutLibrary.txt.
    Useful when the library documentation is updated.
    Only accessible when DEBUG=True or with a valid admin key header.
    """
    from django.conf import settings

    # Simple guard: only allow in debug mode or with a secret header
    admin_key = request.headers.get("X-Admin-Key", "")
    expected_key = getattr(settings, "CHATBOT_REBUILD_KEY", "librisphere-rebuild-2025")

    if not settings.DEBUG and admin_key != expected_key:
        return Response(
            {"error": "Unauthorized. Provide a valid X-Admin-Key header."},
            status=status.HTTP_403_FORBIDDEN,
        )

    try:
        from api.rag_engine import rebuild_vectorstore
        message = rebuild_vectorstore()
        return Response({"status": "success", "message": message}, status=status.HTTP_200_OK)
    except FileNotFoundError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    except Exception as exc:
        logger.error("Vector store rebuild failed: %s", exc, exc_info=True)
        return Response(
            {"error": f"Rebuild failed: {exc}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
