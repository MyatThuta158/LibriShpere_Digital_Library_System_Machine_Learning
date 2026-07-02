from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        """
        Pre-warm the RAG engine when Django starts.
        This loads the HuggingFace embedding model (~8s) at startup time
        so the very first chat request responds in under 1 second.
        """
        # Skip during management commands like migrate, shell, etc.
        import sys
        if any(cmd in sys.argv for cmd in ["migrate", "makemigrations", "shell",
                                            "createsuperuser", "collectstatic"]):
            return

        try:
            logger.info("Pre-warming RAG engine at startup...")
            from api.rag_engine import initialize_rag
            initialize_rag()
            logger.info("RAG engine pre-warmed successfully.")
        except Exception as exc:
            # Don't crash Django startup — just log the warning
            logger.warning("RAG engine pre-warm failed (will retry on first request): %s", exc)
