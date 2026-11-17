"""Interactive FastAPI web service for PepPrism."""

from pepprism.web.app import app, create_app

__all__ = ["app", "create_app"]
