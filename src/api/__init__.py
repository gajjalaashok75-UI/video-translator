"""
Read-only HTTP API for the Video Translator pipeline.

The API is an *additive* application layer. It imports the existing pipeline
modules but never modifies them: `app.py` (Gradio, port 7860) remains the
legacy entry point, and this package exposes the same capability surface to a
future React client.

F1 exposes discovery only — no uploads, no job execution, no SSE. Nothing here
starts a render, downloads a model, or performs inference.
"""

API_VERSION = "1.0.0"

__all__ = ["API_VERSION"]
