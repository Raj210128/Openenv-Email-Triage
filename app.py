"""
app.py — Entry point for Hugging Face Spaces.
HF Spaces looks for app.py with a Gradio or FastAPI app object named `app`.
We re-export from server.py.
"""
from server import app  # noqa: F401

# HF Spaces will pick up `app` and serve it on port 7860
