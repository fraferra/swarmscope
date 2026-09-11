"""Framework adapters. Each is a thin shim onto the raw SDK.

Import lazily: ``from swarmscope.adapters.openai_transport import instrument_openai`` etc.
"""
__all__ = ["openai_transport", "langchain", "crewai"]
