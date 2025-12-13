AI Fact-Checker (RAG-based)
This project is an AI-powered Fact-Checking application built with Streamlit and LlamaIndex. It leverages Retrieval-Augmented Generation (RAG) to verify user claims against a curated local knowledge base.

Unlike standard LLM interactions, this system uses specific guardrails to minimize hallucinations, ensuring that verdicts are strictly based on retrieved evidence from the vector database.

Key Features
RAG Architecture: Uses ChromaDB and LlamaIndex to retrieve relevant context before answering.

Multilingual Support: Automatically detects the input language (via langdetect) and generates explanations in the user's native language (Spanish, English, French, etc.).

Verbatim Evidence: extracts exact quotes from source documents to support the verdict (TRUE, FALSE, or NO_INFO).

Hallucination Guardrails: Logic implemented to filter out "NO_INFO" verdicts to prevent the model from inventing sources.

Context Summarization: An optional feature to generate a narrative summary of all retrieved context regarding a specific topic.

Tech Stack
Frontend: Streamlit

Orchestration: LlamaIndex

Vector Database: ChromaDB (Persistent Client)

Embeddings: HuggingFace (intfloat/multilingual-e5-small)

LLM: Custom API integration (running qwen3:8b)

Utils: LangDetect, RegEx cleaning.
