YouTube RAG Chatbot

An advanced RAG-based chatbot for asking questions about YouTube videos
and getting answers grounded in their transcripts.

The project uses LangChain as the core framework and focuses on
improving a basic RAG system with better query processing, retrieval,
reranking, context compression, grounding, guardrails, memory, and
evaluation.

What it does

The application takes a YouTube video URL, extracts its transcript,
processes the content, creates embeddings, and stores the chunks in
Pinecone.

When a user asks a question, the query passes through several retrieval
and generation stages before the final answer is returned.

The system also supports Hindi and Hinglish conversations by normalizing
Hindi transcript content using IndicTrans2.

RAG Pipeline

YouTube Video -> Transcript Extraction -> Translation / Language
Normalization -> Text Splitting -> BGE-M3 Embeddings -> Pinecone
Indexing -> Query Rewriting -> Multi-Query Generation -> Domain
Routing -> Hybrid Retrieval -> MMR -> Reranking -> Contextual
Compression -> Context Optimization -> Qwen3 Answer Generation ->
Answer Grounding -> Guardrails -> Conversation Memory

Key Features

YouTube Transcript Ingestion

Extracts transcript segments and preserves metadata such as video ID,
segment index, start time, end time, and YouTube timestamps.

Multilingual Processing

Hindi transcript content is translated using IndicTrans2 while
preserving the original transcript information and metadata. English
content is kept as it is.

Query Rewriting

Qwen3 rewrites user questions into standalone search queries.
Conversation history can be used to resolve follow-up questions.

Multi-Query Generation

Multiple query variations are generated to improve retrieval coverage.

Hybrid Retrieval

Dense retrieval with BGE-M3 and Pinecone is combined with BM25 sparse
retrieval. Results are fused using weighted Reciprocal Rank Fusion.

MMR

Maximal Marginal Relevance reduces redundant chunks and improves context
diversity.

Reranking

Retrieved documents are reranked so that the most relevant transcript
chunks are prioritized.

Contextual Compression

An LLM extracts only the parts of retrieved chunks that are directly
useful for the current question.

Context Optimization

Duplicate, empty, and very small chunks are removed. Documents are also
limited to a controlled context budget before generation.

Grounded Answers

The answer generator is instructed to answer from the retrieved
transcript context and provide YouTube timestamp citations.

Guardrails

Generated answers are checked for grounding and citation requirements
before being returned.

Conversation Memory

Recent conversation turns are stored so follow-up questions can be
understood in context.

Evaluation

The RAG pipeline was evaluated using RAGAS.

Metric                 Score

Faithfulness            100%
Answer Relevancy       94.7%
Context Precision     ~100%
Context Recall          100%

Evaluation uses Qwen3 8B through Ollama and BGE-M3 embeddings.

Tech Stack

Python

LangChain

Qwen3 8B

Ollama

Pinecone

BGE-M3

IndicTrans2

BM25

Streamlit

RAGAS

YouTube Transcript API

Project Structure

YutubeChatbot/
├── ingestion/
│   ├── youtube.py
│   ├── splitter.py
│   ├── translator.py
│   ├── embeddings.py
│   └── indexer.py
├── retrieval/
│   ├── vector_store.py
│   ├── query_rewriter.py
│   ├── multi_query.py
│   ├── domain_router.py
│   ├── mmr.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── contextual_compressor.py
├── generation/
│   ├── prompt_template.py
│   ├── llm.py
│   ├── answer_generator.py
│   ├── answer_grounding.py
│   ├── guardrails.py
│   └── context_optimizer.py
├── pipeline/
│   └── rag_pipeline.py
├── evaluation/
│   ├── ragas_evaluator.py
│   └── test_ragas.py
├── memory/
│   └── conversation_memory.py
├── app.py
└── tests

Setup

Clone the repository:

git clone https://github.com/aman-pathak1/YT-Chatbot.git
cd YT-Chatbot

Create and activate a virtual environment:

python -m venv venv

Windows PowerShell:

.env\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Make sure Ollama is installed and Qwen3 8B is available locally:

ollama pull qwen3:8b

Create a .env file:

PINECONE_API_KEY=your_pinecone_api_key

Never commit .env or virtual environment folders to GitHub.

Run

streamlit run app.py

Why this project

The goal of this project is to go beyond a basic "YouTube transcript +
LLM" chatbot.

The main focus is the complete RAG workflow: improving the query before
retrieval, combining different retrieval methods, reducing irrelevant
context, generating grounded answers, validating the output, and
maintaining conversation context.

The components are separated into individual modules so they can be
tested, replaced, and improved independently.

Future Improvements

Agentic retrieval

Multimodal video understanding

More advanced citation verification

Persistent conversation memory

Larger evaluation datasets

Production deployment and monitoring

Author

Aman Pathak

GitHub: https://github.com/aman-pathak1
