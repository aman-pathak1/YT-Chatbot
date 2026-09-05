from __future__ import annotations

import logging
import re
import time

import streamlit as st

from ingestion.translator import TranscriptTranslator
from ingestion.splitter import TranscriptSplitter
from ingestion.embeddings import BGEEmbeddings
from ingestion.indexer import YouTubeIndexer

from retrieval.vector_store import PineconeVectorStoreManager
from retrieval.domain_router import DomainRouter
from retrieval.query_rewriter import QueryRewriter
from retrieval.multi_query import MultiQueryGenerator
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.mmr import MMRRetriever
from retrieval.reranker import BGEReranker
from retrieval.contextual_compressor import ContextualCompressor

from generation.context_optimizer import ContextWindowOptimizer
from generation.answer_generator import AnswerGenerator
from generation.answer_grounding import AnswerGrounding
from generation.guardrails import AnswerGuardrails

from pipeline.rag_pipeline import RAGPipeline
from memory.conversation_memory import ConversationMemory


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YT Research Desk",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

INDEX_NAME = "youtube-chatbot"
NAMESPACE = "youtube-transcripts"


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    defaults = {
        "memory": ConversationMemory(
            session_id=f"streamlit-session-{int(time.time())}",
            max_turns=10,
            max_history_characters=12000,
        ),
        "pipeline": None,
        "video_url": "",
        "video_id": None,
        "chunks": [],
        "indexed": False,
        "messages": [],
        "last_result": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# HELPERS
# ============================================================

def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def youtube_thumbnail(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


# ============================================================
# BUILD SHARED COMPONENTS
# ============================================================

@st.cache_resource(show_spinner=False)
def build_components():
    logger.info("Building RAG components...")

    embeddings = BGEEmbeddings()

    vector_store = PineconeVectorStoreManager(
        embedding_model=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )

    return {
        "embeddings": embeddings,
        "vector_store": vector_store,
        "domain_router": DomainRouter(),
        "query_rewriter": QueryRewriter(),
        "multi_query_generator": MultiQueryGenerator(),
        "contextual_compressor": ContextualCompressor(),
        "context_optimizer": ContextWindowOptimizer(),
        "answer_generator": AnswerGenerator(),
        "answer_grounding": AnswerGrounding(),
        "guardrails": AnswerGuardrails(),
        "mmr_retriever": MMRRetriever(vector_store=vector_store),
        "reranker": BGEReranker(),
    }


# ============================================================
# INDEX VIDEO
# ============================================================

@st.cache_data(show_spinner=False)
def index_video(video_url: str):
    video_id = extract_video_id(video_url)

    if not video_id:
        raise ValueError("Invalid YouTube URL. Please provide a valid URL.")

    components = build_components()

    indexer = YouTubeIndexer(
        translator=TranscriptTranslator(),
        splitter=TranscriptSplitter(),
        vector_store=components["vector_store"],
    )

    logger.info(
        "Indexing video for the first time in this Streamlit cache | video_id=%s",
        video_id,
    )

    return indexer.index_video(video_url)


# ============================================================
# CREATE RAG PIPELINE
# ============================================================

def create_pipeline(chunks):
    if not isinstance(chunks, (list, tuple)):
        raise TypeError(
            "create_pipeline() expects the actual chunk_documents list, "
            f"received {type(chunks).__name__}."
        )

    if not chunks:
        raise ValueError("No retrieval chunks were created for this video.")

    # Keep the contract explicit: HybridRetriever and RAGPipeline both
    # require actual LangChain Document objects.
    from langchain_core.documents import Document

    invalid = [
        index
        for index, document in enumerate(chunks)
        if not isinstance(document, Document)
    ]

    if invalid:
        raise TypeError(
            "create_pipeline() received non-Document items at positions: "
            f"{invalid[:10]}"
        )

    components = build_components()

    hybrid_retriever = HybridRetriever(
        vector_store=components["vector_store"],
        documents=chunks,
    )

    return RAGPipeline(
        domain_router=components["domain_router"],
        query_rewriter=components["query_rewriter"],
        multi_query_generator=components["multi_query_generator"],
        hybrid_retriever=hybrid_retriever,
        mmr_retriever=components["mmr_retriever"],
        reranker=components["reranker"],
        contextual_compressor=components["contextual_compressor"],
        context_optimizer=components["context_optimizer"],
        answer_generator=components["answer_generator"],
        answer_grounding=components["answer_grounding"],
        guardrails=components["guardrails"],
        memory=st.session_state.memory,
    )


# ============================================================
# SIDEBAR — CONTROL CENTER
# ============================================================

with st.sidebar:
    st.title("YT Research Desk")
    st.caption("A local-first conversational RAG workspace")

    st.divider()

    st.subheader("1. Load video")

    sidebar_url = st.text_input(
        "YouTube URL",
        value=st.session_state.video_url,
        placeholder="Paste video URL",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button(
        "Analyze video",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Current session")

    if st.session_state.indexed:
        st.success("Video indexed")
        st.caption(f"Video ID: {st.session_state.video_id}")
        st.metric("Retrieval chunks", len(st.session_state.chunks))
    else:
        st.info("No video loaded")

    st.divider()

    st.subheader("Conversation")

    memory = st.session_state.memory
    m1, m2 = st.columns(2)
    m1.metric("Turns", memory.turn_count())
    m2.metric("Messages", memory.message_count())

    if st.button("Clear conversation", use_container_width=True):
        memory.clear()
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    st.subheader("Pipeline")

    pipeline_steps = [
        "Ingestion",
        "Translation",
        "Hybrid retrieval",
        "MMR",
        "Reranking",
        "Compression",
        "Context optimization",
        "Grounding",
        "Guardrails",
        "Memory",
    ]

    for step in pipeline_steps:
        st.write(f"✓ {step}")


# ============================================================
# MAIN HEADER
# ============================================================

left, right = st.columns([5, 1])

with left:
    st.title("🎥 YT Research Desk")
    st.caption(
        "Turn a YouTube video into a searchable knowledge base and have a grounded conversation with it."
    )

with right:
    if st.session_state.indexed:
        st.success("READY")
    else:
        st.info("WAITING")

st.divider()


# ============================================================
# VIDEO ANALYSIS
# ============================================================

if analyze_clicked:
    if not sidebar_url.strip():
        st.warning("Paste a YouTube URL first.")
    else:
        video_id = extract_video_id(sidebar_url)

        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            try:
                with st.status("Analyzing video...", expanded=True) as status:
                    st.write("Loading transcript...")
                    result = index_video(sidebar_url)

                    chunks = result.chunk_documents

                    if not isinstance(chunks, list):
                        raise TypeError(
                            "Indexer must return chunk_documents as "
                            "list[Document]. Please use the updated "
                            "ingestion/indexer.py."
                        )

                    st.write(
                        f"Created {len(chunks)} retrieval chunks."
                    )

                    st.write("Building conversational RAG pipeline...")
                    pipeline = create_pipeline(chunks)

                    st.session_state.video_url = sidebar_url
                    st.session_state.video_id = video_id
                    st.session_state.chunks = chunks
                    st.session_state.pipeline = pipeline
                    st.session_state.indexed = True
                    st.session_state.memory.clear()
                    st.session_state.messages = []
                    st.session_state.last_result = None

                    status.update(
                        label="Video ready",
                        state="complete",
                        expanded=False,
                    )

                st.rerun()

            except Exception as exc:
                logger.exception("Video analysis failed.")
                st.error(f"Video analysis failed: {exc}")


# ============================================================
# VIDEO WORKSPACE
# ============================================================

if st.session_state.video_id:
    st.subheader("Video workspace")

    video_col, details_col = st.columns([1.55, 1])

    with video_col:
        st.image(
            youtube_thumbnail(st.session_state.video_id),
            use_container_width=True,
        )

    with details_col:
        st.markdown("#### Active source")
        st.code(st.session_state.video_id, language=None)

        st.metric(
            "Indexed chunks",
            len(st.session_state.chunks),
        )

        st.write("The transcript is indexed in Pinecone and ready for retrieval.")

        if st.session_state.video_url:
            st.link_button(
                "Open on YouTube",
                st.session_state.video_url,
                use_container_width=True,
            )

    st.divider()


# ============================================================
# CHAT AREA
# ============================================================

st.subheader("Conversation")
st.caption("English • Hindi • Hinglish • Follow-up questions supported")

if not st.session_state.indexed:
    st.info("Start by loading a YouTube video from the sidebar.")

    st.markdown("#### What this workspace can do")
    a, b, c = st.columns(3)
    a.info("🔎 Find relevant moments")
    b.info("🧠 Remember follow-ups")
    c.info("📌 Ground answers in video context")

else:
    if not st.session_state.messages:
        st.markdown("#### Try a question")

        suggestions = [
            "Give me a concise summary of this video.",
            "What are the main concepts explained?",
            "Isme sabse important point kya hai?",
        ]

        s1, s2, s3 = st.columns(3)
        for column, prompt in zip((s1, s2, s3), suggestions):
            with column:
                if st.button(prompt, use_container_width=True):
                    st.session_state.pending_question = prompt
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask anything about the video...")

    if pending_question and not question:
        question = pending_question

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                stream_gen = st.session_state.pipeline.run_stream(
                    question=question,
                    documents=st.session_state.chunks,
                )
                answer = st.write_stream(stream_gen)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                logger.exception("Question processing failed.")
                st.error("I couldn't process that question. Please try again.")
                logger.error("RAG error: %s", exc)


# ============================================================
# INSPECTOR
# ============================================================

if st.session_state.last_result is not None:
    result = st.session_state.last_result

    st.divider()
    st.subheader("Answer inspector")

    tab1, tab2, tab3 = st.tabs(["Evidence", "RAG trace", "Quality"])

    with tab1:
        documents = result.optimized_documents

        if not documents:
            st.warning("No source documents were returned.")
        else:
            for index, document in enumerate(documents, start=1):
                metadata = document.metadata
                timestamp = (
                    metadata.get("youtube_timestamp")
                    or metadata.get("chunk_start_time")
                    or "Unknown"
                )

                with st.expander(f"Source {index} · {timestamp}"):
                    st.caption(f"Timestamp: {timestamp}")
                    st.write(document.page_content)

    with tab2:
        st.write(f"**Domain:** {result.domain}")
        st.write(f"**Rewritten query:** {result.rewritten_query}")

        st.markdown("**Generated queries**")
        for query in result.generated_queries:
            st.write(f"• {query}")

        trace_metrics = st.columns(4)
        trace_metrics[0].metric(
            "Retrieved",
            len(result.retrieved_documents),
        )
        trace_metrics[1].metric(
            "Reranked",
            len(result.reranked_documents),
        )
        trace_metrics[2].metric(
            "Compressed",
            len(result.compressed_documents),
        )
        trace_metrics[3].metric(
            "Final context",
            len(result.optimized_documents),
        )

    with tab3:
        q1, q2, q3 = st.columns(3)
        q1.metric("Grounding", f"{result.grounding_score:.2f}")
        q2.metric("Grounded", "Yes" if result.is_grounded else "No")
        q3.metric("Guardrails", "Passed" if result.guardrails_passed else "Failed")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("YT Research Desk · LangChain · Ollama · Pinecone · BGE-M3")