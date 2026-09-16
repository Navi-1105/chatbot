import contextlib
import importlib
import os
import sys
import uuid
from functools import lru_cache


FEATURE_NAME = "sre-documentation-assistant"
TRACE_NAME = "sre-docs-rag-answer"
GEMINI_MODEL = "gemini-2.5-flash"


def new_session_id():
    return str(uuid.uuid4())


def _configured():
    return all(
        os.getenv(name)
        for name in (
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_BASE_URL",
        )
    )


def _import_langfuse_sdk():
    """Import the SDK even when a local ./langfuse checkout exists."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    existing = sys.modules.get("langfuse")
    existing_file = getattr(existing, "__file__", None)
    existing_paths = list(getattr(existing, "__path__", [])) if existing else []

    should_restore_existing = existing is not None
    if existing_file is None and any(path.startswith(project_root) for path in existing_paths):
        sys.modules.pop("langfuse", None)
    else:
        should_restore_existing = False

    original_path = list(sys.path)
    try:
        sys.path = [
            path
            for path in sys.path
            if path
            and os.path.abspath(path) not in {project_root, os.getcwd()}
        ]
        module = importlib.import_module("langfuse")
        return module.get_client, module.propagate_attributes
    except Exception:
        if should_restore_existing:
            sys.modules["langfuse"] = existing
        return None, None
    finally:
        sys.path = original_path


@lru_cache(maxsize=1)
def _langfuse_parts():
    if not _configured():
        return None, None, None

    get_client, propagate_attributes = _import_langfuse_sdk()
    if get_client is None:
        return None, None, None

    try:
        client = get_client()
    except Exception:
        return None, None, None

    return client, propagate_attributes, os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")


@contextlib.contextmanager
def rag_trace(query, entrypoint, session_id=None, top_k=None):
    langfuse, propagate_attributes, environment = _langfuse_parts()
    if langfuse is None:
        yield None
        return

    metadata = {
        "feature": FEATURE_NAME,
        "entrypoint": entrypoint,
    }
    if top_k is not None:
        metadata["top_k"] = top_k

    with langfuse.start_as_current_observation(
        as_type="span",
        name="answer-question",
        input={"question": query},
        metadata=metadata,
    ) as root_span:
        with propagate_attributes(
            session_id=session_id,
            tags=["rag", "sre-assistant"],
            metadata=metadata,
            environment=environment,
            trace_name=TRACE_NAME,
        ):
            yield root_span


@contextlib.contextmanager
def retriever_observation(query, top_k):
    langfuse, _, _ = _langfuse_parts()
    if langfuse is None:
        yield None
        return

    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="retrieve-context",
        input={
            "query": query,
            "top_k": top_k,
        },
    ) as span:
        yield span


@contextlib.contextmanager
def generation_observation(prompt):
    langfuse, _, _ = _langfuse_parts()
    if langfuse is None:
        yield None
        return

    with langfuse.start_as_current_observation(
        as_type="generation",
        name="generate-answer",
        model=GEMINI_MODEL,
        input=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    ) as generation:
        yield generation


def flush():
    langfuse, _, _ = _langfuse_parts()
    if langfuse is not None:
        langfuse.flush()


def retrieval_output(results):
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    output = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        item = {
            "rank": index + 1,
            "source": metadata.get("source"),
            "title": metadata.get("title"),
            "section": metadata.get("section"),
            "snippet": document[:500],
        }
        if index < len(distances):
            item["distance"] = distances[index]
        output.append(item)

    return output


def usage_details(response):
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_token_count", None)
    cached_tokens = getattr(usage, "cached_content_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    thoughts_tokens = getattr(usage, "thoughts_token_count", None)
    total_tokens = getattr(usage, "total_token_count", None)

    details = {}
    if prompt_tokens is not None:
        details["input"] = max(prompt_tokens - (cached_tokens or 0), 0)
    if cached_tokens:
        details["input_cached_tokens"] = cached_tokens
    if output_tokens is not None:
        details["output"] = output_tokens
    if thoughts_tokens:
        details["output_reasoning_tokens"] = thoughts_tokens
    if total_tokens is not None:
        details["total"] = total_tokens

    return details or None
