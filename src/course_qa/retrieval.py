import math

from course_qa.config import AppConfig
from course_qa.contracts import RetrievedPassage
from course_qa.embeddings import Embedder, validate_vectors
from course_qa.store import SnapshotStore


def retrieve_passages(
    question: str,
    config: AppConfig,
    embedder: Embedder,
    project: str | None = None,
    max_distance: float | None = None,
) -> tuple[str, list[RetrievedPassage]]:
    """Retrieve candidate evidence without generating an answer."""
    question = question.strip()

    if not question:
        raise ValueError("Enter a nonempty question.")

    if embedder.count_tokens(question) > config.max_question_tokens:
        raise ValueError("The question exceeds the configured token limit.")

    if project is not None:
        project = project.strip()

        if not project:
            raise ValueError("The project filter cannot be empty.")

    if max_distance is not None and (
        not math.isfinite(max_distance) or max_distance < 0
    ):
        raise ValueError("max_distance must be finite and nonnegative.")

    store = SnapshotStore(config.storage_dir)

    with store.locked():
        manifest, collection = store.open_active(
            embedding_fingerprint=embedder.fingerprint,
            dimension=embedder.dimension,
        )

        query_vector = embedder.embed_query(question)

        validate_vectors(
            [query_vector],
            expected_count=1,
            dimension=embedder.dimension,
        )

        passages = store.query(
            collection=collection,
            vector=query_vector,
            top_k=config.top_k,
            project=project,
        )

    for passage in passages:
        if not math.isfinite(passage.distance):
            raise ValueError("Retrieval returned a non-finite distance.")

    if max_distance is not None:
        passages = [passage for passage in passages if passage.distance <= max_distance]

    return manifest["index_id"], passages
