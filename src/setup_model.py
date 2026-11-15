"""Provision the embedding model once; normal application runs stay local."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sentence_transformers import SentenceTransformer

from course_qa.embeddings import LocalSentenceEmbedder
from course_qa.paths import project_root

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
PROJECT_ROOT = project_root()


def setup_model(destination: Path) -> None:
    """Download a pinned model, validate locally, then publish the directory.

    Refuse to overwrite existing files. Temporary downloads are cleaned up if
    loading, saving or validation fails; no course notes are sent to the hub.
    """
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. "
            "Use the existing model or choose a different --output path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} at revision {MODEL_REVISION}")
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )

    with TemporaryDirectory(
        prefix=".embedding-setup-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "model"
        model.save_pretrained(
            str(staged), safe_serialization=True, create_model_card=False
        )
        (staged / "provisioning.json").write_text(
            json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION}, indent=2)
            + "\n",
            encoding="utf-8",
        )

        # Exercise the exact local-only adapter used by ingestion and retrieval.
        embedder = LocalSentenceEmbedder(staged)
        vector = embedder.embed_query("A webcam frame contains image pixels.")
        if len(vector) != 384:
            raise ValueError("Expected a 384-dimensional embedding.")
        print(f"Local smoke check passed: {len(vector)} dimensions.")
        print(f"Embedding input limit: {embedder.max_tokens} tokens.")
        print(f"First eight vector values: {vector[:8]}")
        if destination.exists():
            raise FileExistsError(
                "Destination appeared during setup; refusing to replace it."
            )
        staged.rename(destination)

    print(f"Saved model: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and validate the embedding model once."
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "models/embeddings"
    )
    args = parser.parse_args()
    try:
        setup_model(args.output)
    except KeyboardInterrupt:
        print("\nModel setup interrupted.")
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Model setup failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
