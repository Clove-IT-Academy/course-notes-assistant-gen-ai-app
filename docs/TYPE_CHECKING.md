# Type-checking repairs - 12 November 2026

The source keeps the existing Sentence Transformers embedding adapter and local Chroma store. No generation provider was added. The existing character-based chunker and application scope are unchanged.

## Editor setup

Open `course-notes-qa-assistant` as the VS Code workspace folder. Select `venv/bin/python` using **Python: Select Interpreter** if an interpreter was already selected: VS Code's saved selection can override the new default. Reload the window if diagnostics are stale.

`pyrightconfig.json` checks `src` in strict mode, resolves the `src` import layout, and points the command-line checker at the existing Python 3.14 environment. The `.vscode/settings.json` file provides an interpreter default and the same import path. No diagnostic rule has been disabled.

## What changed

- Fixed the embedding docstring indentation, missing comma and `expected_count` argument.
- Validated optional model dimensions and token limits, counted special tokens, and used the installed Sentence Transformers 6.2.0 dimension API.
- Narrowed the library's multimodal encoder overload to the text-to-NumPy call this adapter actually uses. Returned vectors still undergo runtime validation.
- Corrected Chroma keyword/result names, `dimension`, and the two-item `open_active` return value. Both `SnapshotStore` and the historical `SnapShotStore` spelling refer to the corrected class.
- Added `TypedDict` records for chunk metadata, manifests and ingestion outcomes. Stored JSON and database metadata are validated before becoming application records.
- Added explicit collection/iterator annotations and a typed Chroma query call. Optional result fields are checked before indexing.
- Close Chroma clients before releasing the process lock; use the public client close operation available in the installed Chroma 1.5.9.
- Fixed unchanged ingestion to return `status="unchanged"` rather than accidentally overwriting it with the manifest's `ready` status.

The metadata keys and manifest JSON shape remain the same. The corrected `transformers_version` fingerprint key changes the embedding identity; rebuild any index created with the former fingerprint using your ingestion script.

## Reproduce checks

The verification used Pyright 1.1.414, Python 3.14.7, Chroma 1.5.9, Sentence Transformers 6.2.0, pytest 9.2.0 and Ruff 0.17.3. Pyright was installed in a temporary tool directory during this repair; it was not installed into the project environment. To install it locally if desired:

```bash
./venv/bin/python -m pip install pyright==1.1.414
./venv/bin/python -m pyright
PYTHONPATH=src ./venv/bin/python -m pytest -q tests
./venv/bin/python -m compileall -q src
./venv/bin/python -m ruff check src tests
./venv/bin/python -m ruff format --check src tests
```

All 11 application files pass strict checking without ignores. Regression tests exercise real temporary Chroma storage, fresh-process reopen, source removal, failed-update recovery, deterministic preparation and context assembly. Embedding calls use a controlled model fixture; no model download, real embedding inference or generation quality is claimed. Chroma emits an upstream asyncio deprecation warning under Python 3.14; it does not fail these checks.
