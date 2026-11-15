# Course Notes Q&A Assistant

Local terminal application for approved Markdown, UTF-8 text and text-PDF notes.
Sentence Transformers finds passages in persistent Chroma; Qwen2.5-1.5B-Instruct
judges evidence sufficiency. The configured classroom default displays verified
indexed excerpts, not the model's rewritten explanations. It supports unseen
questions through retrieval; it is not a prepared-answer lookup.

## Install

From this application directory, using Python 3.14 (the tested version):

```bash
python3.14 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
```

`requirements.lock` captures the tested macOS ARM64/Python 3.14 environment,
including development tools. Other platforms/Python versions are not certified;
the direct dependencies are declared in `pyproject.toml`. The package supports
Python 3.12–3.14 syntactically, but installation/model execution must be checked
on the target platform. On PowerShell, activation is `venv\Scripts\Activate.ps1`.

The existing `venv` and downloaded models can be reused; do not recreate/download
them merely to run an updated source file. Setup downloads weights explicitly:

```bash
course-notes-qa-setup-embeddings
course-notes-qa-setup-generation
# For already downloaded generation weights:
course-notes-qa-setup-generation --check-only
```

Setup refuses to overwrite existing models. Embeddings and generation are
separate pretrained models, not models trained from scratch by this application.
Generation weights occupy about 3.1 GB on disk and CPU float32 execution needs
substantially more working memory. A 16 GiB Mac was tested; no lower minimum or
latency guarantee has been established. Keep `models/`, `storage/` and sensitive
reports out of Git. Initial setup needs network access; runtime model loading
uses local files only. Chroma anonymized telemetry is disabled. No notes are sent
to an external generation service; this is not a general network-isolation claim.

## Select notes

Edit `configs/sources.yaml`. Its approved roots are trusted configuration, not
instructions supplied by notes. Paths resolve relative to the manifest, including
symlink containment checks. The default source refers to the sibling Webcam
project in the parent training repository. For a standalone checkout, point it
to your own approved notes. A PDF entry may include:

```yaml
schema_version: 1
sources:
  - source_id: lecture-pdf
    title: Lecture notes
    project: course
    document_type: student_notes
    path: ../notes/lecture.pdf
    approved_root: ../notes
    excluded_pages: [2]
```

For Markdown/text use `excluded_lines: [[10, 12]]`; PDFs use sorted, unique,
one-based physical `excluded_pages`. Included empty/scanned pages, encrypted PDFs,
malformed PDFs, and limit violations fail explicitly. There is no OCR. PDF line
numbers refer to extracted text within the cited physical page; extraction order
can differ from visual layout. Tables/formulas may require a reviewed text export.
The default limits are 20 files, 10 MiB/file, 100 MiB total, 2 million extracted
characters and 200 physical pages/PDF. Excluded pages still count toward page limits.

## Run

From this directory (or use the original `python src/app.py` entry point):

```bash
course-notes-qa preview
course-notes-qa ingest
course-notes-qa status
course-notes-qa retrieve "How are color webcam frames represented?"
course-notes-qa ask "How are color webcam frames represented?"
course-notes-qa chat --project webcam-filter-app
```

Use `--settings /absolute/path/app.yaml` with ingest/retrieve/ask/chat/evaluate.
`COURSE_QA_HOME` selects a checkout root for default paths and model setup.
Preview uses character budgets without loading models; final ingestion uses the
actual embedding tokenizer, so preview and indexed chunk counts can differ.
Querying never ingests automatically. Unchanged ingestion reuses the active index.

`configs/app.yaml` selects `chunking_strategy: boundary` and
`answer_mode: verified_excerpts`. `markdown_blocks` remains an opt-in experiment:
it regressed the development generation results and is not the classroom default.
Changing processing or embeddings changes index identities and can rebuild data.

Verified output is labeled `evidence_selected`. Text is checked against indexed
hashes, supplied context and metadata, with source references. This prevents
model-authored rewriting in displayed excerpts; it does not prove relevance,
completeness, source truth or correct abstention. The model can still be wrong
about whether evidence is sufficient. Source text is data, never executable tools.
Use `--answer-mode generated` only for explicit synthesis comparisons: factual
errors were observed despite valid citations. Operational errors are not abstentions.

Chat loads the models once, treats every question independently and saves no
question history. Blank input is skipped. `/quit`, `/exit` or EOF exits normally;
Ctrl+C exits with code 130. Terminal control characters are escaped on output.

## Snapshot maintenance

```bash
course-notes-qa snapshots
course-notes-qa rollback SNAPSHOT_ID
course-notes-qa cleanup --keep 2
course-notes-qa cleanup --keep 2 --apply
```

Maintenance is model-free and uses the process lock. Rollback validates an existing
snapshot's database/count/embedding identity before atomic publication. Cleanup
previews by default and preserves the active snapshot plus the newest requested
number of inactive snapshots. Only `--apply` deletes complete app snapshots;
unknown/incomplete folders and original notes are not cleanup targets. Back up
important data before deliberate deletion. Never remove a live `.lock`; after an
abnormal process stop, confirm no process uses storage before removing a stale lock.

## Verification

```bash
python -m pytest -q
python -m ruff check src tests
course-notes-qa evaluate --mode retrieval --output /tmp/retrieval.json
course-notes-qa evaluate --mode answers --output /tmp/answers.json
```

Reports refuse overwrites and may contain source text: store them appropriately.
Evaluation defaults to development. Freeze corpus/model/prompt/configuration
before explicitly selecting `--split held_out`; do not tune on held-out results.
Status agreement, retrieval coverage and exact copying are distinct from factual
accuracy. See `eval/reports/release-verification.json` when present for the actual
run evidence and unresolved gates, not a universal performance guarantee.

Exit codes: 0 command completed/normal abstention; 2 invalid input/configuration;
3 missing/incompatible index; 4 extraction/storage/operational failure; 5 generation
or output-validation failure; 130 interruption. Evaluation completion does not by
itself mean quality gates passed. One upstream Chroma Python 3.14 deprecation
warning may appear; it is not an application error.

Models/source licenses and permissions remain the operator's responsibility.
The repository owner maintains dependencies and reviews model/prompt/corpus
changes against the development cases. Report bugs with the command, versions
and sanitized error category, avoiding private notes or credentials.
