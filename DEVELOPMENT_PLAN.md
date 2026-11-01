# Course Notes Q&A Assistant - Development Plan

Prepared: 1 November 2026 | Status: documentation-only baseline; implementation not started.

## 1. Purpose and source of truth

Implement the terminal RAG application defined in [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md), also available as the [architecture PDF](output/pdf/Course_Notes_QA_Assistant_System_Architecture.pdf). The supplied four-page lesson plan governs the learning scope. The architecture defines component contracts; this plan defines implementation order, deliverables and evidence gates.

Follow the existing folder's practice of explicit phases, testable boundaries, decisions and acceptance criteria. This is a new application: do not copy other apps' completed checkboxes, dependency versions or measured results. No application code, Chroma index, provider configuration or verified environment is delivered by this planning work.

## 2. Intended outcome and scope

A student can prepare approved course notes once, reopen their local index, ask independent questions from a terminal and inspect cited answers or a clear insufficient-evidence outcome. They can demonstrate retrieval, explain one improvement and distinguish retrieval failures from generation failures.

Deliver Markdown/text and text-PDF ingestion, explicit embedding and generation adapters, local persistent Chroma, provenance-aware chunks, retrieval inspection, bounded API use, independent-question CLI loop and a small evaluation suite. Keep the web UI, backend server, agents, browsing, OCR, fine-tuning, multi-user hosting and conversational memory outside v1.

Use a modular Python package with pure loaders/chunking helpers, a storage adapter, provider adapters and service-level orchestration. Prefer direct SDK calls and standard CLI facilities to an unnecessary orchestration framework.

### Knowledge-base audit and source ownership

The historical source audit is part of this app documentation:

- [Source recommendations](docs/knowledgebase-audit/knowledgebase-recommendations.md): core notes, supplemental references, exclusions, preparation guidance and starter evaluation questions.
- [Complete file inventory](docs/knowledgebase-audit/file-inventory.csv): 147 files classified in the 29 October 2026 scan, with selection reasons and preparation notes.

The audit is a historical baseline, not a live manifest. Its `path` values are relative to the training folder; `path_from_inventory` values resolve from the audit directory. Original course materials stay in sibling app directories and must be checked again before ingestion. Exclude the audit itself from the default course-notes index.

Phase 0 must turn selected recommendations into an allowlisted source manifest with stable source IDs, resolved paths, document types, permissions, exclusion rules and fresh hashes. Seed the evaluation set from the audit questions, then expand it to the required 20 cases. Phase 2 validates extraction and cleaning rather than assuming every audited document is ingestion-ready.

## 3. Architecture contracts to preserve

| Area | Responsibility and invariant |
| --- | --- |
| `cli.py` | Parse commands, display safe output, handle exit/EOF/Ctrl+C; no retrieval policy. |
| `contracts.py`, `config.py` | Typed records, validated budgets, explicit provider/data-flow settings. |
| `loaders.py`, `cleaning.py` | Approved files to traceable segments; no original-source mutation or external calls. |
| `chunking.py` | Token-bounded chunks, preserved locators, deterministic identity. |
| `embeddings.py` | One chosen embedding space for documents and questions; verify model identity and vectors. |
| `store.py` | Chroma persistence/query, conversion to passage records, snapshot lifecycle and access lock. |
| `ingestion.py` | Dry run, complete staged build, verification and active-pointer publication. |
| `retrieval.py`, `context.py` | Candidate ranking/filtering, deduplication, usable evidence and token budgeting. |
| `prompts.py`, `generation.py` | Evidence-only prompt and bounded API transport; no tools or browsing. |
| `citations.py` | Validate source labels and render metadata-owned locators; no model-generated paths. |
| `service.py` | Answer/abstain/error orchestration independent of terminal. |
| `evaluation.py` | Retrieval/answer measurements and explicit reports; no silent tuning on held-out cases. |

The embedding fingerprint, distance metric, processing configuration and corpus hashes belong to the index contract. A changed contract creates a new snapshot. Querying never creates an empty collection or ingests documents implicitly. API/network activity occurs only through explicitly configured adapters and deliberate model provisioning.

## 4. Ordered development phases

### Phase 0 - Freeze scope, corpus and evaluation intent

**Goal:** Make success and input ownership explicit before implementation.

1. Confirm CLI-only, single-user scope and independent questions. Adopt architecture FR-1 through FR-10.
2. Recheck the core-source files from `docs/knowledgebase-audit/knowledgebase-recommendations.md` against `file-inventory.csv`. Begin with Webcam Filter notes; plan later expansion to Motion/Object/Time Series notes.
3. Write a source inventory with document type, approved root, owner/permission, declared snapshot date and excluded regions. Treat architecture/plan supplements as optional.
4. Draft 20 evaluation cases and their expected claims/source sections: 10 single-source, 4 multi-passage, 4 absent-evidence, 2 adversarial. Split 12 development and 8 held-out with every type represented in both; reserve held-out cases before tuning.
5. Record open provider, platform, data-handling, budget and maintenance decisions. Use the architecture's numeric limits as provisional, not measured optimums.

**Deliverables:** product brief, refreshed selected-source inventory and allowlisted manifest specification derived from the audit, evaluation specification and decision log.

**Exit gate:** required sources are readable/authorized, supported outcomes are defined, and the evaluation split is frozen. Provider-specific processing waits for Phase 1's data-flow selection.

### Phase 1 - Environment and provider feasibility spike

**Goal:** Resolve compatibility, data flow and API contracts early.

1. Create a project-local environment; select Python/OS based on actual Chroma, reader, tokenizer and SDK compatibility. Pin the tested set and capture reproducible installation commands.
2. Select one embedding mode/model/revision and one generation provider/model. Record embedding dimension, normalization, distance metric, model input/output limits and tokenizers.
3. Verify explicitly configured embedding requests and a tiny temporary Chroma round trip with approved synthetic text. Restart the process and retrieve again.
4. Verify one bounded generation request, parsed response envelope, usage reporting where available, timeout/error behavior and credential redaction.
5. Check the provider's current data retention/processing terms and document exactly which notes/questions leave the machine. If local embeddings are selected, provision the model explicitly and verify no per-query download occurs.
6. Check telemetry and model downloads for the selected dependencies. Document or disable unwanted behavior; avoid unsupported no-network claims.
7. Confirm the package/CLI layout in architecture section 10. Do not build features on unresolved embedding compatibility.

**Deliverables:** dependency lock, environment note, adapter contracts, chosen data-flow mode, spike evidence and revised defaults.

**Exit gate:** approved synthetic end-to-end adapter/persistence smoke passes, model/tokenizer budgets are known and no external processing mode is implicit. If live API access is unavailable, continue local/fake-based work but keep this gate explicitly incomplete before live/demo claims.

### Phase 2 - Manifest validation and document extraction

**Goal:** Convert approved files into inspectable, correctly located text.

1. Define manifest schema and safe config loading. Resolve paths relative to the manifest and verify approved roots after symlink resolution.
2. Implement size/count/page/extraction-output limits and allowlisted extensions. Provide dry-run validation with no API calls or index mutation.
3. Implement UTF-8 text/Markdown reading with original line numbers and PDF extraction with one-based physical pages.
4. Implement explicit exclusion rules for generation instructions, appendices and PDF-preparation sections. Preserve removal/locator mapping; leave originals unchanged.
5. Detect empty/scanned/unreadable/encrypted PDFs, malformed text and missing required sources. Fail clearly rather than silently indexing incomplete evidence.
6. Create tiny synthetic fixtures covering multi-page extraction, headings inside code fences, blank/scanned pages, invalid encoding and escaped paths.

**Deliverables:** config/contracts, loaders/cleaning, fixtures and extraction preview.

**Exit gate:** approved examples produce useful text and correct locators; boundary failures are controlled and cause no external calls. A human checks representative cleaned excerpts before embedding them.

### Phase 3 - Chunking and provenance

**Goal:** Produce reproducible passages suitable for retrieval and citation.

1. Split Markdown on headings outside code fences and text/PDF on paragraph/page structure. Keep small tables and worked examples together where possible.
2. Start at target 650/max 800 tokens with up to 100 overlap within a section, lowering caps for the selected embedding model when necessary. Include any embedding prefix in counting.
3. Split oversized sections/blocks explicitly, preserve heading ancestry and page/line locators, and avoid carrying excluded instructions into overlap.
4. Define source hashes, processing fingerprints and chunk IDs exactly as architecture section 5 specifies. Store compatible scalar metadata.
5. Inspect overlapping chunks for redundancy and sample answers spanning two sections. Test identical-input determinism and changed-content identity.

**Deliverables:** chunker, chunk preview, metadata schema and deterministic fixtures.

**Exit gate:** nonempty unique chunks meet actual model limits, locator references reconstruct the indexed evidence and original files are untouched. Exact same files/config produce the same IDs.

### Phase 4 - Explicit embeddings and safe local persistence

**Goal:** Build/reopen a complete local snapshot without stale or duplicate material.

1. Implement bounded embedding batches and shape/finite-value validation. Provide vectors explicitly to Chroma; prohibit implicit default embeddings.
2. Implement the single-command process lock and app-owned storage paths. Missing/incompatible active indexes are explicit errors.
3. Build in an isolated snapshot directory, write a manifest and collection, then close/reopen and verify counts, source metadata and retrieval smoke behavior.
4. Publish only by atomically replacing the active pointer after validation. Inject failures before write, mid-batch and before publication.
5. Skip unchanged source/config fingerprints. Rebuild all changed/removed-source cases into a new snapshot; retain old successful snapshots for rollback.
6. Implement explicit rollback/cleanup maintenance behavior with containment checks; cleanup never removes active data or original notes.

**Deliverables:** embedding/store/ingestion adapters, manifest format and persistence/recovery integration checks.

**Exit gate:** fresh-process reopen works; same-input ingest does not duplicate records; edited/removed sources disappear from the newly active snapshot; interrupted builds leave the previous index usable. Vector/model mismatch is rejected even when dimensions match.

### Phase 5 - Retrieval inspection and context budgeting

**Goal:** Retrieve evidence before adding generated answers.

1. Implement explicit query embedding and project metadata filters. Convert Chroma results into aligned typed passages with raw distances and ranks.
2. Start with six candidates, deduplicate overlap and select at most four. Keep passages from multiple sources when the question requires them.
3. Use development questions to inspect missing/irrelevant results. Tune chunking, top-k or evidence cutoff one factor at a time; thresholds are metric/model specific.
4. Build labeled context including project, title, section/page and passage text. Count the complete prompt reservation and remove whole low-priority chunks when over budget.
5. Return insufficient evidence without calling generation when nothing usable remains. Provide safe `retrieve` output for inspection.

**Deliverables:** retriever, context builder, baseline retrieval measurements and inspection workflow.

**Exit gate:** gold-section coverage is measured on the development split; selected context fits the actual model budget; cross-project cases retrieve all required evidence or are recorded as failures. Do not use generation to hide poor retrieval.

### Phase 6 - Grounded generation and citation validation

**Goal:** Return supported answers, honest abstention and controlled model failures.

1. Version an evidence-only prompt. Treat retrieved content as untrusted data; prohibit tool use, following source instructions and invention of absent project results.
2. Require an answer/insufficient-evidence response envelope and citations on factual paragraphs. Support parsing/validation even if native structured output is unavailable.
3. Map source labels only to the passages actually sent. Reject unknown/missing labels and malformed envelopes; suppress invalid answer text.
4. Distinguish model-declared insufficient evidence from API outage, context overflow and invalid response. No automatic repair-generation loop in v1.
5. Implement bounded transient retries/deadlines and redacted errors. Test failures with fakes; make live checks explicit and opt-in.
6. Manually check factual support and contradictory/planned-versus-implemented material. Valid citation IDs alone do not prove that an answer is true.

**Deliverables:** prompt, generation/citation adapters, Q&A service and answer-quality review.

**Exit gate:** deterministic fake scenarios cover answer/abstain/error, invalid citations never reach the terminal, no-evidence scenarios skip generation, and live approved cases demonstrate grounded answers with accurate locators.

### Phase 7 - Complete CLI and lifecycle

**Goal:** Expose a usable terminal workflow.

1. Wire ingest, status, retrieve, ask, chat and evaluate to services. Validate CLI values before side effects; document proposed exit codes from architecture section 8.
2. Print active snapshot and external-data mode. Show concise answer plus sources; expose retrieved evidence only through explicit inspection/report options.
3. Implement independent question-loop semantics, blank input, Unicode, project selection, quit, EOF and interruption. Sanitize control/escape characters from untrusted terminal text.
4. Close clients and release the lock on every exit. Recover from question-level API errors without corrupting index state.
5. Verify no silent ingestion, implicit model switch or question/answer history persistence.

**Deliverables:** installable CLI, safe output formatting, usage documentation and CLI integration checks.

**Exit gate:** a new user completes ingest -> status -> retrieve -> ask -> chat -> exit using documented commands; fresh-process persistence works and all terminal exits release resources.

### Phase 8 - Evaluate, diagnose and improve

**Goal:** Demonstrate quality with evidence and one justified improvement.

1. Run development cases, saving explicit reports containing retrieved IDs, expected-section coverage, response status, claims/citations review, latency and usage when available.
2. Diagnose failures by layer: extraction, exclusions, chunking, embedding, retrieval, context, prompt or model behavior. Change one relevant factor and record before/after results.
3. Run injection fixtures, mismatched index, no index, empty/unreadable source, provider failure, interrupted publication, source removal and token-overflow checks.
4. Freeze corpus/model/prompt/config/dependencies before opening held-out cases. Evaluate once without tuning; replace held-out cases if they become development inputs.
5. Apply the proposed gates below, report raw counts and retain honest limitations. Do not call a small curated sample a general accuracy guarantee.

**Deliverables:** 20-case suite, development and held-out results, improvement note and limitation register.

**Exit gate:** deterministic checks pass; at least 80% of answerable cases in each split retrieve all required sections; displayed citations all resolve; reviewed accepted answers have no unsupported factual claims; all deliberately unanswerable cases abstain; adversarial fixtures do not produce the prohibited behavior. Fix release-blocking failures and rerun affected checks.

### Phase 9 - Reproducible delivery and ownership

**Goal:** Make the application usable by another student.

1. Complete README: purpose, supported environment, source rights, external-data flow, keys, setup, ingestion, retrieval, asking, evaluation, errors, costs/limits and privacy.
2. Verify setup in a clean project environment using only the README, then demonstrate the main flow on every platform claimed supported. Pin versions only after verification.
3. Review the distribution for secrets, private sources, storage, evaluation content and accidental environment/cache files. Ship sanitized fixtures and example manifests.
4. Record version, measured limitations, known incomplete work and rollback instructions. Assign maintainer, issue route and dependency/model-update review owner.
5. Regenerate the architecture PDF if the design changed, and synchronize decision/verification records. Keep proposed scope distinct from demonstrated behavior.

**Deliverables:** versioned source handoff, clean-install evidence, verification record, safe examples and ownership note.

**Exit gate:** another student can install, ingest and ask with sources; claims match actual evidence; source/API privacy behavior is documented; all FR requirements are accounted for.

## 5. Verification and requirements traceability

| Requirement | Main phases | Meaningful verification |
| --- | --- | --- |
| FR-1 approved ingestion | 0, 2 | Allowlist, traversal/symlink rejection, type/limit failures, no source mutation |
| FR-2 traceable chunks | 2-3 | Exact page/line mapping, fences/tables, deterministic IDs, instruction exclusions |
| FR-3 reopen persistence | 1, 4 | Real temporary Chroma and fresh-process restart with compatible fingerprints |
| FR-4 bounded retrieval | 5 | Relevant-section coverage, filters, duplicate context, multi-source evidence |
| FR-5 grounded citations | 6, 8 | Citation-label validation plus human entailment/locator review |
| FR-6 abstention/errors | 5-8 | Missing evidence, API timeouts, invalid output and incompatibility as distinct outcomes |
| FR-7 CLI lifecycle | 7, 9 | Commands, exit codes, EOF/Ctrl+C, independent questions and cleanup |
| FR-8 evaluation | 0, 8 | Frozen split, separate retrieval/answer metrics, before/after evidence |
| FR-9 privacy/limits | 1-2, 6-9 | Fake-transport call assertions, redacted logs, token/call bounds, approved live requests |
| FR-10 safe replacement | 4, 7 | Injected failures, no duplicates, changed/removed sources, old snapshot remains usable |

Use synthetic/non-sensitive fixtures for automated tests. Keep default tests offline with fake model adapters; test real Chroma locally. Live API checks must be explicit, bounded and reported separately. Testing document generation does not count as testing the future assistant.

## 6. Risk register

| Risk | Planned response |
| --- | --- |
| Irrelevant nearest neighbors or incomplete cross-source evidence | Inspect retrieval, use expected sections, tune on development split and allow abstention. |
| Citation exists but does not support claim | Human claim-level review; preserve evidence and avoid claiming automated truth validation. |
| Embedded instructions contaminate answers | Explicit source exclusions, data/instruction boundary, no tools, adversarial fixtures; document residual risk. |
| Incorrect PDF extraction or locators | Page-aware parsing, human preview, reject unsupported scans; no invented OCR/diagram content. |
| Stale/mixed embedding space or partial updates | Full fingerprint, immutable build snapshots, staged verification and atomic pointer. |
| API cost, outage or leaked content | Approved mode, actual token budgets, deadlines/retries, dry run, redaction and no implicit fallback. |
| Dependency/model changes break behavior | Verified pins, explicit revisions and repeated persistence/retrieval/grounding checks after change. |
| Good demo overstates quality | Frozen held-out cases, raw counts, separate failure types and bounded claims. |

## 7. Milestones and dependencies

- M1: Feasibility and source/evaluation policy approved by evidence - Phases 0-1.
- M2: Traceable clean chunks - Phases 2-3.
- M3: Reopenable, recoverable index - Phase 4.
- M4: Measured retrieval without generation - Phase 5.
- M5: Grounded answer/abstention with valid citations - Phase 6.
- M6: Complete CLI - Phase 7.
- M7: Evaluated and reproducible student handoff - Phases 8-9.

Follow the dependency order. README drafting and synthetic test-case preparation may accompany implementation, but integration waits for agreed contracts. Do not schedule calendar deadlines until provider access and compatibility are established. No agent delegation is required by this plan.

## 8. Decision log and open questions

| Decision | Status | Baseline / evidence needed |
| --- | --- | --- |
| Product surface | Design baseline | Python CLI; no separate UI/backend. |
| Corpus | Design baseline | Webcam first, four core guides next; supplements explicitly opt-in. |
| Storage lifecycle | Design baseline | Local Chroma; complete isolated rebuilds, active pointer, single-command lock. |
| Embedding provider/model/revision | Open | Select after data-flow and compatibility spike; explicit document/query identity. |
| Generation provider/model | Open | Verify API, output parsing, tokenizer, limits, pricing and processing terms. |
| OS/Python/dependencies | Open | Select and pin only after installation/restart checks. |
| Retrieval threshold | Open | Calibrate with selected embedding metric and development cases. |
| Chunk/retrieval/token limits | Proposed | Architecture defaults; revise with measured quality and actual model limits. |
| Evaluation targets | Proposed | Adopt before tuning; small-sample demo gates, not production guarantees. |
| Source data permissions/retention | Open | Confirm for selected external processing mode before live note requests. |
| Latency and API spend budget | Open | Measure and agree before release claims. |
| Storage location and maintenance owner | Open | Assign before handoff. |

Record changes here with reason, affected FRs, migration impact and checks. Changing a model/provider/prompt/corpus must not silently reuse incompatible index data or old evaluation claims.

## 9. Progress and final acceptance

- [x] Read lesson plan and reference project documentation.
- [x] Identify initial course-note sources and preparation caveats.
- [x] Write system architecture and this implementation plan.
- [ ] Phase 0: freeze corpus/evaluation specification.
- [ ] Phase 1: verify dependencies, providers and data mode.
- [ ] Phase 2: extract/clean source documents with correct locators.
- [ ] Phase 3: produce deterministic token-bounded chunks.
- [ ] Phase 4: build/reopen/replace snapshots safely.
- [ ] Phase 5: inspect and evaluate retrieval.
- [ ] Phase 6: generate and validate grounded answers.
- [ ] Phase 7: complete CLI lifecycle.
- [ ] Phase 8: evaluate, improve and pass held-out gates.
- [ ] Phase 9: verify clean setup and hand off ownership.

Final acceptance requires the full CLI demo, accurate source references, sufficient-evidence and absent-evidence examples, persistent restart, documented failure handling, frozen evaluation results and one explained improvement. Record exact commands, environment, model/config/index versions and remaining limitations in a future VERIFICATION.md. No runtime tests or model results have been claimed in this planning baseline.
