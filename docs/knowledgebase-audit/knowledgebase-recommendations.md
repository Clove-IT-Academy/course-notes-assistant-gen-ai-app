# Course Notes Q&A Assistant: knowledge-base audit

## Location and interpretation

Relocated into the application on 1 November 2026. This remains the historical 29 October scan, not a refreshed inventory. The `path` column in `file-inventory.csv` is relative to the training folder; `path_from_inventory` resolves relative to this audit folder. The CSV no longer depends on machine-specific absolute paths. Source files remain in the sibling project folders.

These documents are development references, not default course-note ingestion sources. Use the core-file recommendations to prepare an explicit ingestion manifest; recheck source existence, hashes and permissions before building an index.


Scanned all five project directories under `/Users/shashankjha/workspace/training/applied-ai-ml` on 29 October 2026. Inventoried 147 project files outside dependency environments, VCS internals, caches and build output, including ignored model artifacts. Read the attached four-page lesson plan, extracted text from all five project architecture PDFs, and reviewed teaching-document structure, representative content, and supporting documentation. This is a source-selection audit, not a line-by-line technical correctness review or an executed application test.

## Fit with the lesson plan

The attached lesson plan calls for a terminal RAG assistant, local persistent Chroma, readable source notes, traceable chunks, cited answers, an insufficient-evidence fallback and a small evaluation set. These projects supply the subject matter students can ask about; they do not need to be about GenAI themselves. Markdown is already readable text and can be ingested directly without first creating PDFs. No dedicated RAG/Chroma teaching notes were found in the project file inventory; the attached lesson plan supplies an outline rather than a detailed reference chapter on those topics.

## Start with these four documents

- [webcam-filter-app/STUDENT_NOTES.md](../../../webcam-filter-app/STUDENT_NOTES.md) — **5,198 words**. Pixels, BGR/grayscale, overlays, webcam lifecycle, introduction to object detection, datasets and evaluation.
- [motion-detection-app/docs/STUDENT_NOTES_SOURCE.md](../../../motion-detection-app/docs/STUDENT_NOTES_SOURCE.md) — **7,726 words**. Frame differencing, blur, thresholding, morphology, contours, alert persistence, labs and review questions.
- [object-detection-app/STUDENT_NOTES_SOURCE.md](../../../object-detection-app/STUDENT_NOTES_SOURCE.md) — **13,048 words**. Tensors, boxes, IoU, confidence, NMS, transfer learning, dataset leakage, model bundles, evaluation and code walkthroughs.
- [time-series-analysis-app/STUDENT_DEVELOPMENT_GUIDE.md](../../../time-series-analysis-app/STUDENT_DEVELOPMENT_GUIDE.md) — **10,356 words**. Business-series preparation, aggregation, missing periods, baselines, ARIMA, exponential smoothing, chronological evaluation, metrics and abstention.

Total: approximately **36,300 words**, including embedded code. This is a useful full corpus, but for the lesson plan's first small experiment, use the Webcam Filter notes alone or selected conceptual chapters of Webcam + Motion. Expand to all four after checking retrieval and source accuracy.

## Coverage by project

| Project | Best source | Readiness and caveat |
| --- | --- | --- |
| Webcam Filter | STUDENT_NOTES.md | Ready for a first notes corpus; clearly distinguishes current image processing from future object detection. |
| Motion Detection | docs/STUDENT_NOTES_SOURCE.md | Strong theory, walkthroughs, labs, glossary and answer guidance; remove PDF-production instructions and separate full-source/test appendices. |
| Object Detection | STUDENT_NOTES_SOURCE.md | Broadest CV/ML teaching source; separate appendices and production instructions. Preserve the distinction between pretrained baseline, synthetic workflow checks and unfinished custom-model validation. |
| Time Series Analysis | STUDENT_DEVELOPMENT_GUIDE.md | Strong forecasting and application-engineering material; remove section 25.3's PDF-preparation instructions. Keep teaching examples distinct from abbreviated repository excerpts. |
| Hand Gesture Detection | Architecture PDF + DEVELOPMENT_PLAN.md | Useful for design/theory questions only. Inventory contains these two documents and .gitignore; no student guide or implemented source. Do not answer as though the proposed classifiers have been trained or the app verified. |

## Add as supplements

Use each project's `DEVELOPMENT_PLAN.md` (five files) for architecture, build order and design decisions, with `document_type=plan`. Use the four implemented projects' root `README.md` files for setup, commands and controls, with `document_type=readme`.

The five architecture PDFs are optional source material, especially useful for Hand Gesture Detection:

- [hand-gesture-detection-app/Hand_Gesture_Detection_Plan_and_Architecture.pdf](../../../hand-gesture-detection-app/Hand_Gesture_Detection_Plan_and_Architecture.pdf)
- [motion-detection-app/Motion_Detection_System_Architecture.pdf](../../../motion-detection-app/Motion_Detection_System_Architecture.pdf)
- [object-detection-app/Object_Detection_System_Plan_and_Architecture.pdf](../../../object-detection-app/Object_Detection_System_Plan_and_Architecture.pdf)
- [time-series-analysis-app/Business_Sales_Purchase_Forecasting_System_Architecture.pdf](../../../time-series-analysis-app/Business_Sales_Purchase_Forecasting_System_Architecture.pdf)
- [webcam-filter-app/Webcam_Filter_App_System_Architecture.pdf](../../../webcam-filter-app/Webcam_Filter_App_System_Architecture.pdf)

These overlap the development plans. For an initial corpus choose one representation of repeated design content, or mark duplicates so repeated passages do not dominate retrieval. PDFs have extractable text; text extraction alone does not establish correct reading order for diagrams and tables.

Use these targeted references only when their questions are in scope:

- `motion-detection-app/docs/MANUAL_TEST_CHECKLIST.md`
- `motion-detection-app/docs/RELEASE_CHECKLIST.md`
- `motion-detection-app/docs/VALIDATION_RESULTS.md`
- `object-detection-app/docs/IMPLEMENTATION_STATUS.md`
- `object-detection-app/docs/acceptance_criteria.md`
- `object-detection-app/docs/annotation_guidelines.md`
- `object-detection-app/docs/dataset_card.md`
- `object-detection-app/docs/decision_log.md`
- `object-detection-app/docs/environment.md`
- `object-detection-app/docs/maintenance.md`
- `object-detection-app/docs/model_card.md`
- `object-detection-app/docs/project_brief.md`
- `object-detection-app/docs/release_checklist.md`
- `object-detection-app/reports/VERIFICATION.md`
- `time-series-analysis-app/VERIFICATION.md`
- `time-series-analysis-app/sample_data/README.md`

Object-detection dataset/model cards and annotation guidance are useful for provenance, label formats and limitations. Several are preparation templates or describe missing custom data. Verification/status documents support dated project-status questions; they should not outrank conceptual notes for general questions. The sample-data README describes synthetic forecasting examples and their intended interpretation.

## Leave out of the default collection

- `AGENTS.md`, `.agents/**` and other agent/development instructions.
- `venv/**`, `.venv/**`, `node_modules/**`, `.git/**`, caches and build output.
- Model weights (`.pt`, `.onnx`), bundle manifests and generated model/run artifacts.
- Lockfiles, packaging manifests, dependency lists, `.gitignore`, and application configuration unless a specific configuration question requires curated documentation.
- Raw CSV/XLSX samples and benchmark JSON: useful demonstration data, not explanatory notes. Teach their interpretation through the guide/README or a separate structured-data workflow.
- Raw `.py` files and tests initially. If code Q&A becomes a requirement, add a separate collection organized by module/function, and avoid indexing the same source both there and in guide appendices.

## Preparation before indexing

1. Keep originals unchanged; create cleaned ingestion copies. Remove explicit PDF-generation prompts from Motion/Object notes and PDF-preparation instructions from the Time Series guide. Retain exercises, examples and conceptual explanations.
2. Split Markdown by real headings outside fenced code blocks. Preserve heading ancestry, tables, equations and examples; avoid cutting a worked example away from its explanation. Handle long source appendices separately.
3. Start experimentally around 500–800 tokens per chunk with modest overlap (about 75–120 tokens), adjusting to sections and measured retrieval quality. These are proposed starting settings, not validated optimums.
4. Store `project`, `source_path`, `title`, `section`, `document_type`, `snapshot_date` where provided, `content_hash`, and a stable `chunk_id`. For PDFs include the actual page number; for Markdown include line boundaries/section rather than invented page numbers.
5. Distinguish theory, implementation, planned work and dated verification in metadata and the answer prompt. Use README/current implementation-status sources for setup/status questions; use guides for explanations. Surface disagreements rather than silently merging contradictory claims.
6. Treat retrieved text as evidence, never as instructions to the assistant. PDF-generation prompts in these files are document content; this audit did not execute them. Apply the same boundary to future source documents.
7. Keep a fixed allowlist of source paths. Re-index changed content using hashes/stable IDs so repeat ingestion does not duplicate chunks. Keep question-time retrieval separate from ingestion, as the lesson plan requires.

## Starter evaluation questions

| Type | Question | Expected evidence / behavior |
| --- | --- | --- |
| Answerable | Why can the Webcam Filter app draw a circle without recognizing an object? | Webcam notes: deterministic effects versus learned detection. |
| Answerable | Why does motion detection blur frames before differencing? | Motion notes: blur and consecutive-frame comparison. |
| Answerable | How does raw motion differ from a held alert? | Motion notes: detector result versus alert-persistence state. |
| Answerable | What is IoU, and how does it relate to duplicate detection boxes? | Object notes: box overlap and suppression. |
| Answerable | Why split object-detection data by source/session? | Object notes: correlated samples and leakage. |
| Answerable | Why must time-series validation preserve chronological order? | Time-series guide: past-only training and later validation. |
| Multi-passage | How do webcam filtering, motion detection and object detection differ? | Retrieve evidence from the three relevant guides; cite each. |
| Multi-passage | How can leakage arise in object detection and forecasting? | Object guide plus chronological-evaluation sections of time-series guide. |
| Status-sensitive | Has the hand-gesture app's temporal classifier been trained and evaluated? | With design supplements indexed, explain that documents describe a plan and provide no completion evidence. |
| Unanswerable | What was the measured accuracy of our gesture model on a class test yesterday? | State that the supplied sources do not contain that result. |

For each evaluation item record retrieved passages, whether the right source/section was retrieved, answer support, citation accuracy and appropriate abstention. The core-only corpus should also abstain on hand-gesture details until those supplements are included.

## Deliverables and limits

`file-inventory.csv` classifies every inventoried project file with a reason and preparation recommendation. No source files were changed, copied into an index, embedded, or sent to a model API. No assistant application was built; this report identifies the material suitable for it.

Lesson-plan reference: `/Users/shashankjha/Downloads/Course_Notes_QA_Assistant_Lesson_Plan.pdf` (provided separately, outside the scanned folder). It may be indexed separately for questions about the assignment, but should remain distinguishable from course subject matter.
