# Development answer evaluation — 14 November 2026

## Outcome

The current application does not meet the planned answer-quality gate. Valid
JSON and citations did not prevent incorrect explanations or unsupported answers.
This is an assistant content review, not student/instructor sign-off. The raw
report deliberately retains `human_review: pending` and `release_gate: not_assessed`.

Baseline report: [development-answers-2026-11-14.json](../eval/reports/development-answers-2026-11-14.json).

Command run from the application directory using its venv:

```bash
./venv/bin/python src/app.py evaluate --mode answers --split development \
  --output /tmp/course-qa-development-answers-20261114.json
```

The resulting JSON was copied unchanged into `eval/reports/`. No source code,
configuration, evaluation annotations, model weights or index was changed.
The eight held-out cases were not executed or used for this review.

## Measurements

| Measurement | Result |
| --- | --- |
| Development cases | 12 |
| Expected versus actual status matches | 8/12 (66.7%) |
| Expected abstentions correctly returned | 0/4 |
| Runtime/generation errors | 0 |
| Returned citations passing locator/hash checks | 30/30 |
| Sum of per-question latency, excluding model loading | 234.63 seconds |
| Median per-question latency | 20.95 seconds |
| Per-question latency range | 8.94–32.64 seconds |

All eight answerable cases returned `answered`. This does not mean eight correct
answers. All four cases requiring abstention incorrectly returned `answered`.
The earlier retrieval-only development result was 8/8 complete evidence groups
covered among six candidates; it does not establish that selected evidence was
used correctly by generation.

Every returned reference points to the run's snapshot, has valid line bounds,
and matches the current notes' SHA256. These checks establish traceability, not
support for every statement or relevance of every cited passage.

## Case-by-case content review

| Case | Assessment | Evidence and next concern |
| --- | --- | --- |
| w01 | Core distinction supported; explanation incomplete | Correctly says drawing a circle is not object detection. Does not explain the calculated center/radius as clearly as notes lines 129 and 210. |
| w02 | Factual error and missing expected detail | Correct array shape and uint8 range, but calls color-channel values grayscale intensity and omits BGR order. Cited lines 104–127 explicitly distinguish these concepts. |
| w03 | Factual error | Claims conversion averages all channels; cited line 127 describes a weighted combination. Also omits the exact conversion call and explicit output shape. |
| w04 | Supported corner definitions; expected detail omitted | Correct xyxy boundaries, but omits width = x2-x1 and height = y2-y1 from the expected claims and notes line 257. |
| w05 | Core distinction present; wording needs revision | Describes low-score filtering versus overlap suppression, but “multiple instances of the same object” is ambiguous. Notes line 280 explains suppression of duplicate candidates, not removal of distinct real objects. |
| w06 | Contradictory explanation; inspect context boundary too | Says training loss is reduced “during inference” while also claiming weights remain fixed. Notes lines 238 and 244 distinguish training updates from fixed-weight inference. Cited S4 ends at line 243, immediately before the explicit training paragraph; the other cited ranges cover split roles and training examples. Split roles are otherwise described. |
| w07 | Factual error | Says Shape analyzes detected objects' appearance/proportions. Cited line 210 says the circle uses a hand-calculated center/radius while a detector predicts boxes/classes. |
| w08 | Core explanation supported | Precision/recall definitions and threshold tradeoff agree with cited notes lines 270–280 and 453–472. Wording could state the false-alarm/missed-object tradeoff more directly. |
| w09 | Failed abstention; placeholder output | Returns “Your supported explanation [S1].” rather than stating that yesterday's gesture-model accuracy is absent. |
| w10 | Failed abstention; placeholder output | Same placeholder instead of explaining that no course generation-service API key is established by the notes. |
| w11 | Failed abstention; placeholder output | Same placeholder instead of stating that this morning's measured laptop frame rate is unavailable. |
| w12 | Failed abstention; instruction resistance needs work | Echoes the request to invent results and substitutes generic experiment advice. It does not fabricate a numerical accuracy, but does not clearly reject the instruction or return the required insufficient-evidence status. This fixture alone does not demonstrate injection resistance. |

No aggregate factual-accuracy score is assigned: the review distinguishes wrong
claims, omissions and ambiguous wording instead of treating status agreement as
accuracy. Instructor review is still required.

## Diagnosis and next controlled change

The prompt's final answered-only template literally contains “Your supported
explanation [S1].” Three unsupported cases copied that placeholder. This is a
plausible prompt-induced bias, not proof that changing the prompt will solve the
problem. The decoding grammar permits both answer and abstention; this run never
selected abstention.

The next source file to review is `src/course_qa/prompts.py`:

1. Remove the copyable answered-only placeholder and balance the two outcomes.
2. Make the evidence-sufficiency decision explicit before explanatory content.
3. Require faithful paraphrases, preserve implementation/training distinctions,
   and prohibit replacing missing measurements with generic advice.
4. Increment the prompt version. Keep the model, grammar, corpus, retrieval and
   limits fixed for a controlled comparison.
5. Rerun all 12 development cases and review their claims. Preserve this baseline.

If grounding failures persist, diagnose context selection, decoding behavior or
model capability separately. Do not hide failures with prepared answers,
question-specific matching, changed gold labels or permissive citation checks.
Do not open the held-out split while tuning against development failures.

## Run identity and reproducibility limits

- Snapshot: `a9eef3052a56416c975524d8572bf3f0`.
- Embedding fingerprint: `6f427e87f2bd1280b894538b3ceef2d17d2a9fc875debd000c6e5fca743be5a7`.
- Generation model: `Qwen/Qwen2.5-1.5B-Instruct`, provisioned revision
  `fb163acb387a011a0cd205b259aa1b253299a05d`, CPU float32.
- Prompt version: `course-notes-grounded-v2`.
- Limits: 4096 context tokens, 384 output tokens, 120-second cooperative time limit.
- Corpus: `webcam-only-v1`; sole source `webcam-filter-app/STUDENT_NOTES.md`.
- Source SHA256: `c22618abe800e2064e309c3d618377841376012d00b17bc3c455748154ef91a6`.
- Evaluation fixture SHA256: `5a866cdbd7aad967d8978d532f858a7cc336652c8c82b1a66f252b4d56077318`.
- `configs/app.yaml` SHA256: `54b6f4f837477038653ebeaeaace75aa114b6bbb04e2c73b9be7eb6edf6ccf04`.
- `configs/sources.yaml` SHA256: `7bebd82dc33b1c658ad478c053b6d1e2c1dfcb012279f027722bc73d681cf791`.
- `prompts.py` SHA256: `5931bf0d541e95e05acfc6ad47ee13103f3f66c009f028b3d8712ae8196ed04a`.
- `local_generation.py` SHA256: `74bae06caec71b887231a9cd37ad2392836ba08b22718f671268a8add4acbf6e`.
- Python 3.14.7; torch 2.15.0; transformers 5.22.1;
  sentence-transformers 6.2.0; chromadb 1.5.9; lm-format-enforcer 0.11.3.

This is one development run on one machine with a curated single-document
corpus. Dependencies are not yet captured in a reproducible installation lock,
model files were not rehashed for this review, and latency is not a platform
guarantee. Automated response-format validation is not a factual-grounding test.
