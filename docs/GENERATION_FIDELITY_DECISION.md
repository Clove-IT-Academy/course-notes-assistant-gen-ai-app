# Generation-fidelity diagnosis and next comparison

Date: 15 November 2026. This is a development diagnosis, not a model-quality
guarantee or a change to the application's generation/output strategy.

## Finding

The correct evidence reaches the current local Qwen2.5-1.5B-Instruct model.
Source inspection verified the BGR description and the comparison between a
calculated circle overlay and learned detection. Yet prompt-v4 development
answers still contradict those passages or cite the wrong supporting sources.

A further diagnostic removed the prefix-token filter for development cases
w02 and w07 only. The local model, evidence, chat prompt, greedy decoding,
token budget and cooperative time limit were retained. The normal JSON parser,
EOS guard and citation validator were still applied to the resulting outputs.
The change existed only in a temporary diagnostic process, not source code.

| Case | Native decoding observation | Validation outcome |
| --- | --- | --- |
| w02 | Still describes the color channels as RGB rather than the source's BGR. Uses typographic quotes inside its JSON label list. | Rejected as invalid JSON. |
| w07 | Still describes Shape as modifying detected objects instead of a calculated circle overlay. Lists S2 without a matching inline citation. | Rejected for inline/list citation mismatch. |

[Raw diagnostic report](../eval/reports/native-decoding-diagnostic-2026-11-15.json).
Its raw-response fields are explicitly diagnostic, unvalidated text. Neither
response was accepted or displayed by the production CLI as a valid answer.
Do not reuse these raw strings as supported course explanations.

The observations do not support removing constrained decoding as a fix. They
also do not prove every error is inherent to the model: prompting, context
selection and decoding can still interact. This small experiment establishes
only that these two rewriting failures persist without the format filter.

## Runtime constraints measured locally

The execution environment reports 16 GiB physical memory. The installed torch
build includes MPS, but `torch.backends.mps.is_available()` returns false in this
environment. Do not assume Apple GPU execution is available here. The current
adapter explicitly loads CPU float32 weights.

Any larger-model experiment needs a memory/runtime check, an isolated model
directory, accurate model identification and an explicitly pinned revision.
The current adapter's hard-coded model ID must not silently describe a different
model. Model loading, compatibility and measured answer quality are separate
gates. A larger parameter count alone does not establish better course answers.

## Reviewable next directions

1. **Compare a larger local generation model.** Preserve the current model,
   active 58-chunk snapshot, prompt-v4 baseline and validators. Establish a
   suitable runtime before downloading/loading a candidate. Run the development
   set, review claims against supplied passages, and adopt the candidate only
   if it improves the measured quality gates. Do not switch models automatically.
2. **Keep the current model and use verified source excerpts.** Return selected
   text only after checking it is copied from the indexed evidence and attaching
   its source references. This limits unsupported rewriting, but changes the
   product from free-form synthesis to evidence excerpts. Relevance, completeness,
   unsupported source content and instruction attacks still need evaluation.
   This is not a prepared question-to-answer lookup.

The user chose to keep the current model because students have varied laptop
specifications. The next implementation therefore uses verified source excerpts
as an explicit output mode while retaining generated mode for comparison. The
resource tradeoffs and the algorithm for comparing future adapters are recorded
in [EBOOK_NOTES.md](EBOOK_NOTES.md). No larger model has been benchmarked here.

No new model was downloaded, no provider was changed, and no model/CLI configuration
was changed during this diagnosis. The previously evaluated 58-chunk index remains
active. Held-out cases were not executed. Excerpt-mode adoption is a separate
review decision after its development evaluation.
