# Notes to incorporate into the course ebook

This is a writing backlog and decision record, not the finished ebook.

## Local models and student hardware

Keep Sentence Transformers as the embedding model and Qwen2.5-1.5B-Instruct as
the generation model. They serve different purposes: embeddings find related
passages; the language model attempts to interpret the question and evidence.
Neither was trained from scratch for this application. Saving pretrained weights
locally is model provisioning, not training our own foundation model.

The current application was exercised on a 16 GiB machine with CPU float32
generation. That is a tested environment, not a minimum requirement or a promise
that all student laptops can run it. Include a clean-install/resource check and
state the eventual supported hardware explicitly.

Smaller models generally reduce weight memory and computation, but their
instruction following, factual fidelity and citation selection can be limited.
Our development runs exposed BGR/RGB confusion, incorrect grayscale explanations,
and misinterpretation of the implemented Shape effect even when correct evidence
was present. Valid JSON and citation IDs did not prove factual support. Removing
the decoder's format filter did not fix two tested rewriting failures.

## Larger models: opportunity and resource tradeoff

Write that a larger or better-suited model **may** improve grounded synthesis.
Do not say it would have performed better: no larger-model comparison has been
run for this application. Parameter count alone is not a quality guarantee.

An initial weight-memory estimate is:

```text
weight bytes ≈ parameter count × bytes per parameter
float32: 4 bytes; float16/bfloat16: 2 bytes
int8: approximately 1 byte; 4-bit: approximately 0.5 bytes
```

Quantization metadata and runtime overhead make the quantized estimates
approximate. Actual peak memory also includes activations, attention/KV cache,
the tokenizer, embeddings, framework buffers and the operating system. Reduced
precision or quantization needs a compatible, tested runtime and may change
accuracy or speed. More parameters generally require more computation; latency
depends on hardware, kernels, context/output length and precision. Never assume
GPU availability merely because a laptop has an integrated GPU.

For example, 3 billion parameters in float32 need roughly 12 GB of weight memory
alone, before those other allocations. Disk download size is a separate measure.
Avoid promising that an additional model will fit or be faster without testing.

## Algorithm for adding and comparing another local generation adapter

1. Preserve the current model, configuration and evaluation baseline. Make the
   candidate an explicit option; do not silently fall back or replace weights.
2. Check memory, CPU/GPU availability, Python/library compatibility, license and
   model context limits. Choose a precision/runtime supported by the machine.
3. Provision a pinned model revision into a separate directory. Load locally
   with the established trust/download policy and record accurate model identity.
4. Implement the existing generation contract, including exact chat-template
   token counting, bounded execution, response parsing and clear failures.
5. Keep embedding/index identities unchanged when only generation changes.
   Changing embeddings requires a compatible new index; changing generation
   requires re-budgeting its tokenizer/context, not automatically re-embedding.
6. Freeze corpus, retrieval, prompt, limits and cases for a controlled comparison.
   Run development cases; review factual claims, citation support, abstention,
   adversarial behavior, peak memory and latency. Keep raw counts and failures.
7. Adopt a candidate only after its measured gates pass. Preserve rollback
   settings and disclose remaining limitations. Open held-out cases only after
   tuning is complete; do not turn them into development feedback.

This is an extension plan, not a feature already implemented or benchmarked.

## Verified source-excerpt mode

The current low-resource alternative keeps the same generation model. It uses
its existing bounded response to judge evidence sufficiency, but discards the
generated explanation when source-excerpt mode is selected. The application
returns the whole bounded evidence context so the model's citation choice cannot
silently omit a supplied definition.

Before display, each passage must match its indexed content hash and the text
and metadata actually supplied in the context. Source references are constructed
from stored metadata. The result is labeled `evidence_selected`, not `answered`.
Abstention receives a fixed application message rather than unverified model prose.
Malformed model responses and operational failures remain errors, not silent
fallbacks to excerpts. Terminal controls are escaped when displaying source text.

This prevents model-authored rewriting in displayed excerpts, but does not prove
relevance, completeness, source truth or a correct sufficiency decision. Source
material itself can contain errors or untrusted instructions. Model-selected
abstention can also be wrong. Test these limitations rather than claiming that
excerpt verification eliminates hallucinations or prompt injection.

The mode handles unseen questions through semantic retrieval; it is not a lookup
of prepared questions and answers. It trades free-form synthesis for inspectable
source passages. The generated mode remains available for teaching comparisons.

Example, from the parent workspace with the venv active:

```bash
python course-notes-qa-assistant/src/app.py ask \
  "How are color webcam frames represented in NumPy?" \
  --answer-mode verified_excerpts

python course-notes-qa-assistant/src/app.py chat --answer-mode verified_excerpts

python course-notes-qa-assistant/src/app.py evaluate --mode answers \
  --answer-mode verified_excerpts --split development \
  --output /tmp/course-qa-excerpt-evaluation.json
```

Generated output remains the CLI default until a separate adoption decision.
Evaluations report evidence-selection status agreement separately from answer
accuracy. Retain human-review requirements and do not claim a release gate passed
merely because copied text has a matching hash.
