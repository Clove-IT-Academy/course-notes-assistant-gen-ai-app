"""Local Transformers adapter for explicitly provisioned Qwen2.5 weights."""

import json
import re
from collections.abc import Callable
from importlib import import_module
from itertools import combinations
from pathlib import Path
from time import monotonic
from typing import Protocol, cast

import torch

from course_qa.generation import (
    GenerationError,
    GenerationResult,
    parse_generation_response,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class _LocalLoader(Protocol):
    def from_pretrained(self, path: str, **kwargs: object) -> object: ...


class _Tokenizer(Protocol):
    eos_token_id: int | None
    all_special_ids: list[int]

    def __len__(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...

    def batch_decode(
        self, sequences: list[list[int]], *, skip_special_tokens: bool
    ) -> list[str]: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


class _CausalModel(Protocol):
    def eval(self) -> object: ...

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        max_time: float,
        pad_token_id: int,
        eos_token_id: int,
        use_cache: bool,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]],
    ) -> torch.Tensor: ...


def _evidence_labels(user_prompt: str) -> list[str]:
    """Read only the application-prepared evidence JSON, not question citations."""
    _, marker, payload = user_prompt.partition(
        "Evidence (application-prepared JSON):\n"
    )
    if not marker:
        raise GenerationError(
            "Generation requires the application-prepared evidence block."
        )
    decoded: object = json.JSONDecoder().raw_decode(payload)[0]
    if not isinstance(decoded, dict):
        raise GenerationError("Evidence block must be an object.")
    data = cast(dict[str, object], decoded)
    entries = data.get("evidence")
    if not isinstance(entries, list):
        raise GenerationError("Evidence block must contain a passage list.")
    labels: list[str] = []
    for entry in cast(list[object], entries):
        if not isinstance(entry, dict):
            raise GenerationError("Evidence passage must be an object.")
        label = cast(dict[str, object], entry).get("source_label")
        if not isinstance(label, str) or re.fullmatch(r"S[1-9][0-9]*", label) is None:
            raise GenerationError("Evidence contains an invalid source label.")
        if label in labels:
            raise GenerationError("Evidence contains duplicate source labels.")
        labels.append(label)
    if len(labels) > 4:
        raise GenerationError(
            "This teaching adapter supports at most four evidence passages."
        )
    return labels


def _response_pattern(labels: list[str]) -> str:
    """Allow plain single-paragraph answers with matching inline/list citations.

    The model chooses answer versus abstention, the explanation and which source
    subset to cite. Grammar forces syntax only, never factual support. Quotes,
    backslashes, brackets and control characters are excluded from free text;
    the grammar itself emits the final citation suffix and JSON punctuation.
    """
    body = r'[^"\\\[\]\x00-\x1f]+'
    branches = [
        re.escape('{"status": "insufficient_evidence", "answer": "')
        + body
        + re.escape('", "cited_labels": []}')
    ]
    for size in range(1, len(labels) + 1):
        for selected in combinations(labels, size):
            suffix = " " + " ".join(f"[{label}]" for label in selected) + "."
            branches.append(
                re.escape('{"status": "answered", "answer": "')
                + body
                + re.escape(suffix + '", "cited_labels": ' + json.dumps(selected) + "}")
            )
    return "(" + "|".join(branches) + ")"


class _AllowedTokens(Protocol):
    allowed_tokens: list[int]


class _Enforcer(Protocol):
    def get_allowed_tokens(self, token_sequence: list[int]) -> _AllowedTokens: ...


class _FormatLibrary(Protocol):
    def RegexParser(self, pattern: str) -> object: ...

    def TokenEnforcer(self, data: object, parser: object) -> object: ...

    def TokenEnforcerTokenizerData(
        self,
        tokens: list[tuple[int, str, bool]],
        decoder: Callable[[list[int]], str],
        eos_token_id: int,
        use_bitmask: bool,
        vocab_size: int,
    ) -> object: ...


class LocalQwenGenerator:
    """Run one local model on CPU, with no implicit download or fallback.

    max_time is a cooperative generation limit checked between decoding steps,
    not a hard process deadline. Model loading is outside that limit. CPU memory
    and latency must be measured on the student's machine before demo claims.
    """

    model_id: str = MODEL_ID

    def __init__(
        self,
        model_dir: Path,
        *,
        context_window_tokens: int = 4096,
        max_generation_seconds: float = 60.0,
    ) -> None:
        import math

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_dir = model_dir.resolve(strict=True)
        if not model_dir.is_dir():
            raise ValueError("Generation model path must be a directory.")
        if context_window_tokens <= 0:
            raise ValueError("Context window must be positive.")
        if not math.isfinite(max_generation_seconds) or max_generation_seconds <= 0:
            raise ValueError("Generation time limit must be finite and positive.")
        config = cast(_LocalLoader, AutoConfig).from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        model_type: object = getattr(config, "model_type", None)
        model_limit: object = getattr(config, "max_position_embeddings", None)
        if model_type != "qwen2":
            raise ValueError(
                "This adapter requires a Qwen2-family causal language model."
            )
        if not isinstance(model_limit, int) or isinstance(model_limit, bool):
            raise TypeError("The model must declare its context limit.")
        if context_window_tokens > model_limit:
            raise ValueError("Configured context window exceeds the model limit.")
        self.context_window_tokens = context_window_tokens
        self.max_generation_seconds = max_generation_seconds
        self.tokenizer = cast(
            _Tokenizer,
            cast(_LocalLoader, AutoTokenizer).from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
            ),
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Generation tokenizer must declare an EOS token.")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.model = cast(
            _CausalModel,
            cast(_LocalLoader, AutoModelForCausalLM).from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
            ),
        )
        self.model.eval()
        self._tokenizer_data: object | None = None

    def _format_filter(
        self, labels: list[str]
    ) -> Callable[[int, torch.Tensor], list[int]]:
        formats = cast(_FormatLibrary, import_module("lmformatenforcer"))
        # Use LM Format Enforcer's core API: its bundled Transformers helper
        # imports a tokenizer location removed by Transformers 5.
        if self._tokenizer_data is None:
            tokenizer = self.tokenizer
            zero = tokenizer.encode("0", add_special_tokens=False)[-1]
            special_ids = set(tokenizer.all_special_ids)
            tokens: list[tuple[int, str, bool]] = []
            regular_ids = [i for i in range(len(tokenizer)) if i not in special_ids]
            # Batch decoding avoids hundreds of thousands of Python/Rust calls.
            after_zero_texts = tokenizer.batch_decode(
                [[zero, i] for i in regular_ids], skip_special_tokens=False
            )
            alone_texts = tokenizer.batch_decode(
                [[i] for i in regular_ids], skip_special_tokens=False
            )
            for token_id, prefixed, alone in zip(
                regular_ids, after_zero_texts, alone_texts, strict=True
            ):
                after_zero = prefixed[1:]
                tokens.append((token_id, after_zero, len(after_zero) > len(alone)))
            self._tokenizer_data = formats.TokenEnforcerTokenizerData(
                tokens,
                lambda ids: tokenizer.decode(ids, skip_special_tokens=False).rstrip(
                    "�"
                ),
                self.eos_token_id,
                False,
                len(tokenizer),
            )
        enforcer = cast(
            _Enforcer,
            formats.TokenEnforcer(
                self._tokenizer_data, formats.RegexParser(_response_pattern(labels))
            ),
        )

        def allowed_tokens(batch_id: int, sent: torch.Tensor) -> list[int]:
            if batch_id != 0:
                raise GenerationError("Only single-question generation is supported.")
            ids = [int(token.item()) for token in sent]
            return enforcer.get_allowed_tokens(ids).allowed_tokens

        return allowed_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _format_request(self, system_prompt: str, user_prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_request_tokens(self, system_prompt: str, user_prompt: str) -> int:
        """Count exactly the chat template used by generate, including role tokens."""
        return self.count_tokens(self._format_request(system_prompt, user_prompt))

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> GenerationResult:
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("System and user prompts must be nonempty.")
        if max_output_tokens <= 0:
            raise ValueError("Output token limit must be positive.")
        try:
            formatted = self._format_request(system_prompt, user_prompt)
            ids = self.tokenizer.encode(formatted, add_special_tokens=False)
            if len(ids) + max_output_tokens > self.context_window_tokens:
                raise GenerationError(
                    "Prompt and output reservation exceed the generation context window."
                )
            token_filter = self._format_filter(_evidence_labels(user_prompt))
            input_ids = torch.tensor([ids], dtype=torch.long, device="cpu")
            started = monotonic()
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_output_tokens,
                    do_sample=False,
                    max_time=self.max_generation_seconds,
                    pad_token_id=self.eos_token_id,
                    eos_token_id=self.eos_token_id,
                    use_cache=True,
                    prefix_allowed_tokens_fn=token_filter,
                )
            generated = [int(token.item()) for token in output[0, len(ids) :]]
            elapsed = monotonic() - started
            if not generated or generated[-1] != self.eos_token_id:
                limits: list[str] = []
                if len(generated) >= max_output_tokens:
                    limits.append("output token limit reached")
                if elapsed >= self.max_generation_seconds:
                    limits.append("cooperative time limit reached")
                reason = "; ".join(limits) or "model stopped without EOS"
                raise GenerationError(
                    f"Generation incomplete: {reason}; produced {len(generated)}/"
                    f"{max_output_tokens} output tokens in {elapsed:.1f}s "
                    f"for {len(ids)} input tokens. No answer was accepted."
                )
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            return parse_generation_response(raw)
        except GenerationError:
            raise
        except (OSError, ValueError, RuntimeError) as error:
            raise GenerationError(
                "Local generation failed; no answer was accepted."
            ) from error
