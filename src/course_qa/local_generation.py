"""Local Transformers adapter for explicitly provisioned Qwen2.5 weights."""

from pathlib import Path
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

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...

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
    ) -> torch.Tensor: ...


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

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

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
            formatted = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(formatted, add_special_tokens=False)
            if len(ids) + max_output_tokens > self.context_window_tokens:
                raise GenerationError(
                    "Prompt and output reservation exceed the generation context window."
                )
            input_ids = torch.tensor([ids], dtype=torch.long, device="cpu")
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
                )
            generated = [int(token.item()) for token in output[0, len(ids) :]]
            if not generated or generated[-1] != self.eos_token_id:
                raise GenerationError(
                    "Generation ended before EOS; output may be incomplete."
                )
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            return parse_generation_response(raw)
        except GenerationError:
            raise
        except (OSError, ValueError, RuntimeError) as error:
            raise GenerationError(
                "Local generation failed; no answer was accepted."
            ) from error
