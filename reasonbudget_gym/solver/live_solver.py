"""Live solver: DeepSeek-R1-Distill-Qwen-7B via transformers (requires GPU)."""

from reasonbudget_gym.solver.base_solver import BaseSolver, SolverResult, grade_answer, extract_boxed_answer


class LiveSolver(BaseSolver):
    """Solver using local inference with DeepSeek-R1-Distill-Qwen-7B.
    Two-stage generation: <think> up to budget_tokens, then </think> + final answer.
    """

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self._model.eval()

    def solve(self, question: str, ground_truth: str, budget_tokens: int) -> SolverResult:
        """Generate think block up to budget_tokens, then final answer; grade vs ground_truth."""
        self._ensure_loaded()
        import torch

        # Build prompt: User question + Assistant <think>
        prompt = f"<|User|>{question}<|Assistant|><think>"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs.input_ids.shape[1]

        # Find </think> token id if present
        think_end = self._tokenizer.convert_tokens_to_ids("</think>")
        if think_end is None or think_end == self._tokenizer.unk_token_id:
            think_end = self._tokenizer.eos_token_id

        with torch.no_grad():
            think_out = self._model.generate(
                inputs.input_ids,
                max_new_tokens=budget_tokens,
                do_sample=False,
                eos_token_id=think_end,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        tokens_used = think_out.shape[1] - prompt_len

        # Force-close </think> and generate final answer
        if self._tokenizer.decode(think_out[0][-3:]) != "</think>":
            think_out = self._tokenizer(
                self._tokenizer.decode(think_out[0]) + "</think>",
                return_tensors="pt",
            ).input_ids.to(self._model.device)
        answer_out = self._model.generate(
            think_out,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        full_text = self._tokenizer.decode(answer_out[0], skip_special_tokens=True)
        answer = extract_boxed_answer(full_text)
        was_correct = grade_answer(answer, ground_truth)
        return SolverResult(
            answer=answer,
            tokens_used=tokens_used,
            was_correct=was_correct,
            response_text=full_text,
        )
