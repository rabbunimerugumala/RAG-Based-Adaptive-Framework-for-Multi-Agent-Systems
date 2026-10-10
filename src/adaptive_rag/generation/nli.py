"""Optional local Natural Language Inference model used by the Verifier Agent.

Default: cross-encoder/nli-deberta-v3-xsmall (Apache-2.0, ~70M parameters, ~285 MB download). It runs
on CPU so the GPU stays available for the LLM. Label names are read from the model config rather
than hard-coded, so other NLI cross-encoders can be configured.
"""

from __future__ import annotations

import threading


class NLIModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.name = model_name
        labels = {i: name.lower() for i, name in self.model.config.id2label.items()}
        try:
            self._idx = {
                "entailment": next(i for i, n in labels.items() if n.startswith("entail")),
                "contradiction": next(i for i, n in labels.items() if n.startswith("contra")),
            }
        except StopIteration as exc:
            raise ValueError(f"{model_name} does not expose entailment/contradiction labels: {labels}") from exc
        self._lock = threading.Lock()

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[dict[str, float]]:
        """pairs = [(premise/evidence, hypothesis/claim)] -> [{'entailment': p, 'contradiction': p}]."""
        out: list[dict[str, float]] = []
        torch = self._torch
        with self._lock, torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                enc = self.tokenizer(
                    [p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                    max_length=384, padding=True, return_tensors="pt",
                ).to(self.device)
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
                for row in probs:
                    out.append({k: float(row[idx]) for k, idx in self._idx.items()})
        return out
