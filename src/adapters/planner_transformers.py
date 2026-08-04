"""Transformers-based planner adapter using small HF instruction models.

Loads a seq2seq model (e.g., google/flan-t5-small) and generates a simple scene plan
from the user's idea/description. Availability is true only if transformers and the
model can be loaded successfully.
"""
from typing import Dict, Any
from src.abstractions import Planner
import logging

log = logging.getLogger(__name__)


class PlannerTransformers(Planner):
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.model_name = model_name
        self.config = config or {}
        self._model_id = self.config.get("model_id", "google/flan-t5-small")
        self._device = "cuda" if self._cuda_available() else "cpu"
        self._tokenizer = None
        self._model = None
        self.available = False
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_id)
            if self._device == "cuda":
                self._model = self._model.to("cuda")
            self.available = True
            log.info("Loaded planner model %s on %s", self._model_id, self._device)
        except Exception as e:
            log.info("PlannerTransformers not available: %s", e)
            self.available = False

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def run(self, idea: str, **kwargs) -> Dict[str, Any]:
        """Generate a shot/scene plan from the input idea using the loaded model.

        Returns a dict with 'title' and 'scenes' list where each scene: {id, duration, description}.
        """
        if not self.available:
            raise RuntimeError("PlannerTransformers: model not available")
        prompt = self._build_prompt(idea)
        try:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self._device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            gen = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = self._tokenizer.decode(gen[0], skip_special_tokens=True)
            # parse into sentences and create scenes heuristically from model output
            scenes = []
            import re
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            for i, s in enumerate(sents[:10]):
                scenes.append({"id": f"scene-{i+1}", "duration": max(3, min(12, len(s.split())//2)), "description": s})
            if not scenes:
                scenes = [{"id": "scene-1", "duration": 5, "description": text or idea}]
            return {"title": f"Plan: {idea[:40]}", "scenes": scenes}
        except Exception as e:
            log.error("PlannerTransformers generation failed: %s", e)
            raise

    def _build_prompt(self, idea: str) -> str:
        # Simple instruction-style prompt for flan-t5
        return f"Create a short scene-by-scene shot plan for the following idea:\n\n{idea}\n\nList the scenes with short descriptions and approximate duration in seconds."