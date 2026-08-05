"""Run the smallest real creative slice: text -> validated scene_plan.json.

This command intentionally stops after planning. It does not download TTS,
image, video, or music models. The output uses the repository's canonical
scene_plan artifact contract.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def build_scene_plan(text: str, style_playbook: str = "cinematic_documentary") -> Dict:
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Creative prompt/text cannot be empty.")

    scenes = []
    cursor = 0.0
    for index, sentence in enumerate(sentences[:20], start=1):
        duration = max(3.0, min(10.0, round(len(sentence.split()) / 2.5, 1)))
        end = round(cursor + duration, 1)
        scenes.append({
            "id": f"scene-{index}",
            "type": "broll",
            "description": sentence,
            "start_seconds": cursor,
            "end_seconds": end,
            "framing": "cinematic composition",
            "movement": "subtle camera movement",
            "shot_language": {
                "shot_size": "medium_wide",
                "camera_movement": "dolly_in",
                "lens_mm": 50,
                "lighting_key": "natural",
                "depth_of_field": "medium",
                "color_temperature": "neutral",
            },
            "shot_intent": "Visually communicate the narration and maintain viewer attention.",
            "narrative_role": "deliver_payload" if index < len(sentences) else "resolution",
            "information_role": sentence,
            "hero_moment": index == max(1, (len(sentences) + 1) // 2),
            "texture_keywords": ["cinematic", "natural", "detailed"],
            "required_assets": [{"type": "visual", "description": sentence, "source": "generate"}],
        })
        cursor = end

    return {
        "version": "1.0",
        "style_playbook": style_playbook,
        "scenes": scenes,
        "metadata": {"planner": "first_creative_heuristic", "source": "text"},
    }


def validate_scene_plan(plan: Dict) -> None:
    if plan.get("version") != "1.0":
        raise ValueError("scene_plan.version must be '1.0'.")
    scenes = plan.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scene_plan.scenes must be a non-empty array.")
    required = {"id", "type", "description", "start_seconds", "end_seconds"}
    allowed_types = {"talking_head", "broll", "animation", "character_scene", "diagram", "text_card", "transition", "generated", "screen_recording"}
    for scene in scenes:
        missing = required - scene.keys()
        if missing:
            raise ValueError(f"Scene {scene.get('id', '?')} missing: {sorted(missing)}")
        if scene["type"] not in allowed_types:
            raise ValueError(f"Invalid scene type: {scene['type']}")
        if scene["start_seconds"] < 0 or scene["end_seconds"] < 0 or scene["end_seconds"] < scene["start_seconds"]:
            raise ValueError(f"Invalid scene timing: {scene['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Creative idea or narration text")
    parser.add_argument("--project", default="projects/first-creative")
    parser.add_argument("--style-playbook", default="cinematic_documentary")
    args = parser.parse_args()

    plan = build_scene_plan(args.text, args.style_playbook)
    validate_scene_plan(plan)
    output = ROOT / args.project / "artifacts" / "scene_plan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scene plan PASS: {output}")
    print(f"Scenes: {len(plan['scenes'])}")
    print(f"Duration: {plan['scenes'][-1]['end_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
