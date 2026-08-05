"""
Director for first complete creative video.
Generates director_plan.json and shot_plan.json under projects/first-creative-video/.
Usage: python tools/director_first_creative.py --idea "What would happen if Earth suddenly stopped rotating?"
"""
import json
import argparse
from pathlib import Path

DEFAULT_IDEA = "What would happen if Earth suddenly stopped rotating?"

def build_plan(idea: str):
    out = Path('projects/first-creative-video')
    out.mkdir(parents=True, exist_ok=True)

    director = {
        'idea': idea,
        'title': 'If the Earth Stopped',
        'estimated_total_duration_sec': 45,
        'shots_count': 5,
        'notes': 'Vertical-slice: mix of generated images and short clips, narration-driven pacing.'
    }

    # A five-shot plan totaling ~45s
    shots = [
        {
            'shot_id': 'shot_01',
            'purpose': 'Establish scope and mystery',
            'narration': 'A hundred years ago, humanity looked toward the stars and wondered whether we were alone.',
            'visual_purpose': 'Cosmic perspective, Earth silent',
            'visual_description': 'Earth from space, slow rotation implied then frozen; stars, faint aurora',
            'camera': 'extreme-wide, slow zoom in',
            'mood': 'mysterious, reverent',
            'duration_sec': 7,
            'transition': 'fade_to_black',
            'generation_prompt': 'A cinematic photorealistic view of Earth from space at night, slight aurora, high detail, cinematic lighting, 35mm wide, film grain, dramatic color grading',
            'continuity': 'match color palette for sky/aurora'
        },
        {
            'shot_id': 'shot_02',
            'purpose': 'Introduce consequence',
            'narration': 'Tonight, something answered.',
            'visual_purpose': 'City losing power and lights going out',
            'visual_description': 'Aerial cityscape at dusk; lights flicker and then many go out; long exposure streaks',
            'camera': 'aerial push-in to street level',
            'mood': 'tension, confusion',
            'duration_sec': 10,
            'transition': 'crossfade',
            'generation_prompt': 'A cinematic aerial cityscape at dusk with lights flickering out, photoreal, dramatic clouds, subtle motion blur, warm to cold color shift',
            'continuity': 'lining up horizon and time-of-day with previous shot'
        },
        {
            'shot_id': 'shot_03',
            'purpose': 'Human reaction',
            'narration': 'The signal came from a world no telescope had ever seen before.',
            'visual_purpose': 'Close-up faces, strangers looking up',
            'visual_description': 'Crowd on a street looking up, mixed emotions, hands shielding eyes, cinematic shallow depth of field',
            'camera': 'medium close, handheld slight jitter',
            'mood': 'concern, wonder',
            'duration_sec': 9,
            'transition': 'cut',
            'generation_prompt': 'Cinematic close-up of diverse people looking up in the street, warm skin tones, shallow DOF, filmic lighting',
            'continuity': 'use similar color temperature to street lights in shot 02'
        },
        {
            'shot_id': 'shot_04',
            'purpose': 'Reveal signal origin',
            'narration': 'And buried inside that transmission was a message meant for us.',
            'visual_purpose': 'Strange sky phenomenon linking to unknown world',
            'visual_description': 'A strange, soft glowing vertical streak in the night sky above horizon, subtle geometric structure inside',
            'camera': 'slow crane up to sky',
            'mood': 'mystery, awe',
            'duration_sec': 8,
            'transition': 'dissolve',
            'generation_prompt': 'Night sky above horizon with a soft glowing vertical streak and subtle geometric structure, high detail, cinematic colors, volumetric light',
            'continuity': 'align horizon and skyline from shot 02/03'
        },
        {
            'shot_id': 'shot_05',
            'purpose': 'Resolve / contemplative close',
            'narration': 'A message meant for us.',
            'visual_purpose': 'Quiet aftermath, contemplative close-up',
            'visual_description': 'Solo person standing on rooftop watching the sky, silhouette, slow pull back',
            'camera': 'slow pull back to reveal city',
            'mood': 'contemplative, unresolved',
            'duration_sec': 11,
            'transition': 'fade_out',
            'generation_prompt': 'Silhouette of a person on a rooftop watching a glowing sky, cinematic backlight, high contrast, emotional atmosphere',
            'continuity': 'maintain skyline silhouette across shots'
        }
    ]

    director_path = out / 'director_plan.json'
    shot_path = out / 'shot_plan.json'
    director['shots'] = [ { 'shot_id': s['shot_id'], 'purpose': s['purpose'], 'duration_sec': s['duration_sec']} for s in shots ]

    with director_path.open('w', encoding='utf-8') as f:
        json.dump(director, f, indent=2)
    with shot_path.open('w', encoding='utf-8') as f:
        json.dump(shots, f, indent=2)

    return director_path, shot_path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--idea', default=DEFAULT_IDEA)
    args = p.parse_args()
    a,b = build_plan(args.idea)
    print('Wrote', a, b)
