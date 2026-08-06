"""
Assemble visuals, narration, music, and subtitles into final.mp4 using FFmpeg.
Inputs are expected under projects/first-creative/ (images/video per shot, narration.wav, music.wav, subtitles.srt).
"""
import subprocess
from pathlib import Path
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument('--project', default='projects/first-creative')
args = p.parse_args()

# sanitize project path (strip accidental surrounding quotes passed on some shells)
proj_arg = args.project.strip("'\"")
proj = Path(proj_arg)
proj.mkdir(parents=True, exist_ok=True)
out = proj / 'final.mp4'
shot_dir = proj / 'assets'
shot_dir.mkdir(parents=True, exist_ok=True)
shots = sorted(shot_dir.glob('*'))

narration = proj / 'narration.wav'
music = proj / 'music.wav'
subs = proj / 'subtitles.srt'

# Build ffmpeg inputs and concat script
concat_list = proj / 'concat.txt'
with concat_list.open('w', encoding='utf-8') as f:
    for i, shot in enumerate(shots, start=1):
        # assume each shot is a video clip; if image, create from image with duration
        if shot.suffix.lower() in ['.mp4', '.mov', '.webm']:
            f.write(f"file '{str(shot.resolve().as_posix())}'\n")
        else:
            # create a temp video from image
            tmp = proj / f'shot_{i:02d}.mp4'
            cmd = ['ffmpeg','-y','-loop','1','-i',str(shot),'-c:v','libx264','-t','5','-pix_fmt','yuv420p',str(tmp)]
            subprocess.check_call(cmd)
            f.write(f"file '{str(tmp.resolve().as_posix())}'\n")

# Concatenate
intermediate = proj / 'concat.mp4'
subprocess.check_call(['ffmpeg','-y','-f','concat','-safe','0','-i',str(concat_list),'-c','copy',str(intermediate)])

# Mix narration and music with balanced volumes
mixed = proj / 'mixed_audio.wav'
subprocess.check_call(['ffmpeg','-y','-i',str(narration),'-i',str(music),'-filter_complex','[0:a]volume=1.0[a0];[1:a]volume=0.4[a1];[a0][a1]amix=inputs=2:duration=shortest',str(mixed)])

# Final mux with subtitles burned in
subprocess.check_call(['ffmpeg','-y','-i',str(intermediate),'-i',str(mixed),'-c:v','libx264','-c:a','aac','-vf',f"subtitles={subs.as_posix()}",str(out)])
print('WROTE', out)
