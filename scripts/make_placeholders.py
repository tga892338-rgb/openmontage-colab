from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
out = Path('projects/first-creative-video/assets')
out.mkdir(parents=True, exist_ok=True)
colors = [(30,30,60),(40,60,30),(60,30,30),(20,40,80),(80,30,60)]
for i,col in enumerate(colors, start=1):
    img = Image.new('RGB',(1280,720),col)
    d=ImageDraw.Draw(img)
    text = f'Shot {i}'
    try:
        font = ImageFont.truetype('arial.ttf',80)
    except Exception:
        font = ImageFont.load_default()
    try:
        w,h = d.textsize(text,font=font)
    except AttributeError:
        try:
            bbox = d.textbbox((0,0), text, font=font)
            w = bbox[2]-bbox[0]
            h = bbox[3]-bbox[1]
        except Exception:
            w,h = font.getsize(text)
    d.text(((1280-w)/2,(720-h)/2),text,fill=(255,255,255),font=font)
    img.save(out/f'shot_{i:02d}.jpg')
print('WROTE images to', out)

# create silent wavs
import wave, struct
from pathlib import Path
narr = Path('projects/first-creative-video/narration.wav')
music = Path('projects/first-creative-video/music.wav')

def write_silence(path,duration,rate=22050,channels=1):
    nframes = int(duration*rate)
    with wave.open(str(path),'w') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        chunk = struct.pack('<h',0)*1024
        written=0
        while written < nframes:
            to_write = min(1024,nframes-written)
            wf.writeframes(chunk[:to_write*2])
            written += to_write
    print('WROTE',path)

write_silence(narr,45)
write_silence(music,50)
