"""
Script generation engine for YouTube video production.

Converts research data into engaging, retention-optimized scripts using Gemini.
Implements:
- Hook optimization (first 3 seconds matter)
- Pacing for retention (hook → problem → solution pattern)
- Scene/B-roll markers for editing
- Call-to-action optimization
- Voiceover pacing (130-150 wpm for clarity)
"""

import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import logging

# Prefer new google-genai SDK if present, otherwise fall back to older package.
try:
    import google.genai as genai
except Exception:
    try:
        import google.generativeai as genai
    except Exception:
        genai = None
import requests
from tools.production_config import config, credentials

logger = logging.getLogger(__name__)


@dataclass
class ScriptSegment:
    """A single segment of the script."""
    type: str  # "hook", "intro", "main", "transition", "call_to_action"
    voiceover: str  # What the narrator says
    duration_seconds: int  # Estimated reading time
    suggested_broll: List[str]  # B-roll suggestions
    visual_notes: str  # Direction for on-screen visuals
    retention_cue: Optional[str] = None  # What keeps viewers watching


@dataclass
class VideoScript:
    """Complete video script."""
    title: str
    topic: str
    duration_seconds: int  # Total estimated duration
    word_count: int
    segments: List[ScriptSegment]
    research_data: Dict[str, Any]  # Original research for reference
    cta: str  # Call-to-action
    retention_score: float  # 0-100 (based on hooks, pattern, pacing)
    cost_usd: float = 0.001  # Gemini API call
    
    def total_voiceover_words(self) -> int:
        """Count total words in voiceover."""
        return sum(len(seg.voiceover.split()) for seg in self.segments)


class ScriptGenerator:
    """Generates scripts from research data."""
    
    def __init__(self):
        self.use_groq = False
        self.gemini_model = None
        
        # Initialize Gemini client if SDK is present
        if credentials.GEMINI_API_KEY and genai is not None:
            try:
                genai.configure(api_key=credentials.GEMINI_API_KEY)
                try:
                    self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                    logger.info("Gemini API ready")
                except Exception:
                    # Some SDKs expose a different client class — fall back to raw client
                    self.gemini_model = getattr(genai, 'Client', None)
            except Exception as e:
                logger.warning("Gemini not available: %s, will use Groq", e)
                self.use_groq = True
        else:
            if credentials.GEMINI_API_KEY and genai is None:
                logger.warning("GEMINI_API_KEY present but google-genai SDK not installed")

        if not self.gemini_model and not credentials.GROK_API_KEY and not self.use_groq:
            raise RuntimeError("Neither GEMINI_API_KEY nor GROK_API_KEY (Groq) configured")

        if self.use_groq or not self.gemini_model:
            logger.info("Using Groq API for script generation")
            self.use_groq = True
    
    def _word_count_to_seconds(self, word_count: int, wpm: int = 140) -> int:
        """Convert words to spoken duration (assume 140 wpm for clarity)."""
        return max(3, int(word_count / wpm * 60))
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API (fallback when Gemini is rate-limited)."""
        if not credentials.GROK_API_KEY:
            raise RuntimeError("GROK_API_KEY (Groq) not configured")
        
        # Groq API endpoint - using groq-3-70b (latest as of 2024)
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.GROK_API_KEY}"
        }
        
        payload = {
            "model": "groq-3-70b-tool-use-preview",  # Latest Groq model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            logger.error(f"Groq API error: {e.response.status_code} - {e.response.text}")
            raise
    
    async def generate_script(
        self,
        title: str,
        topic: str,
        research_data: Dict[str, Any],
        video_type: str = "educational",  # educational, analysis, explainer, documentary
        target_duration: int = 600,  # seconds (10 minutes)
        include_youtube_optimization: bool = True
    ) -> VideoScript:
        """
        Generate a retention-optimized script for YouTube.
        
        Tries Gemini first, falls back to Grok if rate-limited.
        """
        
        logger.info(f"Generating script: {title}")
        
        # Build research context
        research_context = json.dumps(research_data, indent=2)[:2000]  # Limit to 2000 chars
        
        # Build prompt for Gemini/Grok
        prompt = f"""
Generate a professional YouTube video script for:

Title: {title}
Topic: {topic}
Video Type: {video_type}
Target Duration: {target_duration} seconds (~{int(target_duration/60)} minutes)

Research Data:
{research_context}

SCRIPT REQUIREMENTS:

1. STRUCTURE:
   - Hook (first 3 seconds): CRITICAL - must make viewers want to keep watching
   - Intro (10-15 seconds): Establish credibility, preview what they'll learn
   - Main Content (bulk): Deliver value with clear segments
   - Transitions: Smooth topic shifts, maintain momentum
   - Call-to-Action (last 20 seconds): Subscribe, like, comment, link

2. RETENTION OPTIMIZATION:
   - Use pattern interrupts (change topic/scene every 15-30 seconds)
   - Ask questions (rhetorical or real) every 30-45 seconds
   - Include surprising facts or counterintuitive points
   - Use "curiosity gaps" - ask then answer within 10 seconds
   - End sentences on cliffhangers to carry viewers forward

3. VOICEOVER:
   - Assume 140 words per minute speaking pace
   - Use conversational tone (not robotic)
   - Emphasize key phrases with pauses
   - Include natural "ums" sparingly if needed

4. B-ROLL MARKERS:
   - Mark where visuals change [B-ROLL: description]
   - Suggest specific type: "movie clip", "stock footage", "screen recording", "text overlay", "animation"
   - Include timing for each visual

5. VISUAL NOTES:
   - Specify on-screen text, graphics, transitions
   - Note color tone, mood, pacing of visuals

6. YOUTUBE OPTIMIZATION (if enabled):
   - Hook should reference "biggest surprise" or controversy
   - Include searchable keywords naturally
   - Pattern: Setup problem -> Build tension -> Reveal solution
   - Specific CTA variations: "Subscribe for..." not generic "Subscribe"

Return response as valid JSON with this exact structure:
{{
  "hook": {{
    "text": "voiceover text for first 3 seconds",
    "tension_level": 1-10,
    "broll_suggestion": "what to show"
  }},
  "segments": [
    {{
      "type": "intro|main|transition|cta",
      "voiceover": "the actual dialogue",
      "estimated_words": number,
      "broll": ["list", "of", "suggestions"],
      "visual_notes": "what appears on screen",
      "retention_cue": "why viewers keep watching this part"
    }}
  ],
  "call_to_action": "specific CTA text",
  "total_estimated_words": number,
  "pacing_notes": "how to maintain tension/flow",
  "retention_opportunities": ["opportunity 1", "opportunity 2"]
}}

Return ONLY valid JSON, no code blocks.
"""
        
        try:
            # Try Gemini first
            if self.gemini_model:
                try:
                    logger.info("Attempting Gemini API...")
                    response = self.gemini_model.generate_content(prompt)
                    response_text = response.text.strip()
                except Exception as e:
                    if "quota" in str(e).lower() or "429" in str(e):
                        logger.warning(f"Gemini rate-limited, switching to Groq: {e}")
                        self.use_groq = True
                        response_text = self._call_groq(prompt)
                    else:
                        raise
            else:
                logger.info("Using Groq API...")
                response_text = self._call_groq(prompt)
            
            # Parse JSON (handle markdown if present)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            script_data = json.loads(response_text)
            
            # Build script segments
            segments = []
            
            # Add hook
            hook_data = script_data.get("hook", {})
            segments.append(ScriptSegment(
                type="hook",
                voiceover=hook_data.get("text", ""),
                duration_seconds=3,
                suggested_broll=["opening sequence"],
                visual_notes="Eye-catching opener",
                retention_cue="Immediate hook to stop scroll"
            ))
            
            # Add main segments
            total_words = 0
            for seg_data in script_data.get("segments", []):
                vo = seg_data.get("voiceover", "")
                word_count = len(vo.split())
                total_words += word_count
                
                segments.append(ScriptSegment(
                    type=seg_data.get("type", "main"),
                    voiceover=vo,
                    duration_seconds=self._word_count_to_seconds(word_count),
                    suggested_broll=seg_data.get("broll", []),
                    visual_notes=seg_data.get("visual_notes", ""),
                    retention_cue=seg_data.get("retention_cue")
                ))
            
            # Calculate retention score (based on hooks and pacing)
            hook_quality = hook_data.get("tension_level", 5) / 10.0 * 40  # Up to 40 points
            segment_count = len(segments)
            segment_diversity = min(100, segment_count * 10)  # Varied pacing
            retention_opportunities = len(script_data.get("retention_opportunities", [])) * 5
            retention_score = min(100, hook_quality + segment_diversity * 0.3 + retention_opportunities)
            
            # Create VideoScript
            script = VideoScript(
                title=title,
                topic=topic,
                duration_seconds=sum(seg.duration_seconds for seg in segments),
                word_count=total_words,
                segments=segments,
                research_data=research_data,
                cta=script_data.get("call_to_action", "Subscribe for more videos"),
                retention_score=retention_score,
                cost_usd=0.001
            )
            
            logger.info(f"Script generated: {total_words} words, {script.duration_seconds}s, retention {retention_score:.1f}/100")
            return script
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response text: {response_text[:500] if 'response_text' in locals() else 'N/A'}")
            raise
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            raise
    
    def _export_segment_to_dict(self, segment: ScriptSegment) -> Dict[str, Any]:
        """Convert segment to dict for JSON export."""
        return {
            "type": segment.type,
            "voiceover": segment.voiceover,
            "duration_seconds": segment.duration_seconds,
            "suggested_broll": segment.suggested_broll,
            "visual_notes": segment.visual_notes,
            "retention_cue": segment.retention_cue
        }
    
    def export_script(self, script: VideoScript, output_file: Optional[Path] = None) -> Path:
        """
        Export script as JSON for downstream tools.
        
        This becomes the input for storyboarding, asset collection, and composition.
        """
        
        if not output_file:
            safe_title = "".join(c for c in script.title if c.isalnum() or c in " -_").strip()
            safe_title = safe_title.replace(" ", "_")[:50]
            output_file = config.SCRIPTS_DIR / f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "title": script.title,
            "topic": script.topic,
            "created_at": datetime.now().isoformat(),
            "duration_seconds": script.duration_seconds,
            "duration_minutes": script.duration_seconds / 60,
            "word_count": script.word_count,
            "retention_score": script.retention_score,
            "segments": [self._export_segment_to_dict(seg) for seg in script.segments],
            "call_to_action": script.cta,
            "research_summary": {
                "sources": list(script.research_data.keys()),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Script exported: {output_file}")
        return output_file
    
    def export_voiceover_text(self, script: VideoScript, output_file: Optional[Path] = None) -> Path:
        """
        Export just the voiceover text for TTS processing.
        
        One segment per line, marked with timing and type.
        """
        
        if not output_file:
            safe_title = "".join(c for c in script.title if c.isalnum() or c in " -_").strip()
            safe_title = safe_title.replace(" ", "_")[:50]
            output_file = config.SCRIPTS_DIR / f"{safe_title}_voiceover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            cumulative_seconds = 0
            for i, segment in enumerate(script.segments):
                # Write segment metadata
                f.write(f"\n# SEGMENT {i+1}: {segment.type.upper()}\n")
                f.write(f"# Time: {cumulative_seconds}s - {cumulative_seconds + segment.duration_seconds}s\n")
                f.write(f"# B-roll: {', '.join(segment.suggested_broll)}\n")
                f.write(f"\n{segment.voiceover}\n")
                cumulative_seconds += segment.duration_seconds
        
        logger.info(f"✓ Voiceover text exported: {output_file}")
        return output_file


# ============================================================================
# BATCH SCRIPT GENERATION
# ============================================================================

async def generate_batch_scripts(
    topics: List[Dict[str, Any]],
    research_data_dir: Optional[Path] = None
) -> List[VideoScript]:
    """
    Generate scripts for multiple topics in parallel.
    
    topics: [
        {"title": "...", "topic": "...", "research_data": {...}},
        ...
    ]
    """
    
    logger.info(f"📝 Generating {len(topics)} scripts in parallel")
    
    generator = ScriptGenerator()
    tasks = []
    
    for topic in topics:
        tasks.append(generator.generate_script(
            title=topic.get("title"),
            topic=topic.get("topic"),
            research_data=topic.get("research_data", {}),
            video_type=topic.get("video_type", "educational"),
            target_duration=topic.get("target_duration", 600)
        ))
    
    scripts = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter valid scripts
    valid_scripts = [s for s in scripts if isinstance(s, VideoScript)]
    errors = [s for s in scripts if isinstance(s, Exception)]
    
    if errors:
        logger.warning(f"⚠️  {len(errors)} script generation tasks failed")
    
    logger.info(f"✓ Generated {len(valid_scripts)}/{len(topics)} scripts")
    return valid_scripts


# ============================================================================
# CLI EXAMPLE
# ============================================================================

async def main():
    """Example: Generate script from research."""
    
    generator = ScriptGenerator()
    
    # Example research data (from research_engine output)
    research_data = {
        "overview": "Oppenheimer explores the creation of the atomic bomb during WWII",
        "cast": ["Cillian Murphy", "Robert Downey Jr.", "Emily Blunt"],
        "director": "Christopher Nolan",
        "themes": ["Scientific ethics", "Political power", "Consequences of innovation"]
    }
    
    script = await generator.generate_script(
        title="Oppenheimer: How One Film Changed Cinema Forever",
        topic="Christopher Nolan's Oppenheimer - filmmaking analysis",
        research_data=research_data,
        video_type="analysis",
        target_duration=600
    )
    
    # Export
    script_file = generator.export_script(script)
    voiceover_file = generator.export_voiceover_text(script)
    
    print(f"\n✓ Script generated:")
    print(f"  Total duration: {script.duration_seconds}s ({script.duration_seconds/60:.1f}m)")
    print(f"  Word count: {script.word_count}")
    print(f"  Retention score: {script.retention_score:.1f}/100")
    print(f"  Files: {script_file}, {voiceover_file}")


if __name__ == "__main__":
    asyncio.run(main())
