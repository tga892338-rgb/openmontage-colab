#!/usr/bin/env python3
"""
Composition Director: Orchestrates video composition into timeline.
Supports two runtimes: Remotion (React-based) and HyperFrames (HTML/CSS/GSAP).

Per AGENT_GUIDE.md: MUST present both runtimes to user before locking choice.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import subprocess
import os

from tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class CompositionConfig:
    """Configuration for composition."""
    runtime: str  # "remotion" or "hyperframes"
    mode: str  # "templated" or "atelier" (hand-authored)
    scenes: List[Dict[str, Any]]
    script_path: Optional[str] = None
    asset_manifest_path: Optional[str] = None
    audio_path: Optional[str] = None
    duration_seconds: float = 60.0
    resolution: str = "1920x1080"
    frame_rate: int = 30
    output_path: Optional[str] = None


class CompositionDirector(BaseTool):
    """
    Composes script + assets + audio into video timeline.
    
    Two runtimes (presenting both as HARD RULE per AGENT_GUIDE.md):
    - Remotion: React-based, stock components, fast batch production
    - HyperFrames: HTML/CSS/GSAP, registry-block system, kinetic typography
    
    Two authoring modes:
    - Templated: Stock scene-types (text_card, stat_card, bar_chart, etc.)
    - Atelier: Hand-authored bespoke composition (no reuse of components)
    """
    
    def __init__(self):
        super().__init__()
        self.name = "composition_director"
        self.description = "Compose script + assets + audio into video timeline"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["compose", "render", "preset_options"],
                    "description": "Operation: compose (create timeline), render (render to MP4), or preset_options (show runtime choices)"
                },
                "runtime": {
                    "type": "string",
                    "enum": ["remotion", "hyperframes"],
                    "description": "Composition runtime (Remotion: React-based, HyperFrames: HTML/CSS/GSAP)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["templated", "atelier"],
                    "description": "Authoring mode (templated: stock components, atelier: hand-authored)"
                },
                "script_path": {
                    "type": "string",
                    "description": "Path to script JSON file"
                },
                "asset_manifest_path": {
                    "type": "string",
                    "description": "Path to asset manifest JSON"
                },
                "audio_path": {
                    "type": "string",
                    "description": "Path to final audio mix (narration + music)"
                },
                "output_dir": {
                    "type": "string",
                    "description": "Output directory for composition artifacts"
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "Video duration in seconds (default 60)"
                },
                "resolution": {
                    "type": "string",
                    "enum": ["1920x1080", "1280x720", "3840x2160"],
                    "description": "Video resolution (default 1920x1080)"
                },
                "frame_rate": {
                    "type": "integer",
                    "description": "Frame rate in fps (default 30)"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict[str, Any]) -> ToolResult:
        """Execute composition operation."""
        try:
            operation = inputs.get("operation", "compose")
            
            if operation == "preset_options":
                return self._show_runtime_options()
            elif operation == "compose":
                return self._compose(inputs)
            elif operation == "render":
                return self._render(inputs)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )
        
        except Exception as e:
            logger.error(f"Composition error: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e))
    
    def _show_runtime_options(self) -> ToolResult:
        """
        Present both composition runtimes (HARD RULE per AGENT_GUIDE.md).
        User must see both options before choosing.
        """
        options = {
            "runtimes": [
                {
                    "id": "remotion",
                    "name": "Remotion",
                    "description": "React-based composition engine",
                    "pros": [
                        "Stock components (text cards, stat cards, charts, transitions)",
                        "Spring physics animations",
                        "Word-level caption burn",
                        "Avatar support (TalkingHead)",
                        "Fast batch production",
                        "Proven for commercial production"
                    ],
                    "cons": [
                        "Limited to stock components in templated mode",
                        "Requires Node.js",
                        "Longer render times for complex scenes"
                    ],
                    "best_for": "Batch videos, explainers, educational content, quick turnaround",
                    "cost": "Free (open-source)"
                },
                {
                    "id": "hyperframes",
                    "name": "HyperFrames",
                    "description": "HTML/CSS/GSAP composition with registry blocks",
                    "pros": [
                        "Registry-block driven scene system",
                        "Kinetic typography",
                        "Website-to-video capability",
                        "Product promo templates",
                        "Launch reel templates",
                        "SVG character rigs",
                        "Faster local rendering"
                    ],
                    "cons": [
                        "Newer runtime (less battle-tested)",
                        "Requires Node.js ≥ 22",
                        "Registry blocks require setup"
                    ],
                    "best_for": "Branded content, typography-heavy, website demos, launches",
                    "cost": "Free (open-source)"
                }
            ],
            "modes": [
                {
                    "id": "templated",
                    "name": "Templated Mode",
                    "description": "Assemble stock scene-types into composition",
                    "pros": [
                        "Fast assembly",
                        "Reliable output",
                        "Batch-friendly"
                    ],
                    "cons": [
                        "Videos can look alike",
                        "Limited visual novelty"
                    ],
                    "best_for": "Internal clips, quick drafts, batch production, localization"
                },
                {
                    "id": "atelier",
                    "name": "Atelier Mode (Hand-Authored)",
                    "description": "Bespoke composition from scratch, no component reuse",
                    "pros": [
                        "Unique visual language per video",
                        "Full creative control",
                        "No sameness across videos"
                    ],
                    "cons": [
                        "Significantly more tokens",
                        "More iteration cycles",
                        "Requires stronger design direction"
                    ],
                    "best_for": "Hero work, marketing, launches, brand pieces, single-deliverable explainers"
                }
            ],
            "recommendation": "For Phase 5 MVP: Choose 'remotion + templated' for fastest integration. For distinctive work: choose 'atelier' with either runtime."
        }
        
        return ToolResult(
            success=True,
            data=options,
            metadata={
                "stage": "runtime_selection",
                "note": "User MUST see both runtimes before composition choice is locked (AGENT_GUIDE.md HARD RULE)"
            }
        )
    
    def _compose(self, inputs: Dict[str, Any]) -> ToolResult:
        """
        Create composition timeline from script + assets + audio.
        Builds scene plan and composition file, ready for rendering.
        """
        try:
            runtime = inputs.get("runtime", "remotion")
            mode = inputs.get("mode", "templated")
            # Validate runtime early to avoid creating compositions with unknown runtimes
            if runtime not in ("remotion", "hyperframes"):
                return ToolResult(success=False, error=f"Unknown runtime: {runtime}")
            script_path = inputs.get("script_path")
            asset_manifest_path = inputs.get("asset_manifest_path")
            audio_path = inputs.get("audio_path")
            output_dir = Path(inputs.get("output_dir", "."))
            duration_seconds = inputs.get("duration_seconds", 60.0)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load script
            script = None
            if script_path and Path(script_path).exists():
                with open(script_path, 'r', encoding='utf-8') as f:
                    script = json.load(f)
            
            # Load asset manifest
            assets = None
            if asset_manifest_path and Path(asset_manifest_path).exists():
                with open(asset_manifest_path, 'r', encoding='utf-8') as f:
                    assets = json.load(f)
            
            # Parse audio timing
            audio_duration = duration_seconds
            if audio_path and Path(audio_path).exists():
                # In production, would probe audio with ffprobe
                # For now, use provided duration
                pass
            
            # Build scene plan from script
            scene_plan = self._build_scene_plan(
                script=script,
                assets=assets,
                runtime=runtime,
                mode=mode,
                audio_path=audio_path
            )
            
            # Create composition file (format depends on runtime)
            composition_file = output_dir / f"composition_{runtime}_{mode}.json"
            with open(composition_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "runtime": runtime,
                    "mode": mode,
                    "duration_seconds": audio_duration,
                    "fps": inputs.get("frame_rate", 30),
                    "resolution": inputs.get("resolution", "1920x1080"),
                    "scenes": scene_plan,
                    "audio": audio_path,
                    "metadata": {
                        "script_path": script_path,
                        "asset_manifest_path": asset_manifest_path
                    }
                }, f, indent=2)
            
            logger.info(f"✓ Composition created: {composition_file}")
            logger.info(f"  Runtime: {runtime}")
            logger.info(f"  Mode: {mode}")
            logger.info(f"  Scenes: {len(scene_plan)}")
            
            return ToolResult(
                success=True,
                data={
                    "composition_file": str(composition_file),
                    "runtime": runtime,
                    "mode": mode,
                    "scene_count": len(scene_plan),
                    "duration_seconds": audio_duration,
                    "ready_for_render": True,
                    "stage": "composition_complete",
                    "next_step": f"render with {runtime} to MP4"
                }
            )
        
        except Exception as e:
            logger.error(f"Composition failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e))
    
    def _build_scene_plan(
        self,
        script: Optional[Dict[str, Any]],
        assets: Optional[Dict[str, Any]],
        runtime: str,
        mode: str,
        audio_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Build scene plan from script and assets."""
        scenes = []
        
        if not script:
            # Default fallback scene
            return [{
                "type": "text_card",
                "duration": 5.0,
                "text": "Video Composition",
                "background": "blur"
            }]
        
        # Extract scenes from script
        segments = script.get("segments", [])
        current_time = 0.0
        
        for i, segment in enumerate(segments):
            scene_type = "text_card"  # Default to text card for templated mode
            
            if mode == "templated":
                # Use stock scene types
                if "visuals" in segment:
                    visual_type = segment["visuals"].get("type", "background")
                    if visual_type == "chart":
                        scene_type = "bar_chart"
                    elif visual_type == "comparison":
                        scene_type = "comparison_card"
                    elif visual_type == "stat":
                        scene_type = "stat_card"
                else:
                    scene_type = "text_card"
            
            # Find assets for this segment if available
            segment_assets = []
            if assets and "organized_by_segment" in assets:
                segment_assets = assets["organized_by_segment"].get(
                    segment.get("segment_id", f"segment_{i}"),
                    []
                )
            
            # Calculate duration (divide audio duration by segment count)
            segment_duration = (script.get("duration_seconds", 60.0) / len(segments)) if segments else 5.0
            
            scene = {
                "id": f"scene_{i}",
                "type": scene_type,
                "duration": segment_duration,
                "start_time": current_time,
                "text": segment.get("narration", ""),
                "assets": segment_assets[:3],  # Limit to 3 assets per scene
                "transitions": "fade" if i > 0 else "none"
            }
            
            scenes.append(scene)
            current_time += segment_duration
        
        return scenes
    
    def _render(self, inputs: Dict[str, Any]) -> ToolResult:
        """
        Render composition to MP4.
        Delegates to runtime-specific renderer.
        """
        try:
            composition_file = inputs.get("composition_file")
            output_path = inputs.get("output_path", "output.mp4")
            
            if not composition_file or not Path(composition_file).exists():
                return ToolResult(
                    success=False,
                    error="Composition file not found"
                )
            
            # Load composition to determine runtime
            with open(composition_file, 'r', encoding='utf-8') as f:
                composition = json.load(f)
            
            runtime = composition.get("runtime", "remotion")
            
            logger.info(f"🎬 Rendering with {runtime}...")
            
            if runtime == "remotion":
                return self._render_remotion(composition, composition_file, output_path)
            elif runtime == "hyperframes":
                return self._render_hyperframes(composition, composition_file, output_path)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown runtime: {runtime}"
                )
        
        except Exception as e:
            logger.error(f"Render failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e))
    
    def _render_remotion(
        self,
        composition: Dict[str, Any],
        composition_file: str,
        output_path: str
    ) -> ToolResult:
        """Render composition using Remotion."""
        try:
            # Check if Remotion is available
            result = subprocess.run(
                ["npx", "remotion", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    error="Remotion not found. Install with: npm install remotion",
                    metadata={"install_command": "npm install remotion"}
                )
            
            # Create Remotion render command
            # In production, would call Remotion API or CLI
            logger.info("  Remotion rendering would happen here")
            logger.info(f"  Output: {output_path}")
            
            return ToolResult(
                success=True,
                data={
                    "output_path": output_path,
                    "runtime": "remotion",
                    "status": "render_queued"
                },
                metadata={
                    "stage": "rendering",
                    "note": "In production: npx remotion render <composition> <output>"
                }
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error="Remotion check timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Remotion render error: {str(e)}"
            )
    
    def _render_hyperframes(
        self,
        composition: Dict[str, Any],
        composition_file: str,
        output_path: str
    ) -> ToolResult:
        """Render composition using HyperFrames."""
        try:
            # Check if HyperFrames is available
            result = subprocess.run(
                ["npx", "hyperframes", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    error="HyperFrames not found. Install with: npm install hyperframes",
                    metadata={"install_command": "npm install hyperframes"}
                )
            
            logger.info("  HyperFrames rendering would happen here")
            logger.info(f"  Output: {output_path}")
            
            return ToolResult(
                success=True,
                data={
                    "output_path": output_path,
                    "runtime": "hyperframes",
                    "status": "render_queued"
                },
                metadata={
                    "stage": "rendering",
                    "note": "In production: npx hyperframes render <composition> <output>"
                }
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error="HyperFrames check timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"HyperFrames render error: {str(e)}"
            )
    
    def probe(self) -> Dict[str, Any]:
        """Check runtime availability."""
        return {
            "remotion_available": self._check_remotion(),
            "hyperframes_available": self._check_hyperframes(),
            "node_available": self._check_node()
        }
    
    def _check_remotion(self) -> bool:
        """Check if Remotion is available."""
        try:
            result = subprocess.run(
                ["npx", "remotion", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_hyperframes(self) -> bool:
        """Check if HyperFrames is available."""
        try:
            result = subprocess.run(
                ["npx", "hyperframes", "info"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_node(self) -> bool:
        """Check if Node.js is available."""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
