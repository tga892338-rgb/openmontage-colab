"""
Research automation engine for zero-cost video production.

Orchestrates parallel research across:
- Gemini (topic research, factual validation)
- TMDB (movie data, cast, reviews)
- TVMaze (TV show data)
- Internet Archive (public domain sources)
- Pixabay (stock images)
- Pexels (stock footage metadata)

All free tier APIs. Caches results to avoid re-fetching.
"""

import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import hashlib
import logging
from dataclasses import dataclass, asdict

import aiohttp
# Prefer the new google-genai SDK (module: google.genai). Fall back to
# the older google.generativeai if present. If neither is available, set
# `genai` to None and handle gracefully later.
try:
    import google.genai as genai
except Exception:
    try:
        import google.generativeai as genai
    except Exception:
        genai = None

from tools.production_config import config, credentials

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Structured research output."""
    topic: str
    research_type: str  # "general", "movie", "tv_show", "public_domain"
    data: Dict[str, Any]
    sources: List[str]
    timestamp: str
    cost_usd: float = 0.0


class ResearchCache:
    """Simple file-based cache for research results."""
    
    def __init__(self, cache_dir: Path = config.CACHE_DIR / "research"):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash_query(self, query: str, research_type: str) -> str:
        """Generate cache key from query."""
        key = f"{research_type}:{query}".lower()
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, query: str, research_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached research if available."""
        cache_file = self.cache_dir / f"{self._hash_query(query, research_type)}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None
    
    def save(self, query: str, research_type: str, data: Dict[str, Any]) -> None:
        """Save research to cache."""
        cache_file = self.cache_dir / f"{self._hash_query(query, research_type)}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)


class ResearchEngine:
    """Orchestrates all research tasks."""
    
    def __init__(self):
        self.cache = ResearchCache()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Configure Gemini (guard if genai SDK is unavailable)
        if credentials.GEMINI_API_KEY and genai is not None:
            try:
                genai.configure(api_key=credentials.GEMINI_API_KEY)
                # Some SDK variants expose GenerativeModel differently; guard it.
                try:
                    self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                except Exception:
                    # Older/newer SDKs may expose different factories; assign raw client
                    self.gemini_model = getattr(genai, 'Client', None)
            except Exception as e:
                logger.warning("Gemini initialization failed: %s", e)
                self.gemini_model = None
        else:
            if credentials.GEMINI_API_KEY and genai is None:
                logger.warning("GEMINI API key present but google-genai SDK not installed")
            else:
                logger.info("GEMINI_API_KEY not configured")
            self.gemini_model = None
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Create async HTTP session if needed."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    # ========================================================================
    # GENERAL TOPIC RESEARCH (Gemini)
    # ========================================================================
    
    async def research_general_topic(self, topic: str, focus_areas: Optional[List[str]] = None) -> ResearchResult:
        """
        Research a general topic using Gemini.
        
        Args:
            topic: What to research (e.g., "AI in healthcare")
            focus_areas: Optional specific aspects to focus on
        
        Returns:
            ResearchResult with factual information and sources
        """
        
        # Check cache first
        cached = self.cache.get(topic, "general")
        if cached:
            logger.info(f"✓ Using cached research for: {topic}")
            return ResearchResult(
                topic=topic,
                research_type="general",
                data=cached,
                sources=cached.get("sources", []),
                timestamp=datetime.now().isoformat(),
                cost_usd=0.0  # Cached = no API call
            )
        
        if not self.gemini_model:
            raise RuntimeError("Gemini API not configured")
        
        logger.info(f"🔍 Researching: {topic}")
        
        # Build prompt
        focus_text = ""
        if focus_areas:
            focus_text = f"\n\nFocus on these specific areas:\n" + "\n".join(f"- {a}" for a in focus_areas)
        
        prompt = f"""
Research and provide comprehensive information about: {topic}

Structure your response as JSON with:
- overview: 2-3 sentence summary
- key_facts: List of 5-7 important facts
- historical_context: Brief historical background if relevant
- current_state: Current status/trends (2024)
- applications_or_significance: Why this matters
- notable_people_or_entities: Key figures/organizations
- sources_and_citations: List of reputable sources to cite

Be factual, cite sources, avoid hallucinations. Focus on accuracy over entertainment.
{focus_text}

Return ONLY valid JSON, no markdown code blocks.
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Handle markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            research_data = json.loads(response_text)
            
            # Cache the result
            self.cache.save(topic, "general", research_data)
            
            return ResearchResult(
                topic=topic,
                research_type="general",
                data=research_data,
                sources=research_data.get("sources_and_citations", []),
                timestamp=datetime.now().isoformat(),
                cost_usd=0.001  # Gemini free tier (estimate)
            )
        
        except Exception as e:
            logger.error(f"Error researching {topic}: {e}")
            raise
    
    # ========================================================================
    # TMDB RESEARCH (Movies & TV Shows)
    # ========================================================================
    
    async def research_tmdb_movie(self, movie_title: str) -> ResearchResult:
        """
        Research a movie using TMDB API.
        
        Returns cast, plot, release date, ratings, reviews, etc.
        """
        
        cached = self.cache.get(movie_title, "tmdb_movie")
        if cached:
            logger.info(f"✓ Using cached TMDB data for: {movie_title}")
            return ResearchResult(
                topic=movie_title,
                research_type="movie",
                data=cached,
                sources=["TMDB (The Movie Database)"],
                timestamp=datetime.now().isoformat(),
                cost_usd=0.0
            )
        
        if not credentials.TMDB_API_KEY:
            logger.warning("TMDB_API_KEY not configured")
            return ResearchResult(topic=movie_title, research_type="movie", data={}, sources=[], timestamp=datetime.now().isoformat())
        
        logger.info(f"🎬 Fetching TMDB data for: {movie_title}")
        
        session = await self._ensure_session()
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": credentials.TMDB_API_KEY,
            "query": movie_title
        }
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    search_data = await resp.json()
                    
                    if search_data.get("results"):
                        movie = search_data["results"][0]  # Get first result
                        movie_id = movie["id"]
                        
                        # Fetch full movie details + credits
                        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
                        
                        async with session.get(details_url, params={"api_key": credentials.TMDB_API_KEY}) as d_resp:
                            details = await d_resp.json() if d_resp.status == 200 else {}
                        
                        async with session.get(credits_url, params={"api_key": credentials.TMDB_API_KEY}) as c_resp:
                            credits = await c_resp.json() if c_resp.status == 200 else {}
                        
                        research_data = {
                            "title": details.get("title"),
                            "overview": details.get("overview"),
                            "release_date": details.get("release_date"),
                            "runtime_minutes": details.get("runtime"),
                            "rating": details.get("vote_average"),
                            "genres": [g["name"] for g in details.get("genres", [])],
                            "director": next((p["name"] for p in credits.get("crew", []) if p["job"] == "Director"), None),
                            "cast": [{"name": a["name"], "character": a["character"]} for a in credits.get("cast", [])[:5]],
                            "budget": details.get("budget"),
                            "revenue": details.get("revenue"),
                            "production_companies": [c["name"] for c in details.get("production_companies", [])],
                        }
                        
                        self.cache.save(movie_title, "tmdb_movie", research_data)
                        
                        return ResearchResult(
                            topic=movie_title,
                            research_type="movie",
                            data=research_data,
                            sources=["TMDB (The Movie Database)"],
                            timestamp=datetime.now().isoformat(),
                            cost_usd=0.0
                        )
        
        except Exception as e:
            logger.error(f"Error fetching TMDB data for {movie_title}: {e}")
        
        return ResearchResult(topic=movie_title, research_type="movie", data={}, sources=[], timestamp=datetime.now().isoformat())
    
    # ========================================================================
    # PUBLIC DOMAIN RESEARCH (Internet Archive)
    # ========================================================================
    
    async def research_public_domain(self, subject: str, media_type: str = "movies") -> ResearchResult:
        """
        Find public domain content on Internet Archive.
        
        media_type: "movies", "audio", "texts", "images"
        """
        
        logger.info(f"📚 Searching public domain {media_type}: {subject}")
        
        session = await self._ensure_session()
        url = f"https://archive.org/advancedsearch.php"
        params = {
            "q": f'(subject:"{subject}" OR title:"{subject}") AND mediatype:{media_type}',
            "output": "json",
            "rows": 50,
            "sort": "downloads desc"
        }
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    docs = data.get("response", {}).get("docs", [])
                    
                    results = []
                    for doc in docs[:10]:  # Top 10 results
                        results.append({
                            "title": doc.get("title"),
                            "identifier": doc.get("identifier"),
                            "description": doc.get("description"),
                            "creator": doc.get("creator"),
                            "date": doc.get("date"),
                            "mediatype": doc.get("mediatype"),
                            "access_url": f"https://archive.org/details/{doc.get('identifier')}"
                        })
                    
                    research_data = {
                        "subject": subject,
                        "results": results,
                        "total_found": data.get("response", {}).get("numFound", 0)
                    }
                    
                    return ResearchResult(
                        topic=subject,
                        research_type="public_domain",
                        data=research_data,
                        sources=["Internet Archive (archive.org)"],
                        timestamp=datetime.now().isoformat(),
                        cost_usd=0.0
                    )
        
        except Exception as e:
            logger.error(f"Error searching Internet Archive: {e}")
        
        return ResearchResult(topic=subject, research_type="public_domain", data={}, sources=[], timestamp=datetime.now().isoformat())
    
    # ========================================================================
    # BATCH RESEARCH
    # ========================================================================
    
    async def batch_research(self, topics: Dict[str, Dict[str, Any]]) -> List[ResearchResult]:
        """
        Research multiple topics in parallel.
        
        topics: {
            "general_topic": {"type": "general", "focus_areas": [...]},
            "movie_title": {"type": "tmdb_movie"},
            "public_domain_subject": {"type": "public_domain", "media_type": "movies"}
        }
        """
        
        logger.info(f"🔄 Starting batch research for {len(topics)} topics")
        
        tasks = []
        for topic, params in topics.items():
            research_type = params.get("type", "general")
            
            if research_type == "general":
                tasks.append(self.research_general_topic(
                    topic,
                    focus_areas=params.get("focus_areas")
                ))
            elif research_type == "tmdb_movie":
                tasks.append(self.research_tmdb_movie(topic))
            elif research_type == "public_domain":
                tasks.append(self.research_public_domain(
                    topic,
                    media_type=params.get("media_type", "movies")
                ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, ResearchResult)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        if errors:
            logger.warning(f"⚠️  {len(errors)} research tasks failed")
        
        logger.info(f"✓ Completed batch research: {len(valid_results)}/{len(topics)} successful")
        
        return valid_results
    
    # ========================================================================
    # EXPORT RESEARCH
    # ========================================================================
    
    def export_research(self, results: List[ResearchResult], output_file: Optional[Path] = None) -> Path:
        """
        Export research results as JSON for script generation.
        """
        
        if not output_file:
            output_file = config.CACHE_DIR / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "research_count": len(results),
            "results": [
                {
                    "topic": r.topic,
                    "research_type": r.research_type,
                    "data": r.data,
                    "sources": r.sources,
                    "timestamp": r.timestamp,
                    "cost_usd": r.cost_usd
                }
                for r in results
            ],
            "total_cost_usd": sum(r.cost_usd for r in results)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"✓ Research exported to: {output_file}")
        return output_file


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Example: Run research for a movie analysis video."""
    
    engine = ResearchEngine()
    
    try:
        # Example topics
        topics = {
            "Oppenheimer": {"type": "tmdb_movie"},
            "AI in filmmaking": {"type": "general", "focus_areas": ["CGI", "VFX", "automation", "cost reduction"]},
        }
        
        # Run batch research
        results = await engine.batch_research(topics)
        
        # Export for script generation
        research_file = engine.export_research(results)
        print(f"\n✓ Research complete: {research_file}")
        print(f"  Total cost: ${sum(r.cost_usd for r in results):.4f}")
        
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
