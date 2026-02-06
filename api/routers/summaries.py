"""Résumé automatique par chapitre via LLM."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from api.dependencies import job_store
from api.errors import APIError, ErrorCode
from api.models import JobStatus

router = APIRouter(prefix="/api/v2", tags=["Summaries"])


class SummaryOptions(BaseModel):
    provider: str = "ollama"  # ollama, openai, gemini, anthropic
    model: str = ""  # Auto-select if empty
    max_length: int = Field(default=150, ge=50, le=500)
    style: str = "descriptive"  # descriptive, teaser, keywords
    language: str = "fr"


class ChapterSummary(BaseModel):
    chapter_index: int
    chapter_title: str = ""
    summary: str = ""
    keywords: List[str] = []


# In-memory summaries store (keyed by job_id)
_summaries_store: dict[str, list[dict]] = {}


@router.post("/jobs/{job_id}/summaries")
async def generate_summaries(
    job_id: str, options: SummaryOptions, background_tasks: BackgroundTasks
) -> dict:
    """Génère des résumés pour tous les chapitres d'un job."""
    job = await job_store.get(job_id)
    if not job:
        raise APIError(ErrorCode.NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)

    sum_job_id = await job_store.create("summaries")

    async def process():
        try:
            await job_store.update(sum_job_id, status=JobStatus.processing, progress=10, phase="summarizing")

            # Try to use LLM Enhancer for summaries
            summaries = []
            try:
                from src.llm_enhancer import LLMEnhancer, LLMConfig, LLMProvider

                provider_map = {
                    "ollama": LLMProvider.OLLAMA,
                    "openai": LLMProvider.OPENAI,
                    "gemini": LLMProvider.GEMINI,
                    "anthropic": LLMProvider.ANTHROPIC,
                }
                config = LLMConfig(
                    provider=provider_map.get(options.provider, LLMProvider.OLLAMA),
                    model=options.model or None,
                )
                enhancer = LLMEnhancer(config)

                # Get text from job result
                result = job.get("result", {})
                text = result.get("original_text", "")

                if text:
                    # Split into chapters (simple heuristic)
                    chapters = _split_chapters(text)
                    for i, chapter_text in enumerate(chapters):
                        if len(chapter_text.strip()) < 50:
                            continue
                        try:
                            prompt = _build_summary_prompt(chapter_text[:3000], options.style, options.language, options.max_length)
                            summary_text = enhancer._call_llm(prompt)
                            summaries.append({
                                "chapter_index": i,
                                "chapter_title": f"Chapitre {i + 1}",
                                "summary": summary_text.strip(),
                                "keywords": [],
                            })
                        except Exception:
                            summaries.append({
                                "chapter_index": i,
                                "chapter_title": f"Chapitre {i + 1}",
                                "summary": chapter_text[:200] + "...",
                                "keywords": [],
                            })

                        await job_store.update(
                            sum_job_id,
                            progress=10 + int(80 * (i + 1) / len(chapters)),
                            phase=f"chapter_{i + 1}",
                        )
            except ImportError:
                # Fallback: simple extractive summary
                pass

            if not summaries:
                summaries = [{
                    "chapter_index": 0,
                    "chapter_title": "Résumé",
                    "summary": "Résumé non disponible (LLM requis)",
                    "keywords": [],
                }]

            _summaries_store[job_id] = summaries

            await job_store.update(
                sum_job_id,
                status=JobStatus.completed,
                progress=100,
                phase="done",
                result={"summaries_count": len(summaries)},
            )
        except Exception as e:
            await job_store.update(sum_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": sum_job_id, "message": "Génération des résumés démarrée"}


@router.get("/jobs/{job_id}/summaries")
async def get_summaries(job_id: str) -> dict:
    """Récupère les résumés d'un job."""
    summaries = _summaries_store.get(job_id, [])
    return {
        "job_id": job_id,
        "summaries": [ChapterSummary(**s) for s in summaries],
        "total": len(summaries),
    }


def _split_chapters(text: str) -> list[str]:
    """Découpe le texte en chapitres (heuristique)."""
    import re
    chapters = re.split(r'\n#{1,3}\s+', text)
    if len(chapters) <= 1:
        # Try splitting by "Chapitre" markers
        chapters = re.split(r'\n(?=Chapitre\s+\d+)', text, flags=re.IGNORECASE)
    if len(chapters) <= 1:
        # Split by large paragraph breaks
        chunks = text.split('\n\n\n')
        if len(chunks) > 1:
            chapters = chunks
    return [c.strip() for c in chapters if c.strip()]


def _build_summary_prompt(text: str, style: str, language: str, max_length: int) -> str:
    """Construit le prompt LLM pour le résumé."""
    style_instructions = {
        "descriptive": "Résume ce chapitre de manière descriptive et factuelle.",
        "teaser": "Crée un résumé accrocheur qui donne envie de lire, comme une quatrième de couverture.",
        "keywords": "Identifie les thèmes et mots-clés principaux de ce chapitre.",
    }
    instruction = style_instructions.get(style, style_instructions["descriptive"])

    return f"""{instruction}

Texte du chapitre:
{text}

Consignes:
- Langue: {'français' if language == 'fr' else 'English'}
- Maximum {max_length} mots
- Pas de spoilers majeurs
- Style {'descriptif' if style == 'descriptive' else 'accrocheur' if style == 'teaser' else 'mots-clés'}

Résumé:"""
