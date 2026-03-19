"""
ResearcherAgent — fetches and indexes arXiv papers for the current domain/task.

Steps:
1. Search arXiv with 2 different query angles
2. Select top 3-4 most relevant papers by abstract
3. Download PDFs and store locally
4. Extract key insights via LLM
5. Return updated context with research_papers + research_insights
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import httpx

from .core import BaseAgent

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts"))


class ResearcherAgent(BaseAgent):

    async def run(self, context: dict) -> dict:
        domain       = context.get("domain", "vision")
        category_id  = context.get("category_id", domain)
        dataset_type = context.get("dataset_type", "")
        generation   = context.get("generation", 0)
        depth        = context.get("depth", 0)

        await self.emit_progress("agent_start", "Researcher searching arXiv...", generation, depth)
        self.log_step("Starting arXiv search", {"domain": domain, "dataset_type": dataset_type})

        # Build two complementary search queries
        query1 = f"{domain} neural architecture deep learning {dataset_type} 2024"
        query2 = f"novel {domain} model architecture benchmark {dataset_type}"

        results1, results2 = await asyncio.gather(
            self.fetch_arxiv_papers(query1, max_results=5),
            self.fetch_arxiv_papers(query2, max_results=5),
        )

        # Deduplicate by arxiv_id
        seen   = set()
        papers = []
        for p in results1 + results2:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                papers.append(p)

        # Select most relevant 3-4 using LLM
        selected = await self._select_papers(papers, domain, dataset_type)

        # Download PDFs
        session_dir = ARTIFACTS_DIR / self.research_session_id
        papers_dir  = session_dir / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)

        downloaded = []
        for paper in selected:
            pdf_path = await self._download_pdf(paper, papers_dir)
            if pdf_path:
                paper["local_pdf"] = str(pdf_path)
                downloaded.append(paper)

        # Extract research insights
        insights = await self._extract_insights(downloaded, domain)

        await self.emit_progress(
            "agent_done",
            f"Researcher indexed {len(downloaded)} papers",
            generation, depth,
            {"papers": [p["title"] for p in downloaded]},
        )

        context["research_papers"]   = downloaded
        context["research_insights"] = insights
        return context

    async def _select_papers(self, papers: list[dict], domain: str, dataset_type: str) -> list[dict]:
        if len(papers) <= 4:
            return papers

        abstracts = "\n\n".join(
            f"[{i}] {p['title']}\n{p['abstract'][:500]}"
            for i, p in enumerate(papers)
        )
        prompt = f"""
You are selecting the most relevant papers for a {domain} / {dataset_type} neural architecture research task.

Papers:
{abstracts}

Select the 3-4 most relevant by index. Return ONLY a JSON array of integers, e.g. [0, 2, 4].
"""
        raw = await self.call_llm(prompt, cache_key=f"select_{domain}_{dataset_type}")
        try:
            import json
            start   = raw.find("["); end = raw.rfind("]") + 1
            indices = json.loads(raw[start:end])
            return [papers[i] for i in indices if 0 <= i < len(papers)]
        except Exception:
            return papers[:4]

    async def _download_pdf(self, paper: dict, papers_dir: Path) -> Path | None:
        pdf_url  = paper.get("pdf_url", f"https://arxiv.org/pdf/{paper['arxiv_id']}")
        filename = papers_dir / f"{paper['arxiv_id'].replace('/', '_')}.pdf"

        if filename.exists():
            return filename

        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(pdf_url, headers={"User-Agent": "obsidian-networks-research/1.0"})
                if resp.status_code == 200:
                    filename.write_bytes(resp.content)
                    return filename
        except Exception as e:
            self.log_step(f"PDF download failed for {paper['arxiv_id']}", {"error": str(e)})
        return None

    async def _extract_insights(self, papers: list[dict], domain: str) -> str:
        if not papers:
            return f"No papers retrieved for domain: {domain}"

        abstracts = "\n\n".join(
            f"**{p['title']}** ({p['arxiv_id']})\n{p['abstract'][:800]}"
            for p in papers
        )
        prompt = f"""
You are an expert {domain} ML researcher. Read these paper abstracts and extract:
1. Key architectural innovations
2. Novel training techniques
3. Benchmark results and what they imply
4. Mechanisms that could be applied to new architectures

Papers:
{abstracts}

Write a concise 3-5 paragraph research summary focused on actionable insights for architecture design.
"""
        return await self.call_llm(prompt, force_claude=True, max_tokens=1500)
