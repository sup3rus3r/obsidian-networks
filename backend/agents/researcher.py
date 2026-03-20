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
        domain           = context.get("domain", "vision")
        category_id      = context.get("category_id", domain)
        dataset_type     = context.get("dataset_type", "")
        task_description = context.get("task_description", "")
        generation       = context.get("generation", 0)
        depth            = context.get("depth", 0)
        failed_patterns  = context.get("failed_patterns", [])

        await self.emit_progress("agent_start", "Researcher searching arXiv...", generation, depth)
        self.log_step("Starting arXiv search", {"domain": domain, "generation": generation})

        # Ask the LLM to generate targeted search queries based on the actual goal
        # and what has failed in previous generations — same approach as the main platform.
        queries = await self._generate_search_queries(
            domain, task_description, generation, failed_patterns
        )

        search_tasks = [self.fetch_arxiv_papers(q, max_results=5) for q in queries]
        all_results  = await asyncio.gather(*search_tasks)

        results1, results2 = all_results[0], all_results[1] if len(all_results) > 1 else []

        # Deduplicate by arxiv_id
        seen   = set()
        papers = []
        for p in results1 + results2:
            if p["arxiv_id"] not in seen:
                seen.add(p["arxiv_id"])
                papers.append(p)

        # Select most relevant 3-4 using LLM
        selected = await self._select_papers(papers, domain, task_description or dataset_type)

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

        # Ingest full PDF text into the session vectorstore so downstream agents
        # can query specific content rather than relying on abstract summaries.
        session_dir = ARTIFACTS_DIR / self.research_session_id
        await self._ingest_papers_to_vectorstore(downloaded, session_dir)

        # Extract research insights (used as a lightweight summary in logs/UI)
        insights = await self._extract_insights(downloaded, domain, task_description)

        await self.emit_progress(
            "agent_done",
            f"Researcher indexed {len(downloaded)} papers",
            generation, depth,
            {"papers": [p["title"] for p in downloaded]},
        )

        context["research_papers"]   = downloaded
        context["research_insights"] = insights
        return context

    async def _generate_search_queries(
        self,
        domain: str,
        task_description: str,
        generation: int,
        failed_patterns: list[dict],
    ) -> list[str]:
        """Ask the LLM to generate 2 targeted arXiv search queries for this specific goal."""
        failed_summary = ""
        if failed_patterns:
            failed_mutations = []
            for f in failed_patterns[-5:]:
                failed_mutations.extend(f.get("mutations", []))
            if failed_mutations:
                failed_summary = f"\nPreviously tried (avoid papers that only cover): {', '.join(set(failed_mutations))}"

        prompt = (
            f"You are selecting arXiv search queries to find papers for a neural architecture research task.\n\n"
            f"Domain: {domain}\n"
            f"Research goal: {task_description}\n"
            f"Generation: {generation} (0 = first search, higher = need fresh angles){failed_summary}\n\n"
            f"Generate exactly 2 arXiv search queries that will find the most relevant papers "
            f"for this specific goal. Queries should be complementary — cover different angles. "
            f"For generation > 0, search for approaches different from what has been tried.\n\n"
            f"Return ONLY a JSON array of 2 strings. Example: [\"efficient CNN image classification 2024\", \"attention pooling vision architecture\"]"
        )
        try:
            raw    = await self.call_llm(prompt, force_claude=True, max_tokens=200)
            start  = raw.find("["); end = raw.rfind("]") + 1
            import json
            queries = json.loads(raw[start:end])
            if isinstance(queries, list) and len(queries) >= 2:
                return [str(q) for q in queries[:2]]
        except Exception as e:
            self.log_step("Query generation failed — using fallback", {"error": str(e)})

        # Fallback if LLM call fails
        return [
            f"{task_description[:80]} neural architecture 2024",
            f"novel {domain} deep learning architecture",
        ]

    async def _ingest_papers_to_vectorstore(self, papers: list[dict], session_dir: Path) -> None:
        """Extract full text from downloaded PDFs and ingest into session FAISS vectorstore."""
        try:
            import pypdf
            from vectorstore import ingest_text, get_lock
        except ImportError:
            self.log_step("vectorstore/pypdf not available — skipping full-text ingest", {})
            return

        async with get_lock(str(self.research_session_id)):
            for paper in papers:
                pdf_path = paper.get("local_pdf")
                if not pdf_path:
                    continue
                try:
                    reader = pypdf.PdfReader(pdf_path)
                    pages  = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                    full_text = "\n\n".join(pages)
                    if not full_text.strip():
                        continue
                    ingest_text(
                        session_dir,
                        full_text,
                        source_url   = paper.get("url", pdf_path),
                        source_title = paper.get("title", ""),
                    )
                    self.log_step(f"Ingested PDF: {paper.get('title', '')[:60]}", {"chars": len(full_text)})
                except Exception as e:
                    self.log_step(f"PDF ingest failed for {paper.get('arxiv_id', '?')}", {"error": str(e)})

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

    async def _extract_insights(self, papers: list[dict], domain: str, task_description: str = "") -> str:
        if not papers:
            return f"No papers retrieved for domain: {domain}"

        abstracts = "\n\n".join(
            f"**{p['title']}** ({p['arxiv_id']})\n{p['abstract'][:800]}"
            for p in papers
        )
        goal_line = f"\nResearch goal: {task_description}" if task_description else ""
        prompt = f"""
You are an expert {domain} ML researcher with a focus on discovering what has NOT yet been tried.{goal_line}

Read these paper abstracts. Your job is NOT to summarise what these papers did — it is to identify \
what their findings make plausible that nobody has built yet.

For each key idea in the papers, ask: what does this make possible that remains unexplored?

Extract:
1. The underlying mathematical principle behind each innovation (not just the technique itself)
2. Gaps and open questions the authors left unaddressed
3. Combinations across these papers that no single paper explored
4. What benchmark results imply about the design space that is still uncovered

Papers:
{abstracts}

Write a concise 3-5 paragraph summary focused on UNEXPLORED directions that follow logically from \
this work — not on what was already built, but on what this research makes newly worth trying.
"""
        return await self.call_llm(prompt, force_claude=True, max_tokens=1500)
