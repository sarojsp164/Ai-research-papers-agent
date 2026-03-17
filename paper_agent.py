"""
AI Research Paper Agent  (Agentic / ReAct-style)

A fully autonomous agent that:
  1. Searches arXiv, Semantic Scholar, and Papers With Code for fresh papers
  2. Skips papers already sent in a previous run (seen_papers.json)
  3. Filters each paper through an LLM relevance gate
  4. Flags trending papers (recent + high citation velocity)
  5. Deduplicates across sources with metadata merging
  6. LLM-ranks the survivors by novelty and practical impact
  7. Summarizes the top picks
  8. Delivers a clean digest to Telegram

Scheduling: GitHub Actions cron -- every other day at 04:30 UTC (10:00 IST).
The workflow commits seen_papers.json back to the repo so nothing is repeated.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import json
import time
import traceback
import os
import sys
import hashlib

from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────────
#  CONFIG  (loaded from .env or environment)
# ─────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

PAPERS_PER_TOPIC = int(os.getenv("PAPERS_PER_TOPIC", 5))
TOP_K_PER_TOPIC  = int(os.getenv("TOP_K_PER_TOPIC",  3))
DAYS_BACK        = int(os.getenv("DAYS_BACK",         2))

# Path to the persistent seen-papers store (committed back to repo by CI)
SEEN_PAPERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "seen_papers.json")
# Maximum number of paper IDs to remember (rolling window to keep file small)
SEEN_PAPERS_MAX  = 2000

TOPICS = [
    {
        "name": "LLMs & Language Models",
        "categories": ["cs.CL", "cs.AI"],
        "keywords": ["large language model", "LLM", "transformer",
                     "instruction tuning", "RLHF", "RAG"],
    },
    {
        "name": "Computer Vision",
        "categories": ["cs.CV"],
        "keywords": ["image generation", "diffusion model",
                     "object detection", "vision transformer", "CLIP"],
    },
    {
        "name": "ML / DL Innovations",
        "categories": ["cs.LG", "stat.ML"],
        "keywords": ["deep learning", "neural network", "optimization",
                     "self-supervised", "foundation model"],
    },
    {
        "name": "New AI Algorithms",
        "categories": ["cs.AI", "cs.NE"],
        "keywords": ["reinforcement learning", "graph neural network",
                     "mixture of experts", "state space model", "mamba"],
    },
]


# =============================================================
#  GROQ LLM HELPER
# =============================================================

_groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL   = "llama-3.1-8b-instant"


def _call_groq(prompt: str, max_tokens: int = 300,
               temperature: float = 0.5) -> str:
    """Call Groq via the official SDK with retry logic."""
    for attempt in range(3):
        try:
            response = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "401" in err or "invalid_api_key" in err.lower() \
               or "authentication" in err.lower():
                raise RuntimeError(
                    f"Groq authentication failed -- check GROQ_API_KEY. ({e})"
                )
            if "429" in err or "rate_limit" in err.lower():
                wait = 15
                print(f"  [RATE-LIMITED] Groq -- waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt == 2:
                raise
            print(f"  [RETRY] Groq call failed: {e} -- retrying...")
            time.sleep(3)
    return ""


# =============================================================
#  SEEN-PAPERS PERSISTENCE
# =============================================================

def _paper_id(paper: dict) -> str:
    """Stable short ID: prefer arXiv id in URL, else hash of normalised title."""
    link = paper.get("link", "")
    # Extract arXiv ID from URL like https://arxiv.org/abs/2401.12345
    if "arxiv.org/abs/" in link:
        return link.split("arxiv.org/abs/")[-1].split("v")[0].strip()
    # Fallback: SHA-1 of lowercased title (first 16 hex chars)
    title = paper.get("title", "").lower().strip()
    return hashlib.sha1(title.encode()).hexdigest()[:16]


def load_seen_papers() -> dict:
    """Load the seen-papers store.  Returns {paper_id: iso_date_string}."""
    if os.path.exists(SEEN_PAPERS_FILE):
        try:
            with open(SEEN_PAPERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_seen_papers(seen: dict):
    """Persist the seen-papers store, keeping only the most recent entries."""
    if len(seen) > SEEN_PAPERS_MAX:
        sorted_items = sorted(seen.items(), key=lambda x: x[1])
        seen = dict(sorted_items[-SEEN_PAPERS_MAX:])
    with open(SEEN_PAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=2)


# =============================================================
#  TOOL: SEARCH SOURCES
# =============================================================

def tool_search_arxiv(categories: list, keywords: list,
                      max_results: int = 5) -> list:
    """Search arXiv for recent papers matching categories + keywords."""
    cat_query  = " OR ".join(f"cat:{c}" for c in categories)
    kw_query   = " OR ".join(f'all:"{k}"' for k in keywords[:4])
    full_query = f"({cat_query}) AND ({kw_query})"

    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get("https://export.arxiv.org/api/query",
                        params=params, timeout=15)
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title     = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract  = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        link      = entry.find("atom:id", ns).text.strip()
        authors   = [a.find("atom:name", ns).text
                     for a in entry.findall("atom:author", ns)]
        published = entry.find("atom:published", ns).text[:10]
        papers.append({
            "title":     title,
            "abstract":  abstract,
            "link":      link,
            "authors":   authors[:4],
            "published": published,
            "source":    "arXiv",
            "citations": None,
            "code_url":  None,
            "trending":  False,
        })
    return papers


def tool_search_semantic_scholar(keywords: list, max_results: int = 5) -> list:
    """Search Semantic Scholar for recent influential papers."""
    query     = " ".join(keywords[:4])
    year_from = (datetime.utcnow() - timedelta(days=DAYS_BACK * 30)).year
    params    = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,abstract,url,authors,year,citationCount,externalIds",
        "year":   f"{year_from}-",
        "sort":   "citationCount:desc",
    }
    time.sleep(2)   # polite delay to avoid 429s on consecutive topic calls
    for attempt in range(3):
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, timeout=20,
        )
        if resp.status_code == 429:
            wait = int(resp.headers.get("retry-after", 15))
            print(f"  [S2 RATE-LIMITED] Waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break

    data   = resp.json().get("data", [])
    papers = []
    for item in data:
        if not item.get("abstract"):
            continue
        arxiv_id     = (item.get("externalIds") or {}).get("ArXiv")
        link         = item.get("url") or ""
        if arxiv_id:
            link = f"https://arxiv.org/abs/{arxiv_id}"
        author_names = [a["name"] for a in (item.get("authors") or [])[:4]]
        cite_count   = item.get("citationCount") or 0
        year         = item.get("year") or 0
        current_year = datetime.utcnow().year
        # Heuristic: trending if high citations for a recent paper
        trending = (cite_count >= 50 and year >= current_year - 1)
        papers.append({
            "title":     (item.get("title") or "").strip(),
            "abstract":  (item.get("abstract") or "").strip(),
            "link":      link,
            "authors":   author_names,
            "published": str(year),
            "source":    "Semantic Scholar",
            "citations": cite_count,
            "code_url":  None,
            "trending":  trending,
        })
    return papers


def tool_search_papers_with_code(keywords: list, max_results: int = 5) -> list:
    """Search Papers With Code for papers that have code implementations."""
    query   = " ".join(keywords[:3])
    params  = {"q": query, "page": 1, "items_per_page": max_results}
    headers = {"User-Agent": "research-agent/1.0 (academic use)"}
    resp    = requests.get(
        "https://paperswithcode.com/api/v1/papers/",
        params=params, headers=headers, timeout=15,
    )
    resp.raise_for_status()
    data    = resp.json()
    results = data.get("results", []) if isinstance(data, dict) else data

    papers = []
    for item in results[:max_results]:
        title    = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        if not title or not abstract:
            continue
        arxiv_id = item.get("arxiv_id")
        url_abs  = item.get("url_abs") or ""
        link     = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url_abs

        code_url = None
        repo_url = item.get("repository") or item.get("url_pdf")
        if isinstance(repo_url, str) and repo_url.startswith("http"):
            code_url = repo_url

        authors = item.get("authors") or []
        if authors and isinstance(authors[0], dict):
            authors = [a.get("name", "") for a in authors]

        papers.append({
            "title":     title,
            "abstract":  abstract,
            "link":      link,
            "authors":   authors[:4],
            "published": item.get("published", ""),
            "source":    "Papers With Code",
            "citations": None,
            "code_url":  code_url,
            "trending":  bool(code_url),  # having code = higher visibility signal
        })
    return papers


# =============================================================
#  TOOL: LLM RELEVANCE FILTER
# =============================================================

def tool_is_relevant(paper: dict, topic: str) -> bool:
    """
    Ask the LLM whether this paper genuinely belongs to the topic.
    Returns True (relevant / include) or False (off-topic / discard).
    """
    prompt = f"""You are a strict AI/ML research curator.

Topic: "{topic}"
Paper title: {paper['title']}
Abstract (first 300 chars): {paper['abstract'][:300]}

Is this paper genuinely relevant to the topic above?
Answer with exactly one word: YES or NO."""

    answer = _call_groq(prompt, max_tokens=5, temperature=0.0)
    return answer.strip().upper().startswith("Y")


# =============================================================
#  TOOL: LLM RANK
# =============================================================

def tool_rank_papers(papers: list, topic: str, top_k: int = 3) -> list:
    """Ask the LLM to rank papers by novelty + impact.  Returns top-k indices."""
    if len(papers) <= top_k:
        return list(range(len(papers)))

    descriptions = []
    for i, p in enumerate(papers):
        meta = []
        if p.get("citations"):
            meta.append(f"citations={p['citations']}")
        if p.get("code_url"):
            meta.append("has code")
        if p.get("trending"):
            meta.append("TRENDING")
        meta_str = (", " + ", ".join(meta)) if meta else ""
        descriptions.append(
            f"[{i}] \"{p['title']}\" (source={p['source']}{meta_str})\n"
            f"    {p['abstract'][:200]}..."
        )

    prompt = f"""You are an AI research curator. Given these papers on "{topic}",
pick the {top_k} MOST important ones: novelty, practical impact, scientific
contribution. Prefer papers with code or high citations. Trending papers are
gaining rapid attention and should be weighted higher.

Papers:
{chr(10).join(descriptions)}

Return ONLY a JSON array of selected indices, e.g. [2, 0, 5].
No explanation -- just the JSON array."""

    raw = _call_groq(prompt, max_tokens=100, temperature=0.2).strip()
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start != -1 and end > start:
        indices = json.loads(raw[start:end])
        valid   = [i for i in indices if isinstance(i, int) and 0 <= i < len(papers)]
        return valid[:top_k]
    return list(range(top_k))


# =============================================================
#  TOOL: SUMMARIZE
# =============================================================

def tool_summarize(paper: dict) -> str:
    """Use Groq LLM to produce a tight 2-line summary of a paper."""
    prompt = f"""You are a research digest writer for AI/ML practitioners.
Keep it extremely concise. No emojis. No filler words.

Paper Title: {paper['title']}
Abstract: {paper['abstract']}

Write EXACTLY 2 lines (no extra text, no bullet symbols, no labels):
Line 1 -- What is unique/core about this paper (1 sentence)
Line 2 -- Why it matters to practitioners (1 sentence)"""

    return _call_groq(prompt, max_tokens=200, temperature=0.3)


# =============================================================
#  TOOL: TELEGRAM DELIVERY
# =============================================================

def tool_send_telegram(message: str) -> str:
    """Send a message to Telegram.  Splits into chunks if too long."""
    url    = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = [message[i:i + 4000] for i in range(0, len(message), 4000)]
    errors = []

    for chunk in chunks:
        payload = {
            "chat_id":                  TELEGRAM_CHAT_ID,
            "text":                     chunk,
            "parse_mode":               "Markdown",
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            # Retry without Markdown if parsing fails
            plain = chunk.replace("*", "").replace("_", "").replace("`", "")
            payload2 = {
                "chat_id":                  TELEGRAM_CHAT_ID,
                "text":                     plain,
                "disable_web_page_preview": True,
            }
            resp2 = requests.post(url, json=payload2, timeout=10)
            if resp2.status_code != 200:
                errors.append(resp2.text[:200])

    if errors:
        return f"Sent with {len(errors)} error(s): {errors[0]}"
    return f"Sent {len(chunks)} message(s) to Telegram successfully."


# =============================================================
#  DEDUPLICATION
# =============================================================

def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def deduplicate_papers(papers: list, threshold: float = 0.75) -> list:
    """Remove near-duplicate papers, keeping the richer record."""
    if not papers:
        return []
    unique = [papers[0]]
    for paper in papers[1:]:
        is_dup = False
        for existing in unique:
            if _title_similarity(paper["title"], existing["title"]) > threshold:
                # Merge metadata into the existing record
                if paper.get("citations") and not existing.get("citations"):
                    existing["citations"] = paper["citations"]
                if paper.get("code_url") and not existing.get("code_url"):
                    existing["code_url"] = paper["code_url"]
                if paper.get("trending"):
                    existing["trending"] = True
                if existing["source"] != paper["source"]:
                    existing["source"] += f" + {paper['source']}"
                is_dup = True
                break
        if not is_dup:
            unique.append(paper)
    return unique


# =============================================================
#  THE AGENT  --  ReAct-style reasoning loop
# =============================================================

class ResearchAgent:
    """
    Autonomous research agent:
      PLAN -> SEARCH -> FILTER (seen + relevance) -> DEDUP -> RANK
           -> SUMMARIZE -> BUILD DIGEST -> DELIVER -> PERSIST SEEN
    """

    def __init__(self):
        self.ranked_papers: dict = {}   # topic -> [(paper, summary), ...]
        self.log:           list = []
        self.seen                = load_seen_papers()
        self.newly_seen:    dict = {}   # ids encountered this run

    def _log(self, msg: str):
        ts   = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log.append(line)
        print(line, flush=True)

    # ------------------------------------------------------------------
    # PHASE 1 -- Search
    # ------------------------------------------------------------------

    def _search_topic(self, topic: dict) -> list:
        name       = topic["name"]
        categories = topic["categories"]
        keywords   = topic["keywords"]
        collected  = []

        self._log(f"THINK: Searching arXiv for '{name}'...")
        try:
            papers = tool_search_arxiv(categories, keywords, PAPERS_PER_TOPIC)
            self._log(f"  ACT: arXiv -> {len(papers)} paper(s)")
            collected.extend(papers)
        except Exception as e:
            self._log(f"  OBSERVE: arXiv failed ({e}) -- continuing")

        self._log(f"THINK: Searching Semantic Scholar for '{name}'...")
        try:
            papers = tool_search_semantic_scholar(keywords, PAPERS_PER_TOPIC)
            self._log(f"  ACT: Semantic Scholar -> {len(papers)} paper(s)")
            collected.extend(papers)
        except Exception as e:
            self._log(f"  OBSERVE: Semantic Scholar failed ({e}) -- continuing")

        self._log(f"THINK: Searching Papers With Code for '{name}'...")
        try:
            papers = tool_search_papers_with_code(keywords, PAPERS_PER_TOPIC)
            self._log(f"  ACT: Papers With Code -> {len(papers)} paper(s)")
            collected.extend(papers)
        except Exception as e:
            self._log(f"  OBSERVE: Papers With Code failed ({e}) -- continuing")

        if not collected:
            self._log(f"REFLECT: No results for '{name}' -- "
                      "broadening and retrying arXiv...")
            try:
                fallback = tool_search_arxiv(categories, keywords[:2],
                                             PAPERS_PER_TOPIC)
                self._log(f"  ACT: Broadened arXiv -> {len(fallback)} paper(s)")
                collected.extend(fallback)
            except Exception as e:
                self._log(f"  OBSERVE: Broadened search also failed ({e})")

        return collected

    # ------------------------------------------------------------------
    # PHASE 2 -- Filter seen papers
    # ------------------------------------------------------------------

    def _filter_seen(self, papers: list, topic_name: str) -> list:
        fresh = []
        for p in papers:
            pid = _paper_id(p)
            if pid in self.seen:
                self._log(f"  SKIP (already sent): {p['title'][:60]}")
            else:
                fresh.append(p)
                self.newly_seen[pid] = datetime.utcnow().date().isoformat()
        self._log(f"  OBSERVE: {len(fresh)}/{len(papers)} papers are new "
                  f"for '{topic_name}'")
        return fresh

    # ------------------------------------------------------------------
    # PHASE 3 -- LLM relevance gate
    # ------------------------------------------------------------------

    def _filter_relevant(self, papers: list, topic_name: str) -> list:
        relevant = []
        for p in papers:
            self._log(f"THINK: Relevance check -- "
                      f"'{p['title'][:55]}...'")
            try:
                if tool_is_relevant(p, topic_name):
                    self._log("  ACT: RELEVANT -- keeping")
                    relevant.append(p)
                else:
                    self._log("  ACT: OFF-TOPIC -- discarding")
            except Exception as e:
                self._log(f"  OBSERVE: Relevance check failed ({e}) "
                          "-- keeping paper")
                relevant.append(p)
        return relevant

    # ------------------------------------------------------------------
    # PHASE 4 -- Dedup & Rank
    # ------------------------------------------------------------------

    def _deduplicate_and_rank(self, topic_name: str, papers: list) -> list:
        self._log(f"THINK: Deduplicating {len(papers)} papers "
                  f"for '{topic_name}'...")
        unique = deduplicate_papers(papers)
        self._log(f"  OBSERVE: {len(unique)} unique papers after dedup")

        if len(unique) <= TOP_K_PER_TOPIC:
            return unique

        self._log(f"THINK: LLM ranking {len(unique)} papers "
                  f"-- picking top {TOP_K_PER_TOPIC}...")
        try:
            top_indices = tool_rank_papers(unique, topic_name, TOP_K_PER_TOPIC)
            ranked      = [unique[i] for i in top_indices]
            self._log(f"  ACT: LLM selected indices {top_indices}")
            return ranked
        except Exception as e:
            self._log(f"  OBSERVE: Ranking failed ({e}) "
                      f"-- using first {TOP_K_PER_TOPIC}")
            return unique[:TOP_K_PER_TOPIC]

    # ------------------------------------------------------------------
    # PHASE 5 -- Summarize
    # ------------------------------------------------------------------

    def _summarize_papers(self, topic_name: str, papers: list) -> list:
        results = []
        for paper in papers:
            self._log(f"THINK: Summarizing '{paper['title'][:60]}...'")
            try:
                summary = tool_summarize(paper)
                self._log("  ACT: Summary generated")
                results.append((paper, summary))
            except Exception as e:
                self._log(f"  OBSERVE: Summarization failed ({e}) -- skipping")
                results.append((paper, "- Summary unavailable"))
        return results

    # ------------------------------------------------------------------
    # PHASE 6 -- Build digest
    # ------------------------------------------------------------------

    def _build_digest(self) -> str:
        """
        Clean, uncluttered digest format:
          1. Paper Title [TRENDING]
             link
             summary (2 lines)
        """
        today = datetime.utcnow().strftime("%B %d, %Y")
        lines = [f"*AI Research Digest -- {today}*", ""]

        any_papers = False
        for topic_name, papers in self.ranked_papers.items():
            if not papers:
                continue
            any_papers = True
            lines.append(f"--- *{topic_name}* ---")
            lines.append("")

            for i, (paper, summary) in enumerate(papers, 1):
                trending_tag = "  [TRENDING]" if paper.get("trending") else ""
                lines.append(f"*{i}. {paper['title']}*{trending_tag}")
                lines.append(paper["link"])
                if paper.get("code_url"):
                    lines.append(f"Code: {paper['code_url']}")
                lines.append(summary)
                lines.append("")  # blank line between papers

        if not any_papers:
            lines.append("No new papers this cycle -- "
                         "all recent papers were already sent.")
            lines.append("")

        lines.append("_arXiv + Semantic Scholar + Papers With Code | Groq LLM_")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PHASE 7 -- Deliver
    # ------------------------------------------------------------------

    def _deliver(self, digest: str):
        self._log("THINK: Sending digest to Telegram...")
        result = tool_send_telegram(digest)
        self._log(f"  OBSERVE: {result}")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        self._log("PLAN: Starting AI Research Paper Agent")
        self._log(f"PLAN: {len(TOPICS)} topics | {DAYS_BACK}-day window | "
                  f"top {TOP_K_PER_TOPIC} papers per topic")
        self._log(f"PLAN: Seen-papers store has {len(self.seen)} entries\n")

        for topic in TOPICS:
            name = topic["name"]
            self._log(f"=== Topic: {name} ===")

            # Search all sources
            raw_papers = self._search_topic(topic)
            if not raw_papers:
                self._log(f"REFLECT: No papers found for '{name}' "
                          "-- skipping.\n")
                self.ranked_papers[name] = []
                continue

            # Drop already-seen papers
            fresh_papers = self._filter_seen(raw_papers, name)
            if not fresh_papers:
                self._log(f"REFLECT: All papers for '{name}' were "
                          "already sent -- skipping.\n")
                self.ranked_papers[name] = []
                continue

            # LLM relevance gate
            self._log(f"THINK: Running relevance filter on "
                      f"{len(fresh_papers)} paper(s)...")
            relevant_papers = self._filter_relevant(fresh_papers, name)
            if not relevant_papers:
                self._log(f"REFLECT: No relevant papers for '{name}' "
                          "after filtering.\n")
                self.ranked_papers[name] = []
                continue

            # Dedup + rank
            top_papers = self._deduplicate_and_rank(name, relevant_papers)

            # Summarize
            summarized = self._summarize_papers(name, top_papers)
            self.ranked_papers[name] = summarized
            self._log(f"REFLECT: '{name}' done -- "
                      f"{len(summarized)} papers ready.\n")

        # Build + deliver
        self._log("=== Building final digest ===")
        digest = self._build_digest()

        self._log("=== Delivering digest ===")
        self._deliver(digest)

        # Persist seen papers (includes all new IDs from this run)
        self.seen.update(self.newly_seen)
        save_seen_papers(self.seen)
        self._log(f"ACT: Saved {len(self.newly_seen)} new paper ID(s) "
                  f"to seen_papers.json (total: {len(self.seen)})")

        self._log("DONE: Agent finished successfully.")


# =============================================================
#  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    agent = ResearchAgent()
    try:
        agent.run()
    except Exception as e:
        ts        = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        error_msg = f"[{ts}] FATAL ERROR: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        sys.exit(1)
