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

# Optional: restrict feedback voting to a single Telegram user ID.
# When set, only this user's 👍/👎 clicks count; others see a polite notice.
# Leave unset to accept votes from everyone.
ADMIN_USER_ID    = os.getenv("ADMIN_USER_ID", "").strip()

# Path to the persistent seen-papers store (committed back to repo by CI)
SEEN_PAPERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "seen_papers.json")
# Maximum number of paper IDs to remember (rolling window to keep file small)
SEEN_PAPERS_MAX  = 2000

# Path to Telegram feedback state (votes + processed update offset)
FEEDBACK_STORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "feedback_store.json")
FEEDBACK_EVENTS_MAX = 5000
FEEDBACK_PAPERS_MAX = 3000
PROCESSED_CALLBACKS_MAX = 10000

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

# Timestamp of last arXiv request -- arXiv requires ≥3 s between requests
_last_arxiv_call: float = 0.0


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


def _default_feedback_store() -> dict:
    return {
        "last_update_id": 0,
        "processed_callback_ids": [],
        "votes": [],
        "papers": {},
    }


def _normalise_feedback_store(raw: dict) -> dict:
    base = _default_feedback_store()
    if not isinstance(raw, dict):
        return base

    base["last_update_id"] = int(raw.get("last_update_id", 0) or 0)

    processed = raw.get("processed_callback_ids", [])
    if isinstance(processed, list):
        base["processed_callback_ids"] = [str(x) for x in processed if x]

    votes = raw.get("votes", [])
    if isinstance(votes, list):
        base["votes"] = [v for v in votes if isinstance(v, dict)]

    papers = raw.get("papers", {})
    if isinstance(papers, dict):
        base["papers"] = papers

    return base


def load_feedback_store() -> dict:
    if os.path.exists(FEEDBACK_STORE_FILE):
        try:
            with open(FEEDBACK_STORE_FILE, "r", encoding="utf-8") as f:
                return _normalise_feedback_store(json.load(f))
        except (json.JSONDecodeError, IOError, ValueError):
            pass
    return _default_feedback_store()


def _trim_feedback_store(store: dict):
    store["votes"] = store.get("votes", [])[-FEEDBACK_EVENTS_MAX:]
    store["processed_callback_ids"] = store.get("processed_callback_ids", [])[
        -PROCESSED_CALLBACKS_MAX:
    ]

    papers = store.get("papers", {})
    if len(papers) > FEEDBACK_PAPERS_MAX:
        sorted_items = sorted(
            papers.items(),
            key=lambda x: (
                (x[1] or {}).get("last_feedback_at")
                or (x[1] or {}).get("last_sent_at")
                or ""
            ),
        )
        store["papers"] = dict(sorted_items[-FEEDBACK_PAPERS_MAX:])


def save_feedback_store(store: dict):
    _trim_feedback_store(store)
    with open(FEEDBACK_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def register_sent_paper(feedback_store: dict, paper: dict,
                        topic_name: str, message_id=None):
    pid = _paper_id(paper)
    papers = feedback_store.setdefault("papers", {})
    entry = papers.get(pid, {})

    user_votes = entry.get("user_votes", {})
    if not isinstance(user_votes, dict):
        user_votes = {}

    fb = entry.get("feedback", {})
    if not isinstance(fb, dict):
        fb = {}
    up = int(fb.get("up", 0) or 0)
    down = int(fb.get("down", 0) or 0)

    entry.update({
        "title": paper.get("title", ""),
        "topic": topic_name,
        "source": paper.get("source", ""),
        "link": paper.get("link", ""),
        "last_sent_at": datetime.utcnow().isoformat(),
        "sent_count": int(entry.get("sent_count", 0) or 0) + 1,
        "last_message_id": message_id,
        "feedback": {
            "up": up,
            "down": down,
            "score": up - down,
        },
        "user_votes": user_votes,
    })
    papers[pid] = entry


def _record_feedback_vote(feedback_store: dict, paper_id: str, vote: str,
                          user_id: str, username: str,
                          callback_id: str, update_id: int) -> bool:
    papers = feedback_store.setdefault("papers", {})
    entry = papers.get(paper_id, {})

    user_votes = entry.get("user_votes", {})
    if not isinstance(user_votes, dict):
        user_votes = {}

    fb = entry.get("feedback", {})
    if not isinstance(fb, dict):
        fb = {}

    up = int(fb.get("up", 0) or 0)
    down = int(fb.get("down", 0) or 0)

    previous = user_votes.get(user_id)
    if previous == vote:
        return False

    if previous == "up":
        up = max(0, up - 1)
    elif previous == "down":
        down = max(0, down - 1)

    if vote == "up":
        up += 1
    else:
        down += 1

    user_votes[user_id] = vote
    entry["user_votes"] = user_votes
    entry["feedback"] = {
        "up": up,
        "down": down,
        "score": up - down,
    }
    entry["last_feedback_at"] = datetime.utcnow().isoformat()
    papers[paper_id] = entry

    feedback_store.setdefault("votes", []).append({
        "paper_id": paper_id,
        "vote": vote,
        "user_id": user_id,
        "username": username,
        "callback_id": callback_id,
        "update_id": update_id,
        "timestamp": datetime.utcnow().isoformat(),
    })
    return True


def _answer_callback_query(callback_query_id: str, text: str):
    if not callback_query_id:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
        payload = {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": False,
        }
        requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException:
        pass


def tool_pull_telegram_feedback(feedback_store: dict) -> dict:
    """Ingest feedback clicks from Telegram callback queries."""
    last_update = int(feedback_store.get("last_update_id", 0) or 0)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {
        "offset": last_update + 1,
        "timeout": 0,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram getUpdates failed: {body}")

    updates = body.get("result", [])
    processed = set(feedback_store.get("processed_callback_ids", []))
    new_votes = 0
    ignored = 0
    max_update_id = last_update

    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            max_update_id = max(max_update_id, update_id)

        callback = update.get("callback_query") or {}
        callback_id = callback.get("id", "")
        if callback_id and callback_id in processed:
            continue

        payload = callback.get("data", "")
        parts = payload.split(":", 2)
        if len(parts) != 3 or parts[0] != "fb" or parts[1] not in {"up", "down"}:
            if callback_id:
                processed.add(callback_id)
            ignored += 1
            continue

        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        target_chat_id = str(TELEGRAM_CHAT_ID)
        if target_chat_id.lstrip("-").isdigit() and chat_id and chat_id != target_chat_id:
            if callback_id:
                processed.add(callback_id)
            ignored += 1
            continue

        vote = parts[1]
        paper_id = parts[2]
        user = callback.get("from") or {}
        user_id = str(user.get("id", "unknown"))
        username = (user.get("username")
                    or user.get("first_name")
                    or "unknown")

        # ── admin-only feedback gate ──
        if ADMIN_USER_ID and user_id != ADMIN_USER_ID:
            _answer_callback_query(
                callback_id,
                "Feedback is restricted to the channel admin.",
            )
            if callback_id:
                processed.add(callback_id)
            ignored += 1
            continue

        changed = _record_feedback_vote(
            feedback_store=feedback_store,
            paper_id=paper_id,
            vote=vote,
            user_id=user_id,
            username=username,
            callback_id=callback_id,
            update_id=int(update_id or 0),
        )
        if changed:
            new_votes += 1
            _answer_callback_query(
                callback_id,
                "Feedback saved. Thanks!",
            )
        else:
            _answer_callback_query(
                callback_id,
                "Already recorded.",
            )

        if callback_id:
            processed.add(callback_id)

    feedback_store["last_update_id"] = max_update_id
    feedback_store["processed_callback_ids"] = list(processed)
    _trim_feedback_store(feedback_store)
    return {
        "new_votes": new_votes,
        "ignored": ignored,
        "updates": len(updates),
        "last_update_id": max_update_id,
    }


def build_feedback_profile(feedback_store: dict, max_examples: int = 8) -> dict:
    """Build compact like/dislike context for ranking prompts."""
    liked = []
    disliked = []
    total_votes = 0

    for meta in (feedback_store.get("papers") or {}).values():
        if not isinstance(meta, dict):
            continue
        fb = meta.get("feedback") or {}
        up = int(fb.get("up", 0) or 0)
        down = int(fb.get("down", 0) or 0)
        votes = up + down
        total_votes += votes
        if votes == 0:
            continue

        title = (meta.get("title") or "").strip()
        if not title:
            continue

        item = {
            "title": title,
            "topic": meta.get("topic", ""),
            "score": up - down,
            "votes": votes,
        }
        if item["score"] > 0:
            liked.append(item)
        elif item["score"] < 0:
            disliked.append(item)

    liked.sort(key=lambda x: (x["score"], x["votes"]), reverse=True)
    disliked.sort(key=lambda x: (x["score"], -x["votes"]))

    return {
        "total_votes": total_votes,
        "liked": liked[:max_examples],
        "disliked": disliked[:max_examples],
    }


# =============================================================
#  TOOL: SEARCH SOURCES
# =============================================================

def tool_search_arxiv(categories: list, keywords: list,
                      max_results: int = 5) -> list:
    """Search arXiv for recent papers matching categories + keywords."""
    global _last_arxiv_call
    # arXiv Terms of Use: minimum 3 s between requests
    elapsed = time.time() - _last_arxiv_call
    if elapsed < 3.5:
        time.sleep(3.5 - elapsed)

    cat_query  = " OR ".join(f"cat:{c}" for c in categories)
    kw_query   = " OR ".join(f'all:"{k}"' for k in keywords[:4])
    # Add date range so DAYS_BACK is respected
    date_end   = datetime.utcnow()
    date_start = date_end - timedelta(days=DAYS_BACK)
    date_filter = (f"submittedDate:[{date_start.strftime('%Y%m%d')}0000"
                   f" TO {date_end.strftime('%Y%m%d')}2359]")
    full_query = f"({cat_query}) AND ({kw_query}) AND {date_filter}"

    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    # arXiv rate-limits aggressively; retry with exponential backoff
    resp = None
    for attempt in range(4):
        try:
            _last_arxiv_call = time.time()
            resp = requests.get("https://export.arxiv.org/api/query",
                                params=params, timeout=30)
            if resp.status_code == 429:
                wait = 5 * (2 ** attempt)       # 5, 10, 20, 40
                print(f"  [arXiv RATE-LIMITED] Waiting {wait}s "
                      f"(attempt {attempt + 1}/4)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout:
            if attempt < 3:
                wait = 5 * (2 ** attempt)
                print(f"  [arXiv TIMEOUT] Retrying in {wait}s "
                      f"(attempt {attempt + 1}/4)...")
                time.sleep(wait)
                continue
            raise
    if resp is None or resp.status_code == 429:
        print("  [arXiv] All retry attempts exhausted.")
        return []

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
    # Use publicationDateOrYear for precise recency (YYYY-MM-DD range)
    date_from = (datetime.utcnow() - timedelta(days=max(DAYS_BACK * 15, 30))
                 ).strftime("%Y-%m-%d")
    date_to   = datetime.utcnow().strftime("%Y-%m-%d")
    params    = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,abstract,url,authors,year,citationCount,externalIds",
        "publicationDateOrYear": f"{date_from}:{date_to}",
        "sort":   "citationCount:desc",
    }
    time.sleep(3)   # polite delay to avoid 429s on consecutive topic calls
    resp = None
    for attempt in range(5):
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params, timeout=30,
            )
        except requests.exceptions.Timeout:
            if attempt < 4:
                print(f"  [S2 TIMEOUT] Retrying in 10s...")
                time.sleep(10)
                continue
            raise
        if resp.status_code == 429:
            wait = int(resp.headers.get("retry-after", 30))
            wait = max(wait, 10 * (attempt + 1))  # escalating backoff
            print(f"  [S2 RATE-LIMITED] Waiting {wait}s "
                  f"(attempt {attempt + 1}/5)...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    if resp is None or resp.status_code == 429:
        print("  [S2] All retry attempts exhausted -- returning empty.")
        return []

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
    headers = {
        "User-Agent": "research-agent/1.0 (academic use)",
        "Accept":     "application/json",
    }
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://paperswithcode.com/api/v1/papers/",
                params=params, headers=headers, timeout=20,
            )
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  [PWC RATE-LIMITED] Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "json" not in content_type:
                print(f"  [PWC] Non-JSON response ({content_type}) -- skipping.")
                return []
            data = resp.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < 2:
                time.sleep(5)
                continue
            raise
        except json.JSONDecodeError:
            print("  [PWC] Invalid JSON in response -- skipping.")
            return []
    else:
        print("  [PWC] All retry attempts exhausted.")
        return []

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

def tool_rank_papers(papers: list, topic: str, top_k: int = 3,
                     feedback_profile: dict = None) -> list:
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

    feedback_block = ""
    if feedback_profile and feedback_profile.get("total_votes", 0) > 0:
        liked = feedback_profile.get("liked", [])
        disliked = feedback_profile.get("disliked", [])
        liked_lines = ("\n".join(
            f"- {x['title']} (topic={x.get('topic', 'unknown')}, score={x['score']})"
            for x in liked
        ) or "- none")
        disliked_lines = ("\n".join(
            f"- {x['title']} (topic={x.get('topic', 'unknown')}, score={x['score']})"
            for x in disliked
        ) or "- none")
        feedback_block = f"""
User preference signal from Telegram feedback (soft constraint):
- Total votes observed: {feedback_profile['total_votes']}
- Liked examples:
{liked_lines}
- Disliked examples:
{disliked_lines}

Prefer papers similar to liked examples and avoid papers similar to disliked
examples, but never sacrifice scientific quality just to match preference."""

    prompt = f"""You are an AI research curator. Given these papers on "{topic}",
pick the {top_k} MOST important ones: novelty, practical impact, scientific
contribution. Prefer papers with code or high citations. Trending papers are
gaining rapid attention and should be weighted higher. Also send papers from ai companies as well [openai, deepmind, google, microsoft, anthropic, meta, nvidia, stability ai].

{feedback_block}

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
    chunks = [message[i:i + 4000] for i in range(0, len(message), 4000)]
    errors = []
    sent = 0

    for chunk in chunks:
        result = tool_send_telegram_message(chunk)
        if result.get("ok"):
            sent += 1
        else:
            errors.append(result.get("error", "unknown Telegram error"))

    if errors:
        return (f"Sent {sent}/{len(chunks)} chunk(s) with "
                f"{len(errors)} error(s): {errors[0]}")
    return f"Sent {sent} message(s) to Telegram successfully."


def tool_send_telegram_message(message: str, reply_markup: dict = None) -> dict:
    """Send one Telegram message and return status + message_id."""
    try:
        me = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
            timeout=10,
        )
        if me.status_code in (401, 404):
            return {
                "ok": False,
                "error": (
                    "Telegram bot token is invalid or revoked. "
                    "Rotate it via @BotFather and update the secret."
                ),
            }
    except requests.exceptions.RequestException as e:
        print(f"  [Telegram] Pre-flight check failed: {e}")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code == 200:
        body = resp.json()
        return {
            "ok": True,
            "message_id": (body.get("result") or {}).get("message_id"),
        }

    try:
        err_body = resp.json()
    except Exception:
        err_body = {"description": resp.text[:300]}
    err_desc = err_body.get("description", "")

    if resp.status_code == 404:
        return {
            "ok": False,
            "error": (
                "Telegram returned 404 -- bot token is invalid or revoked. "
                f"{err_desc}"
            ),
        }
    if resp.status_code == 400 and "chat not found" in err_desc.lower():
        return {
            "ok": False,
            "error": (
                f"Telegram chat ID {TELEGRAM_CHAT_ID} not found. "
                "Start a conversation with the bot first, then retry. "
                f"{err_desc}"
            ),
        }

    plain = message.replace("*", "").replace("_", "").replace("`", "")
    payload2 = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": plain,
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload2["reply_markup"] = reply_markup
    resp2 = requests.post(url, json=payload2, timeout=15)
    if resp2.status_code == 200:
        body2 = resp2.json()
        return {
            "ok": True,
            "message_id": (body2.get("result") or {}).get("message_id"),
        }
    return {
        "ok": False,
        "error": f"HTTP {resp2.status_code}: {resp2.text[:200]}",
    }


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
                     -> SUMMARIZE -> DELIVER (one paper/message + feedback buttons)
                     -> PULL FEEDBACK -> PERSIST STATE
    """

    def __init__(self):
        self.ranked_papers: dict = {}   # topic -> [(paper, summary), ...]
        self.log:           list = []
        self.seen                = load_seen_papers()
        self.newly_seen:    dict = {}   # ids encountered this run
        self.feedback_store       = load_feedback_store()
        self.feedback_profile     = build_feedback_profile(self.feedback_store)

    def _log(self, msg: str):
        ts   = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log.append(line)
        print(line, flush=True)

    def _sync_feedback_from_telegram(self):
        self._log("PLAN: Syncing Telegram feedback updates...")
        try:
            result = tool_pull_telegram_feedback(self.feedback_store)
            self.feedback_profile = build_feedback_profile(self.feedback_store)
            save_feedback_store(self.feedback_store)
            self._log(
                "  ACT: Feedback sync complete -- "
                f"{result['new_votes']} new vote(s), "
                f"{self.feedback_profile['total_votes']} total vote(s)."
            )
            if self.feedback_profile["total_votes"] == 0:
                self._log("  OBSERVE: No feedback yet -- ranking stays neutral.")
        except Exception as e:
            self.feedback_profile = {"total_votes": 0, "liked": [], "disliked": []}
            self._log(
                f"  OBSERVE: Feedback sync failed ({e}) -- using neutral ranking."
            )

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
            top_indices = tool_rank_papers(
                unique,
                topic_name,
                TOP_K_PER_TOPIC,
                self.feedback_profile,
            )
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
    # PHASE 6 -- Deliver
    # ------------------------------------------------------------------

    def _deliver(self):
        self._log("THINK: Sending digest to Telegram (one paper per message)...")

        today = datetime.utcnow().strftime("%B %d, %Y")
        intro = (
            f"*AI Research Digest -- {today}*\n"
            "Send feedback per paper using the buttons below each message."
        )
        intro_result = tool_send_telegram_message(intro)
        if not intro_result.get("ok"):
            raise RuntimeError(f"Telegram intro send failed: {intro_result.get('error')}")

        any_papers = False
        sent_count = 0

        for topic_name, papers in self.ranked_papers.items():
            if not papers:
                continue

            any_papers = True
            tool_send_telegram_message(f"--- *{topic_name}* ---")

            for i, (paper, summary) in enumerate(papers, 1):
                pid = _paper_id(paper)
                trending_tag = "  [TRENDING]" if paper.get("trending") else ""

                lines = [
                    f"*{i}. {paper['title']}*{trending_tag}",
                    paper["link"],
                ]
                if paper.get("code_url"):
                    lines.append(f"Code: {paper['code_url']}")
                lines.append(summary)
                lines.append("_Feedback: tap 👍 or 👎 below_")

                # Telegram limits callback_data to 64 bytes
                cb_up   = f"fb:up:{pid}"[:64]
                cb_down = f"fb:down:{pid}"[:64]
                keyboard = {
                    "inline_keyboard": [[
                        {"text": "👍 Relevant", "callback_data": cb_up},
                        {"text": "👎 Not useful", "callback_data": cb_down},
                    ]]
                }

                result = tool_send_telegram_message(
                    "\n".join(lines),
                    reply_markup=keyboard,
                )

                if not result.get("ok"):
                    self._log(
                        "  OBSERVE: Failed to send paper message "
                        f"'{paper['title'][:45]}...' ({result.get('error')})"
                    )
                    continue

                register_sent_paper(
                    feedback_store=self.feedback_store,
                    paper=paper,
                    topic_name=topic_name,
                    message_id=result.get("message_id"),
                )
                sent_count += 1

        if not any_papers:
            tool_send_telegram_message(
                "No new papers this cycle -- all recent papers were already sent."
            )

        save_feedback_store(self.feedback_store)
        self._log(f"  OBSERVE: Sent {sent_count} paper message(s) with feedback buttons.")

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self):
        self._log("PLAN: Starting AI Research Paper Agent")
        self._log(f"PLAN: {len(TOPICS)} topics | {DAYS_BACK}-day window | "
                  f"top {TOP_K_PER_TOPIC} papers per topic")
        self._log(f"PLAN: Seen-papers store has {len(self.seen)} entries\n")

        self._sync_feedback_from_telegram()
        self._log("")

        for idx, topic in enumerate(TOPICS):
            name = topic["name"]
            self._log(f"=== Topic: {name} ===")

            # Polite inter-topic pause (APIs rate-limit across rapid calls)
            if idx > 0:
                self._log("  (pausing 5 s between topics)")
                time.sleep(5)

            # Search all sources — with adaptive deepening
            raw_papers = self._search_topic(topic)
            fresh_papers = []
            if raw_papers:
                fresh_papers = self._filter_seen(raw_papers, name)

            # Agentic self-correction: if all results are seen, widen search
            if not fresh_papers and raw_papers:
                self._log(f"REFLECT: All {len(raw_papers)} papers for "
                          f"'{name}' were already sent -- deepening search...")
                wider_topic = {
                    **topic,
                    "keywords": topic["keywords"][:2],  # broader terms
                }
                extra = tool_search_arxiv(
                    wider_topic["categories"],
                    wider_topic["keywords"],
                    max_results=PAPERS_PER_TOPIC * 3,
                )
                if extra:
                    self._log(f"  ACT: Widened arXiv -> {len(extra)} paper(s)")
                    fresh_papers = self._filter_seen(extra, name)

            if not fresh_papers:
                reason = ("no results from any source"
                          if not raw_papers
                          else "all papers already sent (even after deepening)")
                self._log(f"REFLECT: {reason} for '{name}' -- skipping.\n")
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

        self._log("=== Delivering digest ===")
        self._deliver()

        # Persist seen papers (includes all new IDs from this run)
        self.seen.update(self.newly_seen)
        save_seen_papers(self.seen)
        self._log(f"ACT: Saved {len(self.newly_seen)} new paper ID(s) "
                  f"to seen_papers.json (total: {len(self.seen)})")
        save_feedback_store(self.feedback_store)
        self._log(
            f"ACT: Feedback store now has "
            f"{self.feedback_profile.get('total_votes', 0)} total vote(s)."
        )

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
