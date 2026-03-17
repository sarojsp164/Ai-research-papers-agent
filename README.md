# AI Research Paper Agent

A fully autonomous ReAct-style agent that searches multiple sources for the
latest AI/ML research, filters and ranks papers using an LLM, summarizes the
top picks, and delivers a clean digest to Telegram -- every other day via
GitHub Actions.

## Features

- **Multi-source search** -- arXiv, Semantic Scholar, Papers With Code
- **Seen-paper memory** -- `seen_papers.json` prevents repeats across runs
- **LLM relevance gate** -- each paper is checked for genuine topicality
- **Trending detection** -- flags papers with high citation velocity or code
- **Fuzzy deduplication** -- merges near-duplicate results across sources
- **LLM ranking** -- picks the top-k by novelty, impact, and code availability
- **3-bullet summaries** -- concise what / innovation / why-it-matters format
- **Telegram delivery** -- auto-chunked messages with Markdown fallback
- **GitHub Actions scheduling** -- runs every other day at 10:00 AM IST (free)

## Agent Loop

```
PLAN  ->  For each topic:
            SEARCH  (arXiv + Semantic Scholar + Papers With Code)
            FILTER  (skip seen papers, then LLM relevance gate)
            DEDUP   (fuzzy title matching, merge metadata)
            RANK    (LLM picks top-k by novelty + impact)
            SUMMARIZE  (LLM 3-bullet summary per paper)
          BUILD digest
          DELIVER to Telegram
          PERSIST seen_papers.json
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env from the template and fill in your keys
cp .env.example .env

# 4. Run the agent
python paper_agent.py
```

## Getting Your API Keys

### Groq
1. Go to https://console.groq.com
2. API Keys -> Create Key
3. Paste into `GROQ_API_KEY` in `.env`

### Telegram Bot
1. Open Telegram, message @BotFather
2. Send `/newbot`, follow the prompts, copy the token -> `TELEGRAM_BOT_TOKEN`
3. Send any message to your new bot
4. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. Find `"chat":{"id":XXXXXXX}` -> `TELEGRAM_CHAT_ID`

## GitHub Actions (Automatic Scheduling)

The workflow at `.github/workflows/daily_digest.yml` runs the agent every
other day at 04:30 UTC (10:00 IST) and commits `seen_papers.json` back
to the repo automatically.

### Setup

1. Push your code to a **private** GitHub repo
2. Go to **Settings -> Secrets and variables -> Actions**
3. Add three repository secrets:

   | Name                 | Value             |
   |----------------------|-------------------|
   | `GROQ_API_KEY`       | your Groq key     |
   | `TELEGRAM_BOT_TOKEN` | your bot token    |
   | `TELEGRAM_CHAT_ID`   | your chat ID      |

4. Done -- GitHub will run it on schedule. You can also click
   **Run workflow** in the Actions tab to trigger it manually.

## Configuration

All settings are in `.env` (or set as environment variables):

| Variable           | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `PAPERS_PER_TOPIC` | 5       | Max papers fetched per source per topic  |
| `TOP_K_PER_TOPIC`  | 3       | Papers kept after LLM ranking            |
| `DAYS_BACK`        | 2       | Look-back window in days                 |

### Adding a Topic

Edit the `TOPICS` list in `paper_agent.py`:

```python
{
    "name": "AI for Healthcare",
    "categories": ["cs.AI", "q-bio.QM"],
    "keywords": ["medical imaging", "clinical NLP", "drug discovery"],
},
```

## Project Structure

```
paper_agent.py                        # Main agent script
requirements.txt                      # Python dependencies
.env.example                          # Environment variable template
.gitignore                            # Git ignore rules
seen_papers.json                      # Persistent seen-paper IDs (auto-updated)
README.md                             # This file
.github/workflows/daily_digest.yml    # GitHub Actions cron workflow
```

## License

MIT
