<div align="center">

# 🇮🇳 Identify What Matters

### Autonomous Real-Time Intelligence on Emerging Trends, Controversy & Grassroots Debates in India

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Managed by uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![OpenRouter LLM](https://img.shields.io/badge/LLM-OpenRouter-6366F1.svg?style=for-the-badge&logo=openai&logoColor=white)](https://openrouter.ai/)
[![Multilingual E5](https://img.shields.io/badge/embeddings-Multilingual--E5-10B981.svg?style=for-the-badge)](https://huggingface.co/intfloat/multilingual-e5-base)
[![Test Suite](https://img.shields.io/badge/tests-29%20passing-success.svg?style=for-the-badge&logo=githubactions&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-FFA116.svg?style=for-the-badge)](LICENSE)

<br/>

[🔍 Overview](#-system-overview) • [⚡ Quickstart](#-quickstart-in-60-seconds) • [⚙️ How It Works](#️-how-it-works) • [🚀 Running the Pipeline](#-running-the-pipeline) • [📊 Dashboards](#-interactive-dashboards) • [🤖 ML Stack](#-local-machine-learning-stack) • [📁 Codebase Map](#-codebase-directory-map)

---

</div>

> 💡 **The Problem:** Standard trend trackers rely on raw keyword frequencies and view counts. This surfaces viral reels, celebrity trivia, and PR campaigns while remaining completely blind to genuine civic friction, polarized controversies, and grassroots citizen debates.
> 
> 🎯 **The Solution:** A discussion-first, NLP-driven intelligence monitor. It measures comment-to-like ratio shocks, reply tree depth, multilingual code-mixed arguments, astroturf coordination, and opposing viewpoints across Indian social feeds (Instagram, Google News in 10 languages, Google Trends, X Trends, and Reddit).

---

## 🌟 Key Capabilities

<table>
<tr>
<td width="50%" valign="top">

### ⚔️ Controversy & Debate Detection
Detects disagreement objectively through comment-to-like ratios, reply cascades, account baseline lifts, and like-weighted stance distribution between competing sides ($1 - |s_1 - s_2|$).

</td>
<td width="50%" valign="top">

### ⚡ Early Breakout Anomaly Radar
Catches breaking controversies in their earliest stages (`spark` $\to$ `pre_breakout`) through velocity surges and cross-account spreading before stories reach mainstream press.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🌐 Multilingual & Code-Mixed NLP
Fine-tuned `multilingual-e5-base` bi-encoder and `mDeBERTa` cross-encoder comprehend Hinglish, Tanglish, and regional Indian languages (Hindi, Tamil, Telugu, Bengali, Malayalam, etc.).

</td>
<td width="50%" valign="top">

### 🛡️ Astroturf & Bot Neutralization
Differentiates between authentic citizen discussions and coordinated bot campaigns using 5-shingle Jaccard comment duplication clustering and account behavior baselines.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📋 Automated Intelligence Briefings
Synthesizes contested topics into objective briefs highlighting the core debate question, key factions, caution flags, and confidence audit checks.

</td>
<td width="50%" valign="top">

### 📊 Rich Visual Dashboards
Generates full-fidelity interactive HTML dashboards featuring 2D Debate Radars, breakout stage badges, sentiment balances, and sentinel account pulse cards.

</td>
</tr>
</table>

---

## ⚡ Quickstart in 60 Seconds

### 1. Prerequisites & Installation

Ensure you have **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)** installed:

```bash
# Clone the repository
git clone https://github.com/username/identify-what-matters.git
cd identify-what-matters

# Install all dependencies with uv
uv sync
```

### 2. Configure Environment

All LLM operations (topic distillation, probe generation, brief formulation, and audit) connect directly via **[OpenRouter](https://openrouter.ai/)**. No corporate VPN or proxy is needed!

Create a `.env` file in the root directory:

```bash
# Required: Your OpenRouter API key
OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxx"

# Optional: Preferred model (defaults to openai/gpt-6.1-sol)
OPENROUTER_MODEL="openai/gpt-6.1-sol"

# Optional: Only needed for live private Instagram API collection
INSTAGRAM_SESSION_ID="your_sessionid_cookie"
```

### 3. Run Your First Test (Instant Mock Mode)

Run the full pipeline using historical playback fixtures—**no Instagram login required**:

```bash
uv run identify-what-matters-trends run --instagram mock
```

🎉 **That's it!** The pipeline executes end-to-end, writes artifacts to `output/runs/<run_id>/`, and generates `report.md`.

---

## ⚙️ How It Works

The engine operates across three clean phases that decouple signal ingestion from deterministic offline analysis:

```mermaid
flowchart TD
    %% Phase 1
    subgraph P1["Phase 1 · Plan"]
        direction TB
        NEWS["Google News RSS<br/>(10 Indian Languages)"] --> EMB1["Multilingual E5<br/>Story Clustering"]
        GT["Google Trends IN"] --> EMB1
        REDDIT["Reddit Discussions<br/>(India Communities)"] --> EMB1
        EMB1 --> LLM1["OpenRouter LLM<br/>Topic Extraction & Queries"]
        LLM1 --> FILTER1["Generic Hashtag Filter<br/>(Strips #viral, #reels)"]
        FILTER1 --> PLAN_OUT[("plan.json<br/>Planned Topics & Probes")]
    end

    %% Phase 2
    subgraph P2["Phase 2 · Collect (Zero LLM calls)"]
        direction TB
        PLAN_OUT --> PROBES["Execute Event Probes<br/>on Instagram"]
        XT["X India Trends"] --> SEARCH["Targeted Keyword Searches"]
        PANEL["63 Sentinel Accounts<br/>(Politics, Civic, Satire)"] --> RADAR["Anomaly Radar<br/>(Comment:Like Ratio Shocks)"]
        RADAR --> TREES["Sample Deep Comment Trees"]
        SEARCH --> TREES
        PROBES --> TREES
        TREES --> CAP_OUT[("instagram_capture.json<br/>Raw Deterministic Capture")]
    end

    %% Phase 3
    subgraph P3["Phase 3 · Measure & Brief"]
        direction TB
        CAP_OUT --> CIVIC["Lightweight Civic Classifier<br/>(civic-me5-base, &tau;=0.74)"]
        CIVIC --> CLUST["Average-Linkage Clustering<br/>(multilingual-e5-base)"]
        CLUST --> BRIEF["OpenRouter Debate Briefing<br/>(Contested Question & Sides)"]
        BRIEF --> LOCAL_ML["Local Stance & Tone Engine<br/>(mDeBERTa + Sarcasm Heuristic)"]
        LOCAL_ML --> SCORE["Lifecycle & Controversy Scoring<br/>(debate_score, trend_score)"]
        SCORE --> REPORT[("Final Deliverables<br/>report.md • report.json • dashboard.html")]
    end

    P1 ==> P2
    P2 ==> P3

    classDef phase fill:#f8fafc,stroke:#3b82f6,stroke-width:2px,color:#0f172a;
    classDef store fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#065f46;
    class P1,P2,P3 phase;
    class PLAN_OUT,CAP_OUT,REPORT store;
```

### Core Analytical Metrics

| Metric | What It Measures | How It's Calculated |
| :--- | :--- | :--- |
| **`debate_score`** | Disagreement intensity | Function of comment:like ratio, replies per comment, lift over account historical norm, total comment volume, and side balance ($1 - \|s_1 - s_2\|$). |
| **`trend_score`** | Cross-platform momentum | Weighted combination: $0.55 \times \text{Instagram Heat} + 0.30 \times \text{News Heat} + 0.15 \times \text{Search Heat}$. |
| **`civic_score`** | Governance & public relevance | Local `civic-me5-base` classifier ($\tau = 0.74$, AUC $0.962$) filtering out sports, celebrity gossip, and ads. |
| **Astroturf Index** | Coordinated amplification | 5-shingle Jaccard cluster size + zero-post account ratios + creation date velocity. |
| **Emergence Stage** | Controversy lifecycle | `spark` $\to$ `pre_breakout` $\to$ `breaking_out` $\to$ `peak_debate` $\to$ `saturated` $\to$ `settled`. |

---

## 📋 Sample Intelligence Brief

Every analyzed discussion produces a structured brief with quantified sides, sentiment, and audit checks:

<details open>
<summary><b>View Sample Brief: CJP Shivaji Park Demonstration (from live run)</b></summary>
<br/>

> ### CJP Protest Against Election Commission in Mumbai
> **Category:** `politics_governance` | **Stage:** `settled` | **Trend Score:** `6.02` | **Debate Score:** `6.96` (High)
> 
> Citizens for Justice and Peace (CJP) held a protest at Shivaji Park, Dadar against the Election Commission, leading to an FIR against CJP organizers. BJP leader Nitesh Rane sharply attacked the organizers, sparking intense polarized debate online.
> 
> **Opinion Balance (Sampled Comments, Like-Weighted):**  
> `[███░░░░░▒▒▒▒▒▒▒·]` (20% Supportive / 31% Critical / 42% Humorous / 7% Neutral)
> 
> **The Core Debate:** *Was CJP's protest at Shivaji Park a legitimate democratic demonstration or an unlawful act of provocation?*
> - **Side 1 (42%):** CJP's protest is a valid democratic act and the police crackdown reflects intolerance of dissent.
>   *Top comment:* `"Bjp k dalla😂😂"` (175 likes)
> - **Side 2 (40%):** CJP overstepped without police permission; grandstanding against senior authorities is irresponsible.
>   *Top comment:* `"इसका अब ज्यादा हो रहा है"` (293 likes)
> - **Side 3 (18%):** Humorous and cynical takes mocking political theatrics on both sides.
> 
> ⚠️ **Caution Flag:** Legality of protest and FIR details are disputed; political rhetoric is elevated ahead of state elections.

</details>

---

## 🚀 Running the Pipeline

### Mode 1: One-Shot Execution (`run`)

Run the complete pipeline (Plan $\to$ Collect $\to$ Measure) in a single command:

```bash
# 🔹 Quick local testing with mock fixtures (recommended for first-time users)
uv run identify-what-matters-trends run --instagram mock

# 🔹 Public trends only (Google News + Google Trends + X + Reddit; skips Instagram)
uv run identify-what-matters-trends run --no-instagram

# 🔹 Live Instagram collection using session cookie
uv run identify-what-matters-trends run --instagram instagrapi

# 🔹 Live collection using your browser's existing session
uv run identify-what-matters-trends run --instagram browser
```

---

### Mode 2: Step-by-Step Modular Workflow

For debugging, tuning prompts, or inspecting intermediate JSON files:

```bash
# Step 1: Discover and plan topics from news and search feeds
uv run identify-what-matters-trends plan

# Step 2: Collect data for the planned topics (saved to instagram_capture.json)
uv run identify-what-matters-trends collect --run-id latest --instagram mock

# Step 3: Measure, cluster, generate briefs, and render report
uv run identify-what-matters-trends measure --run-id latest
```

---

### Mode 3: Continuous Monitoring Daemon (`watch`)

Run continuous background monitoring every $N$ hours to track rising, fading, and breakout trends:

```bash
uv run identify-what-matters-trends watch --every-hours 4 --instagram mock
```

---

## 🌐 Instagram Access Options

| Mode | Flag | Network Requirement | When to Use |
| :--- | :--- | :--- | :--- |
| **Mock** | `--instagram mock` | None (Local fixtures) | Local experimentation, CI/CD, fast development without Instagram accounts. |
| **Private API** | `--instagram instagrapi` | Direct Internet | Automated headless servers; requires `INSTAGRAM_SESSION_ID` in `.env`. |
| **Browser Bridge** | `--instagram browser` | Local WebSocket | Maximum safety; routes requests through your existing browser session. |

<details>
<summary><b>How to use the Browser Bridge (Zero API Blocks)</b></summary>
<br/>

1. Launch collection with the browser option:
   ```bash
   uv run identify-what-matters-trends run --instagram browser --bridge-port 8765
   ```
2. Open [instagram.com](https://www.instagram.com) in Chrome/Firefox and log in normally.
3. Open Developer Tools (`F12` or `Ctrl+Shift+I` / `Cmd+Option+I`) $\to$ **Console**.
4. Paste the contents of [src/identify_what_matters/browser_proxy.js](src/identify_what_matters/browser_proxy.js) and press Enter.
5. The browser will securely proxy requests through your active session!

</details>

---

## 📊 Interactive Dashboards

Transform raw run reports into rich, interactive visual HTML dashboards:

```bash
# Generate comprehensive dashboard from the latest run
uv run python make_full_dashboard.py
```

Open `output/runs/<run_id>/dashboard.html` in your browser:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        IDENTIFY WHAT MATTERS                           │
│  [Debate Radar]   [Breakout Lifecycle]   [Stance Balance]   [Audit]    │
├────────────────────────────────────────────────────────────────────────┤
│  ⚡ BREAKOUT RADAR                                                     │
│     High Controversy                                                  │
│          ▲    * CJP Mumbai Protest                                    │
│          │             * Jantar Mantar Probe                          │
│          │                               * FlyDubai Incident          │
│          └────────────────────────────────────────► High Trend Heat    │
│                                                                        │
│  📊 SENTINEL PANEL OVERPERFORMANCE                                     │
│  • @account_a: 51x baseline comments  [Controversy Spike]              │
│  • @account_b: 23x ratio lift         [Viral Friction]                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Local Machine Learning Stack

In addition to OpenRouter for high-level generative synthesis, the system uses locally hosted fine-tuned transformer models stored under `artifacts/`:

| Component | Model / Architecture | Location | Task & Accuracy |
| :--- | :--- | :--- | :--- |
| **Topic Clustering** | `multilingual-e5-base` | `artifacts/topic-models/` | Event-level semantic grouping. AUC **0.905**, ARI **0.874**. |
| **Civic Classifier** | `civic-me5-base` | `artifacts/civic-models/` | Pre-filters non-civic noise. AUC **0.962**, F1 **0.879** ($\tau=0.74$). |
| **Stance Engine** | `mDeBERTa-v3-base-xnli` | `artifacts/stance-models/stance` | Quantifies agreement/opposition per comment. Macro-F1 **0.755**. |
| **Tone Classifier** | Fine-tuned `m-e5-base` | `artifacts/stance-models/tone` | Classifies `supportive`, `critical`, `humorous`, `neutral`. |
| **Bot Detector** | Jaccard Shingling + Heuristics | [src/identify_what_matters/bot_detector.py](src/identify_what_matters/bot_detector.py) | Uncovers astroturf duplication and inauthentic coordinated rings. |

---

## 📁 Codebase Directory Map

```
identify-what-matters/
├── config/
│   ├── panel.json                     # 63 curated sentinel accounts across Indian discourse
│   └── discovery.json                 # Signal sources and RSS feed endpoints
├── artifacts/
│   ├── topic-models/                  # Fine-tuned multilingual E5 bi-encoder checkpoints
│   ├── civic-models/                  # Trained civic relevance classifier
│   └── stance-models/                 # Fine-tuned mDeBERTa stance and tone weights
├── src/identify_what_matters/
│   ├── trends.py                      # Main pipeline orchestrator, scoring engine, and CLI
│   ├── openrouter.py                  # OpenRouter API client with reasoning continuation
│   ├── llm.py                         # LLM cache, schema validation, and retry handlers
│   ├── sources.py                     # Public signal collectors (Google News, Trends, X, Reddit)
│   ├── encoder.py                     # E5 bi-encoder embedding and agglomerative clustering
│   ├── discussions.py                 # Instagram discussion grouping and flame-post detector
│   ├── stance.py                      # Local stance and tone scoring engine
│   ├── civic_classifier.py            # Local civic relevance classifier wrapper
│   ├── bot_detector.py                # Jaccard shingling and astroturf analysis
│   ├── political_pulse.py             # Partisan favorability and community cross-tab engine
│   ├── instagram_access.py            # Transport abstractions (Mock, Instagrapi, Browser, Replay)
│   └── browser_proxy.js               # DevTools in-page browser bridge snippet
├── output/
│   ├── llm_cache/                     # SHA256-keyed disk cache for LLM queries
│   ├── trends.sqlite                  # Longitudinal observation store and account baselines
│   └── runs/<run_id>/                 # Isolated run artifacts (plan, capture, report, dashboard)
├── tests/                             # Full unit test suite (29 tests)
├── make_full_dashboard.py             # Generates complete interactive HTML dashboard
├── pyproject.toml                     # Project metadata and dependencies
└── README.md                          # You are here!
```

---

## 🧪 Testing & Validation

Run the test suite across all classifiers, clustering algorithms, and scoring logic:

```bash
# Run tests cleanly on CPU (prevents GPU memory contention)
CUDA_VISIBLE_DEVICES="" uv run python -m unittest discover tests -v
```

---

## 🛡️ Ethical Standards & Safeguards

- 🔒 **Privacy by Design**: User handles, telephone numbers, and URLs are scrubbed before sending text to LLMs or saving briefings.
- ⚖️ **Objective Debate Framing**: Controversies are articulated as balanced questions rather than taking partisan stances.
- 🚫 **No Demographic Profiling**: Personal attributes (caste, religion, identity) are never inferred from user text or usernames.
- 🔍 **Audit Verification**: Every brief undergoes an automated LLM judge audit comparing claims against raw post evidence to eliminate hallucinations.

---

<div align="center">

Built with ❤️ for understanding authentic Indian digital discourse.

</div>
