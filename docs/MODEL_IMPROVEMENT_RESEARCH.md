# Research & Root-Cause Analysis: Overcoming the Social Model Performance Ceilings

## 1. Executive Summary & Diagnostic Findings

Across the 7 model families and auxiliary tasks in `identify_what_matters`, rigorous evaluation on unpolluted, cleaned splits revealed three systemic root causes behind sub-optimal real-world performance:

1. **Synthetic-to-Real Domain Gap**:
   - Models trained purely on LLM-generated synthetic comments (even with regional Indic scripts) achieve high synthetic test F1 (>0.85–0.92) but collapse on real Instagram comments (stance real F1: 0.3458–0.3803; tone real F1: 0.5792–0.5887; party real accuracy: 0.537).
   - *Why*: Synthetic comments are structured, syntactically complete, and lexically explicit ("I condemn this policy because..."). Real Instagram comments are 3–15 words, heavily abbreviated, full of cryptic slang, sarcastic inversions without emojis, and highly context-dependent.
2. **Threshold Misalignment in Runtime Encoders**:
   - `filter_post_relevance` in `src/identify_what_matters/stance.py` hardcoded a cosine similarity threshold of `0.45`. The production v5 event encoder (`multilingual-e5-base-events-real-v5`) has a dev-calibrated optimal threshold of `0.590` (and clean dev threshold of `0.555`). Operating at `0.45` polluted the downstream stance clustering pipeline with out-of-domain media (F1 dropped from 0.8738 to 0.8332).
3. **Data Contamination & Evaluation Leakage in Upstream Baselines**:
   - Legacy checkpoints for demographic contrastive models and strategic v4 models used the evaluation benchmark for epoch-by-epoch checkpoint selection, creating artificially inflated metrics that failed to generalize.
   - Sarcasm detection relied on naive regex/emoji rules (`😂🤣😜😏🤡💀`), achieving only 0.521 macro-F1 and failing on sarcasm phrased as literal praise.

---

## 2. Quantitative Model Performance Benchmark (Audited Clean Splits)

| Model Family / Subtask | Metric | Legacy / Baseline | Current State / Verified Candidate | Delta | Action Taken & Production Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Political: Direction** | Macro-F1 | 0.5696 (Transformer) | **0.6239** (Linear Word+Char) | **+0.0543** | Candidate trained with word+char subwords ($C=8.0$); outperforms mDeBERTa on short noisy inputs. |
| **Political: Target** | Macro-F1 | 0.4219 (Baseline) | **0.4665** (Candidate) | **+0.0445** | Subword character n-grams resolve misspellings in regional party names (SP, TMC, DMK). |
| **Political: Issue Cluster** | Macro-F1 | 0.4945 (Baseline) | **0.5306** (Candidate) | **+0.0361** | Character n-grams capture Hinglish issue variants (`mehangai`, `aarakshan`, `ghotala`). |
| **Political: Sentiment** | Macro-F1 | 0.5152 (Baseline) | **0.5364** (Candidate) | **+0.0213** | Balanced subword features improve neutral vs mixed distinction. |
| **Event Relevance (v5)** | Macro-F1 | 0.8332 (at $0.45$) | **0.8738** (at tuned threshold) | **+0.0406** | **Fixed**: `filter_post_relevance()` now dynamically loads the encoder's tuned threshold (`0.59` in v5). |
| **Sarcasm Detection** | Macro-F1 | 0.5212 (Rule-based) | **0.7975** (Calibrated TF-IDF) | **+0.2763** | Candidate available via `TRENDS_SARCASM_MODEL`. |
| **Civic Classifier** | Test AUC / F1 | 0.9635 AUC / 0.8607 F1 | **0.9635 AUC / 0.8607 F1** | Baseline | Strong generalization across 12 Indic languages. |
| **Stance Fine-Tune** | Macro-F1 | 0.6643 (Overall) | 0.6873 (Augmented) | +0.0230 | Overall improved, but real-comment slice regressed ($0.3803 \to 0.3458$). Production weights preserved. |
| **Strategic Multi-Task** | Composite-F1 | 0.8337 (Legacy) | 0.7487 (Clean Dev Selected) | -0.0850 | Legacy baseline had benchmark selection leakage; clean checkpoint provides honest generalization. |

---

## 3. High-Impact Action Items & Architectural Solutions

### A. Dynamic Relevance Threshold Alignment (Implemented)
- **Change**: Updated [src/identify_what_matters/stance.py](src/identify_what_matters/stance.py) to read the active encoder's calibrated threshold via `encoder.threshold()` instead of a hardcoded `0.45`.
- **Impact**: Instantly recovers +4.06% macro-F1 on real event-relevance filtering, preventing off-topic noise from reaching the stance clustering and opinion summarization stages.

### B. Political Classification Upgrade
- **Finding**: For Indian social text (captions and comments), the tuned linear model with word (1-3) and character-wb (3-5) n-grams consistently outperforms sequence-level transformers on sparse targets (+5.4% on direction, +4.5% on target, +3.6% on issue clusters).
- **Recommendation**: Deploy the word+char candidates into `artifacts/political_models/` or keep the hybrid ensemble where linear models handle sparse entity extraction while transformers handle complex sentence discourse.

### C. Resolving the Real-Comment Stance & Tone Gap (Domain Adaptation Strategy)
To improve performance on real-world comments without synthetic distribution collapse:
1. **Curriculum Loss & Domain Weighting**:
   - In `stance_finetune.py`, assign higher sample weights ($3.0\times$) to real Instagram comments (`origin: real_llm_labelled`) relative to synthetic comments ($1.0\times$).
2. **Context-Contrastive Hard Negative Mining**:
   - Real comments often use sarcasm or indirect praise (e.g., *"Waah kya vikas hai"* under a pothole post). Generate pairs specifically contrasting literal meaning vs pragmatic intent.
3. **Pseudo-Labeling Real Comment Pools with Teacher Verification**:
   - Rather than generating synthetic text from scratch, take the 10,000+ unlabelled real comments from `output/collections/` and pass them through Sonnet 4.6 with chain-of-thought verification to produce authentic real-comment training pairs.
