# Topic Embedding Training Results

## Unified Production Models & Final VM Retraining (2026-10-04)

All three core ML components have been aggressively fine-tuned and validated across a dedicated 4x NVIDIA A100-80GB GPU cluster:

1. **Topic Clustering Bi-Encoder (`multilingual-e5-base-events-real-v4`)**:
   - **Trained On**: Synthetic Sonnet v2 multilingual corpus + 1,118 real triplets ($3\times$ repeat) from 6 production runs on GPU 0.
   - **Performance**: Test ROC-AUC **0.9118** and F1 **0.8379** on the real Instagram benchmark; Synthetic in-distribution ARI **0.9285**, retrieval P@1 **0.9907**, pair ROC-AUC **0.9790**.
   - **Deployment**: [artifacts/topic-models/multilingual-e5-base-events-real-v4](artifacts/topic-models/multilingual-e5-base-events-real-v4) with calibrated clustering threshold $\tau=0.45$. Superseded older models pruned.

2. **Civic/Political Relevance Classifier (`civic-me5-base`)**:
   - **Trained On**: 3,682 enriched multilingual captions (1,015 real run captions, 2,640 synthetic event captions, and curated hard negatives) on GPU 1.
   - **Performance**: Test ROC-AUC **0.9635**, F1 **0.8607** with average CPU inference latency of ~7.8 ms.
   - **Integration**: Integrated into [src/identify_what_matters/discussions.py](src/identify_what_matters/discussions.py) and [src/identify_what_matters/civic_classifier.py](src/identify_what_matters/civic_classifier.py) to tag candidate clusters with civic relevance scores.
   - **Deployment**: [artifacts/civic-models/civic-me5-base](artifacts/civic-models/civic-me5-base).

3. **Comment Stance & Tone Models**:
   - **Stance Cross-Encoder (`mDeBERTa-v3-base-xnli`)**: Fine-tuned on GPU 2 with enriched real debate discussions and calibrated threshold $\tau=0.275$. Deployed to [artifacts/stance-models/stance](artifacts/stance-models/stance).
   - **Tone 4-Class Classifier (`xlm-roberta-large`)**: Fine-tuned on GPU 3 with inverse class frequency loss weighting over 4,683 training rows. Deployed to [artifacts/stance-models/tone](artifacts/stance-models/tone). Achieves test Accuracy of **0.7564** and Macro-F1 of **0.7325**.

---

## Comment stance and tone cross-encoders, v2 (2026-10-04)

**Purpose.** Label every sampled comment of a debated topic locally, predicting which debate side it backs and its tone, instead of relying only on the 60 comments per topic that the LLM brief labels.

**Critical Flaws in v1:**
1. **Severe Regional Language Disparities:** In v1, Gujarati test macro-F1 was 0.435 (stance) and Punjabi tone macro-F1 was 0.211 due to underrepresentation of Indic regional debates.
2. **Sarcasm / Irony Vulnerability:** Sarcastic comments had macro-F1 of only 0.537 in stance (vs 0.739 non-sarcastic).
3. **Severe Tone Class Imbalance:** Critical comments constituted over 67% of comments, causing class collapse where models predicted 'critical' overwhelmingly (F1 on humorous was only 0.483).

**Enhancements in v2:**
1. **Regional & Sarcastic Data Augmentation:** Generated 10 targeted regional debate discussions across 7 regional languages (pa, gu, ta, mr, te, kn, bn) with high sarcastic comment density (33 sarcastic comments).
2. **Real Instagram Labels Expansion:** Extracted 21 real Instagram debate discussions with 310 real comments from 6 production runs.
3. **Inverse Class Frequency Loss Weighting:** Applied dynamic inverse class weights `[supportive: 1.333, critical: 0.447, humorous: 2.757, neutral: 1.535]` during Tone fine-tuning to counteract critical class bias.

**Data.** Train / Dev / Test splits now contain:
- Stance pairs: 9,514 train / 1,473 dev / 1,719 test
- Tone rows: 3,419 train / 525 dev / 622 test

**Benchmark Comparison on Held-Out Test Set:**

| Model | Stance Acc / Macro-F1 | Sarcastic Stance F1 | Regional Stance Highlights | Tone Acc / Macro-F1 | Sarcastic Tone F1 | Regional Tone Highlights |
| --- | --- | --- | --- | --- | --- | --- |
| **mDeBERTa-v3-base-xnli (v2 tuned)** | **0.717 / 0.698** | **0.565** (+0.028) | **mr: 1.000, te: 0.922, ta: 0.850, pa: 0.545, gu: 0.479** | 0.531 / 0.535 | 0.465 | te: 0.608, gu: 0.404 |
| **XLM-R large (v2 tuned)** | 0.491 / 0.490 | 0.355 | mr: 0.460, te: 0.438 | **0.772 / 0.748** (+0.040) | **0.564** (+0.078) | **gu: 1.000, te: 1.000, en: 0.803, hi: 0.751, mr: 0.733** |
| multilingual-e5-base (v2 tuned) | 0.652 / 0.646 | 0.504 | mr: 1.000, te: 0.852, ta: 0.778 | 0.703 / 0.686 | 0.557 | te: 1.000, gu: 0.683 |
| MuRIL (v2 tuned) | 0.605 / 0.596 | 0.402 | mr: 0.903, ta: 0.850, pa: 0.667 | 0.381 / 0.438 | 0.246 | gu: 0.333, mr: 0.321 |

**Selected Production Models:**
- **Stance Cross-Encoder:** `mDeBERTa-v3-base-xnli` deployed to [artifacts/stance-models/stance](artifacts/stance-models/stance) (calibrated threshold $\tau=0.225$). Preserves cross-lingual balance, excelling in Tamil (0.850), Telugu (0.922), and Marathi (1.000), while boosting sarcastic F1 to 0.565.
- **Tone Classifier:** `XLM-RoBERTa-large` deployed to [artifacts/stance-models/tone](artifacts/stance-models/tone). Achieves test Macro-F1 of **0.7481** and Accuracy of **0.7717**, excelling across Gujarati (1.000), Telugu (1.000), Hindi (0.751), Marathi (0.733), and English (0.803).

---

## Comment stance and tone cross-encoders, v1 (2026-10-03)

**Purpose.** Label every sampled comment of a debated topic locally, meaning which debate side it backs and its tone, instead of relying only on the 60 comments per topic that the LLM brief labels.

**Data.** `src/identify_what_matters/stance_corpus.py` builds the corpus from two sources:

- **Synthetic (Sonnet 4.6).** 136 calls produced 547k output tokens: 256 fictional contested discussions across 16 political and civic domains. Each discussion has 3 positions and 16 labelled comments: 5 Hinglish, 4 English, 3 Hindi, 3 in a rotating regional language, and 1 more. Each discussion includes at least 3 sarcastic comments, at least 2 that take no side, and 1 spam comment.
- **Real.** 216 comments from 15 debated topics in 3 past runs, labelled by the pipeline's own briefs.

Splits are by generation round or topic hash, so no discussion appears in two splits. This gives 8,936 / 1,497 / 1,584 stance pairs and 3,204 / 536 / 572 tone rows for train / dev / test.

**Training.** `src/identify_what_matters/stance_finetune.py` ran on the A100 VM, one GPU per base model, 3-5 epochs, in 1-4 minutes each. The stance task scores the pair (context + position, comment) and asks whether the comment backs that position. The none-threshold is selected on dev macro-F1.

| Model | Stance acc / F1 (test) | Stance real (58 comments) acc / F1 | Tone acc / F1 (test) | Tone real (60) acc / F1 |
| --- | --- | --- | --- | --- |
| mDeBERTa-xnli zero-shot (baseline) | 0.286 / 0.266 | 0.310 / 0.298 | - | - |
| **mDeBERTa-v3-base-xnli, tuned** | 0.738 / 0.727 | **0.655 / 0.697** | 0.701 / 0.602 | 0.583 / 0.494 |
| XLM-R large, tuned | **0.790 / 0.754** | 0.603 / 0.576 | 0.708 / 0.598 | 0.550 / 0.449 |
| multilingual-e5-base, tuned | 0.651 / 0.649 | 0.603 / 0.576 | **0.773 / 0.737** | **0.600 / 0.546** |
| MuRIL, tuned | 0.299 / 0.236 (did not converge) | - | 0.552 / 0.242 | - |

**Selected models.** The stance model is mDeBERTa: it is best on real comments and half the size of XLM-R large. The tone model is mE5-base. Both are saved in `artifacts/stance-models/{stance,tone}`.

**Status: opt-in, not the default.** On real comments the models agree with the LLM labels only 60-70% of the time, and the real test set has just 5 topics. The pipeline therefore keeps the LLM per-comment labels. The local models run only when `TRENDS_LOCAL_STANCE=1`.

**Promotion gate.** Make the local models the default only when real-comment F1 reaches at least 0.80 on at least 300 real comments. Every new run adds labelled comments (`brief.comment_pool` + `comment_labels`). Re-run `stance_corpus export-real`, then `build`, then `artifacts/stance_corpus/run_vm.sh`.

**Lessons:**

- The mDeBERTa-xnli hub checkpoint is stored in fp16 and diverged to NaN weights until it was loaded with fp32 master weights.
- MuRIL did not learn at these settings.
- Labels are LLM judgements, not human gold.

---

## v3: Real Instagram Labels Distilled From Pipeline Runs (2026-10-03)

Each trend run asks Sonnet which Instagram search and hashtag results are actually about each topic. `src/identify_what_matters/relevance.py` turns those judgements into real-data pairs: a topic anchor (title plus the top headline) and a scrubbed caption, labelled on-topic or off-topic. The data is split by topic hash, so no story appears in both train and test.

| Split | Pairs | On-topic | Topics |
| --- | --- | --- | --- |
| train | 874 | 449 | 37 (413 triplets: anchor, on-topic caption, off-topic caption from the same search) |
| dev | 191 | 45 | 11 |
| test | 229 | 130 | 10 |

The v3 encoder was trained on the VM from `multilingual-e5-base`, using the same recipe as v2: the 576-event Sonnet corpus plus the real triplets repeated 3 times, for 7,719 triplets in total. A second variant continued training the v2 checkpoint for one epoch.

| Model | Real test AUC | Real test F1 (dev threshold) | Gemini OOD ARI | Gemini OOD P@1 |
| --- | --- | --- | --- | --- |
| all-mpnet-base-v2 | 0.735 | 0.693 | 0.042 | 0.572 |
| v2 E5 (synthetic only) | 0.821 | 0.630 | 0.848 | 0.978 |
| **v3 E5 (synthetic + real)** | **0.905** | **0.738** | **0.874** | 0.978 |
| v2 continued on real data, 1 epoch | not downloaded | - | 0.838 | 0.978 |

**Decision:** v3 (`artifacts/topic-models/multilingual-e5-base-events-real`, cosine threshold 0.45) is now the default encoder in `encoder.py`. It improves relevance on real captions by 0.084 AUC without losing synthetic cross-lingual quality.

**Caveats:**

- The labels are LLM judgements, not human gold.
- The test set is 10 topics from two runs on one day, so the AUC interval is wide.

Re-export and retrain as runs accumulate. The metrics are in `artifacts/relevance_real/eval_local.json` and `artifacts/topic_training/v2/eval_ood_real.json`.

---

## v2: Multilingual Same-Event Embeddings (2026-10-03)

### What is being fine-tuned and why

The trend pipeline (`src/identify_what_matters/emergence.py`) groups posts and comments into discussion topics by cosine similarity. It then decides novelty by comparing each current cluster with a past-only baseline. Both steps fail if the encoder groups by language or script instead of by subject. Instagram India text mixes English, Devanagari Hindi, romanized Hinglish, and regional scripts, and comments are short, informal reactions.

Two local measurements set the training target:

1. **Tokenizer.** The `all-mpnet-base-v2` tokenizer cannot represent most Indic scripts. A Tamil phrase became all `[UNK]`. Hindi and Bengali became isolated consonants with dropped vowel signs. No amount of fine-tuning recovers discarded input.
2. **Similarity.** With base MPNet, a Hinglish heatwave comment was closer to a Hindi metro-fare caption (0.49) than to English heatwave captions (0.36-0.48).

The target is therefore an encoder where texts about the **same specific event** have high cosine similarity across English, Hindi, Hinglish, and regional languages, and across caption and comment styles. Texts about **different events in the same domain** must have lower similarity.

### Data: Sonnet 4.6 labelled event corpus

`src/identify_what_matters/topic_corpus.py` generates the corpus through the gateway helper described in the GPU VM guide (section 8), running locally.

- **Scope.** 18 discussion domains (news and civic, economy and cost of living, jobs, exams, technology and AI, sport, cinema, music, fashion and beauty, food prices, health, environment, transport and city life, travel and culture, gaming, agriculture, books and language, science and space) x 8 rounds.
- **Per call.** Each call covers one domain and produces 4 deliberately confusable fictional events with a distinguishing fact. Each event has 6 texts:
  - English caption;
  - Hindi caption in Devanagari;
  - Hinglish caption;
  - a regional-language caption, rotating across bn, ta, te, mr, gu, kn, ml, and pa;
  - a Hinglish viewer comment;
  - an English viewer comment.
- **Validation.** A deterministic validator checks slot order, length, script presence (and Latin-only text for en/hinglish), identifiers (no handles, hashtags, URLs, emails, or phone numbers), and duplicates. Rejected raw responses are kept in `rejected/` and salvaged if a relaxed validator accepts them.
- **Splits.** Rounds 0-5 are train, round 6 is dev, and round 7 is test. Splits are therefore event-disjoint and generation-call-disjoint, and every split covers all 18 domains.
- **Labels.** Positives are text pairs from the same event. Hard negatives are cross-event pairs within the same call (same domain, confusable). Easy negatives are cross-domain pairs.

### Corpus actually produced

| Item | Value |
| --- | --- |
| Sonnet calls | 157 (144 validated, 1 salvaged by the relaxed validator, 14 rejected or failed) |
| Tokens | 100,424 input / 392,757 output |
| Events / texts | 576 / 3,456 across 18 domains and 11 language labels (en, hi, hinglish, bn, ta, te, mr, gu, kn, ml, pa) |
| Train | 432 events, 2,592 texts, 6,480 triplets (anchor, positive, same-domain hard negative) |
| Dev / test | 72 events and 432 texts each, with 1,080 positive and 864 hard-negative pairs each |
| OOD test | 144 events and 864 texts written by **Gemini 2.5 Flash** in a deliberately noisy Instagram style (slang, typos, short reactions), with no event title reused from the Sonnet corpus |

The OOD set tests whether the tuned encoders learned Sonnet's writing habits rather than event identity.

### Training

`topic_finetune.py train` uses `MultipleNegativesRankingLoss` with one same-domain hard negative per pair, 3 epochs, batch 64, learning rate 2e-5, warmup 10%, bf16, and maximum length 128. A custom **event-disjoint batch sampler** ensures that no event appears twice in a batch, as anchor, positive, or negative, so a paraphrase of the anchor's own event is never pushed away as an in-batch negative. Four candidates trained in parallel, one per A100, in 50-65 s each.

### Evaluation protocol

Thresholds are chosen on the **dev** split only and frozen. For the OOD set, the dev split is still the Sonnet dev split. The metrics are:

- **Pair AUC:** same-event pairs against same-domain different-event pairs (hard negatives only).
- **F1:** at the dev-selected threshold.
- **P@1:** the nearest neighbour among all texts in the split, from all domains, is from the same event. The 95% CI comes from an event-level bootstrap.
- **Comment-to-caption P@1:** a comment's nearest caption belongs to the same event. This is what assigns comment discussion to topics.
- **ARI and V-measure:** average-linkage clustering at the dev-selected threshold, compared with the true events. This is exactly what `emergence.py` runs.

### Results: in-distribution test (Sonnet, 72 unseen events)

| Model | Pair AUC | F1 | P@1 [95% CI] | Comment-to-caption P@1 | ARI | V-measure | Clusters / events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all-mpnet-base-v2 (current) | 0.586 | 0.365 | 0.558 [0.516, 0.600] | 0.806 | 0.053 | 0.655 | 111 / 72 |
| paraphrase-multilingual-mpnet-base-v2 | 0.790 | 0.529 | 0.810 [0.773, 0.843] | 0.799 | 0.332 | 0.862 | 234 / 72 |
| multilingual-e5-base | 0.817 | 0.557 | 0.875 [0.840, 0.910] | 0.896 | 0.387 | 0.864 | 195 / 72 |
| LaBSE | 0.840 | 0.623 | 0.857 [0.831, 0.884] | 0.764 | 0.584 | 0.894 | 146 / 72 |
| all-mpnet, tuned | 0.784 | 0.603 | 0.755 [0.727, 0.782] | 0.972 | 0.523 | 0.877 | 139 / 72 |
| multilingual-mpnet, tuned | 0.978 | 0.883 | 0.984 [0.970, 0.995] | 0.972 | 0.932 | 0.980 | 94 / 72 |
| **multilingual-e5-base, tuned** | **0.980** | 0.878 | **0.988 [0.977, 0.998]** | **0.993** | **0.946** | **0.984** | 92 / 72 |
| LaBSE, tuned | 0.978 | 0.875 | 0.982 [0.968, 0.993] | 0.944 | 0.913 | 0.978 | 90 / 72 |

### Results: out-of-distribution test (Gemini, noisy style, 144 unseen events)

| Model | Pair AUC | P@1 | Comment-to-caption P@1 | ARI |
| --- | --- | --- | --- | --- |
| all-mpnet-base-v2 (current) | 0.605 | 0.572 | 0.858 | 0.042 |
| paraphrase-multilingual-mpnet-base-v2 | 0.888 | 0.854 | 0.785 | 0.502 |
| multilingual-e5-base | 0.892 | 0.918 | 0.892 | 0.641 |
| LaBSE | 0.890 | 0.876 | 0.799 | 0.602 |
| all-mpnet, tuned | 0.737 | 0.720 | 0.931 | 0.238 |
| multilingual-mpnet, tuned | 0.972 | 0.958 | 0.927 | 0.803 |
| **multilingual-e5-base, tuned** | **0.986** | **0.978** | **0.944** | **0.848** |
| LaBSE, tuned | 0.981 | 0.961 | 0.906 | 0.776 |

Cross-lingual AUC of English against each language on the in-distribution test:

- **Base MPNet:** about chance for most Indic scripts (en-bn 0.51, en-kn 0.49, en-ta 0.52, en-hi 0.55). Fine-tuning MPNet barely moves these (en-kn 0.47, en-ta 0.52), which confirms the tokenizer ceiling.
- **Tuned E5:** 0.94 (ta) to 1.00 (mr), with en-hinglish at 0.97.

### Decision

**Selected encoder: `multilingual-e5-base` fine-tuned on the Sonnet corpus.** It is stored at `artifacts/topic-models/multilingual-e5-base-events`. Its `topic_threshold.json` has a cosine threshold of 0.425, selected on dev, and the `query: ` input prefix, which `emergence.py` and `main.py` apply automatically.

Reasons:

1. It was best or tied for best on every metric in both test sets.
2. Its OOD margin over the other tuned models is largest on clustering (ARI 0.848 against 0.803 and 0.776), the operation the pipeline depends on.
3. Its base model was already the strongest untuned multilingual baseline.

Under in-distribution test, the bootstrap confidence intervals of the three tuned multilingual models overlap. Its superiority there is not established; the OOD result is the tiebreaker.

Fine-tuning adds about +0.56 ARI in-distribution and +0.21 ARI OOD over base E5. Against the original MPNet setup, it adds +0.81 ARI OOD.

### What these numbers do and do not show

- Both test sets are LLM-written fictional events. They measure whether the encoder groups paraphrases of one event across language, script, and caption or comment style, and separates confusable events in the same domain. They do not measure performance on real Instagram text with real noise:
  - spam, giveaways, and hashtag walls;
  - emoji-only comments;
  - multiple topics per post.
- Clustering on real data will be worse. Before relying on alerts, label 200-300 real posts and comments from the first collection runs into events and re-run `topic_finetune.py evaluate`, as described in Literature Review section 7.
- Per-language test slices are small (6-12 texts per regional language per split), so per-language numbers are indicative only.
- The data is not human gold. No real user content was sent to an LLM.

Artifacts:

- `artifacts/topic_training/v2/eval_v2.json` and `eval_ood_gemini.json`: full metrics, per-language and per-style slices, and thresholds.
- `artifacts/topic_corpus_v2/` and `artifacts/topic_corpus_ood_gemini/`: raw calls, rejected responses, generation state, and built datasets.
- VM working directory: `/home/s0r0qw6/instagram-topic-v2` (all 4 tuned checkpoints).

---

# Appendix: v1 Same-Event Topic Embedding Pilot (superseded)

Report date: 2026-10-03. Machine-recorded UTC timestamps begin on
2026-10-02; the run ID preserves that clock rather than rewriting it.

## Outcome

**PARTIAL DATA; NOT TRAINED; NOT PROMOTABLE.** Actual Sonnet generation ran
on the GPU VM, but the bounded budget did not produce the required corpus.
No GPU optimizer steps ran, no model checkpoint was created, and no base or
tuned evaluation scores are available. The downloaded batch is a data
checkpoint, not a trained model.

| Measurement | Actual result |
| --- | --- |
| Remote run ID | topic-20261002T194847Z |
| Generator | claude-sonnet-4-6, version default |
| Actual gateway calls | 7; one attempt per call, no helper retries |
| Accepted / rejected responses | 1 / 6 |
| Validated synthetic event groups | 8 of required 48 |
| Paraphrases | 32; four per event |
| Domains | 2 of required 12: public transport and agriculture |
| Language labels | en, hi, hinglish, bn, ta |
| Explicit Sonnet hard-negative labels | 24 of planned 144 |
| Reported input / output tokens | 2,563 / 38,659 |
| Reported total tokens | 41,222 |
| Conservatively reserved tokens | 59,256 of 60,000 |
| Initial maximum calls | 8; actual run stopped at 7 |
| Train / dev / test pairs from actual data | Not emitted; corpus incomplete |
| GPU training runs | 0 |
| ROC AUC, cosine separation, threshold F1 | Not measured |
| Human real-data evaluation | Not available |
| Promotion | Refused |

## Concrete Blockers

1. The workspace environment initially lacked `python-dotenv` and the
   helper's deferred signer dependency `cryptography`. Installing
   `python-dotenv` locally succeeded. Installing `cryptography` failed
   because DNS resolution for `pypi.org` failed. No project manifest or
   shared VM environment was modified.
2. Local dependency failures occurred before HTTP requests. The initial
   journal recorded six import failures; a later local journal reserved
   seven attempts that failed before HTTP signing. These are not additional
   successful Sonnet calls. Preflight now checks helper and signer imports
   before reserving calls.
3. The VM's installed environment could call Sonnet. Six responses were
   rejected with `JSONDecodeError`. Their raw bodies were not retained, so
   whether they contained fences, prose, or other malformed JSON is unknown.
   After explicitly forbidding Markdown fences and requiring an opening
   and closing JSON object, call seven produced one strictly valid batch.
   That response had 13,711 characters and no leading Markdown fence.
4. Only 744 reserved tokens remain; another full batch needs more than
   8,000. The code did not reset the journal or exceed the initial budget
   to complete the other five batches. More generation requires a new,
   explicitly authorized budget. Partial exports remain resumable.

The available real sample is old and fashion-biased. It was not sent to
Sonnet or used for supervision. Neither that sample nor these eight
fictional event groups constitute sufficient representative real data.

## Files

- `src/identify_what_matters/topic_training.py`: standalone CLI with `generate`,
  `train`, and `evaluate`; heavy ML and helper imports are lazy.
- `data/topic_labels_pilot_20261003.jsonl`: eight validated synthetic event
  groups, containing 32 texts and provenance/language metadata.
- `data/topic_labels_pilot_20261003_negatives.jsonl`: 24 explicit negative
  pair labels with event IDs, domain, languages, provenance, and reasons.
- `artifacts/topic_training/remote-pilot-20261003/batch_0.json`: accepted
  response checkpoint and run metadata.
- `artifacts/topic_training/remote-pilot-20261003/generation_state.json`:
  budget, usage, failure categories, and content-free format diagnostics.
- `artifacts/topic_training/remote-pilot-20261003/incomplete_summary.json`:
  incomplete corpus counts and `training_ready: false`.
- `artifacts/topic_training/remote-pilot-20261003/run_result.json`: verified
  counts; training checkpoint and evaluation metrics are explicitly null.
- `artifacts/topic_training/runtime_checks/`: constructed runtime fixtures,
  **not Sonnet output, not gold, and never used to train a model**.
- Other directories under `artifacts/topic_training/` preserve local failed
  attempt journals or bounded runtime-check outputs. Empty intermediate
  label files are not usable training corpora.

## Implemented Protocol

The complete generation target is 48 event groups across public transport,
agriculture, education, public health, cricket, cinema, technology,
environment, food, music, space science, and civic services. Each event
has English, native-script Hindi, Hinglish, and one regional paraphrase;
regional coverage cycles through Bengali, Tamil, Telugu, Marathi, Gujarati,
and Kannada. These are model-generated labels, not independently verified
translations or human event annotations.

Every domain has four distinguishable events. Sonnet must explicitly mark
all six different-event combinations as hard negatives, with two distinct
paraphrase pairs per combination: 144 explicit negative labels overall.
Positive labels derive from Sonnet-generated same-event group membership;
they are not a second independent adjudication pass. Different events in
the same category are negatives, not merely cross-category contrasts.

Before pairing, deterministic seed 20261003 assigns two events per domain
to train. The remaining two events go to dev for alternating domains and
test for the others. Thus train covers all 12 domains; dev and test each
cover six. This grouping preserves within-domain hard negatives while
making events disjoint. It is not a balanced all-domain evaluation split.
Cross-split event edges are discarded, never reintroduced as pairs.
Complete runtime fixtures yielded 336 train, 168 dev, and 168 test pairs:
144/192 train positives/negatives and 72/96 in each holdout. These counts
describe verified pairing logic, **not the actual incomplete Sonnet corpus**.

Training uses `CosineSimilarityLoss` on explicit labels 1 and 0, not
MultipleNegativesRankingLoss. It requires exactly one visible CUDA GPU,
batch size 8 or 16, one epoch, learning rate 2e-5, and maximum length 128.
An existing checkpoint output directory is rejected instead of overwritten.

Evaluation encodes only independent dev/test texts for both base and tuned
models. Each model selects its own F1 threshold on dev, freezes it for test,
and reports ROC AUC, positive/negative cosine means, separation, precision,
recall, and F1. Language slices contain pairs involving that language and
overlap; metrics require at least five positives and five negatives.
The heuristic generalization gate requires tuned test F1 within 0.05 of
tuned dev F1 and no regression from base test F1 or AUC. Even passing that
gate cannot permit promotion without human-labeled representative real
evaluation. Pair scores are event-correlated, not independent observations.

## VM Evidence And Commands

IAP SSH connected to `llm1-4a100-80g`, project `wmt-et-converse-dev`, zone
`us-central1-a`. All four A100-SXM4-80GB GPUs were idle at the initial probe
(1 MiB used each, 0% utilization, no compute processes). No other job was
stopped, no training was detached, and no shared dependencies were upgraded.

Existing VM interpreter: `/home/s0r0qw6/model-py-java/.venv/bin/python`.
It reports PyTorch 2.6.0+cu124, sentence-transformers 5.7.0, CUDA available,
and compatible `SentenceTransformer.fit` and `CosineSimilarityLoss` APIs.
Accelerate, datasets, sklearn, and helper dependencies are installed.
The cached base `/home/s0r0qw6/model-py-java/artifacts/all-mpnet-base-v2`
is English-focused; no multilingual E5 artifact appeared in the targeted
top-level probe. Even a completed MPNet pilot would not establish native
Indian-language suitability.

Only the new CLI was uploaded, to the isolated path
`/home/s0r0qw6/instagram-topic-pilot-20261003`. Generation outputs were
created there and downloaded with IAP `gcloud compute scp`. No
`model-py-java` files were overwritten.

The bounded remote generation command was:

```sh
cd /home/s0r0qw6/instagram-topic-pilot-20261003
/home/s0r0qw6/model-py-java/.venv/bin/python topic_training.py generate \
  --helper-dir /home/s0r0qw6/model-py-java \
  --output corpus --labels-prefix topic_labels_pilot_20261003 \
  --max-calls 7 --token-cap 60000
```

The following commands are **not executed** and require a complete,
validated corpus and a fresh check that GPU 0 is still unused. Otherwise
select another verified free GPU or defer; do not interrupt existing work.

```sh
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/s0r0qw6/model-py-java/.venv/bin/python topic_training.py train \
  --dataset corpus \
  --base-model /home/s0r0qw6/model-py-java/artifacts/all-mpnet-base-v2 \
  --output checkpoint --batch-size 8 --epochs 1

CUDA_VISIBLE_DEVICES=0 \
  /home/s0r0qw6/model-py-java/.venv/bin/python topic_training.py evaluate \
  --dataset corpus \
  --base-model /home/s0r0qw6/model-py-java/artifacts/all-mpnet-base-v2 \
  --tuned-model checkpoint --device cuda --output evaluation.json
```

## Verification And Privacy

Focused runtime checks passed for sanitizer behavior, schema rejection,
hard-negative edge coverage, event/text-disjoint splits, strict format
instructions, budget-exhausted offline export, module syntax, and the
downloaded actual Sonnet batch. Editor diagnostics reported no errors.
No tests were created. GPU training and numerical evaluation remain
unverified because no complete training-ready corpus exists.

Only synthetic, sanitized task instructions were submitted. The sanitizer
removes handles, hashtags, URLs, emails, long phone-like numeric strings,
PAN-like IDs, registration-like IDs, UUIDs, and common labeled identifiers.
Generated texts/reasons containing those patterns are rejected. The prompt
also excludes personal names and addresses; regex checks are not a general
PII detector. Credentials are loaded only by the imported helper. Prompts,
credential values, raw gateway errors, and rejected response bodies are
not logged. No live Instagram call or browser operation occurred.