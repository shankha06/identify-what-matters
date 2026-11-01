# Party Model Audit and Root-Cause Analysis

Scope: the 17-party mDeBERTa-v3 classifier (`artifacts/demographics_models/bert-party-support-17`), its training data, and the trainers that produce it. Read-only audit. No production artifact was changed and no model was retrained.

## 1. Summary

- The reported 17-party validation scores (accuracy 0.8755, macro-F1 0.8802) do not hold on real held-out text. On 188 real-world gold rows that were not in the training pool, accuracy is 0.537 and macro-F1 is 0.606.
- The "100% gold accuracy" in `finetuning_v2_results.json` is contaminated: gold rows were used for best-checkpoint selection, and gold text overlaps the training pool.
- Root causes, in order of impact: (1) the training distribution is synthetic and does not match real posts; (2) label semantics are noisy (topic and opposition cues are labelled as party support); (3) the "None" class is over-represented and absorbs real civic posts; (4) the validation split is optimistic because of duplicates and gold leakage; (5) several classes have very few real examples.
- CJP (Cockroach Janta Party) has been added to the label definitions and detection code. No CJP training rows exist yet, so the production model cannot predict CJP. A retrain is required and was not run.

## 2. Measured results

| Evaluation set | Rows used | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| Production 17-party model, `model_config.json` (val) | 6527 | 0.8755 | 0.8802 | Split method not verified; duplicates exist in the training pool (849 extra rows) |
| Production model, `real_world_gold_standard_v2` (held-out) | 188 (4 leaked rows removed) | 0.537 | 0.606 | Real posts; 86 of 188 are None |
| Production model, `gold_standard_benchmark` | 48 (4 leaked rows removed) | 1.000 | 1.000 | Used for checkpoint selection, so not a clean held-out |
| 7-class `bert-party-support` (`model_config.json`) | 4854 | 0.7613 | 0.7638 | `docs/ARCHITECTURE.md` claims 0.8203; metrics conflict |

Per-class recall on the real-world held-out set (188 rows), from the eval run:

- Strong: NCP (2/2), RJD (2/2), JDU (2/2), BRS (2/2), BSP (2/2), AIADMK (2/2), Shiv Sena (2/3), TDP (2/3). These are tiny supports, so the numbers are unstable.
- Weak: BJP (7/19 recall), Congress (9/21), AAP (4/15), TMC (4/9), DMK (3/10), None (51/86).
- Precision problems: CPIM predicted 16 times, with 1 correct. TMC precision 0.27. YSRCP and CPIM each have support of 1.

Confusion highlights on real data:

- None to CPIM (10), None to Congress (8), None to TMC (6), None to AAP/BJP (3 each). Non-political civic and lifestyle posts are pushed into parties.
- DMK to None (5 of 10). AAP to None (3 of 15).
- Congress to BJP (4), Congress to AAP (2).

## 3. Root-cause analysis

### RC1. Training data is mostly synthetic, real text is a small fraction (high impact)

- `party_train_17_all.jsonl` (6177 rows): 3043 `synthetic_regional`, 1049 `synthetic`, 1006 `anchor`, 531 `lean`, 409 `gold`, 120 `curated_v2`, 19 `feedback_curated`. Only 409 rows are real gold text.
- Synthetic rows are generated from prompts by a language model. Their style is clean and political. Real posts are noisy, code-mixed, and often apolitical. The model learns the generator's style, not the party signal.
- Evidence: 0.8755 on the validation set versus 0.537 on real posts. The validation composition was not separately verified, so this is a strong signal, not a proven cause.

### RC2. Label semantics are inconsistent (high impact)

- Seed rows in `src/identify_what_matters/demographics_trainer.py` assign party labels from topic or opposition cues. Examples: "delhi rent has doubled" becomes AAP; "nifty at all time high" becomes BJP. A complaint about governance is not a statement of party support.
- The model therefore learns topic-to-party shortcuts. On real text, a civic complaint about a park or pension portal receives a party label (for example, a Delhi park complaint predicted as AAP, a grievance predicted as CPIM).
- Candidate label noise to verify manually: `party_train_all.jsonl` line 409 (Shiv Sena UBT labelled BJP), line 1228 (Shiv-flavoured text labelled None), line 189 (labelled AAP).

### RC3. The "None" class is both over-represented and ill-defined (high impact)

- `synthetic_dataset.jsonl` (22402 rows): None 10832 (48%). `synthetic_demographics_balanced_v2.jsonl`: None 4052 of 7955 (51%).
- None is defined as "no party support". Real-world gold None rows include many civic complaints and lifestyle posts. The model treats any civic or governance text as partisan (None recall 0.59 on real data).

### RC4. Class imbalance and missing classes (medium-high impact)

- Synthetic set: TDP 40 rows, Shiv Sena 34, TMC 728, SP 557, DMK 416. Compared with BJP 4049 and Congress 2682.
- The synthetic demographics dataset includes an `Other` class (1717 rows) that the 17-party set does not have. Its `metadata.json` lists 8 party classes including `Other`, while the 17-party model has none.
- Per-class F1 on real data for TDP, Shiv Sena, NCP, RJD, JDU is unreliable because the supports are 2 to 3 rows.
- Iteration logs show accuracy rising from 0.53 (phase 1) to 0.88 (phase 4). Whether this came from weighting, sharpening, or data was not isolated.

### RC5. Evaluation is contaminated and optimistic (high impact; process)

- Gold leakage in training data: all 904 gold rows in the root `party_train_all.jsonl` are the same texts as `party_gold_corpus/gold_samples.jsonl`. 409 texts in `party_train_17_all.jsonl` match `gold_samples.jsonl`. 4 each match `gold_standard_benchmark.jsonl` and `real_world_gold_standard_v2.jsonl`.
- Checkpoint selection used gold: `train_17_party_vm_v2.py` chose the best checkpoint on `gold_standard_benchmark.jsonl`. The reported best gold accuracy of 1.0 is therefore not a held-out result.
- Duplicates: `party_train_17_all.jsonl` has 849 duplicate extra rows, the root set has 603, and synthetic has 18. The split method was not verified; if identical texts fall on both sides of the split, validation is inflated.
- Metric discrepancies: 7-class 0.7613 (model_config) versus 0.8203 (ARCHITECTURE.md). `docs/TRAINING_RESULTS.md` also contains metrics that do not match artifacts.

### RC6. Text quality (medium impact)

- Root `party_train_all.jsonl`: 306 rows under 15 characters; minimum length 2. Short texts carry almost no signal and add noise.
- Median length 149 characters; the set is short-form, which is the right shape for social posts, but the short rows should be filtered or reviewed.

### RC7. Calibration (medium impact)

- On real data, 15 of 188 predictions have confidence below 0.5, and many wrong predictions are confident (for example, 0.95 to 0.99 on None rows predicted as a party). Softmax confidence is not a usable abstention signal yet.

### RC8. Domain shift (structural)

- Real posts use code-mixed Hinglish, English-only phrasing, and sarcasm. Synthetic generation does not reproduce this distribution. A single synthetic-to-real gap explains most of the 0.88 to 0.54 drop.

## 4. CJP (Cockroach Janta Party) expansion

Decisions (autonomous, because the user was unavailable; please confirm):

- Expansion is labelled "Cockroach Janta Party". Confirmed by the user as the official name. `README.md` sample brief corrected from "Citizens for Justice and Peace" to match.
- Scope: code and data preparation only. No VM retraining. Production artifact `bert-party-support-17` is unchanged.

Changes made:

- `src/identify_what_matters/party_lean_generator.py`: `PARTIES` includes CJP; `PARTY_SCENARIOS["CJP"]` (5 scenarios, drafted and needs review); `AGE_DIST["CJP"]`; prompt context. Scenarios reference the Bengaluru and Mumbai protests already present in the repo corpus.
- 17-class lists in `train_17_party_vm_v2.py`, `train_17_party_vm.py`, `train_iteration_17_party.py`, `run_4_iterations.py`, `scripts/eval_vm_gold.py`: CJP inserted before None.
- 7-class lists in `train_eval_party_bert.py` and `train_party_bert_remote.py`: CJP added.
- `train_17_party_vm_v2.py` ANCHORS: "cockroach janta party", "cockroachjantaparty", "cockroachisback" to CJP.
- `src/identify_what_matters/political_classifier.py`: `ENTITY_TARGET_PATTERNS["CJP"]`; `IS_POLITICAL_RE` now includes "cockroach janta party".
- `src/identify_what_matters/political_corpus.py`, `make_full_dashboard.py`, `src/identify_what_matters/political_pulse.py` (`_OPP_ALLIED`): CJP added.
- Tests: `tests/test_cjp_party.py` (3 tests, passing). The bare acronym "CJP" is intentionally not matched, because it is ambiguous.

Important caveats:

- The production 17-class model has no CJP output. Until retrained with CJP data, CJP posts will be predicted as some other class (typically None or a topic-adjacent party).
- Inference labels come from `config.id2label`, so the production model keeps its 17 labels. Changing label lists in training scripts does not change the running model.
- The `demographics.py` and `political_pulse.py` fallback lists (`idx2party` with 7 classes) remain unchanged on purpose. They only apply when `id2label` is missing.
- Held-out gold must stay free of CJP rows until a clean CJP evaluation set exists.

## 5. Improvement roadmap

Priority order. Items 1 to 3 are data and evaluation hygiene and should happen before any retrain.

1. Clean evaluation. Build one held-out real-world test set, grouped by near-duplicate text, that is never used for checkpoint selection or training. Use `real_world_gold_standard_v2` minus its 4 leaked rows (188 rows) as the starting point. Report per-class support and confidence intervals, because most classes have fewer than 20 rows.
2. Deduplicate and group-split. Remove exact duplicates (603, 849, 18). Group-split near-duplicates before creating train and val. Keep a `source` tag on every row to detect leakage.
3. Fix label definitions. Re-label seed rows so party label means support or endorsement, not topic. Add an explicit abstain or "Other" policy. Review the candidate noisy rows listed in RC2.
4. Real-data expansion. Collect or label more real posts for TDP, Shiv Sena, NCP, RJD, JDU, AIADMK, BRS, BSP, YSRCP, CPIM, and CJP. Use hard negatives: civic complaints and lifestyle posts labelled None.
5. Teacher-student distillation. Use Sonnet 4.6 (gateway, see `synthetic-model-training/docs/calling_sonnet_4_6.md`) or Muse Glimmer 30B on the VM via vLLM as a teacher to label unlabelled real posts. Train the mDeBERTa student on teacher labels plus verified gold. Measure the student against the clean held-out set only.
6. Reinforcement learning with verifiable rewards (RLVR). Use exact-match party labels on verified gold as the reward for a generative labeller. Keep the reward set separate from the eval set.
7. Active learning and hard-negative mining. Use low-confidence and high-disagreement rows (for example, the 15 low-confidence real predictions) to select the next labelling batch.
8. Calibration. Fit temperature scaling on the clean validation set, then add a per-class threshold or an abstain option for low-confidence predictions.
9. Documentation. Correct `docs/TRAINING_RESULTS.md` and `docs/ARCHITECTURE.md` to use measured held-out numbers, and label synthetic-validation numbers as such.

## 6. Reproducing the held-out eval

- Local `uv` cannot reach pypi.org from this machine (DNS failure), so the eval ran on the GPU VM in CPU mode.
- Inputs were copied to `/tmp/party_eval/` on the VM. Model: `bert-party-support-17`. Data: `party_train_17_all.jsonl`, `real_world_gold_standard_v2.jsonl`, `gold_standard_benchmark.jsonl`. Script: the eval script with `ROOT` set to `/tmp/party_eval`.
- Command on the VM: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/s0r0qw6/model-py-java/.venv/bin/python eval.py`. Do not set `CUDA_VISIBLE_DEVICES=` (empty), because it breaks the transformers import.
- The GPU was already in use (about 57 GB) by another process, so the eval avoided it.

## 7. Teacher-student run (Muse Glimmer 30B, v3 candidate)

Status: candidate only. Production `bert-party-support-17` was not replaced.

Steps completed:
- Teacher: Muse Glimmer 30B served by vLLM on the VM (GPU2). Low reasoning, final-channel parsing, strict whole-answer label match. Greedy decoding.
- Teacher on the frozen 188 rows: accuracy 0.6702, macro-F1 0.7633, unparsed 0. Caveat: this set was inspected while debugging the teacher output format, so it is not a clean estimate.
- Real pool: 1436 unlabeled texts (reddit titles and comments, Instagram captions). Teacher labels: 1435 parsed, 1 unparsed. Label counts: None 1250, BJP 109, Congress 35, BRS 13, TMC 11, NCP 6, DMK 4, CPIM 3, BSP 2, AAP 1, TDP 1.
- Training set `train_v3.jsonl`: 6763 rows (5328 from `party_train_17_all`, 1435 teacher-labelled real). 849 exact duplicates removed. Frozen texts excluded.
- Student: `microsoft/mdeberta-v3-base`, fp32 (transformers 5.x loads this checkpoint as fp16 by default, which gave NaN loss). Internal 10% monitor split from the training data only.

Results on the frozen 188 (baseline is production `bert-party-support-17`):

| Run | Config | Frozen acc | Frozen macro-F1 |
|---|---|---|---|
| Production baseline | n/a | 0.5372 | 0.6056 |
| v3 | 3 epochs, lr 2e-5 | 0.3777 | 0.2078 |
| v3b | 10 epochs, lr 3e-5 | 0.5266 | 0.6394 |

Diagnosis of v3: underfit. Training loss was still falling at epoch 3. Congress recall was 0 (0 of 21 correct; only 1 Congress prediction in total). Monitor accuracy rose from 0.356 to 0.541 across epochs.

Caveats for v3b:
- The 10-epoch, lr 3e-5 setting was chosen after the v3 frozen result was seen. The frozen set is therefore no longer a clean estimate for v3b. Selection used the monitor trend only (monitor acc 0.858, macro-F1 0.867 at epoch 10), but the choice was informed by frozen results.
- The 2-row accuracy gap (0.5266 vs 0.5372) is within noise for 188 rows. The macro-F1 gain (+0.034) rests on small classes with 1 to 3 gold rows.
- Per-class: BSP precision 0.14 (14 predicted, 2 gold). BSP has only 2 teacher labels in the pool, so the cause is unclear and may be a sparse-class effect (not verified). Congress F1 0.23. None recall 0.60 (precision 0.81), so None-vs-party confusion is still the largest error source.

Recommendation: do not replace production on this evidence. Next steps that need a clean test:
1. Build a new held-out set that is never used for training or config choice.
2. Add more teacher labels for the small parties (BSP, TDP, NCP, RJD, JDU, CPIM, BRS, Shiv Sena, AIADMK).
3. Decide the None policy (RC3) before the next retrain.

## 8. v4: 18 labels (CJP added), smaller students

Status: candidates only. Production `bert-party-support-17` was not replaced.

Label convention (confirmed by the user): a party label means the post takes a side. Criticism with no side is `None`.

Steps completed:
- Teacher: Muse Glimmer 30B, 18 labels (CJP added as "a youth-led anti-establishment movement").
- Real pool relabel: 1436 texts, 1436 parsed. Labels: None 1237, BJP 99, Congress 34, CJP 23, BRS 13, TMC 12, NCP 7, CPIM 3, DMK 3, BSP 2, AAP 1, SP 1, Shiv Sena 1. Sparse classes have almost no real rows.
- CJP synthesis: 794 unique posts generated by Muse. Teacher agreed CJP on 474 of 794 (59.7%). Kept 474. Teacher labels for the rejected 320 were None.
- Dataset (`scripts/party_v4_build_dataset.py`): train 6929 (5328 from `party_train_17_all`, 1222 real, 379 CJP synthetic). Silver 309 (214 real, 95 CJP). Frozen overlap 0. Frozen texts excluded from every source.
- Silver is a weak selection set. It is 86% None plus CJP. Most party classes have 0 to 15 silver rows. Treat silver macro-F1 as a rough signal only.
- Frozen 188 has no CJP rows, so CJP is untested on the frozen set.

Results on frozen 188 (baseline is production `bert-party-support-17`):

| Run | Params | Config | Frozen acc | Frozen macro-F1 | Notes |
|---|---|---|---|---|---|
| Production | about 280M | n/a | 0.5372 | 0.6056 | reference |
| e5-small fine-tune | 117.7M | 6 epochs, lr 5e-5, sqrt class weight, selected epoch 5 | 0.5372 | 0.5046 | batch-1 GPU latency 8.9 ms, peak train 2.6 GB |
| mDeBERTa-v3-base reference | 278.8M | 6 epochs, same data and recipe, selected epoch 4 | 0.5160 | 0.5546 | same data, same recipe |
| e5-small embeddings plus LR | 117.7M (frozen) | C=10 selected on silver | 0.4681 | 0.5327 | no fine-tuning |
| Multilingual-MiniLM-L12 fine-tune | small | same recipe, selected epoch 6 | 0.4149 | 0.3989 | underfit at 6 epochs |

Interpretation:
- The mDeBERTa reference on the v4 data scores below production. The gap is therefore not caused by model size alone. Data (teacher labels, CJP synthetic, silver-based selection) and the training recipe matter at least as much.
- e5-small matches production accuracy at about 40% of its parameters. Its macro-F1 is lower.
- The frozen set has 188 rows and many classes with 1 to 3 gold rows. One row changes accuracy by about 0.5 points. Differences of this size are noise.
- Frozen was used for earlier selection in v3 and v3b. Treat it as a reporting set only from this point. It is not clean for any selection.

Not yet done (planned): None-bias or per-class threshold tuned on silver, ensembling, checkpoint averaging. These need the trainer to save logits.

Recommendation: do not replace production. The next step with the best expected value is a clean labelled eval set with more rows for the sparse parties, followed by a larger silver set. Without that, small-model comparisons cannot separate real differences from noise.

## 9. v5: distillation, bigger student, and RLCD/RLVR research

Status: in progress. Production `bert-party-support-17` was not replaced. Frozen 188 is contaminated (used for selection in v3 and v3b) and is reporting-only.

Progress:
- Done: `scripts/party_v5_soft_label.py`. Reads the teacher's top-20 logprobs at the first answer token and maps them to the 18 labels. Test on 12 rows: the soft argmax matched the strict teacher label on 12/12.
- Finding: the teacher's first answer token is near-deterministic (chosen token probability about 1.0). Soft targets are therefore almost one-hot and add little beyond hard labels. Ambiguous first-token groups (B for BJP/BRS/BSP, C for Congress/CPIM/CJP, S for SP/Shiv Sena) are resolved to the teacher's strict answer, so within-group confusion is not captured.
- Done: `scripts/party_v5_train.py`. Loss = alpha times KL(teacher soft, student) plus (1 minus alpha) times CE(original label), with label smoothing 0.1 and sqrt class weight. alpha=0 reproduces the hard-label baseline. Checkpoint selection on silver; frozen scored once.
- Running: teacher labelling of all 6929 train rows, including the 5328 original-label rows, on GPU2. Output: `train_v5_soft.jsonl`.
- Staged: xlm-roberta-large (560M, 4.3 GB) in the VM Hugging Face cache as the bigger student.
- Not started: v5 training runs. Planned: xlm-roberta-large with alpha 0.7 against alpha 0; e5-small with alpha 0.7 against the v4 hard baseline (macro-F1 0.5046).

Research: RLVR and RLCD

Sources:
- RLVR, DeepSeek-R1 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)). Pure RL with rewards from a rule-based verifier on math, code, and STEM. The abstract states that reasoning patterns of large models can be used to guide smaller models.
- RLVR debate ([arXiv:2506.14245](https://arxiv.org/abs/2506.14245)). Reports that RLVR extends the reasoning boundary on math and code under CoT-Pass@K. Notes the open question of whether RLVR improves reasoning or only sampling efficiency.
- RLCD ([arXiv:2307.12950](https://arxiv.org/abs/2307.12950), ICLR 2024). Builds preference pairs from two outputs generated under a positive and a negative prompt, with no human labels. Reported to beat RLAIF and context distillation at 7B and 30B simulation scales.

Can RLVR improve performance here?
- Possibly, for a generative teacher or student. It is unlikely to help the current encoder classifier. For an encoder, a 0/1 label reward carries the same information as CE on the same labels.
- The verifier here is label match, and the gold is weak: 309 silver rows, 86% None, most parties under 15 rows. A reward built on this gold can be exploited, for example by collapsing to None. Label quality is the limit, not the optimiser.
- Required order: a clean labelled eval set first, then RLVR on a generative model.

How RLCD could carry more context to the student:
- Hypothesis: the teacher's logprobs are near one-hot, so they give no uncertainty signal by default. Contrast prompting can recover it. Query the teacher with a positive prompt (which party does the author take a side for?) and a negative prompt (which party does the author attack?). The difference between the two answers separates support from attack, which is the axis the label convention depends on.
- Transfer: train the student on the contrast pair, either as a preference pair or as a target distribution built from both answers. Cost: about two teacher passes. The estimated time is about one hour per pass for 6929 rows, based on the observed throughput.
- Not tested. The expected benefit is largest on criticism texts labelled None versus texts labelled with a party.

RLCD test result (`scripts/party_v6_rlcd_contrast.py`, inference-time contrast on the Muse teacher):
- Positive prompt is the current teacher prompt. Negative prompt asks which party the author attacks. Contrast score = log(pos) minus lam times log(neg).
- Silver (309, selection set): positive prompt acc 0.9903, macro-F1 0.8648. Contrast at lam 0 to 0.75 is identical. Lam 1.0 drops to acc 0.8188. Silver is 86% None and does not separate methods.
- Frozen 188 (reporting only; contaminated by earlier selection): positive prompt acc 0.6702, macro-F1 0.7633, matching the earlier teacher result. Contrast at lam 0 to 0.75 is identical. Lam 1.0 drops to acc 0.5053.
- Contrast changed 0 of 188 frozen predictions at lam 0.5. The positive first-token distribution is saturated, so the negative term does not move the argmax.
- All 62 frozen errors are gold party, predicted None. There are 102 party-labelled frozen rows, so the teacher misses 61 percent of them by answering None. No error is a party-to-party confusion. This is a recall problem on texts that support a party. The attack prompt cannot address it.
- Scope: this tests RLCD-style contrast at inference time only. It does not run RLCD training (preference model plus RL) on the teacher.

Conclusion: RLCD-style contrast does not improve the teacher. The hypothesis above is rejected for this teacher and prompt setup.

Decision: RLCD is not pursued further as a teacher improvement. Open question: whether frozen gold uses a looser support convention than the strict "side taken" rule in the teacher prompt. Resolve that before any further teacher work. Production `bert-party-support-17` is unchanged.

Convention check (6 of the 62 missed frozen rows, read directly):
- The gold labels follow the user-confirmed rule ("the post takes a side"). Several gold party labels rest on implicit side-taking: a Congress leader criticising government policy (labelled Congress), sarcasm about a government scheme (labelled AAP), and a mocking slogan post (labelled Congress or BJP).
- So the 62 misses are not just noise. The strict teacher rule misses implicit side-taking. Some gold labels are still ambiguous, so the convention needs a written decision before any RL reward is built.

RLCD v7 (`scripts/party_v7_rlcd_side.py`): convention-axis contrast. Positive prompt = lenient "takes a side, including implicit". Negative prompt = strict "explicit side only". The contrast cancels on explicit praise and only acts on implicit cases.
- Silver (309): pos acc 0.9482, macro-F1 0.6913; lam 0 to 0.75 identical; lam 1.0 acc 0.7282.
- Frozen 188 (reporting only): pos acc 0.6649, macro-F1 0.7603; party recall 0.3922 at lam 0 to 0.75 (the 62 misses persist); lam 1.0 party recall 0.3137 and 8/86 gold None predicted party.
- Result: no change. The lenient prompt does not recover implicit cases, and the contrast is inert again because the first-token distribution is saturated. Inference-time contrast is rejected for this teacher.

v5 teacher labelling finished: `train_v5_soft.jsonl` has 6929 rows. Hard parsed 6926, soft parsed 6894, and the soft argmax equals the hard label on 6883 of 6926 rows. Soft targets are near one-hot, as expected.

New-age RL: verified sources and what each implies here.
- GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)): group-relative advantage from sampled answers. Needs sampling variance, which this near-deterministic teacher lacks at T=0. Requires sampling at T>0.
- DAPO ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)): clip-higher, dynamic sampling, token-level loss. Use if GRPO is run; dynamic sampling drops groups with no reward variance.
- Dr. GRPO ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)): removes length and std normalisation bias. Apply if rewards depend on output length.
- GSPO ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)): sequence-level ratios. Relevant for MoE policies.
- TTRL ([arXiv:2504.16084](https://arxiv.org/abs/2504.16084)): majority vote over samples as a pseudo-reward on unlabelled text. Risk here: the majority is None, so TTRL would reinforce the recall gap. Do not use without a None-balancing term.
- Spurious rewards ([arXiv:2506.10947](https://arxiv.org/abs/2506.10947)): random rewards improved Qwen through clipping bias, not reward signal. Any RL run needs a random-reward control.
- Entropy mechanism ([arXiv:2505.22617](https://arxiv.org/abs/2505.22617)): Clip-Cov and KL-Cov prevent entropy collapse. Relevant if RL collapses to None.
- LSC reward transform ([arXiv:2402.00742](https://arxiv.org/abs/2402.00742)): log-sigmoid centred reward. Emphasises poorly performing outputs, a fit for the recall gap.
- DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)): preference pairs without sampling. Pairs: gold-party rows where the teacher said None (chosen = party, rejected = None), with a balancing set for None rows.
- On-policy distillation (Thinking Machines blog, 2025): student samples, teacher per-token reverse KL as a dense reward. Reported 9 to 30 times cheaper than RL. Best cheap fit for a generative student.

Verification note: three IDs drafted in earlier notes (2507.19671, 2508.18147, 2601.00001) resolve to unrelated papers. They are removed and must not be cited.

Teacher diagnostics (`scripts/party_v8_teacher_diag.py`, frozen is reporting only):
- Reasoning effort does not change recall. Frozen party recall: low 0.3922, medium 0.3725, high 0.3922. Gold None predicted as party: 0 of 86 at every setting.
- Sampling does not recover the misses. Frozen, 8 samples at T=0.8: 98.4 percent of rows are unanimous. Pass@8 party recall is 0.4020 against majority-vote recall 0.3922, so only 1 extra row is recovered. Silver: pass@8 1.0, majority recall 0.9593.
- Implication: the 62 misses are a deterministic prior, not sampling noise. GRPO or TTRL would get almost no reward variance on these rows, and TTRL's majority would reinforce None. Sampling-based group RL is not the lever for this teacher.

GEPA-style rule search (`scripts/party_v9_gepa_prompt.py`, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457), ICLR 2026 Oral): the teacher reflects on its own errors and rewrites the rules block. The dev set is 120 party and 80 None rows sampled from `train_v4.jsonl`, excluding any silver or frozen text. A candidate is accepted only if dev macro-F1 rises by at least 0.005 and the None false-positive rate stays at or below 0.10. Silver and frozen are scored once after the search. Caveat: train labels come from the synthetic production corpus, so the search may move the teacher toward that convention, not the gold convention.
- Dev baseline (round 0): macro-F1 0.8786, acc 0.9000, None false positives 0.0125.
- Result: dev macro-F1 rose to 0.9549 (accepted in rounds 1 and 2). Silver regressed: acc 0.9773 to 0.9385, macro-F1 0.8504 to 0.7489, None FP 0.0108 to 0.0215. Frozen gained two rows (acc 0.6702 to 0.6809), within noise. The searched rules were long keyword lists from synthetic errors (for example "Chhatra Nyay = Congress"), so the search optimized the synthetic convention. Not adopted.

Next (v10): out-of-fold check on frozen. Five stratified folds. Each fold searches rules on the other four and accepts a candidate only if dev macro-F1 gains by at least 0.01, None FP stays at or below 0.10, and silver macro-F1 does not drop. The held-out fold is scored once. Out-of-fold predictions over all 188 rows are the estimate, with silver used only as a guard.

Other candidate: intrinsic-reward RL ([arXiv:2505.19590](https://arxiv.org/abs/2505.19590), Intuitor, ICLR 2026). It uses self-certainty as the reward. It is not suited here: the teacher is confident and wrong on the misses, so self-certainty would reward the error.

v10 out-of-fold (VM, `party_v10_cv_search.py`, five folds, held-out 46, 42, 36, 33, 31 rows): no candidate rule was promoted in any fold. Frozen baseline and frozen searched are identical (acc 0.6702, macro-F1 0.7633, party recall 0.3922, None FP 0.0000). The out-of-fold rule search gives no gain. A written convention decision is still required before any label is treated as gold.

v5 distillation (`scripts/party_v5_train.py` on VM, `train_v5_soft.jsonl`, selection on silver only, frozen 188 scored once after selection, reporting only):
- e5-small, alpha 0.7 (teacher KL plus CE): selected epoch 4, silver acc 0.9159, silver macro-F1 0.3862. Frozen acc 0.5900, macro-F1 0.5400.
- e5-small, alpha 0 (hard labels, same script and seed): selected epoch 4, silver acc 0.9223, silver macro-F1 0.4158. Frozen acc 0.5479, macro-F1 0.4960.
- Reading: on the selection set alpha 0.7 is lower by 0.030 silver macro-F1. Silver has 309 rows and is 86% None, so this is within noise. The frozen gap favours alpha 0.7, but frozen is contaminated for selection and is reporting only. Soft teacher targets do not show a clear gain on the clean guard. Not promoted.
- XLM-R large, alpha 0 (hard labels, lr 1e-5): selected epoch 6, silver acc 0.8867, silver macro-F1 0.4578. Frozen acc 0.6011, macro-F1 0.6279 (baseline 0.5372 and 0.6056 at the same run). Reporting only.
- XLM-R large, alpha 0.7: selected epoch 6, silver acc 0.8932, silver macro-F1 0.3967 (alpha 0 is 0.4578 on the same guard). Frozen acc 0.6436, macro-F1 0.6604. The silver guard favours alpha 0, and frozen is reporting only. Not promoted. Distillation (v5) is closed with no winner.

## 10. Roadmap pass (section 5 items)

- Item 2, dedup and group split: done. `scripts/party_clean_splits.py` writes `artifacts/party_clean_splits/`. Groups are word 3-gram Jaccard >= 0.8 across train, silver, and frozen. Exact overlap across splits is 0. Train loses 6 rows (4 within-train duplicates, 2 near-duplicates of silver). Eval sets are unchanged. Label conflicts: 0. Per-class train loss is at most 2 rows (None, TDP). Limit: the 0.8 threshold does not catch paraphrase leakage, and that has not been measured.
- Item 1, clean held-out set: blocked. Frozen 188 has been used for selection (v3b, teacher debugging, v9 and v10 reporting), so it is no longer clean. A clean set needs new human-labelled real posts, especially for sparse parties and None hard negatives. Teacher labels were not used as gold.
- Item 3, label review of the three RC2 candidates:
  - Line 189 (labelled AAP): correct. The sarcasm praises AK (AAP). Removed from the candidate list.
  - Line 409 (labelled BJP): sarcasm about the Shiv Sena UBT and Congress alliance. BJP support is not established. Needs human review.
  - Line 1228 (labelled None): explicit Shiv Sena endorsement. Likely a wrong label. Not changed, because edits to training data need your confirmation.
  - Full relabel of seed rows: not done.
- Item 4, real-data expansion: not started. Needs human-labelled real posts.
- Item 5, teacher-student distillation: done as far as useful (section 9). No candidate beats the hard-label baseline on the silver guard.
- Item 6, RLVR: not started. The verifier is too weak (section 9) and depends on items 1 and 3.
- Item 7, active learning: not started. Waits on a calibrated model (item 8).
- Item 8, calibration: blocked. Temperature scaling needs a clean validation set (item 1).
- Item 9, documentation: `docs/ARCHITECTURE.md` party row now reports the measured 7-class numbers (accuracy 0.7613, macro-F1 0.7638) and says the earlier 82.03% / 0.8244 is not reproduced. `docs/TRAINING_RESULTS.md` party_support row is marked as synthetic validation and not reproduced on real text. Other rows in those docs belong to other model families and were not verified in this pass.
- Production `bert-party-support-17` is unchanged.
- Frozen 188 archived: moved to `artifacts/party_clean_splits/archive/frozen_clean_contaminated.jsonl` (all 188 rows kept). Still used as an exclusion list by the dedupe scripts, not for scoring or selection. Scripts `prepare_party_news_pool.py`, `build_heldout_sonnet.py` and `build_party_heldout_pool.py` now read the archive path.

### Decisions executed in this pass

- Labelling: Sonnet 4.6 (`claude-sonnet-4-6`, via `synthetic-model-training/tools/sonnet_label.py`). Instructions: 18 labels including CJP (official name Cockroach Janta Party), implicit side-taking counts as a party label, None is the absence of side-taking, one label per post.
- Line 1228 of `party_train_all.jsonl` relabelled None to Shiv Sena (user approved). Same row in `artifacts/party_v3` and `artifacts/party_v4` (`train_v4.jsonl`, 6929 rows) updated.
- Line 409 kept as BJP (user confirmed the sarcasm reading).
- Official name in README and section 4: Cockroach Janta Party (CJP).

### Clean candidate pool and Sonnet held-out

- Pool: `artifacts/party_v6/candidates.jsonl`, 863 rows after the group-split filter (2258 raw). Sonnet labelled 862. `cand_00289` (Marathi CJP post) failed twice with a malformed reply and is excluded. Not hand-labelled.
- Held-out: `artifacts/party_v6/heldout_sonnet.jsonl`, 300 rows, unique ids. Keeps all 164 non-None Sonnet labels plus a seeded None sample (seed 17, 136 rows). Overlap check against train, silver, and frozen removed 0 rows.
- Held-out composition: None 136, BJP 56, Congress 38, CJP 35, NCP 8, TMC 6, DMK 6, AAP 3, BSP 2, BRS 2, Shiv Sena 2, CPIM 2, TDP 1, YSRCP 1, AIADMK 1, RJD 1. The party/None ratio is about 1:1, not the natural distribution. Precision estimates must be reweighted.
- Limit: this set holds every party-positive Sonnet label. Training on it contaminates it for evaluation. Use it for final reporting only.

### Relevance e5 retrain (`scripts/train_relevance_e5_vm.py`)

- Two crashes fixed: `str.splitlines()` split on U+2028 inside captions (now iterates the file handle), and GPU 2 ran out of memory because another user's vLLM held 73.7 GB (relaunched on GPU 3).
- Test set: 567 rows, 44 topics. Baseline AUC 0.8733, F1 at dev threshold 0.7903, best F1 0.8218, P@1 1.0. Fine-tuned AUC 0.8893, F1 at dev threshold 0.7845, best F1 0.8137, P@1 0.8864.
- Result is mixed. The P@1 gap is about 5 of 44 topics, which is noisy. Not promoted.

### Production scored on the clean sets (no retrain)

- Scorer: `scripts/party_v11_score.py` (VM copy of production model, 1,115,314,348 bytes, matches local). Probabilities written per row, then metrics computed locally.
- Held-out (300): accuracy 0.2500, macro-F1 0.1703. Bootstrap 95% CI on accuracy [0.197, 0.297]. Excluding the 35 CJP rows (no CJP class in production): accuracy 0.2830, macro-F1 0.1877.
- Party-vs-None view on held-out: precision 0.617, recall 0.933. Production labels many None posts as TMC (34), CPIM (14), BJP (13), Congress (11).
- Silver (309): accuracy 0.2136, macro-F1 0.1882. Silver contains 98 CJP rows the model cannot predict, so this is a ceiling effect.
- Frozen 188: accuracy 0.5372, macro-F1 0.6056. Matches section 4, so the pipeline reproduces the known baseline.
- Reading: the held-out set is much harder than frozen. Sonnet labels include many hard None posts and 35 CJP rows. The production model is weak on real text. Earlier frozen-based figures overstate it.

### None-bias check (no retrain, silver guard only)

- A single additive bias on the None logit was swept on silver (non-CJP, n=211). Silver accuracy rises from 0.3128 (bias 0) to 0.4834 (bias 0.5), but 88% of silver non-CJP rows are None, so the gain mostly rewards predicting None.
- Frozen (reporting only) moves from 0.5372 to 0.5532 at bias 0.4. That is about 3 rows and within noise.
- Not applied. The held-out set was not used to choose the bias.

### Open items

- Any model that beats production needs training data that is not in the held-out. The 164 party positives in held-out cannot be used for training. Needs a decision before a retrain.
- RLVR (item 6), active learning (item 7), and calibration (item 8) remain blocked on the same clean-data issue.
- Production `bert-party-support-17` is unchanged.

## 11. v12 party dataset: news crawl, Sonnet labels, and retrains

### Data

- Google News RSS crawl (en, hi, mr): `scripts/crawl_party_news_rss.py`, 2528 raw rows in `artifacts/party_v7/news_raw.jsonl`.
- Dedupe against all existing splits with `group_rows` in `scripts/party_clean_splits.py` (word 3-gram Jaccard >= 0.8): `scripts/prepare_party_news_pool.py`. 2528 kept, 0 overlap.
- Sonnet 4.6 labels (`tools/sonnet_label.py`, same 18-label instructions as section 10): 2498 valid. 30 rows failed twice with malformed outputs (labels `a` or `s`). They are excluded, not hand-labelled.
- Frozen 188 is archived and not used for scoring.
- v12 splits from `scripts/party_v12_prepare.py` in `artifacts/party_v12/`:
  - `train.jsonl`: 8924 rows (includes news and CJP).
  - `dev.jsonl`: 497 rows, selection only.
  - `eval.jsonl`: 300 rows, the clean held-out from `heldout_sonnet.jsonl`. Reporting only.
  - Cross-split duplicate groups: 0.

### Results on the clean eval (300 rows, 18 labels)

| Run | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Production `bert-party-support-17` | 0.2500 | 0.1703 | No CJP class. Excluding CJP rows: 0.2830 / 0.1877 |
| `e5-small`, seed 42 | 0.5733 | 0.3361 | 6 epochs, lr 5e-5, batch 32, label smoothing 0.1, sqrt class weight |
| `e5-small`, seed 1 | 0.5567 | 0.3464 | Same config |
| `e5-small`, seed 2 | 0.5533 | 0.2600 | Same config |
| `e5-small`, seed 3 | 0.5733 | 0.3412 | Same config, selected epoch 5 |
| `e5-small`, seed 4 | 0.5600 | 0.3188 | Same config, selected epoch 4 |
| `e5-small`, seed 5 | 0.5400 | 0.2435 | Same config, selected epoch 6 |
| Production init, 18 labels | 0.5633 | 0.2770 | Start from production weights, head re-initialised (17 to 18), 5 epochs, lr 2e-5 |

- `e5-small` over 6 seeds (42, 1, 2, 3, 4, 5): accuracy mean 0.5594 (sd 0.0127), macro-F1 mean 0.3077 (sd 0.0446, range 0.2435 to 0.3464).
- The macro-F1 spread across seeds (about 0.10 range) is larger than the gap between the mean candidate and production-init. No candidate is clearly better than the others.
- Seed ensemble (average of the 6 probability sets, no bias): eval accuracy 0.5633, macro-F1 0.3121. A small gain over the seed mean, within the seed spread. Scored with `scripts/party_ens_score.py` (18 labels, max length 160).
- Per-class bias tuning on dev macro-F1 does not transfer. Dev macro-F1 rises from 0.76 to 0.82, but eval macro-F1 falls to 0.2362. The tuned biases are not used.
- Dev macro-F1 (about 0.76) is far above eval (about 0.31). Dev appears to resemble the training distribution more than eval does; the cause was not checked.
- On eval, the untuned ensemble predicts None for 255 rows (gold 136), BJP for 21 (gold 56), and Congress for 2 (gold 38). The errors concentrate in these classes, so the eval/train distribution gap matters more than seed choice.
- Most of the accuracy comes from the None class (136 of 300 rows; recall about 0.93 to 0.95 for the fine-tuned models). Macro-F1 is driven by classes with 1 to 8 eval rows, so it is noisy.
- The production vs fine-tune comparison is not label-space matched: production cannot predict CJP (35 eval rows). Even excluding CJP, production is at 0.28 accuracy.

### Decision

- Production `bert-party-support-17` is not replaced. The v12 `e5-small` candidates beat production on accuracy (0.5594 mean vs 0.2500) and macro-F1 (0.3077 mean vs 0.1703), but this comparison is not label-space matched (production cannot predict CJP). Macro-F1 variance across seeds is large, and the eval set is only 300 rows. Promotion also needs an 18-label ONNX export and Java-side changes for CJP.
- Next steps, in priority order: human spot-check of a sample of news labels; a larger backbone if a cached one with weights is available; then ONNX export and local parity check.
- Model artifacts are on the VM under `/home/s0r0qw6/party_v12/`: `out_e5s`, `out_e5s_seed1` to `out_e5s_seed5`, `out_prod_init`. Production scores are in `score_prod/eval_probs.jsonl`.
