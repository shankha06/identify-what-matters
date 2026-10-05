# Literature Review: Emerging Discussion on Instagram India Beyond Fashion

**As-of date: 2026-10-03.** This is a literature-informed research design, not a report of topics currently trending on Instagram. Public papers and documentation were fetched for this review; no Instagram private endpoints were called and no live trend collection was performed. No code changes are part of this document.

## 1. Executive Findings and Evidence Boundary

The research question is: **Which discussions are newly gaining attention across an explicitly defined India-relevant Instagram panel, rather than merely being popular, fashionable, or newly encountered by a collector?**

The defensible answer requires a longitudinal sampling frame, multilingual semantic analysis, exposure-aware burst detection, and human evaluation. A collection of high-engagement posts or fashionable hashtags cannot establish either emergence or national representativeness. The literature supports components of this approach, but the sources reviewed do not validate an end-to-end Instagram India emerging-topic detector.

This review uses four evidence labels:

- **Published research:** Findings from papers, limited to their evaluated platforms, languages, tasks, and time periods. An arXiv technical report is identified as such, not silently treated as peer-reviewed evidence.
- **Maintainer or official documentation:** Descriptions of model behavior and platform capabilities; these are version-dependent, not experimental proof of discovery quality.
- **Corporate self-report:** SHEIN's description of its production process, not independent disclosure or validation of proprietary algorithms.
- **Engineering proposal:** The panel quotas, scoring choices, schedules, thresholds, and acceptance gates below. These require local validation and are not results reported in the cited papers.

Principal conclusions:

1. **Separate semantic grouping from temporal emergence.** LDA, Sentence-BERT, and BERTopic help organize text; Kleinberg and EDCoW address temporal bursts. Neither family alone proves novelty [S1-S7].
2. **Broaden the sampling frame before broadening the model.** Include news, politics, cricket, cinema, music, gaming, technology, education, jobs, health, food, travel, civic issues, climate, business, and fashion. Fashion is one stratum, not the organizing ontology.
3. **Keep novelty relative to observation.** A genuinely new event can occur within an old topic; an old discussion can appear new because a language or account was added. Report these cases separately.
4. **Treat India relevance and user geography as different variables.** Indian-language text, a city hashtag, or an Indian subject does not establish author or audience residence.
5. **Official access is incomplete.** Meta's readable documentation supports limited hashtag discovery, professional-account metadata discovery, and authorized-media comment workflows, not unrestricted collection of everyone's comments [S16-S18].
6. **Multilingual performance must be measured locally.** E5, MuRIL, and IndicBERT provide useful starting points but do not guarantee good clustering of transliterated, sarcastic Instagram comments [S8-S12].
7. **Attention is not truth or demand.** Misinformation can be novel and fast-spreading [S14]. Buying-intent comments are not verified sales, and SHEIN's documented feedback loop does not disclose this project's proposed algorithm [S20].

## 2. Annotated Literature and Documentation

The table contains **20 source entries**. Related paper/model-card or documentation pages share an entry where they describe the same method. All successful fetches used below are identified in Section 10. A readable abstract supports only the claims it actually contains; it is not represented as a full-paper read.

| ID | Title, authors/organization, year, and primary URL | Method or documented evidence | Applicability to this project | Limitations and verification status |
| --- | --- | --- | --- | --- |
| S1 | **Bursty and Hierarchical Structure in Streams**, Jon Kleinberg, KDD 2002. [Paper](https://www.cs.cornell.edu/home/kleinber/bhs.pdf); [author's explanation and examples](https://www.cs.cornell.edu/home/kleinber/kdd02.html) | Probabilistic automaton with states corresponding to word frequencies; most-likely state sequences identify high-intensity intervals and significant bursts. | Temporal baseline for terms, entities, or topic occurrence streams; distinguish high background volume from a limited-duration surge. | Paper PDF extraction failed. Author's explanation and bibliography were readable and confirm KDD 2002. This review does not claim to have verified detailed equations or parameter settings from the PDF. Bursty words are not necessarily novel concepts. |
| S2 | **Event Detection in Twitter**, Jianshu Weng and Bu-Sung Lee, ICWSM 2011. [Proceedings](https://ojs.aaai.org/index.php/ICWSM/article/view/14102) | EDCoW constructs wavelet-based word-frequency signals, filters by autocorrelation, and uses modularity-based graph partitioning to form events. | Comparator for synchronously rising event vocabulary, beyond independent hashtag ranking. | Twitter text-stream evidence, not Instagram India evidence. Sensitive to temporal bins, sufficient observations, and tokenization. Site displays a 2021 publication/upload date, but the conference issue is 2011. Abstract and metadata fetched. |
| S3 | **Latent Dirichlet Allocation**, David M. Blei, Andrew Y. Ng, and Michael I. Jordan, 2003. [JMLR](https://www.jmlr.org/papers/v3/blei03a.html) | Hierarchical Bayesian topic mixtures with variational inference and empirical-Bayes estimation. | Interpretable bag-of-words baseline; compare against embedding-based topics rather than presuming neural superiority. | Standard LDA is not a burst detector or online novelty detector. Sparse short comments and code-mixed spelling reduce usable co-occurrence evidence. Abstract and publication metadata fetched. |
| S4 | **BERTopic: Neural topic modeling with a class-based TF-IDF procedure**, Maarten Grootendorst, 2022. [arXiv:2203.05794](https://arxiv.org/abs/2203.05794) | Transformer document embeddings, clustering, and class-based TF-IDF topic representations; paper reports competitive topic benchmarks. | Semantic grouping with inspectable labels and representative documents. | Benchmark coherence does not establish emergence, factual validity, or national coverage. Embedding and clustering choices affect topic boundaries. arXiv abstract fetched; no claim of peer-reviewed publication is made here. |
| S5 | **Dynamic Topic Modeling** and **Online Topic Modeling**, BERTopic maintainer, undated living documentation. [Topics over time](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html); [online](https://maartengr.github.io/BERTopic/getting_started/online/online.html) | `topics_over_time` derives time-specific c-TF-IDF representations from previously fitted topics. `partial_fit` updates compatible incremental components; examples use IncrementalPCA, MiniBatchKMeans, and OnlineCountVectorizer. | Retrospective description versus forward incremental discovery; documentation also discusses merging models and streaming cluster alternatives. | These are distinct workflows. Default batch components do not become incremental automatically. Fixed-K clustering does not guarantee new-topic birth. Full-corpus fitting can leak future information in an early-detection evaluation. Both guides fetched. |
| S6 | **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**, Nils Reimers and Iryna Gurevych, EMNLP 2019. [Paper](https://arxiv.org/abs/1908.10084) | Siamese/triplet training produces sentence embeddings comparable by cosine similarity; evaluated on semantic similarity and transfer tasks. | Efficient semantic grouping and retrieval of prior discussions or paraphrases. | Similarity is not a calibrated novelty probability. Original results do not establish Indic code-mixed performance or validate every later Sentence-Transformers checkpoint. Abstract fetched. |
| S7 | **Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval**, Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin, 2021. [Abstract](https://arxiv.org/abs/2108.08787); [full HTML paper](https://arxiv.org/html/2108.08787v2) | Monolingual Wikipedia retrieval in 11 languages; compares BM25, zero-shot mDPR, and sparse-dense hybrids. | Evidence for evaluating lexical and semantic baselines together and inspecting per-language failure modes. | Retrieval, not Instagram clustering. Languages are Arabic, Bengali, English, Finnish, Indonesian, Japanese, Korean, Russian, Swahili, Telugu, and Thai. Hindi, Gujarati, Marathi, and Tamil are absent. Judgments are incomplete. Full HTML fetched. |
| S8 | **Multilingual E5 Text Embeddings: A Technical Report**, Liang Wang et al., 2024. [Report](https://arxiv.org/abs/2402.05672); [multilingual-e5-large model card](https://huggingface.co/intfloat/multilingual-e5-large) | Contrastive pretraining on one billion multilingual pairs, followed by labeled fine-tuning; small/base/large models. Card describes 100-language support inherited from XLM-R, 1,024-dimensional large embeddings, and input-prefix conventions. | Strong candidate for cross-language semantic retrieval and clustering; compare quality and inference cost across model sizes. | Card warns of low-resource degradation and 512-token truncation. Use `query: ` for symmetric similarity/clustering; asymmetric retrieval uses query/passage prefixes. Benchmark language coverage is not Hinglish validation. Abstract and detailed card fetched. |
| S9 | **MuRIL: Multilingual Representations for Indian Languages**, Simran Khanuja et al., 2021. [Paper](https://arxiv.org/abs/2103.10730) | Indian-language-focused pretraining augmented with translated and transliterated document pairs; reports improvements over mBERT on XTREME and transliterated tests. | Candidate encoder for Indic language adaptation and transliteration-sensitive classification. | A pretrained language model is not automatically a well-calibrated sentence-similarity model. Benchmark gains do not establish robust Instagram slang, mixed-language stance, or topic novelty. Abstract fetched. |
| S10 | **IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages**, Divyanshu Kakwani et al., 2020. [ACL Findings](https://aclanthology.org/2020.findings-emnlp.445/) | Indic corpora, FastText embeddings, compact ALBERT-based models associated with IndicBERT, and IndicGLUE evaluation resources; abstract describes 8.8 billion tokens across 11 Indian languages and Indian English, primarily from news crawls. | Indic-language baselines and task-specific evaluation resources. | News/Wikipedia-oriented tasks differ from Instagram register. Refers to the original resource, not a verified comparison of later IndicBERT versions. Abstract and metadata fetched. |
| S11 | **SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets**, Parth Patwa et al., 2020. [ACL](https://aclanthology.org/2020.semeval-1.100/) | SentiMix provides Hinglish and Spanglish corpora with word-level language identification and sentence sentiment labels; 20K Hinglish and 19K Spanglish examples. | Code-mixed annotation design and a realistic reminder that sentiment is not trivial; reported best Hinglish F1 is 75.0%. | Twitter sentiment is neither Instagram discussion understanding nor emergence detection. Hindi-English does not cover every Indian code-mixed pair; do not transfer the reported F1 as expected local performance. Abstract fetched. |
| S12 | **Big Questions for Social Media Big Data: Representativeness, Validity and Other Methodological Pitfalls**, Zeynep Tufekci, ICWSM 2014. [Paper](https://arxiv.org/abs/1403.7400) | Methodological critique of platform concentration, hashtag sampling, unclear sampling frames, invisible behavior, and external field effects. | Grounds the panel design and limits population-level claims; motivates collecting beyond hashtags. | Not a recipe that statistically repairs an unknown inclusion mechanism. Large sample size cannot cure systematic exclusions. Abstract fetched. |
| S13 | **Mind the Gap: Assessing Temporal Generalization in Neural Language Models**, Angeliki Lazaridou et al., NeurIPS 2021. [Paper](https://arxiv.org/abs/2102.01951) | Transformer-XL performance worsens on future utterances beyond training periods; continual updates can mitigate degradation; streaming WMT/arXiv benchmarks. | Motivates chronological replay, time-specific evaluations, and careful adaptation. | Language-modeling evidence, not a direct measurement of Instagram trend detectors. Does not supply this review's acceptance thresholds. Correct abstract fetched. |
| S14 | **The spread of true and false news online**, Soroush Vosoughi, Deb Roy, and Sinan Aral, Science 2018. [Publisher](https://www.science.org/doi/10.1126/science.aap9559) | Analysis of approximately 126,000 verified story cascades on Twitter, 2006-2017; false news spreads farther, faster, deeper, and more broadly in the analyzed data. | Strong reason to separate attention/novelty from claim verification, especially in politics, health, and disasters. | Observational Twitter study, not a universal causal law or Instagram India effect estimate. Publisher abstract and metadata readable; full article access restricted. |
| S15 | **Learning Transferable Visual Models From Natural Language Supervision**, Alec Radford et al., 2021. [CLIP paper](https://arxiv.org/abs/2103.00020) | Image-text contrastive learning on 400 million internet pairs; zero-shot transfer across visual tasks. | Evidence that visual representations can complement captions; motivates a separately evaluated multimodal extension. | Does not recover Instagram audio, prove image authenticity, or guarantee Indic meme/OCR understanding. Image-text similarity is not verification of the pictured event. Abstract fetched. |
| S16 | **Hashtag Search**, Meta, undated living documentation. [Official guide](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login/hashtag-search) | Facebook Login route: public hashtagged media, top/recent edges; maximum 30 unique hashtags per Business/Creator account in a rolling seven-day period; review for Public Content Access and `instagram_basic`. | Bounded supplemental discovery lane with a recorded query budget. | No Stories hashtag support; emoji hashtag queries unsupported; sensitive/offensive queries may error. Discovery is not exhaustive or random, and cannot be assumed to include arbitrary comment text. Guide fetched. |
| S17 | **Business Discovery**, Meta, undated living documentation. [Official guide](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login/business-discovery) | Basic metadata/metrics for other professional accounts via username-based field expansion; examples include media and engagement counts. | Known-account professional panel and permitted public metadata snapshots. | Age-gated accounts excluded. Guide explicitly says returned media IDs do not grant direct media access. It does not state a follower-relationship prerequisite; do not repeat that unsupported requirement. Guide fetched; endpoint-specific permission details require preflight. |
| S18 | **Comment Moderation**, Meta, undated living documentation. [Official guide](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/comment-moderation) | Comments/replies and webhooks for media owned by app users; distinguishes Instagram Login and Facebook Login permissions and access levels. | Authorized partner/owned-media discussion panel with comment timestamps and reply context where permitted. | Not access to every public post's comments. Advanced Access is required for serving professional accounts not owned/managed by the app operator; authorization still matters. Guide fetched; universal 50-comment caps or ordering guarantees were not established. |
| S19 | **instagrapi project description**, package maintainer, undated living release page. [PyPI](https://pypi.org/project/instagrapi/) | Explicitly describes an unofficial wrapper combining public web and private mobile flows; warns that trust, challenges, device state, and rate limits make private automation fragile. | Feasibility/risk evidence only; not the recommended production collection route. | Package availability or research wording is not Meta approval or permission. Operational, session-security, contractual, and account-enforcement risks remain. PyPI fetched; documentation homepage extraction failed. No library calls executed. |
| S20 | **SHEIN Leverages Digital Thermal Transfer Printing Technology to Support Small-Batch Production and Reduce Water Use**, SHEIN, 2022; **Our Group**, living corporate page. [Announcement](https://www.sheingroup.com/newsroom/shein-leverages-digital-thermal-transfer-printing-technology-to-support-small-batch-production); [corporate overview](https://www.sheingroup.com/our-group) | Announcement describes small orders and increasing or ceasing production based on market feedback; overview describes decisions based on preferences and purchases. | Limited analogy: observe, test with controlled exposure, measure response, then repeat or stop. | Corporate self-report, not an independently validated prediction algorithm. Neither fetched page establishes a universal batch quantity, a three-day turnaround, Instagram embedding architecture, or a comment-to-sales multiplier. Both pages readable. |

## 3. What the Literature Establishes

### 3.1 Temporal bursts, events, and novelty are different targets

Kleinberg's author explanation describes words rising to prominence for bounded periods [S1]. EDCoW instead groups correlated wavelet-derived signals to construct events [S2]. Both give a better methodological starting point than ranking lifetime likes, but neither makes a burst intrinsically new: a familiar cricket tournament, annual admission cycle, recurring protest slogan, or seasonal food discussion can burst again.

**Proposal:** Maintain separate fields for `broad_domain`, `discussion_topic`, `event_instance`, `claim`, and `format_or_audio_template`. A topic might be exam preparation; an event instance might be a particular official schedule change. Several posts may reuse a sound without discussing the same subject. Conversely, one event may span cricket, business, politics, and travel. Allow multiple domain labels rather than forcing every item into one category.

Use five novelty labels:

| Label | Operational meaning | Important counterexample |
| --- | --- | --- |
| New topic candidate | A coherent discussion is materially different from the panel's historical topic archive and has repeated support. | New spelling or a newly covered language is not automatically a new concept. |
| New event in known topic | New entity/time/location/action combination within an established subject. | Another match is an event, not the invention of cricket discussion. |
| Resurgence or seasonal recurrence | An old topic returns above its expected baseline. | A festival burst may be important without being novel. |
| Format diffusion | Visual, meme, sound, or participation template spreads across otherwise different subjects. | Shared audio alone does not imply shared discussion. |
| Coverage-induced apparent novelty | Previously missing accounts, queries, languages, or modalities become visible. | A new Tamil panel produces apparent growth with no behavioral change. |

No finite archive proves global first occurrence. The proper claim is **new or emerging relative to the observed panel and stated lookback**. A historical data gap must lower novelty confidence, not increase it.

### 3.2 Topic models provide organization, not automatic emergence

LDA is a valuable lexical baseline [S3], especially for interpretable vocabulary differences, but standard bag-of-words assumptions omit ordering and context. Short comments offer little word co-occurrence, and comment-level LDA can yield unstable themes. Aggregating within post/thread/time groups may help, but must not erase disagreement or mix unrelated events.

Sentence-BERT enables efficient semantic comparisons [S6]. BERTopic combines document embeddings, clustering, and c-TF-IDF representations [S4]. These support interpretable semantic groups, but cluster size, distance from a centroid, and coherence are different quantities from emergence. Unassigned/outlier documents can contain genuine new discussion and must remain reviewable rather than being discarded as noise.

**Critical distinction [S5]:**

- `topics_over_time` is a retrospective representation of fitted topics at different timestamps. Fitting on the entire historical-and-future evaluation corpus gives future documents influence over the topic space. That is acceptable for retrospective description, not for claims about what could have been discovered earlier.
- `partial_fit` is incremental training with compatible submodels. The documented batch UMAP/HDBSCAN workflow is not simply converted by calling this method after `.fit`.
- The online MiniBatchKMeans example uses a fixed number of clusters. Incremental updates alone therefore do not guarantee an ability to create arbitrarily new topics.
- Model merging or streaming clustering can support new-topic discovery, but cluster identity, splits/merges, drift, and historical comparisons need explicit handling.

**Proposal:** Start with frozen multilingual embeddings and a past-only topic archive, assign incoming material where appropriate, and buffer unmatched material for new-cluster review. Periodic refits should create versioned topic lineage rather than silently renumbering historical topics. Keep raw topic-occurrence counts separate from smoothed word representations so smoothing does not obscure alert timing.

### 3.3 Indic language and code-mixing require local evidence

The multilingual E5 report and model card support its use as a candidate, not a guaranteed winner [S8]. Follow its prefix and normalization conventions and calibrate similarity thresholds on local pairs; the card notes that absolute cosine values may be unusually high. Avoid carrying an arbitrary threshold across embedding models. Long threads must be segmented before the documented 512-token truncation boundary, with chunk-to-thread provenance.

MuRIL explicitly addresses translated/transliterated training signals and discusses Indian-language code-mixing [S9]. IndicNLPSuite provides Indic resources but is primarily news-oriented [S10]. These models are not automatically interchangeable with sentence-embedding models; compare appropriate pooling or task-adapted variants, and record the exact checkpoint and training recipe.

SentiMix demonstrates the relevance of word-level language labels for Hinglish [S11]. Its task is sentiment, so it cannot be cited as proof of topic clustering or truth classification. Mr. TyDi is likewise a retrieval benchmark, with Bengali and Telugu as its Indian-language entries, not a Hindi/Gujarati/Marathi/Tamil benchmark [S7].

**Proposal:** Preserve original text, Unicode scripts, emoji, hashtags, negation, and quoted speech; maintain optional normalized and transliterated views alongside it. Annotate language at span level when feasible. Use explicit mixed/unknown labels for short ambiguous text. Measure native-script Hindi, Romanized Hindi, Hindi-English mixing, English, and selected regional-language strata separately. Expand language support according to actual authorized panel material, not a nominal model-language count.

Do not replace everything with English translation before analysis. Translation may erase slang, puns, place names, sarcasm, and politically relevant distinctions; retain translation provenance and compare original versus translated views in evaluation. A multilingual cluster can bridge paraphrases but must not merge different regional events simply because their captions have similar templates.

### 3.4 Social-data bias is a design constraint

Tufekci identifies hashtag-selected sampling frames, single-platform concentration, and behavior invisible to text-only collection [S12]. For this project, those concerns become specific failure modes:

- Hashtag users differ from non-users; promotional tags and spam can dominate query results.
- Top-media retrieval selects for visibility/engagement; recent-media retrieval still samples an accessible query surface, not all posts.
- Professional-account discovery excludes much ordinary-user discussion. Partner recruitment favors willing, organized accounts.
- Comments come from active responders, not all viewers. Ranking, pagination, moderation, deletions, disabled comments, and collection timing change what is visible.
- A fixed first page overweights whichever ordering the surface provides. A complete authorized traversal is still a census of available comments, not of opinions among viewers.
- One large account or repeated commenters can dominate raw volume. Count independent publishers and deduplicated contributions, not just messages.
- Native-language, regional, non-metropolitan, small-account, image-only, and audio-first discussions can be systematically undercovered.
- Adding accounts or queries increases observed volume without necessarily changing attention. API failures can produce artificial declines and recovery bursts.

Quota balance can improve breadth but does not yield known population inclusion probabilities. Post-stratification cannot recover national prevalence when the inclusion mechanism and population margins are unknown. Report sample composition and panel-relative rates; do not describe the result as a representative survey of India.

## 4. A Multi-Source, Stratified Discussion Panel

**Engineering proposal:** Maintain a stable longitudinal core plus a separately reported exploration lane. Stratify by domain, language/script, source/account type, permitted media format, and geography-evidence tier. This is a purposive sentinel panel, not a probability sample.

| Domain | Candidate panel/source types | Discovery target and recurring confounder |
| --- | --- | --- |
| News | National/regional publishers, local reporters, public primary announcements | New reported events versus syndicated headlines and repeated breaking-news templates |
| Politics | Public institutional accounts, diverse public commentary, authorized community partners | New policies/arguments versus coordinated campaigns; never infer individual political affiliation |
| Cricket | Teams, competitions, sports journalists, regional fan partners | New incidents/tactics versus scheduled fixtures and fan promotion |
| Cinema | Multiple language industries, reviewers, distributors, audience partners | New controversies/reception themes versus release calendars and paid promotion |
| Music | Independent/regional artists, labels, reviewers, festival partners | New listener discussion versus sound-template reuse or organized fandom |
| Gaming | Players, esports organizers, game publishers, community partners | New gameplay/community issues versus update schedules and giveaway spam |
| Technology | Product publishers, independent reviewers, developer communities | New adoption concerns versus launches, affiliate promotion, and recycled AI claims |
| Education | Institutions, educators, student partners, official boards | New learning/exam concerns versus annually repeated admission and exam cycles |
| Jobs | Employers, career educators, worker-community partners, official notices | New hiring/skills concerns versus stale vacancies, recruitment scams, and reposts |
| Health | Public health institutions, qualified communicators, authorized wellness partners | New questions/claims versus unverified cures and commercial endorsements |
| Food | Regional cooks, reviewers, small businesses, community partners | New recipes/practices versus seasonal repetition and restaurant advertising |
| Travel | Transport/tourism bodies, local guides, traveler partners | New disruptions/routes/experiences versus destination promotion and old footage |
| Civic | Municipal/service authorities, local reporters, resident partners | New service complaints versus unresolved recurring grievances |
| Climate | Weather/environment bodies, researchers, local organizations | New impacts/actions versus seasonal weather and miscaptioned disaster media |
| Business | Businesses, entrepreneurs, economic reporters, public filings/notices | New practices/economic concerns versus brand campaigns and unsupported financial claims |
| Fashion | Designers, reviewers, retailers, regional creator partners | New aesthetics/practices versus product launches and purchase prompts |

These examples define collection strata, **not current trending topics**. Do not rank domains by supposed Indian popularity without observed data.

Recommended source lanes:

1. **Authorized Instagram discussion lane:** Partner/owned professional media with permitted caption, comment, and reply collection. Recruit several independent partners per domain where feasible. Authorization applies to the permitted workflow, not to unrelated user data.
2. **Professional metadata lane:** Business Discovery for known eligible accounts and supported public fields. Use it for context and permitted engagement snapshots; missing comment text is an explicit limitation.
3. **Hashtag exploration lane:** Approved queries under the documented budget; record top versus recent surface separately. Do not solve quota constraints with account rotation or access-control evasion.
4. **External event/context lane:** Licensed publisher feeds, official institutional releases, public schedules, and appropriately permitted search/news sources. A news aggregator such as GDELT could be evaluated later, but its current capabilities were not fetched or verified for this review.
5. **Human discovery lane:** Analysts nominate candidates from permitted sources with URL, timestamp, language, and reason. Freeze nominations by date to avoid hindsight-assisted evaluation.

External sources corroborate event context, not Instagram prevalence. Two websites copying one press release count as one underlying source, not independent confirmation. A discussion may genuinely emerge on Instagram before external coverage; absence of news is not proof that discussion is false or unimportant.

### Geography and India relevance

Keep `subject_geography`, `publisher_declared_geography`, `media_location_claim`, and `audience_geography` separate. Assign evidence labels such as explicit verified declaration, post-level location claim, contextual India relevance, conflicting evidence, and unknown. A location tag may refer to the pictured subject, not the person posting. Indian-language material may originate in diaspora communities; English material may originate in India.

**Proposal:** Default residence and audience location to unknown unless permitted, reliable evidence exists. Do not infer city, nationality, caste, religion, or other sensitive attributes from names, language, appearance, or political content. Aggregate region-specific findings only when evidence and sample size support them, and show the unknown-geography fraction alongside every regional report.

## 5. Proposed Emergence Analysis

This section is an engineering design derived from the literature, not a claim that any source implements this exact pipeline.

### 5.1 Observation unit and time semantics

Use a post/thread bundle as the main semantic context, while preserving post and comment units for temporal counting. Captions describe the publisher's framing; comments can introduce different questions, disagreement, jokes, or claims. Analyze them separately before linking themes to avoid assigning every comment to the caption's subject.

Store publication time, comment time, first observation time, and each engagement snapshot time. Lifetime likes observed today are not likes earned today. A single late snapshot cannot reconstruct historical engagement velocity. For replay, information becomes available only at its actual first-observation time, even when an item carries an earlier publication timestamp.

### 5.2 Exposure-aware temporal evidence

For topic $k$, stratum $s$, and interval $t$, define $n_{k,s,t}$ as distinct eligible posts assigned to the topic and $N_{s,t}$ as all eligible observed posts in the same monitored stratum. A descriptive smoothed share is:

$$
p_{k,s,t}=\frac{n_{k,s,t}+\alpha}{N_{s,t}+\alpha+\beta}.
$$

The positive smoothing constants $\alpha$ and $\beta$ must be selected on development data. This is a proposed panel statistic, not Kleinberg's original formula or a population estimator. Track active accounts, request success, query mix, and media availability alongside the denominator. Shares can rise because competing topics fall; therefore show both absolute counts and shares.

Compare current share against a past-only baseline with matching weekday/hour where enough history exists. Use daily bins initially; add finer bins only where sampling density supports them. Compare a simple rolling rate-change baseline, Kleinberg-style burst intervals, and EDCoW-inspired event grouping. Low-count strata require uncertainty intervals and abstention, not dramatic percentage-growth headlines.

Aggregate only with fixed, disclosed panel weights, and also report within-stratum results. Changing weights must create a new series version. A pooled burst should not hide a coverage shift or a burst restricted to one publisher.

### 5.3 Novelty and sustained independent support

Retrieve related material from a past-only archive using lexical terms/entities and multilingual embeddings [S6-S8]. Check whether the difference is a new subject, a new event instance, a new claim, or only paraphrase. An embedding-distance outlier is a review candidate, not proof of novelty. Preserve topic split/merge lineage and inspect representative documents in the original languages.

**Initial candidate gate, proposed:** At least 10 distinct posts from five apparently independent publishers across two daily bins, with a rise above a preselected baseline and no known collection discontinuity. This high-confidence gate will miss small local discussions; retain a separate low-volume analyst-review queue rather than labeling them unimportant. Publisher independence is evidence-based and uncertain: reposting, common ownership, paid campaigns, or near-identical text can reduce effective support.

Distinguish a **candidate** at first support, **emerging** after repeated temporal evidence, **sustained** after persistence, and **resurgent** when history matches. Assign confidence across semantic coherence, temporal growth, independence, historical coverage, and collection quality. Keep claim verification as a separate field. Do not bury missing evidence in a single opaque score.

### 5.4 Comments and conversation quality

Prefer complete permitted pagination or authorized webhook collection with documented coverage. If full collection is infeasible, predeclare a sampling method and preserve pagination/order/time-window metadata. Reservoir sampling is only representative of the traversed accessible stream, not of inaccessible comments or viewers. Store reply linkage where allowed and flag missing parents.

Cap or separately weight repeated contributions by one commenter when identifiers are legitimately available. Count substantive discussion, questions, disagreement, purchase prompts, spam, and emoji-only responses as distinct categories. Do not equate positive sentiment with endorsement, engagement with agreement, or a request for a purchase link with conversion.

Publish a comments-available subset analysis and a captions-only analysis. Never fill missing comment text with zero discussion or silently compare authorized partners with metadata-only accounts as though their measurements were equivalent.

## 6. Multimodal Gaps and Misinformation

Instagram discussion is not reducible to captions. Reels can communicate through speech, on-screen text, image sequences, edits, gestures, music, and prior meme knowledge. Image-only screenshot text is precisely the kind of invisible textual behavior highlighted by Tufekci [S12]. CLIP provides evidence for image-text representation learning [S15], not a complete Reel-understanding system.

**Proposal:** Begin with a clearly labeled text-observable scope. For permitted assets, later compare caption-only, caption-plus-OCR, caption-plus-ASR, and full available-modality variants. Record missingness and confidence for each modality. Evaluate OCR character error rate and ASR word error rate on the local languages/scripts; translation and extraction errors must remain inspectable. Do not claim audio-trend coverage from captions or infer that a soundtrack's reuse measures agreement with a claim.

Deduplicate exact reposts and, where rights and tooling permit, near-duplicate images/video segments. Treat old footage attached to a new caption as a separate verification risk. Do not use faces to infer demographic or geographic attributes.

The Twitter results of Vosoughi et al. show that novelty and diffusion can coexist with falsehood [S14]. They do not supply an Instagram misinformation detector or justify assuming every burst is manipulated.

**Proposal:** Maintain claim states of not assessed, independently supported, disputed, false according to cited verification, and insufficient evidence. Preserve who made the claim and the verification time. Use primary notices, credible independent reporting, and qualified fact-checking where available. High-risk health, civic-emergency, election, and financial claims require human review before public presentation. A trend alert should say that a claim is circulating, not assert the claim as fact. Multiple copied posts or high engagement are not independent factual confirmation.

## 7. Evaluation and Acceptance Criteria

All numerical gates below are **proposed pilot acceptance criteria**, not published findings, achieved results, or universal operating thresholds. Agree them before the locked evaluation, then revise only for a subsequent version.

### 7.1 Temporal benchmark design

Collect a minimum eight-week pilot. Use weeks 1-4 for the initial archive/baseline, weeks 5-6 for development and threshold selection, and weeks 7-8 as a locked chronological holdout. Run at least two past-only development replay origins if sufficient labels exist; never train topic representations or select thresholds using the locked future period. Eight weeks cannot establish annual seasonality; label recurrence confidence as limited unless permitted older records exist.

Recruit multilingual annotators to label coherent topic identity, event identity, novelty type, first observable support, recurrence, independence concerns, and factual-claim status separately. Include alerts from every baseline and random non-alert bundles so missed events are not invisible to evaluation. Use at least 200 adjudicated candidate/non-alert bundles for a pilot, with at least 30 labeled examples per reported priority language/script group where feasible. These are minimum reporting floors, not statistical power guarantees.

Use two independent annotators and adjudication; measure agreement before adjudication. Provide history available at each decision time, not later outcomes. Define a one-to-one event-matching policy before scoring to prevent repeated alerts about one event inflating recall. Keep syndicated/near-duplicate documents grouped during split construction. Report a separate unseen-publisher holdout to expose publisher-template shortcuts.

### 7.2 Metrics and release gates

| Target | Metric and measurement | Proposed acceptance gate |
| --- | --- | --- |
| Sampling integrity | Scheduled eligible collection windows completed, by lane/stratum; known outage flags | At least 95% completion overall and 90% in each supported core stratum; suppress growth claims during unresolved collection discontinuities. |
| Provenance | Records with source lane, access basis, observation/publication times, panel version, and missingness flags | 100% of evidence used in released alerts; no secret-bearing material in research artifacts. |
| Annotation reliability | Krippendorff's alpha for novelty/topic labels before adjudication | At least 0.67 for exploratory use; at least 0.80 before automated external publication of those labels. Otherwise revise guidelines or keep human review. |
| Alert precision | Precision@20 per day for coherent, temporally supported panel-emergence alerts; days with fewer alerts use their actual count | At least 0.80 pooled holdout precision, with a 95% interval reported; no supported priority group below 0.65. Separate topic-novelty precision from new-event precision. |
| Event coverage | Event recall against the adjudicated, panel-observable event inventory | At least 0.60; do not call this recall of all Instagram or all Indian events. Include random non-alert review in the inventory. |
| Early detection | Median and 90th-percentile delay from adjudicated first observable qualifying support to first matched alert | Median at most 24 hours for daily sampling; report the 90th percentile and collection-latency floor. No negative delays caused by future-data leakage. |
| Redundancy | Multiple matched alerts for the same event within seven days | At most 10% redundant alerts after applying the predefined matching policy. |
| Topic quality | Human coherence/label accuracy and B-cubed precision/recall/F1 on labeled document groups; lexical NPMI as a secondary diagnostic | At least 80% human-coherent evaluated topics; no adoption decision based on coherence alone. Report outlier rate and incorrect merges. |
| Language robustness | Same-topic pair retrieval nDCG@10 and novelty precision by native script, Romanization, and code-mixing | Selected multilingual approach improves nDCG@10 by at least 0.05 over the lexical baseline in target groups and satisfies precision floors; report confidence intervals and small-group uncertainty. |
| Historical novelty | Rate of alerts labeled new that adjudicators identify as prior-panel recurrence | At most 15%; report lookback and unknown-history cases separately. |
| Geography integrity | Unsupported residence labels; explicit geography-evidence and unknown rates | Zero residence assignments based solely on language/hashtags; 100% of regional alerts carry evidence tier and unknown fraction. |
| Sensitive factual claims | Released high-risk alerts with verification status and human review | 100%; no unassessed claim presented as established fact. |
| Efficiency | End-to-end latency, inference time, requests consumed, and analyst minutes per accepted alert | P95 processing latency below the chosen 24-hour reporting interval; remain within approved access budgets. Measure analyst effort before setting a cost gate. |

Report macro-averages across domains and languages, not only micro-averages dominated by major accounts. Cluster/bootstrap uncertainty at publisher and day level where appropriate; ordinary document-level resampling understates dependence. If the holdout is too small for a reliable interval, disclose that and withhold strong comparative claims.

### 7.3 Baselines, ablations, and disconfirmation

Compare raw volume ranking, past-normalized lexical growth, LDA plus temporal counts, burst detection, and multilingual embedding topics plus temporal evidence. Keep collection exposure and labeling budgets comparable. Do not compare a model with comments to a captions-only baseline without a modality ablation.

Required ablations: hashtags-only versus full authorized text; captions versus comments; lexical versus dense versus hybrid historical retrieval; English-centric versus multilingual modeling; native text versus translated text; raw versus deduplicated volume; lifetime engagement versus timestamped snapshots; fixed core versus expanded panel; retrospective BERTopic versus past-only replay.

The central falsifiable hypothesis is: **A stratified multilingual panel with exposure-aware temporal evidence improves human-validated emerging-discussion discovery over a fashion/hashtag volume baseline without unacceptable language-specific errors.** Reject or narrow it if apparent gains disappear under fixed-panel replay, unseen-publisher evaluation, deduplication, or per-language scoring. If BERTopic improves coherence but not alert precision or delay, retain it for organization rather than claiming improved emergence detection.

## 8. Concrete Collection Plan and Access Preflight

This is a proposed plan; it was not executed during the review.

### Stage A: Access and governance, before collection

1. Establish the research purpose, retention/deletion rules, participant notices, applicable legal/contractual basis, and permitted downstream uses. Public visibility is not blanket permission to copy, retain, or republish.
2. Obtain documented partner authorization and the relevant Meta app review/access. Record the chosen login product, API version, approved permissions, allowed fields, endpoint constraints, and date of verification. Keep authentication configuration outside research exports; never place tokens, cookies, or passwords in logs or documentation.
3. Verify capabilities through the official references and an authorized operator-run preflight. Capture whether captions, media metadata, comment text, replies, timestamps, and metric snapshots are actually available. No such credentialed preflight was performed for this review.
4. If access is unavailable, proceed only with permitted offline/partner-provided exports or a metadata-only feasibility study. Do not silently substitute private API automation or promise comment coverage.

### Stage B: Register the panel and sampling budget

**Initial target:** 160 core accounts, approximately 10 per domain, subject to lawful access and recruitment. Balance institutional/publisher, creator/expert, and smaller community/business voices within each domain where possible. Use English, Hindi native script, Romanized Hindi/Hinglish, and a set of regional-language strata selected from available partners; record deficits rather than pretending every domain-language cross-product is filled.

Give accounts one primary quota assignment but allow posts multiple content-domain labels. Maintain account additions/removals, declared language, observable source type, consent/access status, eligibility exclusions, and geography-evidence tier in a panel register. Preserve an overlap cohort when changing the panel so trends can be recomputed on the stable core.

For one eligible approved hashtag requester, allocate at most 30 unique queries across any rolling seven days: for example, 16 fixed domain sentinels, eight language/regional exploration queries, and six candidate queries. A single sentinel per domain is weak coverage; this lane supplements the account panel. Track first-query expiry times because changing a calendar week does not reset the rolling limit. Do not issue sensitive/offensive queries or unsupported query forms; retain failures as missingness.

### Stage C: Eight-week longitudinal pilot

Use scheduled six-hour discovery/metadata windows where approved budgets allow, then aggregate daily for the initial analysis. For authorized-media comments, prefer webhooks with periodic permitted reconciliation; otherwise record polling delays and pagination completeness. Observe newly eligible posts and, where supported, take engagement snapshots at first observation, approximately 6 hours, 24 hours, 72 hours, and 7 days. These are proposed measurement times, not guaranteed API availability.

Record per-item source identifier or approved pseudonym, canonical source reference, original text, access lane, query/surface, domain labels, publication/observation times, language/script confidence, parent/reply linkage where allowed, snapshot time, metric availability, and panel/model versions. Record failed requests, partial pages, inaccessible/deleted media, disabled comments, and modality availability. Distinguish not observed, inaccessible, unsupported, and genuinely zero.

Avoid retaining unnecessary user identifiers or raw media; use access-controlled originals only when necessary and permitted. Minimize potentially identifying quotations in public output, especially from ordinary commenters. Apply approved retention and deletion schedules, and retain aggregate evidence only as permitted. Do not store direct messages or private-profile content for this panel.

### Stage D: Analyze, review, and report

Freeze the core-panel baseline and development choices before holdout replay. Issue an internal daily candidate report and a weekly analyst-reviewed synthesis. Each candidate should show topic/event/novelty labels, count/share trajectories, independent-publisher support, language distribution, geographic uncertainty, representative evidence references, lookback, sampling caveats, factual-claim status, and missing modalities.

The first deliverable is a **coverage and feasibility report**, not a national trending leaderboard. Continue to discovery claims only after collection and evaluation gates are met. Publish failed hypotheses and undercovered domains so a high-volume fashion or cinema stratum does not quietly redefine the system's scope.

### Official access conclusions as of this review

The three Meta guides were readable on 2026-10-03 [S16-S18]. That verifies their displayed documentation, **not availability for a particular app, account, jurisdiction, or API version**. No authorized API request was executed. Re-check before implementation because living documentation can change.

The Hashtag Search guide explicitly documents the 30-unique-hashtag rolling limit, Public Content Access review, and `instagram_basic`. The Comment Moderation guide lists `instagram_business_basic` and `instagram_business_manage_comments` for Instagram Login, and `instagram_basic`, `instagram_manage_comments`, and `pages_read_engagement` for Facebook Login, with additional permissions under specified Business Manager conditions. Do not combine the two permission sets or assume the products expose identical discovery capabilities.

Business Discovery shows metadata/metrics through field expansion, not unrestricted direct media/comment access. Its fetched guide does not list a follower relationship as a prerequisite. Endpoint-specific permissions still need verification; absence of that claim in a guide is not proof that every requested target is accessible.

The comment guide is explicitly about media owned by app users. It distinguishes comment and reply operations and recommends webhooks to reduce rate limiting. This review did not verify a universal maximum of 50 comments per query or a universal reverse-chronological order. Those older notes must not become assumptions in a collection design.

The instagrapi maintainer page itself calls the library unofficial and describes private mobile flows and production fragility [S19]. Its installation does not establish compliant access. The recommended collection plan excludes private endpoints and evasion tactics. Vendor licensing also requires review of the vendor's rights and provenance; a paid service is not automatically approved collection.

## 9. SHEIN: Supported Analogy, Unsupported Algorithm Claims

The readable 2022 corporate announcement says SHEIN's small-batch process can produce varied items before increasing output or ceasing production based on market feedback [S20]. Its current corporate overview emphasizes customer preferences and purchases rather than relying on trend forecasts. Together these support a **small-batch test-and-repeat process analogy**, with the important qualification that this is company-reported evidence.

They do not establish that SHEIN uses this project's embeddings, clusters Instagram comments with a specific model, weights purchase requests by a fixed multiplier, or applies a disclosed India-specific trend algorithm. No universal initial batch quantity, three-day production promise, or precise conversion uplift is established by the fetched pages. A process description is not proof of commercial forecasting accuracy.

**Proposal:** Borrow the experimental discipline, not an imagined proprietary implementation: observe a candidate, run a limited authorized content/product experiment, measure response against a control, and decide whether to expand. For discussion discovery, success means timely, coherent, independently supported identification. For merchandising, success requires separate conversion, margin, returns, stockout, and demand measurements. Do not transfer buying-intent measures into politics, health, civic discussion, or general public attention.

## 10. Fetch Audit, Corrections, and Remaining Gaps

### Sources actually fetched and used

Successful content extraction covered all 20 table entries via the following **25 public URLs**. Some returned abstracts or guides rather than complete papers; the table states that boundary.

1. https://www.cs.cornell.edu/home/kleinber/ - author's bibliography confirms the burst paper's 2002 venue/year.
2. https://www.cs.cornell.edu/home/kleinber/kdd02.html - author's burst-method explanation and examples.
3. https://ojs.aaai.org/index.php/ICWSM/article/view/14102 - EDCoW abstract and issue metadata.
4. https://www.jmlr.org/papers/v3/blei03a.html - LDA abstract and metadata.
5. https://arxiv.org/abs/2203.05794 - BERTopic abstract.
6. https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html - dynamic topic guide.
7. https://maartengr.github.io/BERTopic/getting_started/online/online.html - incremental topic guide.
8. https://arxiv.org/abs/1908.10084 - Sentence-BERT abstract.
9. https://arxiv.org/abs/2402.05672 - multilingual E5 report abstract.
10. https://huggingface.co/intfloat/multilingual-e5-large - model card and usage/limitations.
11. https://arxiv.org/abs/2103.10730 - MuRIL abstract.
12. https://aclanthology.org/2020.findings-emnlp.445/ - IndicNLPSuite abstract and publication metadata.
13. https://aclanthology.org/2020.semeval-1.100/ - SentiMix abstract and publication metadata.
14. https://arxiv.org/abs/2108.08787 - Mr. TyDi abstract.
15. https://arxiv.org/html/2108.08787v2 - full Mr. TyDi HTML, including Table 1 languages.
16. https://arxiv.org/abs/1403.7400 - Tufekci methodological abstract.
17. https://arxiv.org/abs/2102.01951 - temporal generalization abstract.
18. https://www.science.org/doi/10.1126/science.aap9559 - misinformation publisher abstract and metadata.
19. https://arxiv.org/abs/2103.00020 - CLIP abstract.
20. https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login/hashtag-search - official guide.
21. https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login/business-discovery - official guide.
22. https://developers.facebook.com/docs/instagram-platform/instagram-api-with-instagram-login/comment-moderation - official guide.
23. https://pypi.org/project/instagrapi/ - maintainer package description and risk statement.
24. https://www.sheingroup.com/our-group - current corporate process overview.
25. https://www.sheingroup.com/newsroom/shein-leverages-digital-thermal-transfer-printing-technology-to-support-small-batch-production - dated small-batch/market-feedback announcement.

The two Kleinberg pages together support S1, and related E5, BERTopic, Mr. TyDi, and SHEIN pages are grouped into source entries rather than counted as additional independent studies.

### Failed, partial, or excluded retrievals

- Kleinberg's PDF at https://www.cs.cornell.edu/home/kleinber/bhs.pdf could not be meaningfully extracted. The attempted `bhs.html` URL returned a multiple-choice/missing-document page. The DOI https://doi.org/10.1145/775047.775061 redirected to https://api.crossref.org/v1/works/10.1145/775047.775061/transform, which returned Not Acceptable. The readable author pages are the evidence used instead.
- https://castorini.github.io/mr.tydi/ and https://aclanthology.org/2021.multirep-1.12/ returned Not Found. Neither is evidence; the full arXiv HTML supplied the verified language list.
- https://arxiv.org/abs/2005.11401 was fetched but is **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**, not the intended temporal-generalization paper. It is excluded from the evidence table; the correct source is S13.
- https://subzeroid.github.io/instagrapi/ failed meaningful extraction, and https://subzeroid.github.io/instagrapi/about.html returned Not Found. S19 uses the readable maintainer-authored PyPI description.
- https://www.sheingroup.com/our-business/our-business-model/, https://www.sheingroup.com/our-business/, https://www.sheingroup.com/our-group/business-model/, and https://www.sheingroup.com/our-business/our-on-demand-business-model/ failed meaningful extraction. The homepage https://www.sheingroup.com/ was readable mainly as navigation, not evidence for batch quantities. S20 uses the current group page and linked announcement.
- The Science page exposed the relevant abstract and publication metadata, but full-article access was restricted. No full-text verification or newly calculated effect sizes are claimed.

### Corrections to prior assumptions

Mr. TyDi does not cover Hindi, Gujarati, Marathi, or Tamil. Business Discovery's fetched guide does not establish a follower prerequisite. Dynamic BERTopic is not synonymous with incremental training. An ordinary authenticated Instagram account does not by itself establish the official professional-account/app-review prerequisites. A public post or discovered media ID does not establish permission to read arbitrary comments. Previously stated comment-page sizes, ordering guarantees, SHEIN production quantities, and proprietary algorithm claims are not carried forward as verified facts.

### Research gaps

This is a targeted methodological survey, not a systematic or exhaustive review of Indian Instagram studies. It does not provide nationally representative demographic estimates, a measured October 2026 trend list, or direct evidence that one model is best on this project's data. Several foundational methods were evaluated on Twitter, Wikipedia, news, or generic vision datasets rather than Indian Instagram discussion.

## 11. Addendum: Novelty, Popularity Dynamics, and Multilingual Embedding Sources

Added 2026-10-03 after the first review. These sources fill the gap between "burst" (Section 3.1) and "new". They also ground the choice of embedding model and training loss used in `src/identify_what_matters/topic_finetune.py`.

| ID | Title, authors, venue, URL | Evidence used | How it is applied here | Limits |
| --- | --- | --- | --- | --- |
| S21 | **Streaming First Story Detection with application to Twitter**, Petrovic, Osborne, Lavrenko, NAACL 2010, pp. 181-189. [ACL Anthology](https://aclanthology.org/N10-1021/) | Title and bibliographic record (abstract/PDF extraction failed). First story detection (FSD) is the Topic Detection and Tracking task of flagging the first document about a previously unseen event, made streaming for Twitter. | `emergence.topic_emergence` computes, for every current-window cluster, its nearest similarity to the past-only baseline (`novelty = 1 - max cos`), the FSD idea applied at cluster level. A cluster with no baseline neighbours above the calibrated threshold is labelled `new`. | Only metadata verified; no claim is made about its LSH details or reported scores. FSD flags the first document; this project requires multi-author support before reporting. |
| S22 | **Trends in Social Media: Persistence and Decay**, Asur, Huberman, Szabo, Wang, 2011. [arXiv:1102.1402](https://arxiv.org/abs/1102.1402) | Abstract: studies Twitter trending topics, says user activity and follower counts do not strongly drive trend formation, and identifies content resonance as a major factor. | Emergence counts **distinct authors**, not followers or raw likes. Lifetime engagement is reported separately as heat, never as emergence. | Twitter 2010-era evidence; not an Instagram India estimate. |
| S23 | **Predicting the Popularity of Online Content**, Szabo and Huberman, 2008. [arXiv:0811.0405](https://arxiv.org/abs/0811.0405) | Abstract: early access measurements predict long-term popularity on Digg/YouTube; the useful observation horizon differs by platform, and fast-decaying content is more predictable. | Engagement velocity uses repeated snapshots (`snapshot_delta`) when two observations exist. Otherwise it uses a labelled single-snapshot age-normalised estimate. Re-observing posts at 6/24/72 h is the recommended schedule. | The decay horizon for Instagram Reels is not established by this source and must be measured. |
| S24 | **Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation**, Reimers and Gurevych, EMNLP 2020. [arXiv:2004.09813](https://arxiv.org/abs/2004.09813) | Abstract: a translated sentence should map to the same vector as its source. A student model is trained to mimic a monolingual teacher, with demonstrations for 50+ languages. | The source of `paraphrase-multilingual-mpnet-base-v2`, the closest multilingual counterpart of the local `all-mpnet-base-v2` (same 768-d MPNet-family space and an XLM-R tokenizer). Candidate base model for fine-tuning. | Translation-parallel objective; Hinglish and Instagram slang are not demonstrated. |
| S25 | **Language-agnostic BERT Sentence Embedding (LaBSE)**, Feng, Yang, Cer, Arivazhagan, Wang, ACL 2022. [arXiv:2007.01852](https://arxiv.org/abs/2007.01852) | Abstract: combines MLM, TLM, dual-encoder translation ranking and additive margin softmax; 83.7% Tatoeba bitext retrieval over 112 languages; 109+ languages released. | Baseline for cross-script same-event retrieval. | Bitext retrieval rewards translation equivalence, not topical same-event grouping of loosely paraphrased posts. |
| S26 | **Efficient Natural Language Response Suggestion for Smart Reply**, Henderson et al., 2017. [arXiv:1705.00652](https://arxiv.org/abs/1705.00652) | Abstract: dual encoders trained so message-response pairs get high dot product, with the batch providing negatives. | Origin of the in-batch-negative objective implemented as `MultipleNegativesRankingLoss`. The project adds one same-domain hard negative per pair. It uses an event-disjoint batch sampler so another text of the same event is never an in-batch negative. | Batch-negative training assumes batch items are true negatives, which motivates the custom sampler. |
| S27 | **Fashion Forward: Forecasting Visual Style in Fashion**, Al-Halah, Stiefelhagen, Grauman, ICCV 2017. [arXiv:1705.06394](https://arxiv.org/abs/1705.06394) | Abstract: unsupervised style discovery from 80,000 Amazon product images over six years. A forecasting model then predicts style popularity. The authors report that visual analysis helps forecasting much more than textual or metadata cues. | Peer-reviewed evidence for the fast-fashion question: the fashion forecasting it supports is visual-attribute time series over sales data, not comment keyword counting. For fashion, the text pipeline here is a discussion signal only; visual style trends need an image model (CLIP, S15) and sales or exposure data. | Amazon product sales, not Instagram; does not describe SHEIN's system. |

### 11.1 What changed in the design because of these sources

1. **Three separate quantities.** *Discussion volume* is the number of distinct authors, posts and comments. *Emergence* is the exposure-normalised rate against a trailing baseline, plus nearest-neighbour novelty (S1, S21, S22). *Heat* is engagement velocity from snapshots (S23). The earlier `velocity_score` (lifetime likes + 4.5 x comments) measured none of these and is retained only as `lifetime_engagement_not_velocity`.
2. **Burst statistic.** For a term or cluster with $c$ distinct current authors, the observed author exposure is $E$. The smoothed baseline rate is $\hat r = (b + \lambda\,\bar r)/(B + \lambda)$, where $b$ is the term's baseline authors, $B$ is the baseline author exposure, and $\bar r$ is the background rate. The expected count is $\mu = \hat r E$, and the reported surprise is the signed root Poisson likelihood ratio $\sqrt{2\,(c\ln(c/\mu) - (c-\mu))}$, which is approximately a z-score. A term needs at least 3 distinct authors and 2 posts; a baseline count of 0 means `new`, and a growth of at least 2x means `rising`. These are engineering choices in the spirit of S1/S2, not formulas from those papers.
3. **Model choice is empirical.** See Section 11.2.

### 11.2 Local empirical evidence (this repository, 2026-10-03)

**Tokenizer coverage.** `all-mpnet-base-v2` uses an English WordPiece vocabulary. Measured locally:

- A Tamil phrase tokenised to only `[UNK]` tokens.
- Hindi and Bengali were split into isolated consonants, with vowel signs dropped or mapped to `[UNK]`.
- Hinglish was split into English sub-words.

Fine-tuning cannot recover script information that the tokenizer discards. A base model with an XLM-R/SentencePiece vocabulary is therefore required for native-script Indian languages.

**Similarity probe.** With base MPNet, a Hinglish heatwave comment scored 0.49 against a Hindi metro-fare caption and 0.51 against a Hinglish metro-fare comment. It scored only 0.36-0.48 against English heatwave captions. In this probe, script and language similarity outweighed topic similarity.

Model comparison results on the Sonnet event corpus are reported in `docs/TRAINING_RESULTS.md`.

Priority next evidence is authorized collection feasibility, a multilingual annotated Instagram discussion sample, longitudinal coverage stability, a past-only event inventory, and modality-specific error analysis. A longer archive is needed for annual recurrence, and underrepresented languages and communities need explicit recruitment rather than extrapolation. Until those gaps are addressed, present results as **observed-panel discussion signals with uncertainty**, not what India as a whole thinks or what Instagram as a whole is discussing.

## 12. Addendum (2026-10-03): Detecting Debate, Not Just Volume

These sources were fetched and read at abstract level or better. They support the discussion-first redesign described in `docs/ARCHITECTURE.md`.

| ID | Source | Finding used | Limitation |
| --- | --- | --- | --- |
| S28 | J. Hessel, L. Lee. *Something's Brewing! Early Prediction of Controversy-causing Posts from Discussion Features.* NAACL 2019, pp. 1648-1659. https://aclanthology.org/N19-1166/ | Controversy means a post splits a community's preferences. Features from the first few comments (tree structure plus text) add predictive power over content-and-rate baselines. Structural features transfer across communities better than content features. | Studied on Reddit, where downvotes are visible. Instagram exposes no dislikes, so a split must be inferred from comment content and reply structure. |
| S29 | J. R. Minot et al. *Ratioing the President.* PLOS ONE 16(4): e0248880, 2021. https://doi.org/10.1371/journal.pone.0248880 | Reply-heavy activity ratios mark contested posts. Ratioed tweets over-represent contentious vocabulary. Response activity stabilises around 1 day, and most inflection points come before about 1.15 days. | Covers two US presidential accounts. The ratio has to be normalised per account. Here, comments relative to likes against the account median stand in for replies relative to retweets. |
| S30 | K. Garimella et al. *Quantifying Controversy on Social Media.* ACM TSC; arXiv:1507.05224. | A graph pipeline (conversation graph, partition, random-walk controversy score) captures controversy, and content features help much less. | Needs an interaction graph. Instagram comment replies are only partly visible through the sampled endpoints. Kept as future work. |
| S31 | CrowdTangle "overperforming" score (industry practice). | Engagement normalised against the account's own expected engagement. | The vendor help page could not be retrieved for this review, so the formula used here is our own (`docs/ARCHITECTURE.md` §2). |

What changed empirically:

- Instagram's Explore grid is personalised and mostly weeks old, so it is unusable as a trend source.
- Fixed public account feeds can be read with one request each.
- Comment payloads include `comment_like_count` and `child_comment_count`, which support endorsement and argument signals.
- X (Twitter) India trend timelines surface debated terms that the news-first pipeline missed.
- An earlier chat claim that "70% of Instagram trends spread via audio" and that "80% of Indian meme humour is in image text" has no source and is withdrawn.