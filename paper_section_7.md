# Section 7. Post-Training: Quality-Filtered SFT and the Multi-Dimensionality of "Quality"

## 7.1 Motivation

Sections 4–6 established a clean two-by-two: **data quality determines the sign of the capacity return**. On the strict-filter v1.1 corpus (9.3M pairs), Big exceeds Base by +0.56 BLEU; on the loose-filter v2 corpus (30M pairs), Big falls below Base by −0.87. The natural follow-up question is whether v2's extra 20M pairs are *recoverable*: can we extract the high-quality subset of v2 and use it to lift v1.1's plateau?

This section tests that hypothesis with **quality-filtered supervised fine-tuning (SFT)**. We score the full v2 corpus with a reference-free quality-estimation (QE) model, take the top-1M pairs by score, deduplicate, and SFT Base v1.1 on this filtered subset. The intent is to retain whatever signal v2 contributes beyond v1's strict cutoffs while discarding the noise.

The result is a clean **negative finding**, and the negative finding is itself the contribution: it constrains what "data quality" means.

## 7.2 Method

**Scoring.** We use CometKiwi-22 (`Unbabel/wmt22-cometkiwi-da`), the WMT22+ standard reference-free QE model based on XLM-RoBERTa-XL. Each `(src, mt)` pair receives a scalar quality score in roughly [0, 1]. Scoring v2's 30,129,500 pairs took 13.8 hours on a single RTX 5090 (≈ 750 pairs/sec at batch size 64), with the score range concentrated in [0.5, 0.92].

**Filtering.** We retain the top 1M pairs by score (top 3.3% of v2). Score range of the retained set: 0.8959–0.9145 — a tight 0.02 band, indicating a dense plateau of near-equivalent high-quality pairs above this cutoff.

**Deduplication.** Sorting and unique-ing the top-1M yields 931,366 unique `(src, tgt)` pairs (7% duplicate rate within the top-1M, mostly UN-document boilerplate). All SFT below uses this 931K-pair deduplicated set.

**SFT setup.** We fine-tune Base v1.1's averaged checkpoint (60M parameters, valid BLEU 30.52, test BLEU 35.31) by resuming training with `--reset-optimizer`. The Noam scheduler resumes at scheduler-step 26,250 (post-warmup), giving an effective peak LR of 2.7e-4 — within the 10–20% of pretraining peak band conventional for SFT. Other hyperparameters are unchanged from pretraining: bf16, label smoothing 0.1, batch 24,576 tokens × 4 accumulation, eval every 2K steps. We allow up to 10K SFT steps (≈ 2 epochs over the 931K SFT set).

**Evaluation.** newstest2013 (valid) and newstest2014 (test) sacrebleu 13a, beam=5, length-penalty=1.0. The same SPM (coverage=1.0) used in v1.1 pretraining is reused.

## 7.3 Results

The SFT run early-stopped at step 111,000 (6K SFT steps) under default patience. Validation BLEU monotonically decreased throughout:

| Global step | Δ steps | Valid BLEU | Δ vs baseline |
|------------|--------|-----------|---------------|
| 105,000 (load) | 0 | 30.52 | — (baseline) |
| 107,000 | 2,000 | 29.44 | **−1.08** |
| 108,000 | 3,000 | 29.46 | −1.06 |
| 111,000 | 6,000 | **28.97** | **−1.55** |

Training loss on the SFT set converged smoothly from ~3.5 (pretraining floor on v1) to 1.94 (SFT floor) — the model fit the SFT distribution well; the loss curve gives no indication of optimization failure. Sample translations during SFT remain fluent and grammatically correct; in particular, accented French characters (`Israël`, etc.) are preserved (no `<unk>`) thanks to the v1.1 SPM. The drop is not a translation-quality collapse. It is a **distribution shift**.

## 7.4 Analysis: why the filter selected what it did

CometKiwi-22 is reference-free: it scores translation quality in absolute terms. Inspection of the top-5 highest-scored pairs (after dedup):

```
1. UN disability statistics (en/fr) — formal report register
2. UNAIDS HIV/AIDS prevalence statistics — formal report
3. Government federal-debt announcement — formal/legislative
4. Finland population statistics — formal/statistical
5. (other UN/legislative variants)
```

The top scorers are **dominated by UN, intergovernmental, and legislative-document parallel data**. This is unsurprising: such corpora contain professionally-translated, syntactically aligned, semantically tight pairs — exactly what a QE metric will rank highest. WMT14 newstest, in contrast, is **news domain**: looser register, shorter sentences, conversational interjections, named-entity-rich.

We thus have two distributions:
- **SFT distribution**: high-quality, formal, document-style
- **Eval distribution**: high-quality, informal, news-style

SFT's effect is to pull the model's output distribution toward the SFT distribution. On the SFT distribution itself, the model improves (loss drops from 3.5 to 1.94, presumably BLEU on UN-style test data would increase). On the eval distribution — where we measure — it gets worse, because every step of SFT trades news-style fit for formal-style fit.

**The model is not deteriorating in absolute quality.** It is being relocated in distribution space, and that relocation crosses the eval distribution.

## 7.5 Implication: "data quality" decomposes

This negative result refines the central claim of the paper. Sections 4–6 showed:

> Capacity return is determined by data quality.

Section 7 forces a decomposition of "quality":

> **Data quality = (translation correctness) × (alignment to target distribution).**
> Both dimensions matter, and they are independent. Reference-free QE metrics measure the first; they do not measure the second.

A naive top-K-by-QE filter optimizes only the first axis, and on a heterogeneous corpus (which v2 is — news, web, UN, Europarl, all mixed) this can move the data far from the target eval distribution. The cleaner the QE filter, the more pronounced the domain skew, because high-quality parallel data is disproportionately UN/legislative.

This generalizes: any post-training step that filters on absolute quality without conditioning on the eval distribution is liable to move the model away from where it should land. The fix is not "better QE" — CometKiwi-22 is doing exactly what it should — but **domain-conditional filtering**: filter by QE within news-domain candidates, or sample uniformly across domains within the QE band, or apply a domain classifier as a hard pre-filter.

## 7.6 Why the v2-pretraining failure mode is *different*

It is worth noting that v2's pretraining failure (Section 5: Base v2 lost 0.79 BLEU vs v1, Big v2 lost 0.87 to Base v2) is **not** the same mechanism as Section 7's SFT failure. v2 pretraining failed because the corpus contains **noisy** pairs — outright mistranslations, misalignments, repeated boilerplate — which inject incorrect supervision signal. Quality-filtering by CometKiwi *does* address that: spike-guard triggers in pretraining and visual inspection of low-scored v2 pairs both confirm v2 has genuine noise.

Section 7's SFT failure is a separate axis: even after the noise is filtered out, the *clean* high-quality remainder is still the wrong distribution for newstest. Quality filtering succeeded at its job; the job was the wrong job.

## 7.7 Limitations and what we did not run

- **Single seed.** We did not repeat the SFT run with different seeds. Variance in SFT outcomes is small relative to the −1.55 BLEU effect, but a 3-seed mean would tighten the bound.
- **Did not test low-LR SFT.** A peak LR ≤ 5e-5 might find a "shallow SFT" point that lifts BLEU on UN-style without moving far enough to hurt newstest. We expect the effect to scale with effective SFT distance from baseline; very-low-LR SFT would be slow to test (need many more steps to converge), and the paper's claim does not require it.
- **Did not test Big v1.1 SFT.** Big has more parameters and may be more or less susceptible to distribution shift. Replication on Big is left to a follow-up; it is not central to the conceptual result.
- **Did not test domain-conditional filtering.** Constructing a news-conditioned top-K filter is a nontrivial sub-project (domain classifier + quality scorer + joint sampling). We flag it as the natural fix and leave it for future work.

## 7.8 Headline result

**Quality-filtered SFT, by itself, does not improve newstest BLEU when "quality" is measured by reference-free QE on a multi-domain corpus.** It costs 1.55 BLEU on Base v1.1. The reason is not that the filter failed — the filter selected genuinely high-quality pairs — but that absolute translation quality and alignment to the eval distribution are independent axes of "quality", and only the first was optimized.

The negative result tightens the paper's prescription:

> **Filter first (for noise), match the evaluation distribution second (for register/domain), and only then scale capacity.** Skipping the second step undoes capacity returns from the first.
