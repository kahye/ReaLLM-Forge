# Korean Hangul Factorizer & POS Multicontext Benchmark Suite

This package organizes the training pipelines, downstream evaluation suites, and comparative report generators for the 24-lane / 25-lane Korean Hangul Compositional Factorizer and POS-Hybrid Multicontext experiments.

## Directory Structure

- **`training/`**: Training pipelines, 20-epoch AdamW & Muon model suites, and loss-weight hyperparameter grid searches:
  - `run_all_20ep_training.py` — 20-epoch AdamW training pipeline across all tokenization & factorization architectures.
  - `run_all_muon_20ep_suite.py` — 20-epoch Muon optimizer training and evaluation suite.
  - `run_nochar_hybrid_pos_experiments.py` — 24-lane No-Char Hangul Hybrid POS experiment runner.
  - `run_pos_byte_experiments.py` — 256-Byte fallback companion stream experiments.
  - `run_new_weighted_nochar_20ep_training.py` — Weighted 24-lane No-Char 20-epoch training.
  - `run_charhead_h11_20ep_training_and_eval.py` & `run_charhead_unweighted_20ep_training_and_eval.py` — Character-head (`lm_head_24`) ablation training and evaluation pipelines.
  - `run_exhaustive_weight_search.py` & `run_charhead_exhaustive_weight_search.py` — Structured multi-task loss-weight searches.
  - `run_opt1_10ep_experiment.py`, `run_option1_sweep.py`, `run_three_hot_59m_experiments.py` — Sweep and baseline comparison runners.

- **`evaluations/`**: Downstream benchmark runners and full-dataset evaluation suites:
  - `run_all_eval_suites.py` — Unified evaluation harness for all checkpoints and decoding strategies (`with_lm_head_24`, `without_lm_head_24` greedy, and joint beam search).
  - `run_ko_hellaswag_1000_all_34_models.py` — Ko-HellaSwag ($N=1,000$) commonsense reasoning evaluation.
  - `run_kormedmcqa_1000_all_34_models.py` & `run_kormedmcqa_all_3009_models.py` — KorMedMCQA medical QA benchmark evaluation.
  - `run_kobalt_cute_1000_all_34_models.py` — KoBALT & CUTE-Hangul ($N=1,000$) character/sub-character manipulation evaluation.
  - `run_noise_resilience_1000_all_34_models.py` — NSMC & UnSmile 80% noise resilience evaluation.
  - `run_seq2seq_1000_all_34_models.py`, `run_seq2seq_translation_benchmark.py`, `run_korean_to_english_seq2seq_eval.py` — Bidirectional EN$\leftrightarrow$KO Seq2Seq translation benchmarks.
  - `run_vocab_tail_and_oov_all_34_models.py`, `run_full_tail_bpc_all_models.py`, `eval_oov_and_tail_bpc.py` — Full 11,172 Hangul syllable tail perplexity (BPC) and OOV stress tests.

- **`reporting/`**: Automated Markdown report builders, head-to-head comparison generators, and qualitative sample analyzers.

## Related Directories in Repository

- **`data/korean_pos_mc/`**: Dataset and multi-lane binary preparation scripts (`prepare_hybrid_pos_lanes.py`, `prepare_pos_and_byte_lanes.py`).
- **`huggingface_model/gemma/`**: Gemma-3 continual pretraining wrapper and runner (`gemma3_continual_model.py`, `run_gemma3_continual_training.py`).
- **`tests/`**: Unit and prototype verification tests (`test_all_mtypes_oov.py`, `test_beam_search_prototype.py`, `test_model4.py`, `test_oov_eval_clean.py`).
