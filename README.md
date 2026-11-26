# GAID corpus: code

This repository contains the code that builds the GAID corpus (Generative-AI Declarations), a collection of authors'
statements about the use or non-use of generative AI in the open-access full text of Europe PMC / PubMed Central
articles first published between 2023-01-01 and 2026-10-31. The dataset is deposited on Zenodo
(https://doi.org/10.5281/zenodo.24553607) and is described in the accompanying Data Descriptor.

## Requirements

Python 3.12 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The annotation, audit and validation steps call language models through OpenAI-compatible endpoints. Copy
`.env.example` to `.env` and fill in:

* `AIHUBMIX_BASE_URL`, `AIHUBMIX_API_KEY`: endpoint for the hosted models;
* `LOCAL_BASE_URL`, `LOCAL_API_KEY`: server for the open-weight models listed in `LOCAL_MODELS` in `scripts/models.py`
  (one of the three reference models).

The model used in each role is set in `scripts/models.py`. Run all scripts from the repository root in isolated mode,
`python -I scripts/<name>.py`, because they parse downloaded files. Intermediate and final data are written to `data/`
(`raw`, `interim`, `validation`, `release`), the cost ledger to `logs/`, and statistics and figures to `paper/`. Long
steps can be interrupted and restarted; finished work is skipped.

## Scripts

| Script | Purpose |
|---|---|
| `01_enumerate_epmc.py` | Sampling frame (`index`, one query per publication day) and candidate articles (`candidates`, full-text query) from Europe PMC |
| `03_fetch_extract.py` | Download JATS XML from the PMC Open Access collection on AWS S3 and keep a superset of text blocks that may hold a declaration (`extract_lib.py`) |
| `03b_retry_failures.py` | Retry articles that could not be downloaded (other S3 versions, then Europe PMC) |
| `03c_latest_versions.py` | List the versions of every candidate and re-parse articles whose latest version differs from the parsed one |
| `04_build_passages.py` | Apply the passage rules (`passage_rules.py`, version 1.4) and collect the distinct passage texts |
| `05_annotate.py` | Annotate every distinct passage with the codebook prompt (`annot_schema.py`, version 1.1) |
| `06_audit_recall.py` | Recall audits: sampling, text retrieval, screening by two models, verification by the reference models, estimates |
| `07_build_release.py` | Release tables (CSV and Parquet) and `build_summary.json` |
| `08_validation.py` | Validation sample, reference models, test-retest, context check, human labels, agreement statistics |
| `09_analysis_figures.py` | Summary statistics (`paper/stats.json`) and figures |
| `11_export_docs.py` | Codebook, data dictionary, lookup tables, validation and model-output files for the release |
| `12_export_excel.py` | One annotation workbook per human annotator |
| `tool_lexicon.py`, `publisher_map.py` | Normalisation of tool names; DOI prefix to publisher |
| `common.py`, `llm_lib.py`, `models.py` | Paths and constants, API client with cost ledger, model roles |
| `pilot_fetch.py`, `pilot_llm.py`, `pilot_round2.py` | Pilot on 336 passages used to develop the codebook and prompt and to try candidate annotator models |
| `show_validation.py` | Prints `validation_results.json` as a table |

`annotation/` holds the guideline given to the human annotators (`标注说明.md`, in Chinese) and the two blank
workbooks produced by `12_export_excel.py`.

## Running the pipeline

The commands below reproduce the release in the order in which the steps depend on each other.

Sampling frame, full texts and passages (no language model):

```
python -I scripts/01_enumerate_epmc.py index
python -I scripts/01_enumerate_epmc.py candidates
python -I scripts/03_fetch_extract.py --source candidates --outdir blocks_v2
python -I scripts/04_build_passages.py
python -I scripts/03b_retry_failures.py
python -I scripts/03c_latest_versions.py list
python -I scripts/03c_latest_versions.py fetch
python -I scripts/04_build_passages.py
```

Annotation and release tables:

```
python -I scripts/05_annotate.py --model deepseek-v4.1-flash --out annot_primary
python -I scripts/05_annotate.py --model qwen3.8-flash --out annot_second
python -I scripts/07_build_release.py
```

Validation (the human annotators fill in the workbooks, which are then placed in `data/validation/human/`):

```
python -I scripts/08_validation.py sample
python -I scripts/08_validation.py refs
python -I scripts/08_validation.py retest
python -I scripts/08_validation.py context
python -I scripts/12_export_excel.py
python -I scripts/08_validation.py check_excel
```

The released validation sample (700 passages) also contains a fourth stratum, drawn with `08_validation.py sample_extra`
from the passages added by rules version 1.3.

Recall audits, shown for stratum A3. The same commands apply to B and C, with samples of 3,000 (A3, B) and 2,000 (C)
articles. Samples A and A2 were used to develop the passage rules, and stratum C is drawn from the release tables.

```
python -I scripts/06_audit_recall.py sample A3 3000
python -I scripts/06_audit_recall.py fetch A3
python -I scripts/06_audit_recall.py screen A3 deepseek-v4.1-flash
python -I scripts/06_audit_recall.py screen A3 qwen-plus-latest
python -I scripts/06_audit_recall.py combine A3 deepseek-v4.1-flash qwen-plus-latest
python -I scripts/06_audit_recall.py licences A3
```

Agreement statistics, figures and documentation:

```
python -I scripts/08_validation.py report
python -I scripts/09_analysis_figures.py
python -I scripts/11_export_docs.py
```

The pilot is run with `pilot_fetch.py`, then `pilot_llm.py build`, `pilot_llm.py run <model> ...` and
`pilot_llm.py report`; the second round with `pilot_round2.py <suffix> <model>` and `pilot_round2.py table`.

## Reproducibility

* Europe PMC and the PMC Open Access collection change over time. The enumeration used for the release was made on
  14-15 November 2026, so a new run will give slightly different counts.
* Language-model outputs are not exactly reproducible. All model outputs used for the release are part of the dataset:
  the labels of both annotators (`annotations.zip`), and the validation sample, reference labels, test-retest labels,
  context check and audit screening and verification files (`validation.zip`). Steps 01-04 and 07 do not call a model;
  with `labels_primary_annotator.jsonl.gz` and `labels_second_annotator.jsonl.gz` unpacked to
  `data/interim/annot_primary.jsonl` and `data/interim/annot_second.jsonl`, the tables are rebuilt without API access.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0 with the exceptions stated in the dataset record.

## Citation

If you use the code or the data, please cite the Data Descriptor and the dataset (https://doi.org/10.5281/zenodo.24553607).
