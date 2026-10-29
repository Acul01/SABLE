# SABLE

Span Ablation for Black-box Label Evaluation. A feasibility study of whether a language model judges a short scientific claim by the sentence a human marked as evidence.

The claims are about invasion biology. The model sees only passages retrieved from the same paper. Deleting that evidence sentence rarely changes the label. A few other sentences do.

## Result

On 67 claims from 48 CC BY papers, a same-paper retriever places the human evidence span in the top five chunks for 40 claims. `gpt-4o-mini` matches the human label on 53 claims at temperature 0.

Deleting the span changes 2 of those 40 labels, and both changes leave the human label. On the 32 claims that stay *supported*, removing the highest-SHAP sentences moves the label; removing the lowest-valued or random sentences largely does not. Those high-valued sentences are only slightly closer to the evidence span than other sentences from the same chunks. Five samples usually agree, including when the agreed label is wrong. Writing the rationale before the label changes 6 labels and leaves the match count at 53.

The full condition tables are the CSV files under `results/`. The write-up lives in `report_paper/` on the machine that produced it and is not part of this repository.

## What is in the repository

| Path | Contents |
|---|---|
| `data/CLAIMS.csv` | 67 claims, labels, and evidence spans |
| `data/SOURCES.csv` | 48 papers, with DOI, journal, theme, and checked license |
| `results/predictions.csv` | Temperature-0 label for each claim |
| `results/intervention.csv` | Label after the evidence span is deleted |
| `results/shap_sentences.csv` | Sentence-level SHAP values for the 32 claims that stay *supported* |
| `results/shap_removal.csv` | Sentences removed in each ablation |
| `results/results.csv` | Temperature-0 labels after those removals |
| `results/sampled_labels.csv` | Every draw from the five-sample runs |
| `results/sampled_summary.csv` | Majority label, agreement, and confidence |
| `results/results_sampled.csv` | Sampled labels by condition |
| `results/similarity_results.csv` | Mean cosine similarity to the evidence span |
| `results/similarity_by_claim.csv` | The same similarity, one row per claim |
| `results/shap_similarity.csv` | Sentence-level similarity |
| `results/order_predictions.csv` | Labels when the rationale is written first |
| `main.ipynb` | The run, from retrieval through the order check |
| `scripts/` | The same steps as importable runners |

`results/intervention_2.csv` is the temperature-0 run that removes the three highest-SHAP sentences. The later tables reuse that condition.

## Data

The full texts come from the Zenodo record [10.5281/zenodo.14525619](https://doi.org/10.5281/zenodo.14525619) (Jennifer D'Souza, TIB, CC BY 4.0). The file committed here is not that corpus. `data/SOURCES.csv` lists the 48 papers kept after a CC BY check of each publisher page or PDF. One of them is CC BY 3.0 (`10.1002/ecs2.2097`).

To rebuild the index, place `publications_tdm.csv` at `data/zenodo/publications_tdm.csv`.

## Run

The tables under `results/` are already in the repository. Regenerating them calls the OpenAI API.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put `OPENAI_API_KEY` in a `.env` file in this directory. The notebook and the scripts load that file by path.

From the repository root:

```bash
python -m scripts.sentence_shap
python -m scripts.sample_labels
python -m scripts.shap_similarity
python -m scripts.order_intervention
python report_paper/make_figures.py
```

`sentence_shap` also rebuilds the Chroma index under `chroma_db/`. Numpy must stay below 2. The figures for the paper are written to `report_paper/figures/`.

## Scope

Sixty-seven claims, one annotator, one evidence span per claim, one model. Retrieval never leaves the source paper. SHAP explains the log probability of the token *supported* under a sparsity penalty that sets most sentences to zero. Confidence is the probability of the first label token on the samples that emitted the majority label. The deletions are edits of the prompt.
