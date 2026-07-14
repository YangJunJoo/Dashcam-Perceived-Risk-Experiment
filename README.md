# Autonomous Vehicles Near-miss Risk Perception: Scripts & Data Replication Package

This directory contains the finalized code scripts, survey configurations, and datasets associated with the working paper:
**"Autonomous Vehicles Change How People Judge Risk in Near-misses"** (Joo et al., 2026). Currently under review (first round). SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5437729](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5437729)

This guide provides a comprehensive overview of the reproduction pipeline, detailing how raw survey responses and video metadata are processed, modeled using a double machine learning causal forest algorithm, and plotted to produce the findings presented in the paper.

---

## Directory Structure

```text
data_and_scripts/
├── README.md               # This replication guide
├── prepare_stimuli.py                # Stimuli video framing
├── reformat_survey_data.py           # Reformat survey questions (long format)
├── preprocess_survey_data.py         # Attention checks, demographic merges
├── tune_causal_forest.py             # ML nuisance model grid search/tuning
├── bootstrap_causal_forest.py        # Bootstrapped CausalForestDML (1,000 iterations)
├── evaluate_causal_forest.py         # HTE analysis, subgroups, and policy trees
├── plot_results.py                   # Sankey diagrams and literature reviews
├── perceived_risk_pipeline.py        # Consolidated end-to-end pipeline script
├── preprocess_existing_data.py       # Helper preprocessing rebuild module
└── data/                             # Experimental datasets
    ├── video_survey_match.xlsx       # Survey video matching configuration (raw)
    ├── metadata.csv                  # Video stimuli metadata annotations (raw)
    ├── df_long.csv                   # Reformatted survey questions (long format, raw)
    ├── prolific_export_688279a66160545bd74e3613.csv # Participant demographic data (raw)
    ├── raw_survey_data.parquet       # Raw survey responses (Parquet)
    ├── tuned_results_df.csv          # Nuisance model tuned parameters cache
    ├── XGB_best_params.json          # XGBoost parameters (JSON)
    ├── subgroup_df_sorted.csv        # Subgroup CATE summaries (combinations of variables)
    ├── bootstrap_cates_array.npz     # CATE bootstrap estimates array (NPZ)
    ├── main/                         # Replicated processed datasets
    │   ├── df_final.parquet          # Cleaned modelling dataset (Parquet)
    │   ├── df_final_4.parquet / df_final_5.parquet / df_final_6.parquet
    │   └── bootstrap_cates_array_4.npz / bootstrap_cates_array_5.npz / bootstrap_cates_array_6.npz
    └── published/                    # Anonymized public datasets
        ├── anonymized_survey+demo+video_matched.csv  # Anonymized responses dataset (9,260 obs)
        ├── video_survey_match.xlsx   # Survey video match sheet (anonymized)
        ├── AsPredicted #239500.pdf   # Official study pre-registration PDF
        ├── Qualtrics Survey Form.qsf # Qualtrics survey template schema
        └── AV_Literature_Review_Categorization_v2.xlsx # Literature categorization data
```

---

## Code Execution Pipeline

To reproduce the study's results from scratch, execute the scripts in the following order:

### Step 1: Stimuli Preparation
* **`prepare_stimuli.py`**  
  Extracts 5-second video clips from the raw Nexar dashcam dataset, varying the time-to-event ($TTE$) from 0.5s to 2.5s to build a within-incident risk gradient.

### Step 2: Survey Formatting & Preprocessing
* **`reformat_survey_data.py`**  
  Reads raw Qualtrics exports, handles response variables, and reformats the survey questions into a long format.
* **`preprocess_survey_data.py`**  
  Performs attention/manipulation checks, drops invalid participants, merges Prolific demographics with survey responses and video metadata, and outputs the modeling dataset (`df_final.parquet` or `anonymized_survey+demo+video_matched.csv`).

### Step 3: Model Tuning & Estimation
* **`tune_causal_forest.py`**  
  Performs grid search to tune machine learning models (Random Forest, Gradient Boosting, XGBoost, and LightGBM) to predict high-risk perception (6+ on 7-point scale). Selects XGBoost as the nuisance model due to performance and speed.
* **`bootstrap_causal_forest.py`**  
  Fits a `CausalForestDML` (EconML) with 1,000 bootstrap iterations using the tuned XGBoost model to calculate average treatment effects (ATE) and conditional average treatment effects (CATEs) with 90% confidence intervals.
* **`evaluate_causal_forest.py`**  
  Analyzes heterogeneous treatment effects (HTE), subgroups, and identifies where automation premiums and penalties occur.

### Step 4: Downstream Plotting
* **`plot_results.py`**  
  Generates literature review Sankey diagrams, sunburst plots of feature importance, and interaction plots.

> [!TIP]
> **Unified Execution Script**: A unified python script **`perceived_risk_pipeline.py`** is provided to automatically run the data preprocessing, model evaluation, bootstrapping, and plotting in a single execution.

---

## Omitted Files

* **Video Stimuli**: The raw video clips (2.9 GB) are excluded to keep the repository lightweight. They can be requested from the authors or downloaded via the Zenodo/OSF repository linked in the paper.

