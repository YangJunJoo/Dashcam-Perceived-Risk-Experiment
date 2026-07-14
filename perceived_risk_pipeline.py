#!/usr/bin/env python3
"""Unified pipeline for the perceived risk analysis workflow.

This script consolidates the logic that was previously scattered across
multiple notebooks in the `final` directory. It performs the following steps:

1. Load and reshape the raw Qualtrics survey export, applying attention checks
   and extracting the perceived risk responses.
2. Merge survey responses with video metadata and Prolific demographic data to
   produce the modelling dataset (`df_final.pkl`).
3. Optionally tune outcome/treatment nuisance models and bootstrap a
   `CausalForestDML` to recover ATE/CATE estimates.
4. Generate key plots (combined ATE distribution with error bars, sunburst, and
   literature Sankey) to match the downstream analysis notebooks.

Outputs are written alongside the original notebooks for compatibility, while
diagnostic artefacts live in `outputs/` by default. Use the CLI flags to
disable expensive steps (e.g., tuning, bootstrapping, plots) or adjust
hyperparameters such as the bootstrap count.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional progress bar
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# Plotting (matplotlib is required; seaborn optional but bundled with notebooks)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling dependencies
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - hard dependency
    raise ImportError("xgboost is required for this pipeline") from exc

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None
except Exception as exc:  # pragma: no cover - optional dependency
    warnings.warn(f"LightGBM import failed ({exc}); skipping LightGBM models.")
    LGBMClassifier = None

try:
    from econml.dml import CausalForestDML
except ImportError as exc:  # pragma: no cover - hard dependency
    raise ImportError("econml is required for the causal forest stage") from exc

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SURVEY_COLUMN_NAMES: List[str] = [
    "start_date",
    "end_date",
    "reponse_type",
    "ip_address",
    "progress",
    "duration_seconds",
    "finished",
    "recorded_date",
    "response_id",
    "last_name",
    "first_name",
    "email",
    "data_reference",
    "latitude",
    "longitude",
    "distribution_channel",
    "user_language",
    "q_attention_direction",
    "q_attention_check",
    "q_vehicle_imagined",
    "q_trust_taxi_skill",
    "q_safe_taxi_overall",
    "q_focus_taxi_ride",
    "q_trust_av_skill",
    "q_safe_av_overall",
    "q_focus_av_ride",
    "q_passenger_in_sdc",
    "q_primary_transport",
    "q_adas_usage",
    "q_tech_confidence",
    "q_driving_experience_years",
    "q_continent",
    "q_gender",
    "q_gender_other",
    "q_age",
    "q_prolific_id",
    "prolific_pid",
    "display_a",
    "display",
    "variable_header",
    "answer_value",
    "ID",
    "question",
]

REFERENCE_CATEGORIES: Dict[str, str] = {
    "AV Experience": "No",
    "Primary Mode": "Motorcycle",
    "ADAS Usage": "Medium",
    "Tech Confidence": "Medium",
    "Driving Experience": "5-10",
    "Continent": "Oceania",
    "Gender": "Other",
    "Age": "25-45",
    "Ethnicity": "Unknown",
    "Student": "Unknown",
    "Employment": "Unknown",
    "Driver Trust": "Medium",
    "Weather": "Clear",
    "Scene": "Other",
    "Light Conditions": "Normal",
    "Road Type": "Residential Area",
    "Time To Event": "2.5",
    "Ego Fault": "No",
    "Point of Impact": "T-bone",
    "Ego Avoidability": "Potentially Avoidable",
    "Ego Maneuver": "Go Straight",
    "Ego Speeding": "No",
    "Ego Violation": "No",
    "Other Maneuver": "Go Straight",
    "Other Speeding": "No",
    "Other Body Style": "Sedan",
    "Other Violation": "No",
    "VRU Involvement": "No",
    "Impact Severity": "Near-Miss",
}

TRUST_ORDER = ["Low", "Medium", "High"]
DISPLAY_LABELS = {"No": "HDV (Control)", "Yes": "AV (Treatment)"}

ID_PATTERN = r"File\.php\?F=([A-Za-z0-9_]+)\s*-\s*(.+)"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def iter_with_progress(iterable: Iterable, **kwargs):
    """Wrap an iterable with tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def ensure_columns(df: pd.DataFrame, expected: List[str]) -> None:
    """Validate the wide dataframe shape before renaming."""
    if df.shape[1] != len(expected):
        raise ValueError(
            f"Unexpected column count ({df.shape[1]}); expected {len(expected)}. "
            "The Qualtrics export format appears to have changed."
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisConfig:
    """File-system configuration for the pipeline."""

    base_dir: Path
    keep_intermediate: bool = True
    outputs_dir_name: str = "outputs"
    raw_survey_path: Path = field(init=False)
    video_survey_match_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    prolific_path: Path = field(init=False)
    literature_excel_path: Path = field(init=False)
    df_long_output_path: Path = field(init=False)
    df_final_output_path: Path = field(init=False)
    tuned_results_path: Path = field(init=False)
    xgb_params_path: Path = field(init=False)
    bootstrap_cates_path: Path = field(init=False)
    outputs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.resolve()

        data_dir = self.base_dir / "data"
        pretest_dir = data_dir / "pre test"
        main_dir = data_dir / "main"

        self.raw_survey_path = main_dir / "raw_survey_data.parquet"
        self.video_survey_match_path = main_dir / "video_survey_match.xlsx"
        self.metadata_path = main_dir / "metadata_ideal.csv"
        self.prolific_path = main_dir / "prolific_export_688279a66160545bd74e3613.csv"
        self.literature_excel_path = data_dir / "AV_Literature_Review_Categorization_v2.xlsx"

        self.df_long_output_path = main_dir / "df_long_PR.csv"
        self.df_final_output_path = main_dir / "df_final.parquet"

        # Keep artefacts alongside notebooks for compatibility
        self.tuned_results_path = self.base_dir / "tuned_results_df.csv"
        self.xgb_params_path = self.base_dir / "XGB_best_params.json"
        self.bootstrap_cates_path = self.base_dir / "bootstrap_cates_array.npz"

        self.outputs_dir = self.base_dir / self.outputs_dir_name
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


def extract_id_question(header: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse the Qualtrics variable header into (unique_id, question)."""
    if not isinstance(header, str):
        return None, None
    match = pd.Series(header).str.extract(ID_PATTERN, expand=True)
    if match.isnull().all(axis=None):
        return None, header
    unique_id, question = match.iloc[0]
    return unique_id, question.strip() if isinstance(question, str) else question


def build_perceived_risk_long(config: AnalysisConfig) -> pd.DataFrame:
    """Recreate the long-form perceived risk dataframe with attention checks."""
    logging.info("Loading raw survey data from %s", config.raw_survey_path)
    if not config.raw_survey_path.exists():
        raise FileNotFoundError(f"Missing raw survey export: {config.raw_survey_path}")
    df_raw = pd.read_parquet(config.raw_survey_path)

    logging.info("Melting Qualtrics export to long format")
    cols_to_drop = [col for col in df_raw.columns if "Timing - " in col]
    df_raw = df_raw.drop(columns=cols_to_drop)
    
    all_columns = df_raw.columns.tolist()
    value_vars = [
        col for col in all_columns if "File.php?F=" in col
    ]
    id_vars = [col for col in all_columns if col not in value_vars]

    df_long = pd.melt(
        df_raw,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="VariableHeader",
        value_name="AnswerValue",
    )

    logging.debug("Extracting unique IDs and question text")
    extracted = df_long["VariableHeader"].apply(lambda x: pd.Series(extract_id_question(x)))
    extracted.columns = ["unique_id", "Question"]
    df_long = pd.concat([df_long, extracted], axis=1)

    df_long = df_long.dropna(subset=["AnswerValue"])

    ensure_columns(df_long, SURVEY_COLUMN_NAMES)
    df_long.columns = SURVEY_COLUMN_NAMES

    numeric_cols = ["q_attention_check", "q_vehicle_imagined", "display", "display_a", "answer_value"]
    for col in numeric_cols:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col], errors="coerce")

    logging.info("Applying attention checks")
    attn_check_one = df_long["q_attention_check"] == 1
    attn_check_two = (
        df_long["q_vehicle_imagined"].astype(float) == df_long["display"].astype(float) + 1
    )

    passed_mask = attn_check_one & attn_check_two
    df_passed = df_long.loc[passed_mask].copy()

    df_perceived = df_passed[df_passed["question"].str.contains("perceive", case=False, na=False)].copy()
    df_perceived["answer_value"] = pd.to_numeric(df_perceived["answer_value"], errors="coerce")

    logging.info("Perceived-risk responses retained: %s rows", len(df_perceived))

    if config.keep_intermediate:
        config.df_long_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_perceived.to_csv(config.df_long_output_path, index=False)
        logging.info("Saved intermediate df_long_PR to %s", config.df_long_output_path)

    return df_perceived


def load_metadata_tables(config: AnalysisConfig) -> pd.DataFrame:
    """Merge Nexar metadata with survey/video mapping."""
    if not config.video_survey_match_path.exists():
        raise FileNotFoundError(f"Missing video survey match file: {config.video_survey_match_path}")
    if not config.metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {config.metadata_path}")

    video_survey_match = pd.read_excel(config.video_survey_match_path)
    video_survey_match["Name"] = video_survey_match["Name"].str.replace("new_", "", regex=False)
    video_survey_match["file_name"] = video_survey_match["Name"].str.split("_").str[0].astype(int)
    video_survey_match["offset"] = (
        video_survey_match["Name"]
        .str.split(".mp4")
        .str[0]
        .str.split("_off")
        .str[1]
        .astype(int)
        / 100
    )
    video_survey_match = video_survey_match.loc[:, ["file_name", "offset", "ID", "Name"]]

    metadata = pd.read_csv(config.metadata_path)
    metadata = metadata.rename(columns={"file name": "file_name"})

    merged = pd.merge(video_survey_match, metadata, on="file_name", how="inner")
    logging.info(
        "Merged video metadata: %s rows (expected to match ID count)", len(merged)
    )
    return merged


def preprocess_responses(
    df_long_pr: pd.DataFrame, metadata_merged: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    """Replicate the preprocessing steps that culminate in df_final.pkl."""
    df = df_long_pr.copy()

    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_columns:
        df = df.drop(columns=constant_columns)
        logging.debug("Dropped constant columns: %s", constant_columns)

    if config.prolific_path.exists():
        df_demo = pd.read_csv(config.prolific_path)
        demo_cols = [
            "Participant id",
            "Ethnicity simplified",
            "Country of birth",
            "Country of residence",
            "Nationality",
            "Language",
            "Student status",
            "Employment status",
        ]
        missing_demo_cols = [col for col in demo_cols if col not in df_demo.columns]
        if missing_demo_cols:
            raise KeyError(f"Missing expected columns in Prolific export: {missing_demo_cols}")
        df_demo = df_demo[demo_cols].copy()
        df_demo.rename(columns={"Participant id": "prolific_pid"}, inplace=True)
        df = df.merge(df_demo, on="prolific_pid", how="left")
    else:
        logging.warning("Prolific demographics file not found at %s; skipping merge", config.prolific_path)

    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    excluded_columns = [
        "response_id",
        "ID",
        "Ethnicity simplified",
        "Country of residence",
        "Student status",
        "Employment status",
    ]
    drop_objects = [col for col in object_columns if col not in excluded_columns]
    df_cleaned = df.drop(columns=drop_objects)

    df_cleaned["Combined_Trust"] = (
        df_cleaned["q_trust_taxi_skill"].fillna(0) + df_cleaned["q_trust_av_skill"].fillna(0)
    )
    df_cleaned["Combined_Safe"] = (
        df_cleaned["q_safe_taxi_overall"].fillna(0) + df_cleaned["q_safe_av_overall"].fillna(0)
    )
    df_cleaned["Combined_Focus"] = (
        df_cleaned["q_focus_taxi_ride"].fillna(0) + df_cleaned["q_focus_av_ride"].fillna(0)
    )

    drop_columns = [
        "q_vehicle_imagined",
        "duration_seconds",
        "latitude",
        "longitude",
        "q_trust_taxi_skill",
        "q_safe_taxi_overall",
        "q_focus_taxi_ride",
        "q_trust_av_skill",
        "q_focus_av_ride",
        "q_safe_av_overall",
    ]
    df_cleaned = df_cleaned.drop(columns=[col for col in drop_columns if col in df_cleaned.columns])

    df_final = df_cleaned.merge(metadata_merged, on="ID", how="left")
    drop_metadata_cols = [
        "Name",
        "file_name",
        "time of event",
        "time of alert",
        "video path",
        "event",
        "bad video",
        "time of first appearance",
        "event time",
        "crash occurance",
    ]
    df_final = df_final.drop(columns=[col for col in drop_metadata_cols if col in df_final.columns])
    df_final = df_final.dropna(subset=["offset"])

    numeric_fields = [
        "answer_value",
        "q_tech_confidence",
        "q_age",
        "q_driving_experience_years",
        "q_adas_usage",
        "Combined_Trust",
        "Combined_Safe",
        "Combined_Focus",
        "display",
        "display_a",
    ]
    for col in numeric_fields:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    df_final["Combined_Trust"] = df_final[["Combined_Trust", "Combined_Safe", "Combined_Focus"]].mean(axis=1)

    df_final["q_tech_confidence"] = np.select(
        [
            df_final["q_tech_confidence"].isin([1, 2]),
            df_final["q_tech_confidence"].isin([3, 4, 5]),
            df_final["q_tech_confidence"].isin([6, 7]),
        ],
        ["Low", "Medium", "High"],
        default="Unknown",
    )
    df_final["answer_value"] = np.select(
        [
            df_final["answer_value"].isin([1, 2]),
            df_final["answer_value"].isin([3, 4, 5]),
            df_final["answer_value"].isin([6, 7]),
        ],
        [0, 1, 2],
        default=np.nan,
    )
    df_final["Combined_Trust"] = np.select(
        [
            df_final["Combined_Trust"] <= 2.5,
            (df_final["Combined_Trust"] > 2.5) & (df_final["Combined_Trust"] <= 5.5),
            df_final["Combined_Trust"] > 5.5,
        ],
        ["Low", "Medium", "High"],
        default="Unknown",
    )
    df_final["q_age"] = np.select(
        [
            df_final["q_age"] <= 25,
            (df_final["q_age"] > 25) & (df_final["q_age"] <= 45),
            (df_final["q_age"] > 45) & (df_final["q_age"] <= 65),
            df_final["q_age"] > 65,
        ],
        ["<25", "25-45", "45-65", ">65"],
        default="Unknown",
    )
    df_final["q_driving_experience_years"] = np.select(
        [
            df_final["q_driving_experience_years"] < 5,
            (df_final["q_driving_experience_years"] >= 5)
            & (df_final["q_driving_experience_years"] <= 10),
            df_final["q_driving_experience_years"] > 10,
        ],
        ["<5", "5-10", ">10"],
        default="Unknown",
    )
    df_final["q_adas_usage"] = np.select(
        [
            df_final["q_adas_usage"].isin([1, 2]),
            df_final["q_adas_usage"].isin([3, 4, 5]),
            df_final["q_adas_usage"].isin([6, 7]),
        ],
        ["1-4", "5-6", "7-10"],
        default="Unknown",
    )
    df_final["Ethnicity simplified"] = np.select(
        [
            df_final["Ethnicity simplified"].isin(["DATA_EXPIRED", "CONSENT_REVOKED", "Other"]),
            df_final["Ethnicity simplified"].isna(),
        ],
        ["Unknown", "Unknown"],
        default=df_final["Ethnicity simplified"],
    )
    df_final["Student status"] = np.select(
        [
            df_final["Student status"].isin(["DATA_EXPIRED", "CONSENT_REVOKED"]),
            df_final["Student status"].isna(),
        ],
        ["Unknown", "Unknown"],
        default=df_final["Student status"],
    )
    df_final["Employment status"] = np.select(
        [
            df_final["Employment status"].isin(
                [
                    "Unemployed (and job seeking)",
                    "Not in paid work (e.g. homemaker', 'retired or disabled)",
                    "Due to start a new job within the next month",
                ]
            ),
            df_final["Employment status"].isin(["DATA_EXPIRED", "CONSENT_REVOKED", "nan", "Other"]),
            df_final["Employment status"].isna(),
        ],
        ["Unemployed", "Unknown", "Unknown"],
        default=df_final["Employment status"],
    )
    df_final["Country of residence"] = np.select(
        [
            df_final["Country of residence"].isin(["DATA_EXPIRED", "CONSENT_REVOKED", "nan", "Other"]),
            df_final["Country of residence"].isna(),
        ],
        ["Unknown", "Unknown"],
        default=df_final["Country of residence"],
    )

    df = df_final.copy()
    target_question = "answer_value"
    df[target_question] = df[target_question].map({0: 0, 1: 0, 2: 1})

    df.columns = (
        df.columns.str.replace("q_", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
    )

    rename_map = {
        "passenger in sdc": "AV Experience",
        "primary transport": "Primary Mode",
        "adas usage": "ADAS Usage",
        "tech confidence": "Tech Confidence",
        "driving experience years": "Driving Experience",
        "continent": "Continent",
        "gender": "Gender",
        "age": "Age",
        "display": "Display",
        "Combined Trust": "Driver Trust",
        "Combined Safe": "Passenger Safety",
        "Combined Focus": "Passenger Distraction",
        "Ethnicity simplified": "Ethnicity",
        "Country of residence": "Country",
        "Student status": "Student",
        "Employment status": "Employment",
        "weather": "Weather",
        "scene": "Scene",
        "light conditions": "Light Conditions",
        "road type": "Road Type",
        "offset": "Time To Event",
        "ego fault": "Ego Fault",
        "point of impact": "Point of Impact",
        "ego vehicle avoidability": "Ego Avoidability",
        "ego maneuver": "Ego Maneuver",
        "ego speeding": "Ego Speeding",
        "ego violation": "Ego Violation",
        "counterpart maneuver": "Other Maneuver",
        "counterpart speeding": "Other Speeding",
        "counterpart body style": "Other Body Style",
        "traffic violation": "Other Violation",
        "VRU involve": "VRU Involvement",
        "impact severity": "Impact Severity",
        "answer value": "Answer Value",
    }
    df.rename(columns=rename_map, inplace=True)

    df["Point of Impact"] = df["Point of Impact"].replace(
        {"rear-end": "Rear-End", "sideswipe": "Sideswipe", "head-on": "Head-On", "T-bone": "T-bone", "other": "Other"}
    )
    df["Ego Avoidability"] = df["Ego Avoidability"].replace(
        {
            "clearly avoidable ": "Clearly Avoidable",
            "potentially avoidable ": "Potentially Avoidable",
            "unavoidable": "Unavoidable",
        }
    )
    df["Road Type"] = df["Road Type"].replace(
        {
            "signalized intersection": "Signalized Intersection",
            "collector road": "Collector",
            "arterial": "Arterial",
            "highway": "Highway",
            "ramp": "Ramp",
            "stop sign intersection": "Stop Sign Intersection",
            "parking lot": "Parking Lot",
            "residential area": "Residential Area",
        }
    )
    df["Ego Fault"] = df["Ego Fault"].replace({1: "Yes", 0: "No"})
    df["AV Experience"] = df["AV Experience"].replace({1: "Yes", 2: "No"})
    df["Primary Mode"] = df["Primary Mode"].replace(
        {1: "Car", 2: "Public Transport", 3: "Bicycle/on foot", 4: "Motorcycle"}
    )
    df["ADAS Usage"] = df["ADAS Usage"].replace({"1-4": "Low", "5-6": "Medium", "7-10": "High"})
    df["Continent"] = df["Continent"].replace(
        {1: "Africa", 2: "Asia", 3: "Europe", 4: "North America", 5: "Oceania", 6: "South America"}
    )
    df["Gender"] = df["Gender"].replace({1: "Male", 2: "Female", 3: "Other", 4: "Other", 5: "Other"})
    df["Age"] = df["Age"].replace({1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+"})
    df["Display"] = df["Display"].replace({1: "Yes", 0: "No"})
    df["Ego Maneuver"] = df["Ego Maneuver"].replace(
        {
            "proceeding straight": "Go Straight",
            "turning": "Turning",
            "lane changing": "Lane Changing",
            "slow down": "Slow Down",
            "stop": "Stop",
            "merging": "Merging",
            "parking": "Parking",
        }
    )
    df["Ego Speeding"] = df["Ego Speeding"].replace({"yes": "Yes", "no": "No"})
    df["Ego Violation"] = df["Ego Violation"].replace(
        {
            "no": "No",
            "right-of-way": "Right-of-way",
            "red-light-running": "Red Light Running",
            "stop sign": "Stop Sign",
            "centerline": "Centerline",
            "off-road-running": "Off-road",
        }
    )
    df["Other Maneuver"] = df["Other Maneuver"].replace(
        {
            "proceeding straight": "Go Straight",
            "turning": "Turning",
            "lane changing": "Lane Changing",
            "slow down": "Slow Down",
            "stop": "Stop",
            "merging": "Merging",
            "parking": "Parking",
            "backing up": "Backing Up",
            "open the door": "Opening Door",
            "waiting": "Waiting",
            "other": "Other",
        }
    )
    df["Other Speeding"] = df["Other Speeding"].replace({"yes": "Yes", "no": "No"})
    df["Other Body Style"] = df["Other Body Style"].replace(
        {"sedan": "Sedan", "van": "Van", "truck": "Truck", "pickup truck": "Pickup Truck", "suv": "SUV", "other": "Other"}
    )
    df["Other Violation"] = df["Other Violation"].replace(
        {
            "no": "No",
            "right-of-way": "Right-of-way",
            "red-light-running": "Red Light Running",
            "stop sign": "Stop Sign",
            "centerline": "Centerline",
            "off-road-running": "Off-road",
            "jaywalking": "Jaywalking",
            "ilegal parking": "Illegal Parking",
            "other": "Other",
        }
    )
    df["VRU Involvement"] = df["VRU Involvement"].replace({"no": "No", "yes": "Yes"})
    df["Impact Severity"] = df["Impact Severity"].replace(
        {"minor": "Minor", "moderate": "Moderate", "major": "Severe", "near-miss": "Near-Miss"}
    )

    drop_post_columns = ["Passenger Safety", "Passenger Distraction"]
    df = df.drop(columns=[col for col in drop_post_columns if col in df.columns])

    df["Answer Value"] = df["Answer Value"].astype(float)

    config.df_final_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.df_final_output_path, index=False)
    logging.info("Saved modelling dataset to %s", config.df_final_output_path)

    return df


# --------------------------------------------------------------------------- #
# Modelling helpers
# --------------------------------------------------------------------------- #


def prepare_model_inputs(
    df_final: pd.DataFrame, drop_driver_trust: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """One-hot encode categorical features for the causal forest."""
    df_model = df_final.copy()

    df_model["Display"] = df_model["Display"].map({"No": 0, "Yes": 1})
    y = df_model["Answer Value"].astype(int)
    t = df_model["Display"].astype(int)

    categorical_features = list(REFERENCE_CATEGORIES.keys())
    missing_features = [col for col in categorical_features if col not in df_model.columns]
    if missing_features:
        raise KeyError(f"Missing expected modelling columns: {missing_features}")

    X_encoded = pd.get_dummies(df_model[categorical_features], dtype=float)
    for feature, reference in REFERENCE_CATEGORIES.items():
        column_to_drop = f"{feature}_{reference}"
        if column_to_drop in X_encoded.columns:
            X_encoded.drop(columns=column_to_drop, inplace=True)

    if drop_driver_trust:
        driver_trust_cols = [col for col in X_encoded.columns if col.startswith("Driver Trust_")]
        X_encoded.drop(columns=driver_trust_cols, inplace=True, errors="ignore")

    X_encoded.columns = X_encoded.columns.str.replace(r"[\[\]<]", "", regex=True)
    return X_encoded, y, t, df_model


def tune_nuisance_models(
    X: pd.DataFrame,
    T: pd.Series,
    y: pd.Series,
    *,
    cv_splits: int,
    n_jobs: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Optional[Dict[str, object]]]:
    """Grid-search baseline classifiers and persist their best parameters."""
    logging.info("Starting hyperparameter tuning with %s-fold CV", cv_splits)
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_grids = {
        "Random Forest": {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 20, 50, None],
            "min_samples_leaf": [1, 2, 5],
            "min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 8, 10],
            "subsample": [0.8, 1.0],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 400, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 8, 10],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [20, 31, 50, 70],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }

    models = {
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
        ),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(random_state=random_state, verbose=-1)
    else:
        logging.warning("LightGBM not installed; skipping LightGBM grid search")
        param_grids.pop("LightGBM", None)

    results: List[Dict[str, object]] = []
    features = pd.concat([X, T.rename("Display")], axis=1)
    best_xgb_params: Optional[Dict[str, object]] = None

    for name, model in models.items():
        grid_params = param_grids[name]
        num_candidates = len(list(ParameterGrid(grid_params)))
        logging.info("Grid-searching %s across %s parameter combinations", name, num_candidates)
        grid = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            cv=kfold,
            scoring="roc_auc",
            n_jobs=n_jobs,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.fit(features, y)
        results.append(
            {
                "Model": name,
                "Best Score (AUC)": grid.best_score_,
                "Best Parameters": grid.best_params_,
            }
        )
        if name == "XGBoost":
            best_xgb_params = grid.best_params_

    tuned_results_df = pd.DataFrame(results)
    return tuned_results_df, best_xgb_params


def get_xgb_best_params(config: AnalysisConfig, fallback: Dict[str, object]) -> Dict[str, object]:
    """Load tuned XGBoost parameters or return the provided fallback."""
    if config.xgb_params_path.exists():
        try:
            import json
            with open(config.xgb_params_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            if isinstance(best_params, dict):
                return best_params
        except Exception:  # pragma: no cover - defensive
            logging.warning("Failed to load %s; falling back to defaults", config.xgb_params_path)
    return fallback


def run_causal_forest_bootstrap(
    X: pd.DataFrame,
    T: pd.Series,
    y: pd.Series,
    *,
    best_params: Dict[str, object],
    n_bootstrap: int,
    random_state: int,
    cv_splits: int,
    n_estimators: int,
    min_samples_leaf: int,
    discrete_outcome: bool = True,
) -> np.ndarray:
    """Bootstrap a causal forest and return the stacked CATE matrix."""
    n_obs = len(y)
    rng = np.random.default_rng(random_state)
    bootstrap_cates: List[np.ndarray] = []

    logging.info("Bootstrapping causal forest (%s iterations)", n_bootstrap)
    iterator = iter_with_progress(range(n_bootstrap), desc="Bootstrap Progress", unit="iter")

    for iteration in iterator:
        sample_indices = rng.choice(n_obs, size=n_obs, replace=True)
        X_boot = X.iloc[sample_indices]
        T_boot = T.iloc[sample_indices]
        y_boot = y.iloc[sample_indices]

        outcome_model = XGBClassifier(
            random_state=random_state + iteration,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
            **best_params,
        )
        treatment_model = XGBClassifier(
            random_state=random_state + iteration,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
        )
        forest = CausalForestDML(
            model_y=outcome_model,
            model_t=treatment_model,
            discrete_treatment=True,
            discrete_outcome=discrete_outcome,
            honest=True,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            criterion="het",
            cv=cv_splits,
            random_state=random_state + iteration,
        )
        try:
            forest.fit(y_boot, T_boot, X=X_boot)
            effect = forest.effect(X)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Bootstrap iteration %s failed: %s", iteration, exc)
            effect = np.full(n_obs, np.nan, dtype=float)
        bootstrap_cates.append(effect)

    bootstrap_array = np.asarray(bootstrap_cates)
    return bootstrap_array


def compute_ate_stats(bootstrap_array: np.ndarray) -> Dict[str, float]:
    """Calculate the mean ATE and 95% confidence interval from bootstraps."""
    bootstrapped_ates = np.nanmean(bootstrap_array, axis=1)
    return {
        "ate_distribution": bootstrapped_ates,
        "point_estimate": float(np.nanmean(bootstrapped_ates)),
        "ci_lower": float(np.nanpercentile(bootstrapped_ates, 2.5)),
        "ci_upper": float(np.nanpercentile(bootstrapped_ates, 97.5)),
    }


def compute_trust_display_ci(
    df_final: pd.DataFrame, *, n_bootstrap: int, random_state: int
) -> pd.DataFrame:
    """Bootstrap percentile CIs for Display x Driver Trust segments."""
    df = df_final.copy()
    df["Display Condition"] = df["Display"].map(DISPLAY_LABELS)
    df["Driver Trust"] = pd.Categorical(df["Driver Trust"], categories=TRUST_ORDER, ordered=True)
    rng = np.random.default_rng(random_state)

    results: List[Dict[str, object]] = []

    grouped = df.groupby(["Driver Trust", "Display Condition"], observed=True)
    for (trust_level, display_condition), group in grouped:
        values = group["Answer Value"].astype(float).to_numpy()
        if values.size == 0:
            continue
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(values, size=values.size, replace=True)
            bootstrap_means.append(np.mean(sample))
        bootstrap_means = np.asarray(bootstrap_means)
        results.append(
            {
                "Driver Trust": trust_level,
                "Display Condition": display_condition,
                "mean": float(values.mean()),
                "ci_lower": float(np.nanpercentile(bootstrap_means, 2.5)),
                "ci_upper": float(np.nanpercentile(bootstrap_means, 97.5)),
                "count": int(values.size),
            }
        )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_combined_ate_figure(
    ate_stats: Dict[str, float],
    trust_ci: pd.DataFrame,
    df_final: pd.DataFrame,
    output_path: Path,
) -> None:
    """Recreate the combined histogram + error-bar figure from the notebook."""
    bootstrapped_ates = np.asarray(ate_stats["ate_distribution"])
    point_estimate = ate_stats["point_estimate"]
    ci_lower = ate_stats["ci_lower"]
    ci_upper = ate_stats["ci_upper"]

    trust_ci = trust_ci.set_index(["Display Condition", "Driver Trust"]).sort_index()
    df = df_final.copy()
    df["Display Condition"] = df["Display"].map(DISPLAY_LABELS)

    trust_order = TRUST_ORDER
    display_categories = sorted(DISPLAY_LABELS.values(), reverse=True)

    distribution = (
        df.groupby(["Display Condition", "Driver Trust"])
        .size()
        .reset_index(name="count")
    )
    total_counts = df["Display Condition"].value_counts().to_dict()
    distribution["percentage"] = distribution.apply(
        lambda row: 100.0 * row["count"] / total_counts.get(row["Display Condition"], 1), axis=1
    )
    dist_map = {
        (row["Display Condition"], row["Driver Trust"]): row for _, row in distribution.iterrows()
    }

    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    sns.histplot(bootstrapped_ates * 100, kde=True, ax=ax1, stat="density", color="steelblue", bins=50)
    ax1.axvline(point_estimate * 100, color="darkred", linestyle="--", linewidth=2)
    ax1.axvspan(ci_lower * 100, ci_upper * 100, color="darkred", alpha=0.1)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("ATE (AV Treatment - HDV Control) (%)")
    ax1.set_ylabel("Density")

    colors = {"HDV (Control)": "darkorange", "AV (Treatment)": "royalblue"}
    markers = {"HDV (Control)": "o", "AV (Treatment)": "s"}

    x_ticks = np.arange(len(trust_order))
    width = 0.2
    for idx, condition in enumerate(display_categories):
        if condition not in trust_ci.index.get_level_values("Display Condition"):
            continue
        data = trust_ci.xs(condition, level="Display Condition").reindex(trust_order)
        if data["mean"].isna().all():
            continue
        x_pos = x_ticks - width / 2 + idx * width
        y_err = [
            data["mean"].to_numpy() - data["ci_lower"].to_numpy(),
            data["ci_upper"].to_numpy() - data["mean"].to_numpy(),
        ]
        ax2.errorbar(
            x=x_pos,
            y=data["mean"],
            yerr=y_err,
            fmt=markers[condition],
            color=colors[condition],
            capsize=4,
            linestyle="None",
            label=condition,
        )

    xticklabels = []
    for trust in trust_order:
        av_row = dist_map.get(("AV (Treatment)", trust))
        hdv_row = dist_map.get(("HDV (Control)", trust))
        av_text = (
            f"AV: {av_row['percentage']:.0f}% (n={int(av_row['count'])})"
            if av_row is not None
            else "AV: 0% (n=0)"
        )
        hdv_text = (
            f"HDV: {hdv_row['percentage']:.0f}% (n={int(hdv_row['count'])})"
            if hdv_row is not None
            else "HDV: 0% (n=0)"
        )
        xticklabels.append(f"{trust}\n{av_text}\n{hdv_text}")

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(xticklabels, fontsize=9)
    ax2.set_ylabel("Average High Risk % (with 95% CI)")
    ax2.set_xlabel("Driver Trust Level (Mediator)")
    ax2.set_ylim(0)
    ax2.legend(title="Vehicle Type")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved combined ATE figure to %s", output_path)


def generate_sunburst(df_final: pd.DataFrame, output_path: Path) -> None:
    """Produce the multi-panel sunburst chart (requires plotly + kaleido)."""
    try:
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:  # pragma: no cover - optional dependency
        logging.warning("Plotly not available; skipping sunburst chart")
        return
    try:
        import plotly.graph_objects as go  # noqa: F401 - required for make_subplots
    except ImportError:  # pragma: no cover
        logging.warning("Plotly graph_objects missing; skipping sunburst chart")
        return

    df_participant = df_final[
        [
            "AV Experience",
            "Primary Mode",
            "ADAS Usage",
            "Tech Confidence",
            "Driving Experience",
            "Continent",
            "Gender",
            "Age",
            "Driver Trust",
            "Ethnicity",
            "Country",
            "Student",
            "Employment",
        ]
    ]
    df_scenario = df_final[
        [
            "Weather",
            "Scene",
            "Light Conditions",
            "Road Type",
            "Time To Event",
            "Ego Fault",
            "Point of Impact",
            "Ego Avoidability",
            "Ego Maneuver",
            "Ego Speeding",
            "Ego Violation",
            "Other Maneuver",
            "Other Speeding",
            "Other Body Style",
            "Other Violation",
            "VRU Involvement",
            "Impact Severity",
        ]
    ]
    df_plot = pd.concat([df_participant, df_scenario], axis=1)

    trans_col = ["AV Experience", "ADAS Usage", "Tech Confidence", "Primary Mode", "Driving Experience", "Driver Trust"]
    demo_col = ["Gender", "Age", "Ethnicity", "Country", "Student", "Employment"]
    env_col = ["Weather", "Scene", "Light Conditions", "Road Type"]
    crash_col = ["Time To Event", "Point of Impact", "VRU Involvement", "Impact Severity"]
    maneuver_col = ["Other Maneuver", "Other Speeding", "Other Body Style", "Other Violation", "Ego Fault", "Ego Avoidability", "Ego Maneuver", "Ego Speeding", "Ego Violation"]

    column_groups = [trans_col, demo_col, env_col + crash_col, maneuver_col]
    titles = ["Mobility<br>Behavior<br>& Attitudes", "Demographics", "Crash<br>Environment", "Maneuver<br>Properties"]
    subplot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}], [{"type": "domain"}, {"type": "domain"}]],
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    for columns_to_plot, pos, title in zip(column_groups, subplot_positions, titles):
        df_melted = df_plot[columns_to_plot].melt(var_name="variable", value_name="value")
        df_melted["variable"] = df_melted["variable"].str.replace(" ", "<br>")
        df_melted["value"] = df_melted["value"].astype(str).str.replace(" ", "<br>")
        df_melted["title"] = title

        sunburst_fig = px.sunburst(df_melted, path=["title", "variable", "value"])

        custom_text_templates = []
        for entry in sunburst_fig.data[0].ids:
            template = "<b>%{label}</b><br>%{percentParent:.0%}" if entry.count("/") == 2 else "<b>%{label}</b>"
            custom_text_templates.append(template)
        sunburst_fig.data[0].texttemplate = custom_text_templates
        fig.add_trace(sunburst_fig.data[0], row=pos[0], col=pos[1])

    fig.update_traces(insidetextorientation="radial", rotation=90)
    fig.update_layout(height=800, width=800, margin=dict(t=0, l=0, r=0, b=0))

    try:
        fig.write_image(str(output_path), scale=3, width=800, height=800)
        logging.info("Saved sunburst chart to %s", output_path)
    except ValueError as exc:  # pragma: no cover - requires kaleido
        logging.warning("Unable to save sunburst chart (is kaleido installed?): %s", exc)


def generate_literature_sankey(config: AnalysisConfig, output_path: Path) -> None:
    """Render the literature review Sankey chart."""
    if not config.literature_excel_path.exists():
        logging.warning("Literature review file not found at %s; skipping Sankey plot", config.literature_excel_path)
        return
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:  # pragma: no cover - optional dependency
        logging.warning("Plotly not available; skipping Sankey plot")
        return

    df = pd.read_excel(config.literature_excel_path)
    sankey_cols = ["Focus", "Methodology", "Stimuli type"]
    focus_order = sorted(df["Focus"].dropna().unique().tolist())
    method_order = ["Survey", "Qualitative + Survey", "Qualitative", "Experiment"]
    stimuli_order = ["No stimuli", "Field study", "VR", "Text", "Text+Video", "Video"]

    all_labels = set(df[sankey_cols[0]].dropna()) | set(df[sankey_cols[1]].dropna()) | set(df[sankey_cols[2]].dropna())
    ordered_labels = [lbl for lbl in (focus_order + method_order + stimuli_order) if lbl in all_labels]
    label_to_id = {label: idx for idx, label in enumerate(ordered_labels)}

    sources, targets, values = [], [], []
    for idx in range(len(sankey_cols) - 1):
        grouped = df.groupby([sankey_cols[idx], sankey_cols[idx + 1]]).size().reset_index(name="count")
        for _, row in grouped.iterrows():
            src, dst = row[sankey_cols[idx]], row[sankey_cols[idx + 1]]
            if src in label_to_id and dst in label_to_id:
                sources.append(label_to_id[src])
                targets.append(label_to_id[dst])
                values.append(int(row["count"]))

    node_inflow = {i: 0 for i in range(len(ordered_labels))}
    node_outflow = {i: 0 for i in range(len(ordered_labels))}
    for src, dst, val in zip(sources, targets, values):
        node_outflow[src] += val
        node_inflow[dst] += val
    node_totals = {i: max(node_inflow[i], node_outflow[i]) for i in range(len(ordered_labels))}
    node_labels = [f"{label} ({node_totals[idx]})" for label, idx in label_to_id.items()]

    node_x, node_y = [], []
    focus_present = [lbl for lbl in focus_order if lbl in all_labels]
    method_present = [lbl for lbl in method_order if lbl in all_labels]
    stimuli_present = [lbl for lbl in stimuli_order if lbl in all_labels]

    focus_pos = {label: i for i, label in enumerate(focus_present)}
    method_pos = {label: i for i, label in enumerate(method_present)}
    stimuli_pos = {label: i for i, label in enumerate(stimuli_present)}

    for label in ordered_labels:
        if label in focus_pos:
            node_x.append(0.0)
            denom = max(len(focus_present) - 1, 1)
            node_y.append(focus_pos[label] / denom if denom else 0.5)
        elif label in method_pos:
            node_x.append(0.5)
            denom = max(len(method_present) - 1, 1)
            node_y.append(method_pos[label] / denom if denom else 0.5)
        else:
            node_x.append(1.0)
            denom = max(len(stimuli_present) - 1, 1)
            node_y.append(stimuli_pos[label] / denom if denom else 0.5)

    sankey = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=10,
                    thickness=10,
                    line=dict(color="black", width=1),
                    label=node_labels,
                    color=px.colors.qualitative.Plotly,
                    x=node_x,
                    y=node_y,
                ),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    sankey.update_layout(width=800, height=500)
    try:
        sankey.write_image(str(output_path), scale=2, width=800, height=500)
        logging.info("Saved literature Sankey to %s", output_path)
    except ValueError as exc:  # pragma: no cover - requires kaleido
        logging.warning("Unable to write Sankey image (is kaleido installed?): %s", exc)


# --------------------------------------------------------------------------- #
# CLI orchestration
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the perceived risk analysis pipeline.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the notebooks and data folder.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip nuisance model hyperparameter tuning.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip causal forest bootstrapping.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generation of plots (matplotlib/plotly).",
    )
    parser.add_argument(
        "--drop-driver-trust",
        action="store_true",
        help="Exclude Driver Trust indicators when fitting the causal forest.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=250,
        help="Number of bootstrap iterations for the causal forest (default: 250).",
    )
    parser.add_argument(
        "--bootstrap-random-state",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling.",
    )
    parser.add_argument(
        "--bootstrap-estimators",
        type=int,
        default=100,
        help="Number of trees in each causal forest fit.",
    )
    parser.add_argument(
        "--bootstrap-min-leaf",
        type=int,
        default=3,
        help="Minimum samples per leaf in the causal forest.",
    )
    parser.add_argument(
        "--tuning-cv",
        type=int,
        default=10,
        help="Number of folds for nuisance model tuning (default: 10).",
    )
    parser.add_argument(
        "--tuning-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV (default: -1 for all cores).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Do not persist intermediate data products (df_long_PR).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    config = AnalysisConfig(base_dir=args.base_dir, keep_intermediate=not args.no_intermediates)

    df_long_pr = build_perceived_risk_long(config)
    metadata_merged = load_metadata_tables(config)
    df_final = preprocess_responses(df_long_pr, metadata_merged, config)

    X, y, T, _ = prepare_model_inputs(df_final, drop_driver_trust=args.drop_driver_trust)

    tuned_results_df: Optional[pd.DataFrame] = None
    tuned_xgb_params: Optional[Dict[str, object]] = None
    best_xgb_params = {
        "colsample_bytree": 1.0,
        "learning_rate": 0.1,
        "max_depth": 4,
        "n_estimators": 200,
        "subsample": 0.8,
    }

    if not args.skip_tuning:
        tuned_results_df, tuned_xgb_params = tune_nuisance_models(
            X,
            T,
            y,
            cv_splits=args.tuning_cv,
            n_jobs=args.tuning_jobs,
            random_state=args.bootstrap_random_state,
        )
        if tuned_results_df is not None:
            tuned_results_df.to_csv(config.tuned_results_path, index=False)
            logging.info("Tuning results saved to %s", config.tuned_results_path)
        if tuned_xgb_params is not None:
            best_xgb_params = tuned_xgb_params
        else:
            logging.warning("XGBoost tuning did not return parameters; using default fallback")
        import json
        with open(config.xgb_params_path, 'w', encoding='utf-8') as f:
            json.dump(best_xgb_params, f, indent=4)
        logging.info("Persisted XGBoost best params to %s", config.xgb_params_path)
    else:
        logging.info("Skipping tuning step; using fallback XGBoost parameters")
        best_xgb_params = get_xgb_best_params(config, best_xgb_params)

    bootstrap_array = None
    if not args.skip_bootstrap:
        bootstrap_array = run_causal_forest_bootstrap(
            X,
            T,
            y,
            best_params=best_xgb_params,
            n_bootstrap=args.n_bootstrap,
            random_state=args.bootstrap_random_state,
            cv_splits=args.tuning_cv,
            n_estimators=args.bootstrap_estimators,
            min_samples_leaf=args.bootstrap_min_leaf,
        )
        np.savez_compressed(config.bootstrap_cates_path, data=bootstrap_array)
        logging.info("Stored bootstrap CATEs at %s", config.bootstrap_cates_path)
    else:
        logging.info("Skipping bootstrap step")

    if args.skip_plots:
        logging.info("Skipping plot generation by request")
        return

    if bootstrap_array is not None:
        ate_stats = compute_ate_stats(bootstrap_array)
        trust_ci = compute_trust_display_ci(
            df_final,
            n_bootstrap=min(1000, args.n_bootstrap),
            random_state=args.bootstrap_random_state,
        )
        combined_plot_path = config.outputs_dir / "combined_analysis_plot_percentile_ci.png"
        plot_combined_ate_figure(ate_stats, trust_ci, df_final, combined_plot_path)
    else:
        logging.warning("Bootstrap results unavailable; skipping ATE figure")

    sunburst_path = config.outputs_dir / "sunburst_chart.png"
    generate_sunburst(df_final, sunburst_path)

    sankey_path = config.outputs_dir / "literature_sankey.png"
    generate_literature_sankey(config, sankey_path)


if __name__ == "__main__":
    main()

