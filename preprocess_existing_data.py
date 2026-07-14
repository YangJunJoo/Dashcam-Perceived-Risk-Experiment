#!/usr/bin/env python3
"""Rebuild the modelling dataset starting from the preprocess_survey_data step.

This script assumes that the long-format perceived-risk survey (`df_long_PR`)
already exists and focuses on reproducing the transformations performed in
`preprocess_survey_data.py`. By default it looks for the same data files that
the script references:

* `data/main/df_long_PR.csv`
* `data/main/video_survey_match.xlsx`
* `data/main/metadata_ideal.csv`
* `data/main/prolific_export_688279a66160545bd74e3613.csv`

The resulting `df_final.parquet` is written back to `data/main/df_final.parquet`, so the
downstream causal forest scripts can be executed without modification.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate df_final.pkl using existing long-form survey data."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the notebooks and data folder.",
    )
    parser.add_argument(
        "--long-path",
        type=Path,
        default=None,
        help="Path to df_long_PR (CSV or pickle). Defaults to data/main/df_long_PR.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path for the resulting df_final.pkl. Defaults to data/main/df_final.pkl.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def load_long_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find df_long_PR at {path}")
    logging.info("Loading perceived-risk responses from %s", path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path)


def load_metadata_tables(base_dir: Path) -> pd.DataFrame:
    match_path = base_dir / "data" / "main" / "video_survey_match.xlsx"
    metadata_path = base_dir / "data" / "main" / "metadata_ideal.csv"

    if not match_path.exists():
        raise FileNotFoundError(f"Missing video_survey_match.xlsx at {match_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata_ideal.csv at {metadata_path}")

    video_survey_match = pd.read_excel(match_path)
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

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.rename(columns={"file name": "file_name"})

    merged = pd.merge(video_survey_match, metadata, on="file_name", how="inner")
    logging.info("Merged metadata rows: %s", len(merged))
    return merged


def preprocess_responses(
    df_long: pd.DataFrame,
    metadata_merged: pd.DataFrame,
    base_dir: Path,
) -> pd.DataFrame:
    df = df_long.copy()

    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_columns:
        logging.debug("Dropping constant columns: %s", constant_columns)
        df = df.drop(columns=constant_columns)

    demo_path = base_dir / "data" / "main" / "prolific_export_688279a66160545bd74e3613.csv"
    if demo_path.exists():
        df_demo = pd.read_csv(demo_path)
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
        logging.warning("Prolific export not found at %s; skipping demographic merge", demo_path)

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

    df_final["Combined_Trust"] = df_final[
        ["Combined_Trust", "Combined_Safe", "Combined_Focus"]
    ].mean(axis=1)

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

    drop_post_columns: List[str] = ["Passenger Safety", "Passenger Distraction"]
    df = df.drop(columns=[col for col in drop_post_columns if col in df.columns])
    df["Answer Value"] = df["Answer Value"].astype(float)

    logging.info("Final dataset shape: %s", df.shape)
    return df


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    base_dir = args.base_dir.resolve()
    long_path = (
        args.long_path
        if args.long_path is not None
        else base_dir / "data" / "main" / "df_long_PR.csv"
    )
    output_path = (
        args.output_path
        if args.output_path is not None
        else base_dir / "data" / "main" / "df_final.parquet"
    )

    df_long = load_long_dataframe(long_path)
    metadata_merged = load_metadata_tables(base_dir)
    df_final = preprocess_responses(df_long, metadata_merged, base_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df_final.to_parquet(output_path, index=False)
    else:
        df_final.to_pickle(output_path)
    logging.info("Saved df_final to %s", output_path)


if __name__ == "__main__":
    main()


