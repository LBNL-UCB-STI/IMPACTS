"""Fleet Step 1: prepare shared EMFAC workflow inputs.

Substeps:
1.1 Build or load the EMFAC-to-BEAM class crosswalk.
1.2 Build the shared EMFAC formatter used by later steps.
1.3 Persist the resolved fleet workflow snapshot for reproducibility.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

# Get the absolute path to the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from impacts.fleet.config import BeamClasses

# Step 1.1: class mapping helpers


def _read_table(path: str) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    table = pv.read_csv(str(source), read_options=pa.csv.ReadOptions(use_threads=True))
    return table.to_pandas()


def _normalize_vehicle_class_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "vehicle_class" not in result.columns and "vehicleCategory" in result.columns:
        result = result.rename(columns={"vehicleCategory": "vehicle_class"})
    return result


def generate_emfac_beam_class_mapping(emfac_activity_file, vehicle_class_output_file, to_filter_out):
    """
    Creates vehicle class mapping and saves it to a JSON file if it doesn't exist.
    If the file exists, loads and returns the existing mapping.

    Args:
        to_filter_out:

    Returns:
        dict: The vehicle class mapping (either newly created or loaded from existing file)
    """
    # Check if the file already exists
    if os.path.exists(vehicle_class_output_file):
        print(f"File {vehicle_class_output_file} already exists. Loading existing mapping.")
        with open(vehicle_class_output_file, 'r') as f:
            return json.load(f)

    # Create the mapping
    mapping = {}

    df = _normalize_vehicle_class_column(_read_table(emfac_activity_file))

    for vehicle in df["vehicle_class"].unique():
        if 'Utility' in vehicle or 'Public' in vehicle:
            mapping[vehicle] = "NotMatched"
        elif 'Port' in vehicle or 'POLA' in vehicle or 'POAK' in vehicle:
            mapping[vehicle] = "NotMatched"
        elif 'SWCV' in vehicle or 'PTO' in vehicle or 'T6TS' in vehicle:
            mapping[vehicle] = "NotMatched"
        elif vehicle in ['LDA', 'LDT1', 'LDT2', 'MDV']:
            mapping[vehicle] = BeamClasses.CLASS_CAR
        elif vehicle in ['MCY']:
            mapping[vehicle] = BeamClasses.CLASS_BIKE
        elif vehicle in ['UBUS']:
            mapping[vehicle] = BeamClasses.CLASS_MDP
        elif 'LHD' in vehicle:
            mapping[vehicle] = BeamClasses.CLASS_2B3_VOCATIONAL
        elif 'Class 4' in vehicle or 'Class 5' in vehicle or 'Class 6' in vehicle:
            mapping[vehicle] = BeamClasses.CLASS_456_VOCATIONAL
        elif 'Class 7' in vehicle or 'Class 8' in vehicle:
            if 'Tractor' in vehicle or 'CAIRP' in vehicle:
                mapping[vehicle] = BeamClasses.CLASS_78_TRACTOR
            else:
                mapping[vehicle] = BeamClasses.CLASS_78_VOCATIONAL
        elif "T7IS" in vehicle:
            mapping[vehicle] = BeamClasses.CLASS_78_TRACTOR
        else:
            mapping[vehicle] = "NotMatched"

    # Print category groupings
    class_groups = defaultdict(list)
    for vehicle, vehicle_class in mapping.items():
        if vehicle_class in to_filter_out:
            mapping[vehicle] = "NotMatched"
        class_groups[mapping[vehicle]].append(vehicle)
    for vehicle_class, vehicles in class_groups.items():
        print(f"Category: {vehicle_class}")
        for vehicle in vehicles:
            print(f"  - {vehicle}")

    return {k: v for k, v in mapping.items() if v != "NotMatched"}


def _sanitize_token(value: object) -> str:
    return str(value).replace(" ", "").replace("_", "").replace("/", "").replace("-", "")


def create_emfac_id(row: pd.Series) -> str:
    return (
        f"{_sanitize_token(row['model_year_group'])}"
        f"{_sanitize_token(row['vehicle_class'])}"
        f"{_sanitize_token(row['fuel'])}"
    )


def categorize_model_year(year, bin_years=None) -> str:
    if bin_years is None:
        bin_years = [1993, 2006, 2018]
    bin_years = sorted(bin_years)
    if year <= bin_years[0]:
        return str(bin_years[0])
    for i in range(len(bin_years) - 1):
        if year <= bin_years[i + 1]:
            return str(bin_years[i + 1])
    return str(bin_years[-1])


def build_emissions_formatter(mapping_config):
    def format_emissions_data(emfac_types: pd.DataFrame) -> pd.DataFrame:
        result_ft_df = emfac_types.copy()
        result_ft_df["mappedClass"] = result_ft_df["vehicle_class"].map(mapping_config["class"]["emfac-ft"])
        result_ft_df.dropna(subset=["mappedClass"], inplace=True)
        result_ft_df["mappedFuel"] = result_ft_df["fuel"].map(mapping_config["fuel"]["emfac-ft"])
        result_ft_df.dropna(subset=["mappedFuel"], inplace=True)

        result_pax_df = emfac_types.copy()
        result_pax_df["mappedClass"] = result_pax_df["vehicle_class"].map(mapping_config["class"]["emfac-pax"])
        result_pax_df.dropna(subset=["mappedClass"], inplace=True)
        result_pax_df["mappedFuel"] = result_pax_df["fuel"].map(mapping_config["fuel"]["emfac-pax"])
        result_pax_df.dropna(subset=["mappedFuel"], inplace=True)

        result_bus_df = emfac_types.copy()
        result_bus_df["mappedClass"] = result_bus_df["vehicle_class"].map(mapping_config["class"]["emfac-bus"])
        result_bus_df.dropna(subset=["mappedClass"], inplace=True)
        result_bus_df["mappedFuel"] = result_bus_df["fuel"].map(mapping_config["fuel"]["emfac-bus"])
        result_bus_df.dropna(subset=["mappedFuel"], inplace=True)

        result_df = pd.concat([result_ft_df, result_pax_df, result_bus_df])
        result_df["model_year_group"] = result_df["model_year"].apply(
            lambda value: categorize_model_year(value, mapping_config["fleet"]["model_year_bins"])
        )
        if "sub_area" in result_df.columns:
            result_df[["county", "area"]] = result_df["sub_area"].str.extract(r"^([^()]+)\s*\(([^)]+)\)")
        else:
            if "county" not in result_df.columns:
                result_df["county"] = ""
            if "area" not in result_df.columns:
                result_df["area"] = ""
        result_df["county"] = result_df["county"].astype(str).str.strip().str.lower()
        result_df["area"] = result_df["area"].astype(str).str.strip()
        result_df["emfacId"] = result_df.apply(create_emfac_id, axis=1)
        return result_df

    return format_emissions_data


def run_step1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 1: prepare the shared EMFAC class mapping and formatter."""
    area = workflow["area"]
    scenario = workflow["scenario"]
    config = workflow["config"]
    work_dir = workflow["work_dir"]

    emfac_activity_file = config["emfac"]["activity_file"]
    vehicle_class_output_file = f"{config['run']['output_dir']}/{area}_vehicle_class_mapping_{scenario}.json"
    emfac_class_map = generate_emfac_beam_class_mapping(
        emfac_activity_file=emfac_activity_file,
        vehicle_class_output_file=os.path.join(work_dir, vehicle_class_output_file),
        to_filter_out=[BeamClasses.CLASS_2B3_VOCATIONAL],
    )
    config["mapping"]["class"]["emfac"] = emfac_class_map
    format_emissions_data = build_emissions_formatter(config["mapping"])

    emissions_work_dir = os.path.join(work_dir, config["run"]["emissions_dir"])
    os.makedirs(emissions_work_dir, exist_ok=True)
    config_path = os.path.join(emissions_work_dir, f"{area}_emissions_config_{scenario}.json")
    with open(config_path, "w") as f:
        json.dump({"area": area, "scenario": scenario, "work_dir": work_dir, "config": config}, f, indent=2)

    workflow["emfac_class_map"] = emfac_class_map
    workflow["format_emissions_data"] = format_emissions_data
    workflow["emissions_config_path"] = config_path
    return workflow
