"""Fleet Step 5: write mapped fleet outputs and per-EMFAC rate files.

Substeps:
5.1 Resolve output file locations for mapped fleet artifacts.
5.2 Prepare the emissions-rate output directory.
5.3 Write per-EMFAC emissions-rate files and attach them to vehicle types.
5.4 Persist mapped carriers, passenger vehicles, and vehicle-type tables.
"""

import json
import logging
import os
import os.path
import shutil
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd
from joblib import Parallel, delayed

# Get the absolute path to the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now use absolute import
from impacts.fleet.config import BeamClasses
pd.set_option('display.max_columns', 20)


def _resolve_freight_input_file(config, prefix):
    directory = Path(str(config["beam"]["freight_directory"])).expanduser().resolve()
    matches = sorted(directory.glob(f"{prefix}--*"))
    if not matches:
        raise FileNotFoundError(f"No file matching '{prefix}--*' found in {directory}")
    return matches[0]


def _mapped_output_path(path_like):
    source = Path(path_like)
    if source.suffix == ".gz" and source.name.endswith(".csv.gz"):
        stem = source.name[:-7]
        return str(source.with_name(f"{stem}--EM.csv.gz"))
    if source.suffix:
        stem = source.name[: -len(source.suffix)]
        return str(source.with_name(f"{stem}--EM{source.suffix}"))
    return str(source.with_name(f"{source.name}--EM"))


def _write_table(frame, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
        return
    compression = "gzip" if target.name.endswith(".csv.gz") else None
    frame.to_csv(target, index=False, compression=compression)


def _resolve_output_paths(scenario, work_dir, config):
    carriers_out_file = str(_mapped_output_path(_resolve_freight_input_file(config, "carriers")))
    ft_vehtypes_out_file = os.path.join(
        work_dir,
        _mapped_output_path(config['beam']['ft_vehicle_types_file']),
    )
    pax_vehtypes_out_file = os.path.join(
        work_dir,
        _mapped_output_path(config['beam']['pax_vehicle_types_file']),
    )
    vehicles_output = os.path.join(work_dir, _mapped_output_path(config['beam']['pax_vehicles_file']))
    emissions_rates_dir = os.path.join(
        os.path.dirname(os.path.join(work_dir, config['beam']['ft_vehicle_types_file'])),
        f"emissions/{scenario.replace('_', '-')}",
    )
    return carriers_out_file, ft_vehtypes_out_file, pax_vehtypes_out_file, vehicles_output, emissions_rates_dir


def _prepare_emissions_rate_directory(emissions_rates_dir):
    try:
        if os.path.exists(emissions_rates_dir):
            shutil.rmtree(emissions_rates_dir)
        os.makedirs(emissions_rates_dir, exist_ok=True)
        logging.info(f"Ready to write new data to the directory {emissions_rates_dir}")
    except Exception as e:
        logging.error(f"Failed to prepare directory {emissions_rates_dir}: {e}")


def _assign_rate_filepaths(vehtypes_with_emfac_id, emissions_rates, emissions_rates_dir):
    results = []
    chunk_size = 100
    for i in range(0, len(vehtypes_with_emfac_id), chunk_size):
        chunk = vehtypes_with_emfac_id.iloc[i:i + chunk_size]
        chunk_results = Parallel(n_jobs=-1, timeout=600)(
            delayed(process_single_vehicle_type)(veh_type, emissions_rates, f"{emissions_rates_dir}/")
            for _, veh_type in chunk.iterrows()
        )
        results.extend(chunk_results)
        del chunk_results

    path_parts = emissions_rates_dir.split('/')
    em_index = path_parts.index("emissions")
    shortened_path = '/'.join(path_parts[em_index:])
    for veh_type_id, emfac_id in results:
        if veh_type_id:
            relative_rates_filepath = f"{shortened_path}/{emfac_id}.csv"
            vehtypes_with_emfac_id.loc[
                vehtypes_with_emfac_id['vehicleTypeId'] == veh_type_id,
                'emissionsRatesFile',
            ] = relative_rates_filepath


def _write_updated_vehicle_types(
    vehtypes_with_emfac_id,
    other_pax_vehicle_types,
    ft_vehtypes_out_file,
    pax_vehtypes_out_file,
):
    logging.info(f"Writing:\n{ft_vehtypes_out_file}\n{pax_vehtypes_out_file}")

    ft_freight_mask = vehtypes_with_emfac_id['vehicleCategory'].isin(BeamClasses.get_freight_classes())
    updated_ft_vehicle_types = vehtypes_with_emfac_id[ft_freight_mask].copy()
    updated_ft_vehicle_types.drop(['emfacId', 'oldVehicleTypeId', 'vehicleClass'], axis=1, inplace=True)
    updated_ft_vehicle_types.to_csv(ft_vehtypes_out_file, index=False)

    updated_pax_vehicle_types_others = other_pax_vehicle_types.copy()
    updated_pax_vehicle_types_others['emissionsRatesFile'] = ""
    updated_pax_vehicle_types = pd.concat(
        [vehtypes_with_emfac_id[~ft_freight_mask].copy(), other_pax_vehicle_types],
        axis=0,
    )
    updated_pax_vehicle_types.drop(['emfacId', 'oldVehicleTypeId', 'vehicleClass'], axis=1, inplace=True)
    updated_pax_vehicle_types.to_csv(pax_vehtypes_out_file, index=False)


# Step 5.3-5.5: build mapped BEAM outputs and attach emissions files

def run_step5(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 5: write mapped fleet outputs and per-EMFAC emissions-rate files."""
    print("\n=== Map EMFAC To BEAM Population ===\n")
    (
        ft_vehtypes_out_file,
        pax_vehtypes_out_file,
        vehicles_output,
        emissions_rates_dir,
    ) = _resolve_output_paths(workflow["scenario"], workflow["work_dir"], workflow["config"])[1:]

    vehtypes_with_emfac_id = pd.concat(
        [workflow["new_ft_vehicle_types"], workflow["new_pax_vehicle_types"]],
        ignore_index=True,
    )
    vehtypes_with_emfac_id = vehtypes_with_emfac_id.fillna("")
    _prepare_emissions_rate_directory(emissions_rates_dir)
    _assign_rate_filepaths(vehtypes_with_emfac_id, workflow["emfac_rates"], emissions_rates_dir)
    _write_updated_vehicle_types(
        vehtypes_with_emfac_id,
        workflow["other_pax_vehicle_types"],
        ft_vehtypes_out_file,
        pax_vehtypes_out_file,
    )
    _write_table(workflow["new_carriers"], _resolve_output_paths(workflow["scenario"], workflow["work_dir"], workflow["config"])[0])
    _write_table(workflow["pax_vehicles"], vehicles_output)
    workflow["fleet_output_paths"] = {
        "carriers": _resolve_output_paths(workflow["scenario"], workflow["work_dir"], workflow["config"])[0],
        "ft_vehicle_types": ft_vehtypes_out_file,
        "pax_vehicle_types": pax_vehtypes_out_file,
        "pax_vehicles": vehicles_output,
        "emissions_rates_dir": emissions_rates_dir,
    }
    return workflow


def process_single_vehicle_type(
        veh_type: Dict[str, Any],
        emissions_rates: pd.DataFrame,
        rates_prefix_filepath: str
) -> Optional[tuple[str, str]]:
    """
    Process and save emissions rates for a single vehicle type.

    Filters the emissions rates for a specific vehicle type identified by its
    vehicleTypeId, removes the emfacId column, and saves the filtered data
    to a CSV file in the specified directory.

    Args:
        veh_type (Dict[str, Any]): Dictionary containing vehicle type information,
            must include 'vehicleTypeId' key
        emissions_rates (pd.DataFrame): DataFrame containing emissions rates data
            with 'emfacId' column matching vehicleTypeId values
        rates_prefix_filepath (str): Directory path prefix where the CSV file
            will be saved

    Returns:
        Optional[str]: The vehicleTypeId if processing was successful, None if
            no emissions data was found or an error occurred

    Raises:
        IOError: If there is an error writing the CSV file
    """
    try:
        veh_type_id = veh_type['vehicleTypeId']
        emfac_id = veh_type['emfacId']

        # Filter emissions_rates for the current vehicle type
        veh_emissions = emissions_rates[emissions_rates['emfacId'] == emfac_id].copy()

        if veh_emissions.empty:
            logging.warning(f"No emissions data found for vehicle type {veh_type_id}")
            return None

        # Generate the file path
        file_path = f"{rates_prefix_filepath}{emfac_id}.csv"

        # Save the emissions rates to a CSV file only if it doesn't exist
        if not os.path.exists(file_path):
            print(f"Writing emissions data to {file_path}")
            logging.info(f"Writing emissions data to {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            veh_emissions.to_csv(file_path, index=False)
            print(f"Created new file: {file_path}")
        else:
            print(f"File already generated: {file_path}")

        return veh_type_id, emfac_id

    except KeyError as e:
        logging.error(f"Missing required key in vehicle type data: {e}")
        return None
    except IOError as e:
        logging.error(f"Error writing emissions data file: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing vehicle type: {e}")
        return None

def print_unmapped(df, mapped_col, col_to_be_mapped):
    unmapped_classes = df[df[mapped_col].isna()][col_to_be_mapped].unique()
    if len(unmapped_classes) > 0:
        unmapped_classes_message = f"The following {col_to_be_mapped} were not mapped to {mapped_col}:\n"
        formatted_list = ""
        current_line = ""
        for vehicle_class in unmapped_classes:
            # Check if adding this class would exceed the line limit
            if len(current_line + vehicle_class) > 115:  # 115 to leave room for comma and space
                formatted_list += current_line.rstrip(", ") + "\n"
                current_line = vehicle_class + ", "
            else:
                current_line += vehicle_class + ", "
        # Add the last line
        if current_line:
            formatted_list += current_line.rstrip(", ")
        print(f"{unmapped_classes_message}{formatted_list}")
