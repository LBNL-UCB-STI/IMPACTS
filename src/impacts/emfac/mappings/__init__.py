from __future__ import annotations

from pathlib import Path

import pandas as pd

MAPPINGS_DIR = Path(__file__).resolve().parent


def load_class_fuel_alternatives() -> dict[tuple[str, str], tuple[str, str]]:
    frame = pd.read_csv(MAPPINGS_DIR / "class_fuel_alternatives.csv")
    return {
        (str(row.source_vehicle_class), str(row.source_fuel)): (
            str(row.alternative_vehicle_class),
            str(row.alternative_fuel),
        )
        for row in frame.itertuples(index=False)
    }


def load_beam_road_type_mapping() -> list[dict[str, object]]:
    frame = pd.read_csv(MAPPINGS_DIR / "beam_road_type_mapping.csv")
    mappings: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        f_class = None if pd.isna(row.f_class) else int(row.f_class)
        mappings.append(
            {
                "beam_road_type": str(row.beam_road_type),
                "road_category": str(row.road_category),
                "carb_road_category": str(row.carb_road_category),
                "f_class": f_class,
            }
        )
    return mappings
