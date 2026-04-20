from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


AUTO_MODE_PREFIXES = ("DRIVE", "SHARED")
DEFAULT_HOUSEHOLDS_FILE = "households.parquet"
DEFAULT_PERSONS_FILE = "persons.parquet"
DEFAULT_PLANS_FILE = "plans.parquet"
DEFAULT_VEHICLES_FILE = "vehicles.csv.gz"
DEFAULT_PLAN_BATCH_SIZE = 250_000


class ProgressBar:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(0, int(total))
        self.current = 0
        self.last_render = 0.0
        self.started = time.time()

    def update(self, increment: int) -> None:
        self.current += max(0, int(increment))
        now = time.time()
        if now - self.last_render < 0.25 and self.current < self.total:
            return
        self.last_render = now
        total = max(1, self.total)
        ratio = min(1.0, self.current / total)
        width = 24
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self.started
        print(
            f"\r{self.label:<18} [{bar}] {ratio * 100:5.1f}% ({self.current:,}/{self.total:,}) {elapsed:5.1f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        self.last_render = 0.0
        self.update(0)
        print(file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze UrbanSim passenger consistency across households, vehicles, and plans. "
            "Plans do not include vehicle_id, so the script compares household cars and vehicle rows "
            "against household auto-plan activity as a proxy."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing households.parquet, persons.parquet, plans.parquet, and vehicles.csv.gz",
    )
    parser.add_argument(
        "--plan-batch-size",
        type=int,
        default=DEFAULT_PLAN_BATCH_SIZE,
        help="Batch size used when scanning plans.parquet",
    )
    return parser.parse_args()


def _normalize_integral_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return text
    if not decimal_value.is_finite():
        return text
    if decimal_value == decimal_value.to_integral_value():
        return format(decimal_value.quantize(Decimal("1")), "f")
    return text


def _normalize_identifier_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_integral_identifier)


def _is_auto_mode(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.upper()
    return text.str.startswith(AUTO_MODE_PREFIXES)


def _resolve_input_paths(input_dir: Path) -> dict[str, Path]:
    paths = {
        "households": input_dir / DEFAULT_HOUSEHOLDS_FILE,
        "persons": input_dir / DEFAULT_PERSONS_FILE,
        "plans": input_dir / DEFAULT_PLANS_FILE,
        "vehicles": input_dir / DEFAULT_VEHICLES_FILE,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required UrbanSim input files:\n" + "\n".join(missing))
    return paths


def _load_households(households_path: Path) -> pd.DataFrame:
    households = pd.read_parquet(households_path)
    if households.index.name != "household_id":
        raise ValueError(
            f"Expected households index name 'household_id', found {households.index.name!r}"
        )
    if "cars" not in households.columns:
        raise ValueError("Households file is missing required column 'cars'")
    result = households[["cars"]].copy()
    result.index = _normalize_identifier_series(result.index.to_series())
    result.index.name = "household_id"
    result["cars"] = pd.to_numeric(result["cars"], errors="coerce").fillna(0).astype(int)
    progress = ProgressBar("households", len(households))
    progress.update(len(households))
    progress.close()
    return result


def _load_person_households(persons_path: Path) -> pd.Series:
    persons = pd.read_parquet(persons_path, columns=["household_id"])
    if persons.index.name != "person_id":
        raise ValueError(f"Expected persons index name 'person_id', found {persons.index.name!r}")
    person_households = _normalize_identifier_series(persons["household_id"])
    person_households.index = _normalize_identifier_series(persons.index.to_series())
    person_households.index.name = "person_id"
    progress = ProgressBar("persons", len(persons))
    progress.update(len(persons))
    progress.close()
    return person_households


def _load_vehicle_counts(vehicles_path: Path) -> pd.Series:
    vehicles = pd.read_csv(vehicles_path, usecols=["household_id"])
    progress = ProgressBar("vehicles", len(vehicles))
    progress.update(len(vehicles))
    progress.close()
    household_ids = _normalize_identifier_series(vehicles["household_id"])
    return household_ids.value_counts().sort_index()


def _aggregate_plan_auto_usage(
    plans_path: Path,
    person_households: pd.Series,
    *,
    batch_size: int,
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(plans_path)
    frames: list[pd.DataFrame] = []
    progress = ProgressBar("plans", parquet_file.metadata.num_rows)
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["person_id", "tour_id", "trip_id", "tour_mode", "trip_mode"],
    ):
        table = pa.Table.from_batches([batch])
        frame = table.to_pandas()
        progress.update(len(frame))
        frame["person_id"] = _normalize_identifier_series(frame["person_id"])
        frame["household_id"] = frame["person_id"].map(person_households)
        frame = frame.loc[frame["household_id"].fillna("").ne("")].copy()
        if frame.empty:
            continue

        auto_trip_rows = frame.loc[_is_auto_mode(frame["trip_mode"])].copy()
        auto_tour_rows = frame.loc[_is_auto_mode(frame["tour_mode"])].copy()

        trip_counts = (
            auto_trip_rows.groupby("household_id", dropna=False)["trip_id"].nunique().rename("auto_trip_ids")
            if not auto_trip_rows.empty
            else pd.Series(dtype="int64", name="auto_trip_ids")
        )
        tour_counts = (
            auto_tour_rows.groupby("household_id", dropna=False)["tour_id"].nunique().rename("auto_tour_ids")
            if not auto_tour_rows.empty
            else pd.Series(dtype="int64", name="auto_tour_ids")
        )
        combined = pd.concat([trip_counts, tour_counts], axis=1).fillna(0).astype(int)
        frames.append(combined)
    progress.close()

    if not frames:
        return pd.DataFrame(columns=["auto_trip_ids", "auto_tour_ids"]).astype(int)

    plan_counts = pd.concat(frames).groupby(level=0).sum()
    plan_counts.index.name = "household_id"
    return plan_counts.sort_index()


def _build_household_report(
    households: pd.DataFrame,
    vehicle_counts: pd.Series,
    plan_counts: pd.DataFrame,
) -> pd.DataFrame:
    report = households.join(vehicle_counts.rename("vehicle_rows"), how="left")
    report = report.join(plan_counts, how="left")
    report["vehicle_rows"] = report["vehicle_rows"].fillna(0).astype(int)
    report["auto_trip_ids"] = report.get("auto_trip_ids", 0)
    report["auto_tour_ids"] = report.get("auto_tour_ids", 0)
    report["auto_trip_ids"] = pd.to_numeric(report["auto_trip_ids"], errors="coerce").fillna(0).astype(int)
    report["auto_tour_ids"] = pd.to_numeric(report["auto_tour_ids"], errors="coerce").fillna(0).astype(int)
    report["vehicle_minus_cars"] = report["vehicle_rows"] - report["cars"]
    report["auto_tours_gt_vehicles"] = report["auto_tour_ids"] > report["vehicle_rows"]
    report["auto_tours_gt_cars"] = report["auto_tour_ids"] > report["cars"]
    report["auto_trips_gt_vehicles"] = report["auto_trip_ids"] > report["vehicle_rows"]
    report["auto_trips_gt_cars"] = report["auto_trip_ids"] > report["cars"]
    report["vehicles_match_cars"] = report["vehicle_rows"] == report["cars"]
    return report.sort_index()


def _build_summary(report: pd.DataFrame) -> dict[str, int | float]:
    return {
        "households": int(len(report)),
        "total_household_cars": int(report["cars"].sum()),
        "total_vehicle_rows": int(report["vehicle_rows"].sum()),
        "total_auto_trip_ids": int(report["auto_trip_ids"].sum()),
        "total_auto_tour_ids": int(report["auto_tour_ids"].sum()),
        "households_vehicle_rows_match_cars": int(report["vehicles_match_cars"].sum()),
        "households_vehicle_rows_do_not_match_cars": int((~report["vehicles_match_cars"]).sum()),
        "households_auto_tours_gt_vehicles": int(report["auto_tours_gt_vehicles"].sum()),
        "households_auto_tours_gt_cars": int(report["auto_tours_gt_cars"].sum()),
        "households_auto_trips_gt_vehicles": int(report["auto_trips_gt_vehicles"].sum()),
        "households_auto_trips_gt_cars": int(report["auto_trips_gt_cars"].sum()),
        "households_with_zero_cars_and_auto_tours": int(report.loc[report["cars"].eq(0), "auto_tour_ids"].gt(0).sum()),
        "households_with_zero_vehicle_rows_and_auto_tours": int(report.loc[report["vehicle_rows"].eq(0), "auto_tour_ids"].gt(0).sum()),
    }


def _print_top_households(report: pd.DataFrame, *, title: str, mask: pd.Series, limit: int = 10) -> None:
    subset = report.loc[mask, ["cars", "vehicle_rows", "auto_trip_ids", "auto_tour_ids"]].head(limit)
    print(title)
    if subset.empty:
        print("  none")
        return
    print(subset.to_string())


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    input_paths = _resolve_input_paths(input_dir)

    households = _load_households(input_paths["households"])
    person_households = _load_person_households(input_paths["persons"])
    vehicle_counts = _load_vehicle_counts(input_paths["vehicles"])
    plan_counts = _aggregate_plan_auto_usage(
        input_paths["plans"],
        person_households,
        batch_size=args.plan_batch_size,
    )
    report = _build_household_report(households, vehicle_counts, plan_counts)
    summary = _build_summary(report)

    print(
        "Plans do not include vehicle_id, so comparisons against plans use unique household auto trip/tour counts "
        "as a proxy, not literal vehicles in plans."
    )
    print(f"Households: {summary['households']:,}")
    print(f"Total household cars: {summary['total_household_cars']:,}")
    print(f"Total vehicle rows: {summary['total_vehicle_rows']:,}")
    print(f"Households where vehicle rows match cars: {summary['households_vehicle_rows_match_cars']:,}")
    print(f"Households where vehicle rows do not match cars: {summary['households_vehicle_rows_do_not_match_cars']:,}")
    print(f"Households where auto tours > vehicle rows: {summary['households_auto_tours_gt_vehicles']:,}")
    print(f"Households where auto tours > cars: {summary['households_auto_tours_gt_cars']:,}")
    print(f"Households where auto trips > vehicle rows: {summary['households_auto_trips_gt_vehicles']:,}")
    print(f"Households where auto trips > cars: {summary['households_auto_trips_gt_cars']:,}")
    _print_top_households(
        report,
        title="Sample households where vehicle rows do not match cars:",
        mask=~report["vehicles_match_cars"],
    )
    _print_top_households(
        report,
        title="Sample households where auto tours exceed vehicle rows:",
        mask=report["auto_tours_gt_vehicles"],
    )
    _print_top_households(
        report,
        title="Sample households where auto tours exceed cars:",
        mask=report["auto_tours_gt_cars"],
    )


if __name__ == "__main__":
    main()
