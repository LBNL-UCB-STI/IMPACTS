from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.tools.beam.sample_passenger import _filter_blocks
from impacts.tools.beam.sample_passenger import _filter_plans
from impacts.tools.beam.sample_passenger import _filter_vehicles
from impacts.tools.beam.sample_passenger import _load_sampled_households
from impacts.tools.beam.sample_passenger import _load_sampled_persons


def test_sampled_urbansim_outputs_stay_aligned(tmp_path: Path) -> None:
    households_path = tmp_path / "households.parquet"
    persons_path = tmp_path / "persons.parquet"
    vehicles_path = tmp_path / "vehicles.csv.gz"
    vehicles2_path = tmp_path / "vehicles2.parquet"
    blocks_path = tmp_path / "blocks.parquet"
    plans_path = tmp_path / "plans.parquet"

    households = pd.DataFrame(
        [{"income": 10, "block_id": 7001}, {"income": 20, "block_id": 8002}],
        index=pd.Index([101, 202], name="household_id"),
    )
    households.to_parquet(households_path)

    persons = pd.DataFrame(
        [
            {"household_id": 101, "PNUM": 1},
            {"household_id": 101, "PNUM": 2},
            {"household_id": 202, "PNUM": 1},
        ],
        index=pd.Index([1001, 1002, 2001], name="person_id"),
    )
    persons.to_parquet(persons_path)

    vehicles = pd.DataFrame(
        [
            {"household_id": "101.0", "vehicle_id": "1", "vehicleTypeId": "a"},
            {"household_id": "202.0", "vehicle_id": "1", "vehicleTypeId": "b"},
        ]
    )
    vehicles.to_csv(vehicles_path, index=False, compression="gzip")
    vehicles.assign(vehicleTypeId=["a2", "b2"]).to_parquet(vehicles2_path, index=False)
    pd.DataFrame(
        [
            {"block_id": 7001, "TAZ": 1},
            {"block_id": 8002, "TAZ": 2},
        ]
    ).to_parquet(blocks_path, index=False)

    plans = pd.DataFrame(
        [
            {"person_id": 1001, "trip_id": 1},
            {"person_id": 1002, "trip_id": 2},
            {"person_id": 2001, "trip_id": 3},
        ]
    )
    plans.to_parquet(plans_path, index=False)

    sampled_households, sampled_household_ids = _load_sampled_households(
        households_path,
        sample_share=0.5,
        seed=0,
    )
    sampled_persons = _load_sampled_persons(persons_path, sampled_household_ids)
    sampled_person_ids = set(sampled_persons.index.astype(str))
    sampled_block_ids = set(sampled_households["block_id"].astype(str))

    sampled_vehicles_path = tmp_path / "sampled-vehicles.csv.gz"
    sampled_vehicles2_path = tmp_path / "sampled-vehicles2.parquet"
    sampled_blocks_path = tmp_path / "sampled-blocks.parquet"
    sampled_plans_path = tmp_path / "sampled-plans.parquet"
    vehicles_rows = _filter_vehicles(
        vehicles_path,
        sampled_vehicles_path,
        sampled_household_ids,
        chunksize=2,
    )
    vehicles2_rows = _filter_vehicles(
        vehicles2_path,
        sampled_vehicles2_path,
        sampled_household_ids,
        chunksize=2,
    )
    blocks_rows = _filter_blocks(
        blocks_path,
        sampled_blocks_path,
        sampled_block_ids,
    )
    plans_rows = _filter_plans(
        plans_path,
        sampled_plans_path,
        sampled_person_ids,
        chunksize=2,
    )

    assert list(sampled_households.index.astype(str)) == ["202"]
    assert list(sampled_persons.index.astype(str)) == ["2001"]
    assert list(sampled_persons["household_id"].astype(str)) == ["202"]

    sampled_vehicles = pd.read_csv(sampled_vehicles_path, compression="gzip")
    sampled_vehicles2 = pd.read_parquet(sampled_vehicles2_path)
    sampled_blocks = pd.read_parquet(sampled_blocks_path)
    sampled_plans = pd.read_parquet(sampled_plans_path)
    assert vehicles_rows == 1
    assert vehicles2_rows == 1
    assert blocks_rows == 1
    assert plans_rows == 1
    assert sampled_vehicles["household_id"].tolist() == [202.0]
    assert sampled_vehicles2["household_id"].tolist() == ["202.0"]
    assert sampled_vehicles2["vehicleTypeId"].tolist() == ["b2"]
    assert sampled_blocks["block_id"].tolist() == [8002]
    assert sampled_plans["person_id"].astype(str).tolist() == ["2001"]
