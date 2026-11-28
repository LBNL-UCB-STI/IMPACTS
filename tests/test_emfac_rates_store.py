from __future__ import annotations

import pandas as pd

from impacts.pipeline.emfac._common import load_rates_store_relative_paths
from impacts.pipeline.emfac.activities.step4_finalize_output import _complete_sparse_counties_for_rates_store


def test_complete_sparse_counties_imputes_missing_counties_for_rate_rows() -> None:
    frame = pd.DataFrame(
        [
            {"county": "Marin", "process": "RUNEX", "speedMph_timeMin": 30, "pm25_gram": 10.0, "nox_gram": 5.0},
            {"county": "Sonoma", "process": "RUNEX", "speedMph_timeMin": 30, "pm25_gram": 14.0, "nox_gram": 7.0},
            {"county": "Alameda", "process": "PMBW", "speedMph_timeMin": 30, "pm25_gram": 20.0, "nox_gram": 9.0},
            {"county": "Marin", "process": "PMBW", "speedMph_timeMin": 30, "pm25_gram": 21.0, "nox_gram": 10.0},
            {"county": "Sonoma", "process": "PMBW", "speedMph_timeMin": 30, "pm25_gram": 22.0, "nox_gram": 11.0},
        ]
    )

    result = _complete_sparse_counties_for_rates_store(
        frame,
        expected_counties=["Alameda", "Marin", "Sonoma"],
    )
    runex = result.loc[(result["process"] == "RUNEX") & (result["speedMph_timeMin"] == 30)].sort_values("county")

    assert runex["county"].tolist() == ["Alameda", "Marin", "Sonoma"]
    assert runex["pm25_gram"].tolist() == [12.0, 10.0, 14.0]
    assert runex["nox_gram"].tolist() == [6.0, 5.0, 7.0]


def test_fleet_uses_rates_store_from_the_activities_manifest(tmp_path) -> None:
    store = tmp_path / "activities/2018-Baseline/rates"
    rates = store / "dataset/emfacId=fixture/fixture.parquet"
    rates.parent.mkdir(parents=True)
    rates.touch()
    result = load_rates_store_relative_paths(
        config={"activities": {"emissions_store_root": str(store)}},
        scenario="2018-Baseline",
        output_root=tmp_path,
        required_emfac_ids=["fixture"],
    )
    assert result == {"fixture": str(rates.relative_to(tmp_path))}
