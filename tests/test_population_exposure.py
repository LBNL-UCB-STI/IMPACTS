from __future__ import annotations

import pandas as pd

from impacts.pipeline.preprocessing.step4_aggregate_population import _classify_urban
from impacts.pipeline.preprocessing.step4_aggregate_population import _counts_from_joined


def test_counts_from_joined_aggregates_per_cell() -> None:
    joined = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "aermod_cell_id": [10, 10, 20, 20, 20],
        }
    )

    result = _counts_from_joined(joined)

    assert result["aermod_cell_id"].tolist() == [10, 20]
    assert result["person_count"].tolist() == [2, 3]
    assert result["source_urban_class"].tolist() == [0, 0]


def test_classify_urban_thresholds() -> None:
    counts = pd.Series([0, 999, 1000, 9999, 10000, 50000])
    result = _classify_urban(counts).tolist()
    assert result == [0, 0, 1000, 1000, 10000, 10000]