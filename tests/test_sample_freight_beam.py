from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.tools.beam.sample_freight_beam import _filter_by_tour_id
from impacts.tools.beam.sample_freight_beam import _sample_tours


def test_sample_freight_outputs_stay_aligned_on_tour_id() -> None:
    tours = pd.DataFrame(
        [
            {"tourId": "b2b-2", "departureTimeInSec": 2},
            {"tourId": "b2b-0", "departureTimeInSec": 0},
            {"tourId": "b2b-1", "departureTimeInSec": 1},
        ]
    )
    carriers = pd.DataFrame(
        [
            {"carrierId": "c0", "tourId": "b2b-0", "vehicleId": "v0"},
            {"carrierId": "c1", "tourId": "b2b-1", "vehicleId": "v1"},
            {"carrierId": "c2", "tourId": "b2b-2", "vehicleId": "v2"},
        ]
    )
    carriers2 = pd.DataFrame(
        [
            {"carrierId": "x0", "tourId": "b2b-0", "vehicleId": "w0"},
            {"carrierId": "x1", "tourId": "b2b-1", "vehicleId": "w1"},
            {"carrierId": "x2", "tourId": "b2b-2", "vehicleId": "w2"},
        ]
    )
    payloads = pd.DataFrame(
        [
            {"payloadId": "p0", "tourId": "b2b-0"},
            {"payloadId": "p1", "tourId": "b2b-0"},
            {"payloadId": "p2", "tourId": "b2b-1"},
            {"payloadId": "p3", "tourId": "b2b-2"},
        ]
    )

    sampled_tours = _sample_tours(
        tours,
        sample_share=0.5,
        seed=0,
    )
    sampled_tour_ids = set(sampled_tours["tourId"].astype(str))
    sampled_carriers = _filter_by_tour_id(carriers, "Carriers file", sampled_tour_ids)
    sampled_carriers2 = _filter_by_tour_id(carriers2, "Extra carriers file", sampled_tour_ids)
    sampled_payloads = _filter_by_tour_id(payloads, "Payloads file", sampled_tour_ids)

    sampled_tour_ids = set(sampled_tours["tourId"].astype(str))
    assert set(sampled_carriers["tourId"].astype(str)) == sampled_tour_ids
    assert set(sampled_carriers2["tourId"].astype(str)) == sampled_tour_ids
    assert set(sampled_payloads["tourId"].astype(str)) == sampled_tour_ids
    expected_payloads = payloads.loc[payloads["tourId"].astype(str).isin(sampled_tour_ids), "payloadId"].tolist()
    assert sampled_payloads["payloadId"].tolist() == expected_payloads
