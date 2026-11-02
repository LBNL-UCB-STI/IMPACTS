"""Reconstruct recorded BEAM electric-road-vehicle charging, not electrification.

RefuelSessionEvent.fuel is energy delivered in joules; its time is disconnect
time and duration is summed active charging time. Where plug-in is available,
spread energy uniformly over the connected interval: sub-session power cycles
are not present in this event extract. Never wrap next-day records into day 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROAD_MODES = {'car', 'car_hov2', 'car_hov3', 'ride_hail', 'ride_hail_pooled', 'bus'}


def reconstruct_sessions(sessions, plugs, electric_types, *, scale=10., end_seconds=86400., quarantine_invalid=False):
    if not np.isfinite(scale) or scale <= 0 or end_seconds != 86400.:
        raise ValueError('Positive scale and a 0-24h representative window required')
    records = sessions.copy().sort_values('time',kind='stable').reset_index(drop=True)
    records['eligible_electric_road_type'] = records.vehicleType.isin(electric_types)
    times = plugs.groupby(['vehicle', 'parkingZoneId'], dropna=False).time.apply(
        lambda x: np.sort(x.to_numpy(dtype=float))).to_dict()
    profiles = {}
    audited = []
    completed = set()
    for row in records.to_dict('records'):
        end, active, joules = (float(row[k]) for k in ['time', 'duration', 'fuel'])
        if not np.isfinite([end, active, joules]).all() or min(end, active, joules) < 0:
            raise ValueError('Invalid native charging time/energy')
        if active == 0 and joules > 0:
            raise ValueError('Positive energy with zero active charging duration')
        candidates = times.get((row['vehicle'], row['parkingZoneId']), np.array([]))
        candidates = candidates[candidates <= end]
        start = float(candidates[-1]) if len(candidates) else end - active
        method = 'plug_in_to_disconnect_mean_power' if len(candidates) else 'end_minus_duration_fallback'
        invalid = start < 0 or end - start + 1e-6 < active
        reason = 'active_duration_exceeds_connection' if invalid else ''
        if invalid and not quarantine_invalid:
            raise ValueError('Active charging duration exceeds matched connection interval')
        key = (row['vehicle'],row['parkingZoneId'],start)
        if not invalid and joules > 0:
            if key in completed:
                if not quarantine_invalid: raise ValueError('Duplicate disconnect for one plug-in')
                invalid = True
                reason = 'duplicate_positive_disconnect_for_same_plugin'
            else:
                completed.add(key)
        # A zero-energy event is retained in the audit, never a synthetic load.
        total_kwh = joules / 3_600_000.
        lo, hi = max(start, 0.), min(end, end_seconds)
        clipped_kwh = total_kwh * max(0., hi - lo) / (end - start) if end > start else 0.
        selected = bool(row['eligible_electric_road_type']) and not invalid
        included = clipped_kwh if selected else 0.
        if included > 0:
            profile = profiles.setdefault(row['vehicleType'], np.zeros(24))
            for hour in range(24):
                overlap = max(0., min(hi, (hour + 1) * 3600.) - max(lo, hour * 3600.))
                profile[hour] += total_kwh / (end - start) * overlap * scale
        row.update(start_seconds=start, end_seconds=end, timing_method=method,
                   quarantined=invalid,
                   exclusion_reason=reason,
                   connection_seconds=end-start, delivered_kwh=total_kwh,
                   included_sample_kwh=included, excluded_sample_kwh=total_kwh-included,
                   included_expanded_kwh=included*scale,
                   mean_connected_kw=total_kwh/(end-start)*3600. if end > start else 0.)
        audited.append(row)
    audit = pd.DataFrame(audited)
    np.testing.assert_allclose(sum(p.sum() for p in profiles.values()),
                               audit.included_expanded_kwh.sum(), atol=1e-8)
    # A plug-in cannot be reused for two positive-energy disconnections.
    positive = audit.loc[audit.delivered_kwh.gt(0) & ~audit.quarantined]
    if positive.duplicated(['vehicle', 'parkingZoneId', 'start_seconds']).any():
        raise ValueError('Ambiguous charging-session pairing')
    return profiles, audit
