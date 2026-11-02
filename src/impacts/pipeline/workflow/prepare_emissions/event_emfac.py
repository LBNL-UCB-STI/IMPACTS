"""Events -> daily link emissions in grams, before annualization/allocation.

Uses the native EMFAC partition store and explicit vehicleTypeId -> emfacId
mapping. No fleet mix, sampling fraction, or missing vehicle mapping is inferred.
Process units follow BEAM VehicleEmissions.scala (see docs/events_emfac.md).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ....common import configure_duckdb_connection, read_table
from ....config.defaults import meters_per_mile

logger = logging.getLogger(__name__)

COUNTIES = dict(zip(
    ['001', '013', '041', '055', '075', '081', '085', '095', '097'],
    ['Alameda', 'Contra Costa', 'Marin', 'Napa', 'San Francisco',
     'San Mateo', 'Santa Clara', 'Solano', 'Sonoma'],
))
POLLUTANTS = {'PM25': 'pm25_gram', 'NOx': 'nox_gram', 'SOx': 'sox_gram',
              'NH3': 'nh3_gram', 'ROG': 'rog_gram', 'BC': 'bc_gram'}
PER_MILE = {'RUNEX', 'PMBW', 'PMTW', 'PRDUST', 'PTOEX'}
PER_HOUR = {'RUNLOSS', 'DIURN', 'IDLEX'}
PER_START = {'STREX', 'HOTSOAK'}
ROAD_MODES = {'car', 'car_hov2', 'car_hov3', 'bus', 'ride_hail', 'ride_hail_pooled'}


def load_vehicle_mapping(paths):
    frames = [read_table(p) for p in paths if p]
    if not frames:
        raise ValueError('Events EMFAC requires configured vehicle type tables.')
    frame = pd.concat(frames, ignore_index=True)
    if not {'vehicleTypeId', 'emfacId'}.issubset(frame.columns):
        raise ValueError('Vehicle type tables require vehicleTypeId and emfacId.')
    frame['vehicleTypeId'] = frame.vehicleTypeId.astype(str).str.strip()
    frame['emfacId'] = frame.emfacId.astype('string').str.strip()
    if frame.vehicleTypeId.duplicated().any():
        for _, group in frame.groupby('vehicleTypeId'):
            if len(group.drop_duplicates()) > 1:
                raise ValueError(f'Conflicting vehicle mapping: {group.vehicleTypeId.iloc[0]}')
    return frame.drop_duplicates('vehicleTypeId').set_index('vehicleTypeId', drop=False)


def _merge_rate_duplicates(frame, pollutant_columns):
    # Passenger/freight stores can contain identical or complementary rows.
    # Neither summing nor averaging conflicting nonzero factors is justified.
    keys = ['county', 'process', 'roadCategory', 'speedMph_timeMin']
    if not frame.duplicated(keys).any():
        return frame
    groups = frame.groupby(keys, dropna=False, sort=False)
    if groups[pollutant_columns].nunique().gt(1).any().any():
        raise ValueError('Conflicting duplicate EMFAC rates; refusing to average them.')
    return groups[pollutant_columns].first().reset_index()


class EmfacRateStore:
    """Read only requested native partitions; retain county/road/activity bins."""

    def __init__(self, root, mapping, pollutants):
        self.root = Path(root)
        self.mapping = mapping
        self.pollutants = list(pollutants)
        if set(pollutants) - POLLUTANTS.keys():
            raise ValueError(f'Unsupported pollutants: {set(pollutants) - POLLUTANTS.keys()}')
        self.cache = {}
        self.lookup_cache = {}

    def rates(self, vehicle_type):
        if vehicle_type not in self.mapping.index:
            raise ValueError(f'No explicit EMFAC vehicle mapping for {vehicle_type!r}')
        emfac_id = self.mapping.loc[vehicle_type, 'emfacId']
        if pd.isna(emfac_id) or not emfac_id:
            raise ValueError(f'Empty emfacId for road vehicle {vehicle_type!r}')
        if emfac_id in self.cache:
            return self.cache[emfac_id]
        dataset = self.root / 'dataset' if (self.root / 'dataset').is_dir() else self.root
        # Resolve an exact partition, not a user-provided glob.
        if Path(str(emfac_id)).name != str(emfac_id) or '/' in str(emfac_id) or '\\' in str(emfac_id):
            raise ValueError('Invalid emfacId partition name')
        files = sorted((dataset / f'emfacId={emfac_id}').glob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'No EMFAC partition for {vehicle_type}: emfacId={emfac_id} in {dataset}')
        frame = pd.concat([pq.ParquetFile(f).read().to_pandas() for f in files], ignore_index=True)
        columns = [POLLUTANTS[p] for p in self.pollutants]
        required = {'county', 'process', 'roadCategory', 'speedMph_timeMin', *columns}
        if required - set(frame.columns):
            raise ValueError(f'EMFAC partition missing columns: {sorted(required - set(frame.columns))}')
        for col in columns:
            values = pd.to_numeric(frame[col], errors='raise')
            if (~np.isfinite(values.dropna())).any() or values.dropna().lt(0).any():
                raise ValueError(f'Invalid emission rate in {emfac_id}/{col}')
            frame[col] = values
        frame = _merge_rate_duplicates(frame, columns)
        frame['county'] = frame.county.astype(str).str.strip().str.lower()
        frame['roadCategory'] = frame.roadCategory.astype('string').str.strip().str.lower()
        frame['process'] = frame.process.astype(str).str.upper()
        unknown = set(frame.process) - PER_MILE - PER_HOUR - PER_START
        if unknown:
            raise ValueError(f'Unsupported EMFAC processes: {sorted(unknown)}')
        self.cache[emfac_id] = frame
        return frame

    def factor(self, vehicle_type, process, county, road, speed, soak_minutes):
        key = (vehicle_type, process, county.lower(), str(road).lower())
        if key not in self.lookup_cache:
            frame = self.rates(vehicle_type)
            rows = frame.loc[frame.process.eq(process) & frame.county.eq(county.lower())]
            exact = rows.roadCategory.eq(str(road).lower()).fillna(False)
            rows = rows.loc[exact] if exact.any() else rows.loc[rows.roadCategory.isna() | rows.roadCategory.eq('')]
            self.lookup_cache[key] = rows
        rows = self.lookup_cache[key]
        if rows.empty:
            raise ValueError(f'Missing EMFAC rate: {vehicle_type}/{process}/{county}/{road}')
        labels = rows.speedMph_timeMin
        if labels.notna().any():
            if labels.isna().any():
                raise ValueError('Mixed binned and unbinned EMFAC rates are ambiguous')
            value = speed if process in PER_MILE else soak_minutes
            centers, contains = [], []
            for label in labels.astype(str):
                if label.startswith(('(', '[')):
                    left, right = map(float, label.strip('[]()').split(','))
                    centers.append((left + right) / 2)
                    contains.append((value >= left if label[0] == '[' else value > left)
                                    and (value <= right if label[-1] == ']' else value < right))
                else:
                    center = float(label)
                    centers.append(center)
                    contains.append(abs(value - center) <= (2.5 if process in PER_MILE else .5))
            # Deterministic nearest-bin fallback, as used in BEAM. Lower center
            # wins a tie. Apply rates before aggregating speeds or soak durations.
            candidates = [i for i, inside in enumerate(contains) if inside] or list(range(len(rows)))
            pos = min(candidates, key=lambda i: (abs(centers[i] - value), centers[i]))
            row = rows.iloc[pos]
        else:
            if len(rows) != 1:
                raise ValueError('Ambiguous unbinned EMFAC rates')
            row = rows.iloc[0]
        # Native BEAM wide tables use null for a pollutant absent from a matched
        # process row. Missing rows/columns/partitions are NEVER treated as zero.
        return np.array([0.0 if pd.isna(row[POLLUTANTS[p]]) else float(row[POLLUTANTS[p]])
                         for p in self.pollutants])


def build_daily_emissions(events_path, network_path, intersection_path, rates_folder,
                          vehicle_paths, pollutants, output_path, allowed_ids=None):
    """Stream sorted road trips into daily gram totals compatible with Step 1.

    Parking is reconstructed only between observed trips. Initial/final parking
    and explicit hotelling are not inferred. See the limitations in the docs.
    """
    mapping = load_vehicle_mapping(vehicle_paths)
    store = EmfacRateStore(rates_folder, mapping, pollutants)
    network = read_table(network_path)
    required = {'linkId', 'linkLength', 'attributeOrigType'}
    if required - set(network):
        raise ValueError(f'Network missing {required - set(network)}')
    if network.linkId.duplicated().any():
        raise ValueError('Duplicate linkId in network')
    network = network.set_index('linkId')
    network_ids = set(network.index)
    link_data = network[['linkLength', 'attributeOrigType']].to_dict('index')
    vehicle_data = mapping.to_dict('index')
    zones = read_table(intersection_path)
    required = {'linkId', 'county_COUNTYFP', 'county_proportion'}
    if required - set(zones):
        raise ValueError(f'County intersection missing {required - set(zones)}')
    zones = zones.groupby(['linkId', 'county_COUNTYFP'], as_index=False).county_proportion.sum()
    if (~np.isfinite(zones.county_proportion)).any() or zones.county_proportion.le(0).any():
        raise ValueError('Invalid county intersection proportions')
    county_weights = {}
    for link, group in zones.groupby('linkId'):
        total = group.county_proportion.sum()
        if total > 1.00001:
            raise ValueError(f'County proportions exceed one for link {link}')
        # Full-link emission rate; grid intersection fractions are applied once
        # later. County-dependent factors are averaged by in-domain length.
        codes = [str(c).zfill(3) for c in group.county_COUNTYFP]
        if set(codes) - COUNTIES.keys():
            raise ValueError(f'Events EMFAC county names require a configured lookup outside SFBAY: {codes}')
        county_weights[int(link)] = [(COUNTIES[c], float(w / total))
                                     for c, w in zip(codes, group.county_proportion)]
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    if partial.exists():
        raise FileExistsError(f'Incomplete prior events conversion exists: {partial}')
    con = duckdb.connect()
    writer = None
    totals = {}
    stats = dict(road_trips=0, out_of_domain_link_traversals=0, observed_parking_intervals=0,
                 initial_final_parking_inferred=False, fleet_expansion_applied=False,
                 annualization_applied=False, units='grams', source=str(Path(events_path).resolve()),
                 min_departure_seconds=None, max_arrival_seconds=None,
                 limitations=['No initial/final parking or initial cold start inferred',
                              'Working idle uses idleTimeFraction times driving hours; no hotelling inferred',
                              'Missing payload field/value assumes zero payload',
                              'County factors are length-weighted within each link',
                              'All supplied road trips retained, including hours beyond 24',
                              'Native matched-row null pollutant entries follow BEAM zero-component convention'])

    def flush():
        nonlocal writer
        if not totals:
            return
        records = [dict(zip(['hour', 'linkId', 'vehicleTypeId', 'process', 'observations', *pollutants],
                            [*key, values[0], *values[1:]])) for key, values in totals.items()]
        table = pa.Table.from_pylist(records)
        if writer is None:
            writer = pq.ParquetWriter(partial, table.schema, compression='zstd')
        writer.write_table(table)
        totals.clear()

    def emit(vtype, process, link, time, seconds, speed, payload=0., soak_minutes=0.):
        if link not in county_weights:
            return
        meta = vehicle_data[vtype]
        length = float(link_data[link]['linkLength'])
        road = link_data[link]['attributeOrigType']
        factor = sum(w * store.factor(vtype, process, county, road, speed, soak_minutes)
                     for county, w in county_weights[link])
        amount = length / meters_per_mile if process in PER_MILE else seconds / 3600 if process in PER_HOUR else 1.
        if process == 'PRDUST':
            mass = pd.to_numeric(meta.get('curbWeightInKg'), errors='coerce')
            if pd.isna(mass) or not np.isfinite(mass) or mass <= 0:
                raise ValueError(f'PRDUST requires positive curbWeightInKg for {vtype}')
            amount *= ((mass + payload) / 907.18474) ** 1.02
        if process == 'IDLEX':
            idle = pd.to_numeric(meta.get('idleTimeFraction'), errors='coerce')
            if pd.isna(idle) or not np.isfinite(idle) or not 0 <= idle <= 1:
                raise ValueError(f'IDLEX requires explicit idleTimeFraction for {vtype}')
            amount *= idle
        key = (int(time // 3600), int(link), vtype, process)
        if key not in totals:
            totals[key] = np.zeros(1 + len(pollutants))
        totals[key] += np.r_[1., factor * amount]

    try:
        configure_duckdb_connection(con, working_dir=target, show_progress=False)
        escaped = str(events_path).replace("'", "''")
        scan = (f"read_parquet('{escaped}')" if str(events_path).endswith('.parquet')
                else f"read_csv_auto('{escaped}', sample_size=-1)")
        columns = {row[0] for row in con.execute(f'DESCRIBE SELECT * FROM {scan}').fetchall()}
        required = {'type', 'vehicle', 'vehicleType', 'departureTime', 'arrivalTime', 'links', 'linkTravelTime', 'mode'}
        if required - columns:
            raise ValueError(f'Events missing required fields: {required - columns}')
        modes = ','.join("'" + m + "'" for m in sorted(ROAD_MODES))
        where = f"type='PathTraversal' AND mode IN ({modes})"
        observed = {row[0] for row in con.execute(f'SELECT DISTINCT vehicleType FROM {scan} WHERE {where}').fetchall()}
        unknown = observed - set(mapping.index)
        if unknown:
            raise ValueError(f'Events vehicleType -> emfacId mapping missing for {len(unknown)} types: {sorted(unknown)}')
        active = observed if allowed_ids is None else observed & set(allowed_ids)
        if not active:
            raise ValueError('No eligible road vehicles in events')
        for vtype in sorted(active):
            store.rates(vtype)  # Fail before expanding millions of links.
        processes_by_type = {v: set(store.rates(v).process) for v in active}
        stats['vehicle_types'] = sorted(active)
        stats['emfac_partitions'] = sorted(store.cache)
        payload = 'PayloadWeightInKg' if 'PayloadWeightInKg' in columns else 'NULL'
        query = f'''SELECT vehicle, vehicleType, departureTime, arrivalTime, links, linkTravelTime,
                    {payload} AS payload FROM {scan} WHERE {where}
                    ORDER BY vehicle, departureTime, arrivalTime'''
        previous = None
        for batch in con.execute(query).to_arrow_reader(20000):
            for row in batch.to_pylist():
                vtype = row['vehicleType']
                if vtype not in active:
                    continue
                links = [int(x) for x in str(row['links'] or '').split(',') if x.strip()]
                times = [float(x) for x in str(row['linkTravelTime'] or '').split(',') if x.strip()]
                if not links or len(links) != len(times) or not np.isfinite(times).all() or min(times) < 0:
                    raise ValueError(f'Malformed links/linkTravelTime for vehicle {row["vehicle"]}')
                missing = set(links) - network_ids
                if missing:
                    raise ValueError(f'Event links absent from supplied network: {sorted(missing)[:10]}')
                dep, arr = float(row['departureTime']), float(row['arrivalTime'])
                if not np.isfinite([dep, arr]).all() or dep < 0 or arr < dep:
                    raise ValueError('Invalid event departure/arrival time')
                payload_value = float(row['payload']) if row['payload'] not in (None, '') else 0.
                if not np.isfinite(payload_value) or payload_value < 0:
                    raise ValueError('Invalid payload weight')
                processes = processes_by_type[vtype]
                if previous and previous['vehicle'] == row['vehicle']:
                    gap = dep - previous['arrival']
                    if gap < 0:
                        raise ValueError(f'Overlapping road trips for vehicle {row["vehicle"]}')
                    if gap > 0:
                        stats['observed_parking_intervals'] += 1
                        prevtype = previous['type']
                        for process in processes_by_type[prevtype] & {'DIURN', 'HOTSOAK'}:
                            emit(prevtype, process, previous['link'], previous['arrival'], gap, 0., soak_minutes=gap/60)
                        if 'STREX' in processes:
                            emit(vtype, 'STREX', links[0], dep, gap, 0., soak_minutes=gap/60)
                clock = dep
                for link, seconds in zip(links, times):
                    if link not in county_weights:
                        stats['out_of_domain_link_traversals'] += 1
                        clock += seconds
                        continue
                    length = float(link_data[link]['linkLength'])
                    if not np.isfinite(length) or length < 0 or (length > 0 and seconds == 0):
                        raise ValueError(f'Invalid link length/travel time for link {link}')
                    speed = length / meters_per_mile / (seconds / 3600) if seconds else 0.
                    for process in processes & (PER_MILE | {'RUNLOSS', 'IDLEX'}):
                        emit(vtype, process, link, clock, seconds, speed, payload_value, seconds/60)
                    clock += seconds
                previous = dict(vehicle=row['vehicle'], type=vtype, arrival=arr, link=links[-1])
                stats['road_trips'] += 1
                stats['min_departure_seconds'] = dep if stats['min_departure_seconds'] is None else min(dep, stats['min_departure_seconds'])
                stats['max_arrival_seconds'] = arr if stats['max_arrival_seconds'] is None else max(arr, stats['max_arrival_seconds'])
                if len(totals) >= 20000:
                    flush()
        flush()
        if writer is None:
            raise ValueError('No in-domain road emissions rows generated')
    finally:
        if writer is not None:
            writer.close()
        con.close()
    partial.rename(target)
    return stats
