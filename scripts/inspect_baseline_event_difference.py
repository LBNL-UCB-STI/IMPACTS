"""Read-only comparison of new CSV events and previous Parquet events."""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'examples/pilates/beam/beam_output/ITERS/it.0/0.events.csv.gz'
OLD = ROOT / 'chargingcode/beam-fc07-v260419-it3/3.events.parquet'


def summarize(path):
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='6GB'")
    source = '''read_csv(?, header=true, all_varchar=true, delim=',', quote='"', escape='"')''' if path.name.endswith('.gz') else 'read_parquet(?)'
    table = db.execute(f"""
        SELECT type,
          CASE WHEN type='PathTraversal' THEN vehicle END vehicle,
          CASE WHEN type='PathTraversal' THEN vehicleType END vehicle_type,
          CASE WHEN type='PathTraversal' THEN mode END travel_mode,
          CASE WHEN type='PathTraversal' THEN primaryFuelType END fuel_type,
          CASE WHEN type='PathTraversal' THEN secondaryFuelType END secondary_fuel_type,
          count(*) n, min(try_cast(time AS DOUBLE)) min_time, max(try_cast(time AS DOUBLE)) max_time,
          sum(CASE WHEN type='PathTraversal' THEN try_cast(length AS DOUBLE) ELSE 0 END) distance_m,
          sum(CASE WHEN type='PathTraversal' AND try_cast(departureTime AS DOUBLE)<86400
               THEN try_cast(length AS DOUBLE) ELSE 0 END) distance_first24_m,
          sum(CASE WHEN type='PathTraversal' THEN try_cast(arrivalTime AS DOUBLE)-try_cast(departureTime AS DOUBLE) ELSE 0 END) travel_s,
          min(CASE WHEN type='PathTraversal' THEN try_cast(departureTime AS DOUBLE) END) first_departure,
          max(CASE WHEN type='PathTraversal' THEN try_cast(arrivalTime AS DOUBLE) END) last_arrival,
          sum(CASE WHEN type='PathTraversal' AND try_cast(length AS DOUBLE)<0 THEN 1 ELSE 0 END) negative_length,
          sum(CASE WHEN type='PathTraversal' AND try_cast(arrivalTime AS DOUBLE)<try_cast(departureTime AS DOUBLE) THEN 1 ELSE 0 END) reversed_time,
          sum(CASE WHEN type='PathTraversal' AND try_cast(departureTime AS DOUBLE)>=86400 THEN 1 ELSE 0 END) departures_after24,
          sum(CASE WHEN type='PathTraversal' AND nullif(cast(emissions AS VARCHAR),'') IS NOT NULL THEN 1 ELSE 0 END) emissions_rows,
          min(CASE WHEN type='PathTraversal' THEN try_cast(startX AS DOUBLE) END) min_x,
          max(CASE WHEN type='PathTraversal' THEN try_cast(startX AS DOUBLE) END) max_x,
          min(CASE WHEN type='PathTraversal' THEN try_cast(startY AS DOUBLE) END) min_y,
          max(CASE WHEN type='PathTraversal' THEN try_cast(startY AS DOUBLE) END) max_y
        FROM {source} GROUP BY ALL
    """, [str(path)]).df()
    db.close()
    paths = table.loc[table.type.eq('PathTraversal')].copy()
    def group_stats(d):
        n = int(d.vehicle.nunique())
        return dict(vehicles=n, traversals=int(d.n.sum()), distance_km=float(d.distance_m.sum()/1000),
                    first24_km_per_vehicle=float(d.distance_first24_m.sum()/1000/n) if n else None,
                    first24_miles_per_vehicle=float(d.distance_first24_m.sum()/1609.344/n) if n else None)
    vt = paths.vehicle_type.fillna('')
    passenger = paths.travel_mode.isin(['car','car_hov2','car_hov3','ride_hail','ride_hail_pooled'])
    cohorts = {'all_pathtraversal': paths, 'passenger_modes_including_freight': paths.loc[passenger],
               'passenger_modes_excluding_freight': paths.loc[passenger & ~vt.str.startswith('ft-')],
               'freight_ft_prefix': paths.loc[vt.str.startswith('ft-')],
               'BEV_fuel_passenger_modes': paths.loc[passenger & paths.fuel_type.fillna('').str.lower().eq('electricity') & paths.secondary_fuel_type.fillna('None').str.lower().eq('none')],
               'PHEV_fuel_passenger_modes': paths.loc[passenger & paths.fuel_type.fillna('').str.lower().eq('electricity') & paths.secondary_fuel_type.fillna('').str.lower().eq('gasoline')],
               'electric_freight': paths.loc[vt.str.startswith('ft-') & paths.fuel_type.fillna('').str.lower().eq('electricity')],
               'BEV_name_passenger_modes': paths.loc[vt.str.startswith('ev-') & passenger],
               'PHEV_name_passenger_modes': paths.loc[vt.str.startswith('phev-') & passenger],
               'electric_primary_fuel': paths.loc[paths.fuel_type.fillna('').str.lower().eq('electricity')]}
    result = dict(path=str(path), total_rows=int(table.n.sum()), min_time=float(table.min_time.min()),
                  max_time=float(table.max_time.max()), event_types={str(k):int(v) for k,v in table.groupby('type').n.sum().items()},
                  cohorts={k:group_stats(v) for k,v in cohorts.items()},
                  vehicle_type_count=int(paths.vehicle_type.nunique()),
                  freight_type_count=int(paths.loc[vt.str.startswith('ft-')].vehicle_type.nunique()),
                  passenger_type_count=int(paths.loc[passenger & ~vt.str.startswith('ft-')].vehicle_type.nunique()),
                  modes={str(k):group_stats(v) for k,v in paths.groupby('travel_mode')},
                  fuel_types={str(k):group_stats(v) for k,v in paths.groupby(['fuel_type','secondary_fuel_type'],dropna=False)},
                  checks={k:int(paths[k].sum()) for k in ['negative_length','reversed_time','departures_after24','emissions_rows']},
                  start_coordinate_bounds=[float(paths.min_x.min()),float(paths.min_y.min()),float(paths.max_x.max()),float(paths.max_y.max())])
    metadata = pd.concat([pd.read_csv(p, usecols=['vehicleTypeId']) for p in
        (ROOT/'examples/pilates/beam/production/sfbay/vehicle-tech').glob('vehicleTypes*.csv')])
    mapped = paths.loc[paths.vehicle_type.isin(metadata.vehicleTypeId)]
    result['local_vehicle_metadata_match'] = dict(types=int(mapped.vehicle_type.nunique()),vehicles=int(mapped.vehicle.nunique()))
    result['charging_event_times'] = table.loc[table.type.str.contains('Charging|Refuel', regex=True),['type','n','min_time','max_time']].to_dict('records')
    print(json.dumps(result, allow_nan=False), flush=True)
    return paths


if __name__ == '__main__':
    print('SCANNING NEW CSV', flush=True)
    new = summarize(NEW)
    print('SCANNING OLD PARQUET', flush=True)
    old = summarize(OLD)
    nids, oids = set(new.vehicle.dropna()), set(old.vehicle.dropna())
    print(json.dumps(dict(shared_vehicle_ids=len(nids & oids), new_only_vehicle_ids=len(nids-oids), old_only_vehicle_ids=len(oids-nids),
        new_vehicle_type_count=len(set(new.vehicle_type.dropna())-set(old.vehicle_type.dropna())),
        removed_vehicle_type_count=len(set(old.vehicle_type.dropna())-set(new.vehicle_type.dropna())))), flush=True)
