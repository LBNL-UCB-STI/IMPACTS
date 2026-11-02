"""Read-only audit of the example's existing BEAM emissions skims."""
from pathlib import Path
import json
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKIMS = ROOT/'examples/pilates/beam/beam_output/ITERS/it.0/0.skimsEmissions.parquet'
POLLUTANTS = ['CH4','CO','CO2','HC','NH3','NOx','PM','PM10','PM25','ROG','SOx','TOG','BC']


def main():
    db = duckdb.connect()
    db.execute('SET threads=4')
    db.execute("SET memory_limit='6GB'")
    source = "read_parquet('" + SKIMS.as_posix() + "')"
    basic = db.execute(f'''SELECT count(*) n_rows, count(DISTINCT linkId) links,
        count(DISTINCT vehicleTypeId) vehicle_types, min(hour) hour_min,max(hour) hour_max,
        min(iterations) iterations_min,max(iterations) iterations_max,
        min(observations) observations_min,max(observations) observations_max,sum(observations) observations_sum,
        count(*) FILTER (WHERE observations IS NULL OR observations<=0) invalid_observations,
        count(*) FILTER (WHERE nullif(trim(emissions),'') IS NULL) blank_emissions,
        count(*) FILTER (WHERE travelTimeInSecond<0 OR parkingDurationInSecond<0) negative_times
        FROM {source}''').df().iloc[0].to_dict()
    print(json.dumps({'basic': {k:int(v) for k,v in basic.items()}}), flush=True)
    ids = db.execute(f'SELECT DISTINCT vehicleTypeId FROM {source}').df()
    meta = pd.concat([pd.read_csv(p,low_memory=False) for p in
        (ROOT/'examples/pilates/beam/production/sfbay/vehicle-tech').glob('vehicleTypes*.csv')],ignore_index=True)
    match = ids.merge(meta[['vehicleTypeId','emfacId','emfacVehicleCategory','emfacFuel','primaryFuelType']],
                      on='vehicleTypeId',how='left',validate='one_to_one')
    print(json.dumps({'metadata_match':dict(types=len(ids),missing_emfac=int(match.emfacId.isna().sum()),
         missing_types=match.loc[match.emfacId.isna(),'vehicleTypeId'].tolist()),
         'vehicle_type_prefix_counts':ids.vehicleTypeId.str.split('-').str[0].value_counts().to_dict()}),flush=True)
    print('AGGREGATING OBSERVATION-WEIGHTED EMISSIONS',flush=True)
    parsed = ','.join(f"try_cast(regexp_extract(emissions,'(^|;){p}:([^;]+)',2) AS DOUBLE) AS {p}" for p in POLLUTANTS)
    values = ','.join(f'sum({p}*observations)/1000 AS {p}_kg, sum({p})/1000 AS {p}_unweighted_kg, '
                     f'count(*) FILTER (WHERE {p} IS NULL OR NOT isfinite({p})) AS {p}_invalid, '
                     f'count(*) FILTER (WHERE {p}<0) AS {p}_negative' for p in POLLUTANTS)
    g = db.execute(f'''WITH parsed AS (SELECT hour,split_part(vehicleTypeId,'-',1) AS vehicle_group,
        process,observations,{parsed} FROM {source})
        SELECT hour,vehicle_group,process,count(*) n_rows,sum(observations) observations,{values}
        FROM parsed GROUP BY hour,vehicle_group,process''').df()
    pollutants = [p+'_kg' for p in POLLUTANTS]
    print(json.dumps({'total_kg':g[pollutants].sum().to_dict(),
        'unweighted_sum_kg_do_not_use':g[[p+'_unweighted_kg' for p in POLLUTANTS]].sum().to_dict(),
        'invalid_counts':g[[p+'_invalid' for p in POLLUTANTS]].sum().astype(int).to_dict(),
        'negative_counts':g[[p+'_negative' for p in POLLUTANTS]].sum().astype(int).to_dict()}),flush=True)
    for col in ['vehicle_group','process']:
        print(json.dumps({col:g.groupby(col)[['n_rows','observations']+pollutants].sum().reset_index().to_dict('records')}),flush=True)
    hours = g.groupby('hour')[pollutants].sum()
    print(json.dumps({'hourly_peak':{p:dict(hour=int(hours[p+'_kg'].idxmax()),kg=float(hours[p+'_kg'].max())) for p in POLLUTANTS}}),flush=True)
    duplicate = db.execute(f'''SELECT count(*) duplicated_keys,coalesce(sum(n-1),0) extra_rows
        FROM (SELECT count(*) n FROM {source} GROUP BY hour,linkId,vehicleTypeId,process HAVING count(*)>1)''').df()
    print(json.dumps({'duplicates':duplicate.to_dict('records')}),flush=True)
    net = pd.read_csv(ROOT/'examples/pilates/beam/beam_output/network.csv.gz',usecols=['linkId'])
    db.register('network_ids',net)
    print(json.dumps({'unmatched_links':db.execute(f'''SELECT count(DISTINCT linkId),count(*) FROM {source}
        WHERE linkId NOT IN (SELECT linkId FROM network_ids)''').fetchone()}),flush=True)


if __name__ == '__main__':
    main()
