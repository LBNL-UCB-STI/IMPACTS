"""Quarantine the ten explicitly approved HF-1EP invalid derived trip rows.

Never modifies the original scenario events or silently accepts a new count.
The unfiltered derived input is retained alongside the filtered input.
"""
from pathlib import Path
import hashlib
import json

import duckdb


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'examples/pilates/impacts/impacts_output/events-sfbay-future-scenarios-20261011/HF-1EP/_work/inputs/derived_events'
INVALID = """departureTime<0 OR arrivalTime<departureTime
    OR parsed_weight IS NULL OR parsed_weight<=0
    OR len(parsed_links)!=len(parsed_times)
    OR list_contains(list_transform(parsed_times,x->x<0),true)"""


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    source = FOLDER / '3.events.parquet'
    backup = FOLDER / '3.events.before_invalid_trip_exclusion.parquet'
    filtered = FOLDER / '3.events.filtered.partial.parquet'
    excluded = FOLDER / 'excluded_invalid_trips.parquet'
    audit_path = FOLDER / 'invalid_trip_exclusion_audit.json'
    for path in (backup, filtered, excluded, audit_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite prior audit/product: {path}')
    original_hash = sha256(source)
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        con.execute('SET preserve_insertion_order=true')
        con.read_parquet(str(source)).create_view('source')
        con.execute("""CREATE VIEW checked AS SELECT *,
            try_cast(weight AS DOUBLE) parsed_weight,
            list_transform(string_split(links,','),x->try_cast(x AS BIGINT)) parsed_links,
            list_transform(string_split(linkTravelTime,','),x->try_cast(x AS DOUBLE)) parsed_times
            FROM source""")
        total = con.execute('SELECT count(*) FROM checked').fetchone()[0]
        count = con.execute(f'SELECT count(*) FROM checked WHERE {INVALID}').fetchone()[0]
        negative = con.execute('SELECT count(*) FROM checked WHERE parsed_weight<0').fetchone()[0]
        if count != 10 or negative != 10:
            raise ValueError(f'Approved scope changed: invalid={count}, negative-weight={negative}')
        detail = con.execute(f"""SELECT vehicle,vehicleType,count(*) excluded_trips,
            min(parsed_weight) minimum_weight_kg,max(parsed_weight) maximum_weight_kg
            FROM checked WHERE {INVALID} GROUP BY ALL ORDER BY vehicle""").fetchdf().to_dict('records')
        con.execute(f"""COPY (SELECT * EXCLUDE(parsed_links,parsed_times),
            'negative_vehicle_weight' AS exclusion_reason FROM checked WHERE {INVALID})
            TO '{excluded.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        con.execute(f"""COPY (SELECT * EXCLUDE(parsed_weight,parsed_links,parsed_times)
            FROM checked WHERE NOT coalesce(({INVALID}),false))
            TO '{filtered.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        actual = con.read_parquet(str(filtered)).count('*').fetchone()[0]
        if actual != total - 10:
            raise AssertionError((total, actual))
    # Retain an exact recoverable copy; neither raw events nor existing AQ outputs change.
    source.rename(backup)
    try:
        filtered.rename(source)
    except Exception:
        backup.rename(source)
        raise
    audit = dict(status='approved_rows_excluded',scenario='HF-1EP',
        approval='User explicitly approved excluding these ten invalid trip records',
        input_rows=total,excluded_rows=10,retained_rows=actual,
        vehicles=detail,backup=str(backup),excluded_records=str(excluded),
        filtered_input=str(source),original_sha256=original_hash,
        filtered_sha256=sha256(source),original_scenario_events_modified=False,
        concentration_rerun=False,
        note='Only invalid trip rows removed; other trips of these vehicles retained. '
             'Exclusion does not validate upstream weight data or reconstruct missing activity.')
    audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
