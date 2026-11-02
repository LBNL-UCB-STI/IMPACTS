"""Stage the October future-scenario ZIP inputs and audit, without computing AQ.

Archives and previous outputs are never modified. Extracted input files retain
their original names. Unrelated Boston/PSRC data and macOS metadata are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

import duckdb
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "chargingcode"
DEFAULT_OUTPUT = ROOT / "examples/pilates/impacts/impacts_output/events-sfbay-future-scenarios-20261011"
SCENARIOS = ("BF-06EP", "BF-1EP", "HF-1EP")
ROAD_MODES = "('car','car_hov2','car_hov3','ride_hail','ride_hail_pooled','bus')"


def extract_member(archive, info, destination):
    relative = PurePosixPath(info.filename)
    if relative.is_absolute() or '..' in relative.parts or ':' in info.filename:
        raise ValueError(f"Unsafe ZIP member: {info.filename}")
    target = destination.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"ZIP member outside destination: {info.filename}")
    if target.exists():
        if target.stat().st_size != info.file_size:
            raise ValueError(f"Existing input size mismatch: {target}")
        # Validate CRC before reusing an existing extracted input.
        import zlib
        crc = 0
        with target.open('rb') as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                crc = zlib.crc32(chunk, crc)
        if crc != info.CRC:
            raise ValueError(f"Existing input CRC mismatch: {target}")
    else:
        archive.extract(info, destination)
    return {"member": info.filename, "bytes": info.file_size, "crc32": info.CRC,
            "extracted_path": str(target)}


def stage(output):
    destination = output / "_inputs"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for prefix in (*SCENARIOS, 'local_run'):
        matches = sorted(SOURCE.glob(prefix + '-*.zip'))
        if len(matches) != 1:
            raise ValueError(f"Expected one ZIP for {prefix}, found {matches}")
        path = matches[0]
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = info.filename
                if info.is_dir() or '/.' in name or name.startswith('__MACOSX'):
                    continue
                if prefix == 'local_run':
                    keep = ('/beam-ft/20260925/' in name and 'SFBAY-2050-' in name)
                    keep |= '/network/sfbay-' in name
                    if not keep:
                        continue
                print('Staging', name, flush=True)
                record = extract_member(archive, info, destination)
                records.append({"archive": str(path), **record})
    (output / 'staged_inputs.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


def audit(output):
    network_path = SOURCE / 'network.csv.gz'
    baseline_path = ROOT / 'examples/pilates/beam/beam_output/network.csv.gz'
    network, baseline = pd.read_csv(network_path), pd.read_csv(baseline_path)
    network_audit = {"new_network": str(network_path), "baseline_network": str(baseline_path),
                     "new_links": len(network), "baseline_links": len(baseline)}
    try:
        fields = [c for c in baseline if c != 'numberOfLanes']
        pd.testing.assert_frame_equal(baseline[fields], network[fields])
        network_audit['baseline_intersection_reuse_validated'] = True
    except (AssertionError, KeyError) as error:
        network_audit.update(baseline_intersection_reuse_validated=False, comparison_error=str(error))
    classification = pd.read_csv(ROOT / 'scenarios/all_scenarios_vehicle_classification.csv')
    baseline_freight = pd.read_csv(
        ROOT / 'examples/pilates/beam/production/sfbay/vehicle-tech/vehicleTypes--frism--2018-Baseline--EM.csv',
        low_memory=False)
    mapped_freight_types = set(baseline_freight.sourceVehicleTypeId)
    reports = []
    for scenario in SCENARIOS:
        path = output / '_inputs' / scenario / '4.events.parquet'
        print('Auditing', scenario, flush=True)
        con = duckdb.connect()
        con.execute("SET threads=4")
        con.execute("SET memory_limit='12GB'")
        con.read_parquet(str(path)).create_view('events')
        types = con.execute(f"""SELECT vehicleType,mode,count(*) AS path_traversals,
            count(DISTINCT vehicle) AS vehicles,min(departureTime) AS first_departure,
            max(arrivalTime) AS last_arrival FROM events
            WHERE type='PathTraversal' GROUP BY vehicleType,mode ORDER BY mode,vehicleType""").fetchdf()
        vehicles = con.execute(f"""SELECT DISTINCT vehicle,vehicleType FROM events
            WHERE type='PathTraversal' AND mode IN {ROAD_MODES}""").fetchdf()
        labels = classification.loc[classification.scenario.eq('SFBAY-2050-' + scenario)]
        freight = vehicles.loc[vehicles.vehicleType.str.startswith('ft-', na=False)].copy()
        linked = freight.merge(labels, left_on='vehicle', right_on='vehicleId', how='left',
                               suffixes=('_event', '_label'), validate='one_to_one', indicator=True)
        folder = output / scenario / 'input_audit'
        folder.mkdir(parents=True, exist_ok=True)
        types.to_csv(folder / 'event_vehicle_types.csv', index=False)
        linked.to_csv(folder / 'freight_classification_match.csv', index=False)
        metadata_path = (output / '_inputs/local_run/beam-ft/20260925/vehicle-tech' /
                         f'vehicletypes--frism--SFBAY-2050-{scenario}.csv')
        metadata = pd.read_csv(metadata_path)
        missing_types = sorted(set(freight.vehicleType) - mapped_freight_types)
        metadata.loc[metadata.vehicleTypeId.isin(missing_types)].to_csv(
            folder / 'freight_types_without_baseline_emfac_mapping.csv', index=False)
        missing_network = con.execute(f"""SELECT count(DISTINCT link) FROM (
            SELECT unnest(string_split(links,',')) AS link FROM events
            WHERE type='PathTraversal' AND mode IN {ROAD_MODES}
        ) WHERE try_cast(link AS BIGINT) IS NOT NULL AND try_cast(link AS BIGINT)
            NOT IN (SELECT linkId FROM read_csv_auto(?))""", [str(network_path)]).fetchone()[0]
        report = dict(scenario=scenario, events=str(path), total_event_rows=pq.ParquetFile(path).metadata.num_rows,
                      road_vehicles=len(vehicles), freight_vehicles=len(freight),
                      freight_matching_classification=int(linked['_merge'].eq('both').sum()),
                      freight_type_disagreements=int((linked['_merge'].eq('both') &
                          linked.vehicleType.ne(linked.vehicleTypeId)).sum()),
                      missing_network_links=missing_network,
                      freight_types_without_baseline_mapping=missing_types,
                      freight_vehicles_without_baseline_mapping=int(
                          freight.vehicleType.isin(missing_types).sum()),
                      freight_types=freight.vehicleType.value_counts().to_dict(),
                      classification_categories=linked.port_category.value_counts(dropna=False).to_dict())
        (folder / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
        con.close()
    result = dict(status='inputs_audited_not_model_results', network=network_audit, scenarios=reports,
                  pending=['Confirm EMFAC calendar year and fleet mapping', 'Confirm sample expansion',
                           '2050 inventory exports located; 2050 project-analysis and BC rates not located',
                           'Check new source-grid coverage before dispersion'])
    (output / 'input_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(network_audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.audit_only:
        stage(args.output)
    audit(args.output)
