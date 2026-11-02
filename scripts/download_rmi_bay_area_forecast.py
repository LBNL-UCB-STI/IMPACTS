"""Archive the public RMI map's daily JSON responses, without changing IMPACTS inputs.

Run with system Python (requests required). Existing payloads are validated/reused.
No authentication, private endpoints, or original vehicle telematics are accessed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'OUTPUT/external-data/rmi-lbnl-bay-area-forecast-20261026'
API = 'https://doe-bay-area-map-server.onrender.com/api'
MAP = 'https://bay-area-forecast.rmi.org/'
REPORT = 'https://rmi.org/download/33072/?tmstv=1761598024'
SCENARIOS = {'aate3': 'Scenario 1: CEC AATE3', '100_zev': 'Scenario 2: 100% freight electrification by 2050'}
DUTIES = {'medium': 'Medium return-to-depot', 'heavy': 'Heavy return-to-depot',
          'drayage': 'Port drayage', 'long_haul': 'Long haul', 'all': 'All duty cycles (server aggregate)'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_daily(raw, year, scenario, duty):
    data = json.loads(raw)
    for key, expected in [('year', year), ('scenario', scenario), ('duty', duty)]:
        if data.get(key) != expected:
            raise ValueError(f'Mismatched {key}: {data.get(key)!r} != {expected!r}')
    for field in ('locations', 'feeders', 'minmaxLocations', 'minmaxFeeders'):
        rows = data[field]
        if len(rows) != 24 or not rows or not rows[0]:
            raise ValueError(f'{field}: not 24 populated hours')
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f'{field}: inconsistent dimensions')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in rows for v in row):
            raise ValueError(f'{field}: nonfinite/non-numeric values')
    if any(v < 0 for row in data['locations'] for v in row):
        raise ValueError('Negative location charging load')
    return data


def fetch(url):
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(15, 75), headers={'User-Agent': 'IMPACTS-public-data-archive/1.0'})
            if response.status_code == 429 or response.status_code >= 500:
                delay = min(float(response.headers.get('Retry-After', 2 ** (attempt + 1))), 30)
                response.raise_for_status()
            else:
                delay = 2 ** (attempt + 1)
                response.raise_for_status()
            return response.content, response.url, response.headers.get('Content-Type', '')
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(locals().get('delay', 2 ** (attempt + 1)))


def write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never overwrite an existing downloaded artifact.
    with path.open('xb') as stream:
        stream.write(raw)


def acquire(out, relative, url, validator=None):
    path = out / relative
    existed = path.exists()
    if existed:
        raw, final_url, mime = path.read_bytes(), None, None
    else:
        raw, final_url, mime = fetch(url)
    parsed = validator(raw) if validator else None
    if not existed:
        write_new(path, raw)
    entry = dict(path=relative, source_url=url, resolved_url=final_url, content_type=mime,
                 bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), checked_utc=utc_now(), reused=existed)
    return entry, parsed, raw


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    entries, errors, summaries = [], [], []
    for relative, url in [('reference/report.pdf', REPORT), ('reference/dashboard.html', MAP)]:
        entry, _, raw = acquire(out, relative, url)
        if relative.endswith('.pdf') and not raw.startswith(b'%PDF'):
            raise ValueError('Report response is not a PDF')
        entries.append(entry)
        if relative.endswith('.html'):
            for src in re.findall(r'<script[^>]+src="([^"]+)"', raw.decode('utf-8')):
                if src.startswith('/assets/'):
                    e, _, _ = acquire(out, 'reference/' + src.rsplit('/', 1)[-1], MAP.rstrip('/') + src)
                    entries.append(e)

    tasks = [(year, scenario, duty) for scenario in SCENARIOS for year in range(2025, 2051) for duty in DUTIES]

    def download(task):
        year, scenario, duty = task
        url = f'{API}/daily/{year}/{scenario}/{duty}'
        relative = f'raw/{scenario}/{year}/{duty}.json'
        e, data, _ = acquire(out, relative, url, lambda raw: validate_daily(raw, year, scenario, duty))
        e.update(year=year, scenario=scenario, duty=duty, hours=24,
                 location_count=len(data['locations'][0]), feeder_count=len(data['feeders'][0]))
        summary = dict(year=year, scenario=scenario, duty=duty,
                       full_dashboard_footprint_hourly_kW=[math.fsum(row) for row in data['locations']])
        return e, summary

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(download, task): task for task in tasks}
        for number, future in enumerate(as_completed(pending), 1):
            try:
                entry, summary = future.result()
                entries.append(entry)
                summaries.append(summary)
                print(f'[{number}/{len(tasks)}] OK {entry["path"]} ({entry["bytes"]:,} bytes)', flush=True)
            except Exception as exc:
                errors.append(dict(task=pending[future],error=str(exc)))
                print(f'[{number}/{len(tasks)}] ERROR {pending[future]}: {exc}', flush=True)

    comparisons = []
    lookup = {(s['scenario'],s['year'],s['duty']):s['full_dashboard_footprint_hourly_kW'] for s in summaries}
    for scenario in SCENARIOS:
        for year in range(2025,2051):
            if all((scenario,year,duty) in lookup for duty in DUTIES):
                diff = [lookup[scenario,year,'all'][hour]-sum(lookup[scenario,year,duty][hour]
                        for duty in DUTIES if duty != 'all') for hour in range(24)]
                comparisons.append(dict(scenario=scenario,year=year,max_absolute_all_vs_components_kW=max(map(abs,diff))))

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    manifest = dict(retrieved_utc=utc_now(),expected_daily_payloads=len(tasks),
                    completed_daily_payloads=len(summaries),status='complete' if not errors else 'incomplete',
                    scenarios=SCENARIOS,duties=DUTIES,years=[2025,2050],entries=entries,errors=errors,
                    all_vs_components_checks=comparisons)
    write_new(out/f'manifest_{stamp}.json',json.dumps(manifest,indent=2).encode())
    write_new(out/f'region_hourly_totals_{stamp}.json',json.dumps(sorted(summaries,key=lambda x:(x['scenario'],x['year'],x['duty'])),indent=2).encode())
    notes = '''RMI-LBNL BAY AREA TRUCK CHARGING FORECAST - PUBLIC MAP SNAPSHOT

Scope: Scenario 1 (aate3) and Scenario 2 (100_zev); years 2025-2050 inclusive;
four truck duty cycles and server-provided All. Each raw JSON contains 24 hours.
No passenger vehicle profiles, original Geotab records, or private data are included.

raw/<scenario>/<year>/<duty>.json: exact public daily API response bytes.
locations[hour][idx]: location charging power in kW as displayed by the map.
The front end joins array index idx to the map geometry's idx attribute.
Location geometries/identifiers are NOT included in this archive: the API returns
indexed arrays, not a self-contained geospatial dataset. County or nine-county
Bay Area filtering requires the matching geometry/index crosswalk.
feeders: preserved exactly as supplied; do not add them to location loads.
minmax*: display ranges, not additional loads. Negative feeder values are preserved.

region_hourly_totals_*.json: derived sum over locations for each hour in the FULL
dashboard footprint. This is NOT necessarily the nine-county SFBAY boundary.
All is a separate aggregate, not a fifth vehicle category. Never add All to its
four components. Manifest reports the aggregate-vs-components discrepancy;
small differences may result from source rounding and are not silently corrected.

The 24 hours describe a modeled daily profile, NOT an annual 8760-hour timeseries.
Vehicle counts are not returned: per-1000-truck normalization cannot be inferred.
These are scenario predictions, NOT measured truck charging validation data.
Years available in the interface do not establish independent simulation for
every year. Report states long-haul Scenario 2 scales Scenario 1 by fleet size.
Timezone/DST and within-hour averaging convention require author confirmation.

reference/report.pdf: original report; dashboard.html and application JS preserve
the public endpoint and selector definitions at download time. These web assets
are for provenance, not a standalone offline map. No website was modified.
manifest_*.json: source URLs, timestamps, byte counts, SHA256, validation/errors.
Existing downloads are reused/validated, never overwritten by this script.

Sources:
https://bay-area-forecast.rmi.org/
https://rmi.org/resources/bay-area-medium-and-heavy-vehicle-electrification-roadmap/
https://rmi.org/download/33072/?tmstv=1761598024
Public frontend API (not a guaranteed stable documented API):
https://doe-bay-area-map-server.onrender.com/api/daily/{year}/{scenario}/{duty}
Contact shown on dashboard: Nocona Sanders, nocona.sanders@rmi.org.
Retain attribution; confirm data reuse terms with authors before redistribution.

No existing IMPACTS settings, inputs, charging outputs, or concentrations changed.
'''
    if not (out/'README.txt').exists():
        write_new(out/'README.txt',notes.encode())
    print(json.dumps(dict(output=str(out),status=manifest['status'],daily_files=len(summaries),
                          total_bytes=sum(e['bytes'] for e in entries),errors=errors),indent=2),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT)
    main(parser.parse_args().output)
