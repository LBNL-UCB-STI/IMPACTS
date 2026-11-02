"""Compute designed Home80 LDV power, then add each collaborator freight case.

Power-only additive scenarios; no traffic and no rescaling of existing 10x loads.
"""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
from pathlib import Path
import hashlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

import continue_beam_charging_pipeline as power
from impacts.pipeline.workflow.power_generation import plot_power_generation_concentration
from impacts.pipeline.postprocess.step4_plot_concentrations import _SF_BAY_COUNTIES

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'examples/pilates/impacts/impacts_output'
OUT = OUTPUTS / 'ldv-home80-freight3-power-combined10x-20261029'
LDV = OUTPUTS / 'events-fc07-existing-ev-charging-code10x-20260928-v2/bev-phev-home80-evi20c-20261012'
FREIGHT = OUTPUTS / 'baseline0-ldv-home80-freight3-10x-20261026/freight_power'
ISRM = ISRM_PATH
SCENARIOS = ['BF-06EP', 'BF-1EP', 'HF-1EP']


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def signature(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_components(data):
    cols = [c for c in data if c.startswith('power_')]
    assert data.inmap_cell_id.is_unique
    assert np.isfinite(data[cols].to_numpy()).all()
    assert (data[cols].to_numpy() >= 0).all()
    np.testing.assert_allclose(data.power_TotalPM25, data.power_PrimaryPM25 + data.power_SecondaryPM25)
    np.testing.assert_allclose(data.power_SecondaryPM25, data[['power_SOA', 'power_pNO3', 'power_pNH4', 'power_pSO4']].sum(axis=1))
    return cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'run_status.json', dict(status='running', traffic_included=False))
    source = LDV / 'comparison_hourly.csv'
    d = pd.read_csv(source).sort_values('hour')
    np.testing.assert_array_equal(d.hour, np.arange(24))
    load = d.Both_Home80_MW.to_numpy()
    np.testing.assert_allclose(load, d.BEV_Home80_MW + d.PHEV_Home80_MW)
    metrics = json.loads((LDV / 'run_summary.json').read_text())
    ref = next(r for r in metrics['seed42_metrics'] if r['scenario'] == 'Both_Home80')
    np.testing.assert_allclose(load.sum(), ref['daily_MWh'])
    assert metrics['scale'] == 10
    target = OUT / 'LDV'
    demand = target / 'charging_demand'
    demand.mkdir(parents=True, exist_ok=True)
    d[['hour', 'BEV_Home80_MW', 'PHEV_Home80_MW', 'Both_Home80_MW']].to_csv(demand / 'ldv_daily_curve_10x.csv', index=False)
    pd.DataFrame(dict(hour_of_year=np.arange(8760), charging_demand_MWh=np.tile(load, 365))).to_parquet(demand / 'sfbay_hourly_charging_demand.parquet', index=False)
    qa = dict(source=str(source), source_sha256=signature(source), fleet_scale_factor=10,
        additional_scale_factor=1, seed=42, annualization_days=365,
        curve_scope='Cristian 3.events existing LDV BEVs and PHEVs; designed Home80/Home80 steady day; already 10x',
        daily_energy_MWh=float(load.sum()), annual_energy_MWh=float(load.sum()*365),
        peak_MW=float(load.max()), peak_hour=int(load.argmax()),
        battery_kWh_per_km=metrics['battery_kWh_per_km'], charging_efficiency=metrics['charging_efficiency'],
        input_BEVs=607, input_PHEVs=937, excluded_BEVs=12, missing_PHEV_itinerary=1,
        energy_boundary='Grid-side charger load already includes 90% charger efficiency; Cambium distribution losses applied separately',
        evi_used_for_dispatch=False, traffic_included=False, baseline0_events_used=False)
    save(demand / 'beam_charging_summary.json', qa)
    power.DEMAND = demand
    folder = target / 'power_generation'
    if not (folder / 'power_generation_summary.json').exists():
        print('LDV HOURLY DISPATCH', flush=True)
        power.dispatch(target)
    summary = json.loads((folder / 'power_generation_summary.json').read_text())
    if not summary.get('power_isrm_completed'):
        print('LDV LAYER0 ISRM', flush=True)
        power.power_isrm(target, ISRM, source_layer_index=0)
    summary = json.loads((folder / 'power_generation_summary.json').read_text())
    summary.update(traffic_included=False, traffic_baseline=None, contribution='Designed existing LDV charging power only',
        charging_events_source=str(ROOT / 'chargingcode/beam-fc07-v260419-it3/3.events.parquet'))
    save(folder / 'power_generation_summary.json', summary)
    ldv = gpd.read_parquet(folder / 'power_generation_isrm_concentrations.parquet').sort_values('inmap_cell_id').reset_index(drop=True)
    cols = validate_components(ldv)
    ldv_plants = pd.read_csv(folder / 'plant_emissions_used_for_concentration.csv')
    settings = yaml.safe_load((power.BASE / 'settings.yaml').read_text(encoding='utf-8'))
    region = gpd.read_parquet(settings['impacts']['analysis']['human_exposure']['ces_tracts_file'])
    county_col = next(c for c in ['County', 'NAME', 'NAMELSAD'] if c in region)
    region = region.loc[region[county_col].astype(str).str.replace(r'\s+County$', '', regex=True).isin(_SF_BAY_COUNTIES)]
    mask = region.to_crs(ldv.crs).geometry.union_all()
    plot_power_generation_concentration(gpd.clip(ldv, mask, keep_geom_type=True), ldv_plants,
        folder / 'power_generation_isrm_total_pm25_sfbay_outline.png', source_layer_index=0,
        method_note='LDV Home80/Home80; 10x; SF Bay land-area mask')
    rows = []
    for scenario in SCENARIOS:
        original = FREIGHT / scenario / 'power_generation'
        fpath = original / 'power_generation_isrm_concentrations.parquet'
        before = signature(fpath)
        fq = json.loads((original / 'power_generation_summary.json').read_text())
        for k in ['method', 'year', 'power_source_layer_index', 'power_isrm_source', 'power_local_local_removal_applied', 'concentration_source_scope']:
            assert fq[k] == summary[k], (scenario, k)
        freight = gpd.read_parquet(fpath).sort_values('inmap_cell_id').reset_index(drop=True)
        assert validate_components(freight) == cols
        assert ldv.crs == freight.crs
        np.testing.assert_array_equal(ldv.inmap_cell_id, freight.inmap_cell_id)
        assert ldv.geometry.to_wkb().equals(freight.geometry.to_wkb())
        combined = ldv[['inmap_cell_id', 'geometry']].copy()
        for c in cols:
            combined['ldv_' + c] = ldv[c]
            combined['freight_' + c] = freight[c]
            combined[c] = ldv[c] + freight[c]
            np.testing.assert_allclose(combined[c] - combined['freight_' + c], combined['ldv_' + c], atol=1e-15)
        validate_components(combined)
        dest = OUT / 'combined' / scenario
        dest.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(dest / 'ldv_freight_power_concentrations.parquet', index=False, compression='zstd')
        combined.to_file(dest / 'ldv_freight_power_concentrations.gpkg', driver='GPKG')
        # Sum only extensive plant quantities; do not add coordinates or rates.
        fp = pd.read_csv(original / 'plant_emissions_used_for_concentration.csv')
        plant_rows = pd.concat([ldv_plants.assign(contribution='LDV'), fp.assign(contribution='Freight')])
        plant_rows.to_csv(dest / 'plant_emissions_by_contribution.csv', index=False)
        quantities = ['added_generation_MWh'] + [f'{p}_kg' for p in power.POLLUTANTS]
        plants = plant_rows.groupby('ORISPL')[quantities].sum().join(plant_rows.drop_duplicates('ORISPL').set_index('ORISPL')[['PNAME', 'LAT', 'LON']]).reset_index()
        plants.to_csv(dest / 'combined_sfbay_plant_emissions.csv', index=False)
        plot_power_generation_concentration(gpd.clip(combined, mask, keep_geom_type=True), plants,
            dest / 'ldv_freight_power_total_pm25_sfbay_outline.png', source_layer_index=0,
            method_note=f'LDV Home80/Home80 + {scenario} freight; both 10x; power only')
        fqload = json.loads((original / 'beam_charging_summary.json').read_text())
        row = dict(scenario=scenario, receptor_cells=len(combined), ldv_daily_MWh=float(load.sum()),
            freight_daily_MWh=fqload['daily_energy_MWh'], combined_daily_MWh=float(load.sum()+fqload['daily_energy_MWh']),
            max_combined_PM25_ug_m3=float(combined.power_TotalPM25.max()),
            max_ldv_PM25_ug_m3=float(ldv.power_TotalPM25.max()), max_freight_PM25_ug_m3=float(freight.power_TotalPM25.max()))
        rows.append(row)
        save(dest / 'combination_audit.json', dict(**row, component_sum_verified=True,
            ldv_source=str(folder), freight_source=str(fpath), freight_source_sha256=before,
            traffic_included=False, units='ug/m3', additional_scale_factor=1, source_layer=0,
            local_local_removal=False, source_scope='Nine-county SFBAY plants only',
            caveats=['LDV from Cristian 3.events, not baseline0 events',
                'Freight from collaborator first-24h BE charging; both sources already 10x',
                'Representative daily profiles repeated 365 times; Cambium 2040/eGRID 2023 proxy',
                'Layer0 temporary ground-release approximation; excludes outside-Bay plants and unallocated generation',
                'PM2.5 components only; NOx emissions do not constitute NO2 concentration, no BC concentration provided']))
        assert signature(fpath) == before
        print('COMBINED', scenario, row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'combined_summary.csv', index=False)
    save(OUT / 'run_status.json', dict(status='complete', ldv_power_complete=True, combined_scenarios=SCENARIOS,
        traffic_included=False, component_sum_verified=True, results=rows))


if __name__ == '__main__':
    main()
