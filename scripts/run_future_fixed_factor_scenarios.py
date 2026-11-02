"""Independent BF/HF traffic comparison with frozen 2018 factors and 10x scale.

Reuses the validated events activity, EMFAC, ISRM and AERMOD implementations.
New zero-tailpipe freight types use pooled same-body-class baseline non-exhaust
factors. PHEV engine-on trips are identified from actual secondary fuel use;
mixed trips are conservatively entirely engine-on (not an energy-share model).
"""
from __future__ import annotations

from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
import argparse
import copy
import gc
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

import run_events_traffic_pipeline as traffic
import run_ldv_freight_products as legacy
from audit_future_scenario_inputs import DEFAULT_OUTPUT, SCENARIOS
from impacts.pipeline.workflow.prepare_emissions.event_emfac import EmfacRateStore

OUT = DEFAULT_OUTPUT
ORIGINAL_RATES = traffic.RATES
REFERENCE = legacy.OUT / 'Freight'
BASE_MIXTURE = legacy.ORIGINAL / 'inputs/vehicle_emfac_mixture.csv'
NONEXHAUST = {'PMBW', 'PMTW', 'PRDUST'}
MODES = "('car','car_hov2','car_hov3','ride_hail','ride_hail_pooled')"
GROUPS = ('Passenger', 'PortRelated', 'NonPort')
# Explicit user approval on 2026-10-15; scoped to HF-1EP only.
ADDITIONAL_APPROVED_ISRM_SOURCES = {'HF-1EP': (1926, 1927, 2078)}


def freight_donors(source, baseline):
    """Return explicitly selected baseline cohorts and process filter policy."""
    raw, _, mode = source.partition('~')
    if not raw.startswith('ft-'):
        return baseline.loc[baseline.sourceVehicleTypeId.eq(raw)].copy(), 'native'
    stem, fuel = raw.rsplit('-', 1)
    if fuel in {'Dsl', 'Gas'}:
        return baseline.loc[baseline.sourceVehicleTypeId.eq(raw)].copy(), 'native'
    if fuel in {'BE', 'H2FC'}:
        d = baseline.loc[baseline.sourceVehicleTypeId.isin([stem+'-Dsl', stem+'-Gas'])].copy()
        d = d.drop_duplicates('emfacId')
        # Priors are preserved from the same original fleet mapping, not equal
        # weights across fuels/categories/model-year cohorts.
        for weight, prior in [('vmt_weight', 'fleetVmtPrior'), ('pop_weight', 'fleetPopulationPrior')]:
            values = pd.to_numeric(d[prior], errors='raise')
            if values.isna().any() or (values < 0).any() or values.sum() <= 0:
                raise ValueError(f'Invalid pooled non-exhaust prior for {source}')
            d[weight] = values / values.sum()
        return d, 'nonexhaust'
    if fuel == 'PHEV' and mode in {'engine_on', 'engine_off'}:
        secondary = 'Gas' if stem in {'ft-ld1', 'ft-ld3'} else 'Dsl'
        d = baseline.loc[baseline.sourceVehicleTypeId.eq(stem+'-'+secondary)].copy()
        return d, 'native' if mode == 'engine_on' else 'phev_off'
    raise ValueError(f'Unsupported freight technology: {source}')


def select_processes(rates, policy):
    if policy == 'native':
        return rates.copy()
    allowed = NONEXHAUST | ({'DIURN'} if policy == 'phev_off' else set())
    d = rates.loc[rates.process.isin(allowed)].copy()
    if not NONEXHAUST.issubset(set(d.process)):
        raise ValueError('Missing non-exhaust processes in donor factors')
    return d


def configure(scenario):
    work = OUT / scenario / '_work'
    traffic.INPUT = work / 'inputs/derived_events'
    traffic.MODES = MODES
    traffic.RATES = work / 'inputs/rates'
    return work


def setup(scenario):
    work = configure(scenario)
    derived = traffic.INPUT
    derived.mkdir(parents=True, exist_ok=True)
    events = OUT / '_inputs' / scenario / '4.events.parquet'
    network = traffic.ROOT / 'chargingcode/network.csv.gz'
    audit = json.loads((OUT / 'input_audit.json').read_text())
    if not audit['network']['baseline_intersection_reuse_validated']:
        raise ValueError('New network requires regenerated intersections')
    labels = pd.read_csv(traffic.ROOT / 'scenarios/all_scenarios_vehicle_classification.csv')
    labels = labels.loc[labels.scenario.eq('SFBAY-2050-'+scenario)].copy()
    if labels.vehicleId.duplicated().any() or not labels.beam_vehicle_id.eq('body-'+labels.vehicleId).all():
        raise ValueError('Ambiguous freight ID bridge')
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        con.register('labels', labels)
        con.read_parquet(str(events)).create_view('events')
        invalid = con.execute(f"""SELECT count(*) FROM events e LEFT JOIN labels l ON e.vehicle=l.vehicleId
            WHERE type='PathTraversal' AND mode IN {MODES} AND starts_with(vehicleType,'ft-')
            AND (l.vehicleId IS NULL OR e.vehicleType<>l.vehicleTypeId)""").fetchone()[0]
        if invalid:
            raise ValueError(f'Invalid freight labels in events: {invalid}')
        bodies = con.execute('SELECT count(*) FROM labels l JOIN (SELECT DISTINCT vehicle FROM events) e ON l.beam_vehicle_id=e.vehicle').fetchone()[0]
        if bodies != len(labels):
            raise ValueError('beam_vehicle_id bridge not present in events')
        invalid_fuel = con.execute("""SELECT count(*) FROM events WHERE type='PathTraversal'
            AND starts_with(vehicleType,'ft-') AND ends_with(vehicleType,'PHEV')
            AND (primaryFuel IS NULL OR secondaryFuel IS NULL OR primaryFuel<0 OR secondaryFuel<0
                 OR (length>0 AND primaryFuel=0 AND secondaryFuel=0)
                 OR primaryFuelType<>'Electricity'
                 OR secondaryFuelType<>CASE WHEN vehicleType IN ('ft-ld1-PHEV','ft-ld3-PHEV') THEN 'Gasoline' ELSE 'Diesel' END)""").fetchone()[0]
        if invalid_fuel:
            raise ValueError(f'Invalid PHEV fuel-mode records: {invalid_fuel}')
        diagnostics = con.execute("""SELECT vehicleType,count(*) trips,
            count(*) FILTER(WHERE primaryFuel>0 AND secondaryFuel>0) mixed_trips,
            count(*) FILTER(WHERE secondaryFuel>0) engine_on_trips,
            sum(length) FILTER(WHERE primaryFuel>0 AND secondaryFuel>0)/1000 mixed_km
            FROM events WHERE type='PathTraversal' AND starts_with(vehicleType,'ft-')
            AND ends_with(vehicleType,'PHEV') GROUP BY 1""").fetchdf()
        diagnostics.to_csv(derived / 'phev_fuel_mode_audit.csv', index=False)
        target = derived / '3.events.parquet'
        if target.exists():
            raise FileExistsError(target)
        con.execute(f"""COPY (SELECT e.type,e.mode,e.vehicle,
            e.vehicleType AS original_vehicle_type,
            e.vehicleType || CASE WHEN starts_with(e.vehicleType,'ft-') AND ends_with(e.vehicleType,'PHEV')
                THEN CASE WHEN e.secondaryFuel>0 THEN '~engine_on' ELSE '~engine_off' END ELSE '' END || '@@' ||
                CASE WHEN NOT starts_with(e.vehicleType,'ft-') THEN 'Passenger'
                     WHEN l.is_port_related THEN 'PortRelated' ELSE 'NonPort' END AS vehicleType,
            departureTime,arrivalTime,weight,links,linkTravelTime
            FROM events e LEFT JOIN labels l ON e.vehicle=l.vehicleId
            WHERE type='PathTraversal' AND mode IN {MODES})
            TO '{traffic.sqlpath(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    shutil.copy2(network, derived / 'network.csv.gz')
    traffic.setup(work)
    labels.to_csv(work / 'inputs/vehicle_classification.csv', index=False)
    traffic.save(work / 'scenario.json', dict(
        scenario=scenario, events=str(events), derived_activity_input=str(target),
        scenario_year=2050, emission_factor_year=2018, fleet_scale=10, input_sample_fraction=.1,
        power_generation=False, transit_included=False, day_window_seconds=[0, 86400],
        interpretation='Fixed-2018-factor traffic/powertrain comparison, NOT a forecast of 2050 emission factors',
        mapping_assumptions=[
            'Passenger and ICE freight retain the original baseline EMFAC cohort factors and priors',
            'BE/H2FC freight: same-body-class pooled baseline PMBW/PMTW/PRDUST only; no regeneration discount',
            'PHEV freight: engine on iff event secondaryFuel>0; mixed trips entirely engine-on conservatively',
            'Engine-off PHEV retains non-exhaust plus gasoline DIURN if present, no exhaust/start/hot-soak',
            'Event gross vehicle mass used for road dust, including payload; no extra payload addition',
            'Baseline annual operating days and representative 0-24h window retained',
            'No unobserved initial cold starts or overnight parking inferred',
        ]))
    print('SETUP',scenario,flush=True)


def mapping(scenario):
    work = configure(scenario)
    baseline = pd.read_csv(BASE_MIXTURE, low_memory=False)
    native_store = EmfacRateStore(ORIGINAL_RATES, baseline.set_index('vehicleTypeId', drop=False), traffic.POLS)
    original_rates = pd.read_parquet(ORIGINAL_RATES.parent/'inventory/sf-emfac-2018-project-analysis-final-freight-rates.parquet')
    with duckdb.connect() as con:
        sources = con.execute('SELECT DISTINCT vehicleType FROM read_parquet(?) ORDER BY 1', [str(traffic.INPUT/'3.events.parquet')]).fetchdf().vehicleType
    frames = []
    written = set()
    for source in sources:
        raw, category = source.split('@@')
        donors, policy = freight_donors(raw, baseline)
        if donors.empty:
            raise ValueError(f'No fixed-factor donor for {source}')
        donors['donor_emfacId'] = donors.emfacId
        donors['donor_vehicleTypeId'] = donors.vehicleTypeId
        donors['rate_policy'] = policy
        donors['original_source_type'] = raw.split('~')[0]
        donors['product_category'] = category
        donors['sourceVehicleTypeId'] = source
        donors['mapping_basis'] = f'Frozen 2018 factors; {policy}; see scenario.json'
        donors['emfacId'] = donors.emfacId.map(lambda e: str(e) if policy == 'native' else f'fixed2018-{policy}-{e}')
        donors['vehicleTypeId'] = source + '--' + donors.emfacId
        for w in ['vmt_weight', 'pop_weight']:
            np.testing.assert_allclose(donors[w].sum(), 1.)
        for _, row in donors.iterrows():
            emfac = row.emfacId
            if emfac in written:
                continue
            try:
                rates = native_store.rates(row.donor_vehicleTypeId)
            except FileNotFoundError:
                rates = original_rates.loc[original_rates.vehicleCategory.eq(row.emfacVehicleCategory)
                    & original_rates.fuel.eq(row.emfacFuel) & original_rates.modelYear.eq(row.emfacResolvedModelYear)].copy()
                if rates.empty:
                    raise
            rates = select_processes(rates, policy)
            partition = traffic.RATES / f'emfacId={emfac}'
            partition.mkdir(parents=True, exist_ok=True)
            rates.to_parquet(partition/'rates.parquet', index=False)
            written.add(emfac)
        frames.append(donors)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(work/'inputs/vehicle_emfac_mixture.csv', index=False)
    for group in ['passenger', 'freight']:
        result.loc[result.assignment_group.eq(group)].to_csv(work/f'inputs/{group}_vehicle_types.csv', index=False)
    traffic.save(work/'mapping_audit.json', dict(source_keys=len(sources), weighted_components=len(result),
        exact_or_filtered_native_partitions=len(written), no_new_numeric_emission_factors_invented=True,
        nonexhaust_policy='Same-body-class baseline cohort proxy; not technology-specific calibrated brake/tire factors'))
    print('MAPPING',scenario,len(result),flush=True)


def prepare(scenario):
    work = configure(scenario)
    source = work/'emissions/prepared_skims_for_grid_allocation.parquet'
    mix = pd.read_csv(work/'inputs/vehicle_emfac_mixture.csv', low_memory=False)
    lookup = mix[['sourceVehicleTypeId', 'product_category', 'original_source_type']].drop_duplicates().rename(columns={'sourceVehicleTypeId':'vehicleTypeId'})
    if lookup.vehicleTypeId.duplicated().any():
        raise ValueError('Duplicate cohort lookup')
    measures = ['totTrips','totVMT']+[f'tons_per_year_{p}' for p in traffic.POLS]
    sums = ','.join(f'sum(e."{c}")*10 AS "{c}"' for c in measures)
    aggregate = work/'emissions/classified_emissions_10x.parquet'
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        con.register('lookup', lookup)
        con.execute(f"""COPY (SELECT m.product_category,e.linkId,e.process,e.source_release_height,e.roadCategory,{sums}
            FROM read_parquet(?) e JOIN lookup m USING(vehicleTypeId) GROUP BY ALL)
            TO '{traffic.sqlpath(aggregate)}' (FORMAT PARQUET, COMPRESSION ZSTD)""", [str(source)])
        grouped = con.execute('SELECT product_category,'+','.join(f'sum("{c}") AS "{c}"' for c in measures)+
                              ' FROM read_parquet(?) GROUP BY 1',[str(aggregate)]).fetchdf()
        grouped.to_csv(OUT/scenario/'annual_emissions_by_category.csv',index=False)
        old = json.loads((work/'emissions_audit.json').read_text())
        for p in traffic.POLS:
            np.testing.assert_allclose(grouped[f'tons_per_year_{p}'].sum(),old[p]*10,rtol=1e-10)
        # Export three disjoint categories plus their combined total.
        for product in (*GROUPS, 'Total'):
            target = OUT/scenario/product
            for folder in ['inputs','emissions','concentrations','exposure','_tmp']:
                (target/folder).mkdir(parents=True,exist_ok=True)
            where = '' if product == 'Total' else f"WHERE product_category='{product}'"
            totals = ','.join(f'sum("{c}") AS "{c}"' for c in measures)
            prepared = target/'emissions/prepared_skims_for_grid_allocation.parquet'
            con.execute(f"""COPY (SELECT linkId,'{product}-height-'||cast(source_release_height AS VARCHAR) AS vehicleTypeId,
                process,source_release_height,roadCategory,{totals} FROM read_parquet(?) {where} GROUP BY ALL)
                TO '{traffic.sqlpath(prepared)}' (FORMAT PARQUET, COMPRESSION ZSTD)""",[str(aggregate)])
            for filename in ['settings.yaml','preprocess_manifest.yaml']:
                text = (REFERENCE/filename).read_text(encoding='utf-8').replace(str(REFERENCE),str(target)).replace(REFERENCE.as_posix(),target.as_posix())
                data = yaml.safe_load(text)
                if filename == 'preprocess_manifest.yaml':
                    data['notes'] = [f'{scenario}/{product}: fixed 2018 factors, 10x applied once to prepared emissions; no power or transit.']
                    p = data['pipeline']
                    p.update(population_sample=1.,freight_sample=1.,transit_sample=1.)
                    for key,path in [('events_input',OUT/'_inputs'/scenario/'4.events.parquet'),
                                     ('network',traffic.ROOT/'chargingcode/network.csv.gz'),
                                     ('rates_folder',traffic.RATES),
                                     ('passenger_vehicle_types_input',work/'inputs/passenger_vehicle_types.csv'),
                                     ('freight_vehicle_types_input',work/'inputs/freight_vehicle_types.csv')]:
                        data['inputs'][key]=dict(kind='local',source_path=str(path),staged_path=str(path),exists=True,optional=False)
                    p['passenger_vehicle_types_file']=str(work/'inputs/passenger_vehicle_types.csv')
                    p['freight_vehicle_types_file']=str(work/'inputs/freight_vehicle_types.csv')
                (target/filename).write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
            record = json.loads((work/'scenario.json').read_text())
            record.update(product=product,prepared_emissions_already_annualized_and_scaled=True)
            traffic.save(target/'scenario.json',record)
            traffic.save(target/'run_status.json',dict(status='emissions_ready',concentration_generated=False))
    print('10X EMISSIONS READY',scenario,flush=True)


def resolve_short_connector_roads(work):
    """Use a documented Local Urban dust proxy only for short synthetic links.

    The geometry and activity stay unchanged. Long/fast/real OSM links with
    missing road classes still fail, rather than receiving a blanket fallback.
    """
    network=pd.read_csv(traffic.ROOT/'chargingcode/network.csv.gz')
    eligible=network.loc[network.attributeOrigType.isna() & network.attributeOrigId.isna()
        & network.linkLength.le(30) & network.linkFreeSpeed.le(2.78),['linkId']]
    with duckdb.connect(str(work/'_tmp/traffic.duckdb')) as con:
        con.register('eligible',eligible)
        missing=con.execute('''SELECT n.linkId,n.linkLength,n.linkFreeSpeed,sum(a.miles) AS sample_day_miles
            FROM running a JOIN network n USING(linkId) JOIN zones z USING(linkId)
            WHERE n.road IS NULL GROUP BY ALL''').fetchdf()
        if not missing.linkId.isin(eligible.linkId).all():
            raise ValueError('Unclassified roads exceed short synthetic connector policy')
        if len(missing):
            missing['dust_road_proxy']='residential (Local Urban)'
            missing.to_csv(work/'inputs/short_synthetic_connector_proxy.csv',index=False)
            con.execute("UPDATE network SET road='residential' WHERE road IS NULL AND linkId IN (SELECT linkId FROM eligible)")
            print('Short synthetic connector proxy:',missing.to_dict('records'),flush=True)


def audit_weights(scenario):
    """Persist invalid inputs without repairing or deleting any event activity."""
    work=configure(scenario)
    with duckdb.connect(str(work/'_tmp/traffic.duckdb'),read_only=True) as con:
        bad=con.execute('''SELECT trip_id,vehicle,source_type,dep,arr,weight_kg,
            len(link_list) AS link_count FROM trips WHERE weight_kg IS NULL OR weight_kg<=0''').fetchdf()
    bad.to_csv(OUT/scenario/'input_audit/invalid_event_weights.csv',index=False)
    traffic.save(OUT/scenario/'run_status.json',dict(status='awaiting_invalid_weight_policy',
        final_concentration_generated=False,invalid_trips=len(bad),vehicles=bad.vehicle.nunique(),
        input_events_unchanged=True,repair_applied=False,
        proposed_policy='Use verified empty-vehicle mass for anomalous negative-weight trips in dust calculation only; needs user confirmation',
        observed_initial_weight_kg=20000,initial_payload_loading_kg=0,
        evidence='Both affected tours start with zero loading and 20000 kg event gross mass; later unload decrements create negative weights. PathTraversal PayloadWeightInKg is null.'))


def coalesced_emissions(work):
    """Distributivity optimization: sum cohort coefficients before activity join.

    Every rate, activity bin, process, county, height, prior and operating-day
    weight is preserved. Verify against the native expanded expression before
    computing the full result. No approximations/rounding/thresholds are added.
    """
    traffic.emissions(work,prepare_only=True)
    sums=','.join(f'sum("{p}"*operation_days*{{weight}}*{{idle}}) AS "{p}"' for p in traffic.POLS)
    keys='source_type,source_release_height,county,process,roadCategory,activity_bin'
    joins='''a.source_type=r.source_type AND a.county=r.county
        AND (r.roadCategory IS NULL OR r.roadCategory='' OR lower(a.road)=r.roadCategory)
        AND (r.activity_bin=0 OR a.{bin}=r.activity_bin)'''
    amount="CASE WHEN r.process='PRDUST' THEN a.dust_miles WHEN r.process IN ('IDLEX','RUNLOSS') THEN a.hours ELSE a.miles END"
    park_amount="CASE WHEN r.process='DIURN' THEN a.hours ELSE a.observations END"
    measures=['totTrips','totVMT']+[f'tons_per_year_{p}' for p in traffic.POLS]
    with duckdb.connect(str(work/'_tmp/traffic.duckdb')) as con:
        con.execute('SET threads=4');con.execute("SET memory_limit='12GB'")
        for label,weight,idle in [('run','vmt_weight',"CASE WHEN process='IDLEX' THEN idle_fraction ELSE 1 END"),('park','pop_weight','1')]:
            processes="('RUNEX','RUNLOSS','PMBW','PMTW','PRDUST','PTOEX','IDLEX')" if label=='run' else "('DIURN','HOTSOAK','STREX')"
            con.execute(f'''CREATE OR REPLACE TABLE {label}_coeff AS SELECT {keys},
                sum(operation_days*{weight}) AS activity_coeff,{sums.format(weight=weight,idle=idle)}
                FROM component_rates WHERE process IN {processes} GROUP BY ALL''')
        def query(running,parking,*,expanded=False):
            frames=[]
            for label,table,act,bincol,weight in [('run',running,amount,'speed_bin','vmt_weight'),('park',parking,park_amount,'soak_bin','pop_weight')]:
                coef=f'r.operation_days*r.{weight}' if expanded else 'r.activity_coeff'
                factor=f'r.operation_days*r.{weight}' if expanded else '1'
                if expanded and label=='run':factor+="*CASE WHEN r.process='IDLEX' THEN r.idle_fraction ELSE 1 END"
                selected=','.join(f'{act}*r."{p}"*{factor}*a.county_weight/{traffic.grams_per_short_ton} AS "tons_per_year_{p}"' for p in traffic.POLS)
                extra=" AND a.process=r.process" if label=='park' else ''
                filter_sql=(" WHERE r.process IN ('RUNEX','RUNLOSS','PMBW','PMTW','PRDUST','PTOEX','IDLEX')" if label=='run' else " WHERE r.process IN ('DIURN','HOTSOAK','STREX')") if expanded else ''
                frames.append(f'''SELECT a.source_type AS vehicleTypeId,a.linkId,r.process,r.source_release_height,a.road AS roadCategory,
                    a.observations*{coef}*a.county_weight AS totTrips,
                    {('a.miles' if label=='run' else '0')}*{coef}*a.county_weight AS totVMT,{selected}
                    FROM {table} a JOIN {('component_rates' if expanded else label+'_coeff')} r ON {joins.format(bin=bincol)}{extra}{filter_sql}''')
            return ' UNION ALL '.join(frames)
        con.execute('CREATE OR REPLACE TEMP TABLE parity_running AS SELECT * FROM running_detail USING SAMPLE 2000 ROWS (reservoir, 42)')
        con.execute('CREATE OR REPLACE TEMP TABLE parity_parking AS SELECT * FROM parking_detail USING SAMPLE 2000 ROWS (reservoir, 42)')
        group_keys='vehicleTypeId,linkId,process,source_release_height,roadCategory'
        totals=','.join(f'sum("{c}") AS "{c}"' for c in measures)
        def aggregate(sql):return f'SELECT {group_keys},{totals} FROM ({sql}) GROUP BY ALL'
        original=con.execute(aggregate(query('parity_running','parity_parking',expanded=True))).fetchdf().set_index(group_keys.split(',')).sort_index()
        optimized=con.execute(aggregate(query('parity_running','parity_parking'))).fetchdf().set_index(group_keys.split(',')).sort_index()
        pd.testing.assert_index_equal(original.index,optimized.index)
        np.testing.assert_allclose(original.to_numpy(),optimized.to_numpy(),rtol=1e-11,atol=1e-10)
        traffic.save(work/'emissions_optimization_parity.json',dict(passed=True,compared_source_process_rows=len(original),
            sampled_running_rows=2000,sampled_parking_rows=2000,
            max_absolute_errors=(original-optimized).abs().max().to_dict(),
            method='Exact weighted-rate coalescence by source type, process, county, height, road and activity bin'))
        print('EXPANDED/COALESCED PARITY PASSED',len(original),flush=True)
        target=work/'emissions/prepared_skims_for_grid_allocation.parquet'
        partial=target.with_suffix('.coalesced.partial.parquet')
        con.execute(f"COPY ({aggregate(query('running_detail','parking_detail'))}) TO '{traffic.sqlpath(partial)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        report=con.execute('SELECT count(*) row_count,'+','.join(f'sum(tons_per_year_{p}) AS "{p}"' for p in traffic.POLS)+
                           ' FROM read_parquet(?)',[str(partial)]).fetchdf().iloc[0].to_dict()
        if any(not np.isfinite(report[p]) or report[p]<0 for p in traffic.POLS):raise ValueError('Invalid emissions')
        partial.replace(target)
        report.update(fleet_expansion=1,power_generation=False,units='short tons/year',source='new events, frozen factors; exact coefficient coalescence')
        traffic.save(work/'emissions_audit.json',report)
        print('EMISSIONS',report,flush=True)


def disperse(scenario, product, stage):
    import geopandas as gpd
    from impacts.pipeline.workflow import step3_compute_aermod_concentrations as aermod
    from impacts.pipeline.workflow import step4_prepare_exposure as exposure
    from impacts.manifest.schema import PipelineConfig
    from gateway_aermod_crop import apply_cropped
    target = OUT/scenario/product
    if stage == 'filter':
        legacy.OUT = OUT/scenario
        try:
            result = legacy.filter_sources(product, additional_approved_sources=
                ADDITIONAL_APPROVED_ISRM_SOURCES.get(scenario, ()))
            audit_path = target/'additional_missing_isrm_audit.json'
            if audit_path.exists():
                report = json.loads(audit_path.read_text(encoding='utf-8'))
                report.update(status='additional_exclusion_approved_and_applied',
                    approval='User explicitly approved on 2026-10-15',
                    approved_additional_source_ids=list(ADDITIONAL_APPROVED_ISRM_SOURCES.get(scenario, ())),
                    aermod_sources_filtered=False,
                    required_action=None)
                traffic.save(audit_path, report)
            return result
        except ValueError as error:
            if str(error) != 'New missing sources outside authorized 103':
                raise
            import zarr
            matrix = zarr.open_group(ISRM_PATH, mode='r')
            source = target/'emissions/beam_emissions_for_inmap.parquet'
            columns = [c for c in pq.read_schema(source).names if c.startswith('tons_per_year_')]
            data = pd.read_parquet(source, columns=['inmap_cell_id', *columns]).groupby('inmap_cell_id')[columns].sum()
            missing = data.loc[~data.index.isin(matrix['source_ids'][:]) & data.abs().sum(axis=1).gt(0)]
            approved = pd.read_csv(legacy.TOTAL/'inputs/missing_isrm_sources.csv').inmap_cell_id
            extra = missing.loc[~missing.index.isin(approved)]
            extra.to_csv(target/'inputs/additional_missing_isrm_sources.csv')
            report = dict(status='blocked_additional_missing_isrm_sources', scenario=scenario,
                product=product, final_concentration_generated=False,
                missing_source_count=len(missing), additional_source_ids=list(map(int, extra.index)),
                additional_emissions_short_tons_year=extra.sum().to_dict(),
                additional_emission_fractions=(extra.sum()/data.sum()).to_dict(),
                completed_stages=['activity','emissions','prepare','allocate'],
                required_action='Supply the additional ISRM source rows or explicitly approve their exclusion; no additional sources have been omitted.')
            traffic.save(target/'additional_missing_isrm_audit.json', report)
            traffic.save(target/'run_status.json', report)
            if product == 'Total':
                traffic.save(OUT/scenario/'run_status.json', report)
            raise
    if stage == 'merge':
        pipeline = PipelineConfig.from_dict(yaml.safe_load((target/'preprocess_manifest.yaml').read_text())['pipeline'])
        data = exposure._build_full_exposure_grid(pipeline=pipeline,
            prepared_inmap=exposure._prepare_inmap_exposure_inputs(gpd.read_parquet(target/'concentrations/beam_inmap_concentrations.parquet')),
            prepared_aermod=exposure._prepare_aermod_exposure_inputs(gpd.read_parquet(target/'concentrations/beam_aermod_concentrations.parquet')),
            working_dir=target/'exposure')
        data.to_parquet(target/'exposure/beam_concentration_distribution.parquet',index=False)
        return
    if stage == 'aermod':
        aermod._apply_kernels = apply_cropped
        def parquet_only(gdf,output_path):
            output_path.parent.mkdir(parents=True,exist_ok=True)
            gdf.to_parquet(output_path,index=False)
        aermod._write_outputs = parquet_only
    traffic.run_stage(target,'emissions' if stage=='allocate' else stage)


def validate(scenario,product):
    target=OUT/scenario/product
    path=target/'exposure/beam_concentration_distribution.parquet'
    pf=pq.ParquetFile(path)
    if any(c.startswith('power_') for c in pf.schema_arrow.names):
        raise ValueError('Unexpected power-generation columns')
    cols=['TotalPM25','PrimaryPM25','SecondaryPM25','BC','NO2','inmap_PrimaryPM25','inmap_SecondaryPM25',
          'aermod_PrimaryPM25','inmap_BC','inmap_NO2','aermod_BC','aermod_NO2',
          'inmap_only_PrimaryPM25','inmap_only_TotalPM25','inmap_only_BC','inmap_only_NO2']
    pop=pd.read_parquet(legacy.TOTAL/'exposure/beam_population_counts.parquet',columns=['aermod_cell_id','person_count']).set_index('aermod_cell_id').person_count
    maxima={c:0. for c in cols};burdens={c:0. for c in cols}
    for batch in pf.iter_batches(columns=['aermod_cell_id',*cols],batch_size=131072):
        d=batch.to_pandas();values=d[cols].to_numpy()
        if not np.isfinite(values).all() or values.min() < -1e-10:
            raise ValueError('Nonfinite/negative concentration')
        for actual,left,right in [('TotalPM25','PrimaryPM25','SecondaryPM25'),
            ('PrimaryPM25','inmap_PrimaryPM25','aermod_PrimaryPM25'),('BC','inmap_BC','aermod_BC'),
            ('NO2','inmap_NO2','aermod_NO2'),('inmap_only_TotalPM25','inmap_only_PrimaryPM25','SecondaryPM25')]:
            np.testing.assert_allclose(d[actual],d[left]+d[right],atol=1e-10)
        np.testing.assert_allclose(d.SecondaryPM25,d.inmap_SecondaryPM25,atol=1e-10)
        if (d.inmap_only_PrimaryPM25+1e-10<d.inmap_PrimaryPM25).any():
            raise ValueError('InMAP-only primary unexpectedly less than nonlocal primary')
        weights=d.aermod_cell_id.map(pop).fillna(0.)
        for c in cols:
            maxima[c]=max(maxima[c],float(d[c].max()))
            burdens[c]+=float((d[c]*weights).sum())
    report=dict(status='complete',scenario=scenario,product=product,concentration_file=str(path),
        grid_cells=pf.metadata.num_rows,units='ug/m3',power_generation=False,transit_included=False,
        fleet_expansion=10,emission_factor_year=2018,component_identities_passed=True,
        population=float(pop.sum()),maxima_ug_m3=maxima,
        population_weighted_concentrations={c:v/float(pop.sum()) for c,v in burdens.items()},
        partial_isrm=json.loads((target/'partial_isrm_audit.json').read_text()))
    exclusion = OUT/scenario/'_work/inputs/derived_events/invalid_trip_exclusion_audit.json'
    if exclusion.exists():
        report['input_trip_exclusion'] = json.loads(exclusion.read_text(encoding='utf-8'))
        report['input_trip_exclusion']['concentration_rerun'] = True
    traffic.save(target/'run_status.json',report)
    if product == 'Total':
        traffic.save(OUT/scenario/'run_status.json',report)
    print(json.dumps(report,indent=2),flush=True)


def full_expanded_parity(scenario):
    """Optional independent full-total check when the expanded table was saved."""
    work=configure(scenario)
    with duckdb.connect(str(work/'_tmp/traffic.duckdb'),read_only=True) as con:
        con.execute('SET threads=4')
        sums=','.join(f'sum(amount*"{p}"*operation_days/{traffic.grams_per_short_ton}) AS "{p}"' for p in traffic.POLS)
        expanded=con.execute('SELECT '+sums+' FROM gram_activity').fetchdf().iloc[0].to_dict()
    optimized=json.loads((work/'emissions_audit.json').read_text())
    for p in traffic.POLS:
        np.testing.assert_allclose(expanded[p],optimized[p],rtol=1e-10,atol=1e-8)
    traffic.save(work/'full_expanded_emissions_parity.json',dict(passed=True,
        expanded_totals=expanded,coalesced_totals={p:optimized[p] for p in traffic.POLS},
        absolute_differences={p:abs(expanded[p]-optimized[p]) for p in traffic.POLS},
        units='short tons/year before 10x expansion'))
    print('FULL EXPANDED EMISSIONS PARITY PASSED',scenario,flush=True)


def run_job(scenario, stage, product='Total'):
    work = OUT/scenario/'_work'
    marker = (work if stage in {'setup','mapping','activity','emissions','prepare'} else OUT/scenario/product)/f'completed_{stage}.json'
    if marker.exists():
        return
    logdir=OUT/scenario/'logs';logdir.mkdir(parents=True,exist_ok=True)
    print('START',scenario,product,stage,flush=True)
    with (logdir/f'{product}_{stage}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',__file__,'--scenario',scenario,'--stage',stage,'--product',product],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    traffic.save(marker,dict(status='complete',stage=stage,product=product))
    print('DONE',scenario,product,stage,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenario',choices=SCENARIOS)
    parser.add_argument('--product',default='Total',choices=(*GROUPS,'Total'))
    parser.add_argument('--stage',required=True,choices=['all','setup','mapping','activity','audit_weights','emissions','prepare','allocate','filter','inmap','aermod','merge','validate','full_parity'])
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stage=='all':
        for scenario in ([args.scenario] if args.scenario else SCENARIOS):
            for stage in ['setup','mapping','activity','emissions','prepare']:
                run_job(scenario,stage)
            for stage in ['allocate','filter','inmap','aermod','merge','validate']:
                run_job(scenario,stage,args.product)
        return
    if not args.scenario:
        parser.error('--scenario required for a single stage')
    work=configure(args.scenario)
    if args.stage=='setup':return setup(args.scenario)
    if args.stage=='mapping':return mapping(args.scenario)
    if args.stage=='activity':return traffic.activity(work)
    if args.stage=='audit_weights':return audit_weights(args.scenario)
    if args.stage=='full_parity':return full_expanded_parity(args.scenario)
    if args.stage=='emissions':
        resolve_short_connector_roads(work)
        return coalesced_emissions(work)
    if args.stage=='prepare':return prepare(args.scenario)
    if args.stage=='validate':return validate(args.scenario,args.product)
    disperse(args.scenario,args.product,args.stage)


if __name__=='__main__':
    main()
