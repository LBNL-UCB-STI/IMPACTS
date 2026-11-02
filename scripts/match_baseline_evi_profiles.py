"""Fetch auditable EVI breakdown, match baseline composition without curve fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
OUT = BASE/'evi-fleet-matched-20261012'
API = 'https://developer.nlr.gov/api/evi-pro-lite/v1'
SPEC = 'https://developer.nlr.gov/docs/transportation/evi-pro-lite-v1/spec.yml'
TYPE_SHARES = {'phev20':.10,'phev50':.15,'bev100':.25,'bev250':.50}
BODY_SHARES = {'sedan':.8,'suv':.2}
CHARGERS = ('home_l1','home_l2','work_l1','work_l2','public_l2','public_l3')


def unpack(payload):
    parts = payload['results']['weekday_load_profile']
    if set(parts) != set(CHARGERS):
        raise ValueError('Unexpected charger components')
    result = {}
    for charger, data in parts.items():
        expected = {f'{t}_{b}' for t in TYPE_SHARES for b in BODY_SHARES
                    if not (charger == 'public_l3' and t.startswith('phev'))}
        if set(data) != expected:
            raise ValueError(f'Unexpected subtype keys for {charger}')
        result[charger] = {}
        for subtype, values in data.items():
            if any(x is None for x in values):
                raise ValueError('Null EVI values are not zeros')
            x=np.asarray(values,dtype=float)
            if x.shape != (96,) or not np.isfinite(x).all() or (x<0).any():
                raise ValueError('Invalid EVI 15-minute power array')
            result[charger][subtype] = x
    return result


def reweight(parts, target_counts, source_fleet=30000, source_shares=None):
    source_shares = TYPE_SHARES if source_shares is None else source_shares
    if set(target_counts) != set(TYPE_SHARES) or any(v<0 for v in target_counts.values()):
        raise ValueError('Invalid target subtype counts')
    output = {}
    for charger, data in parts.items():
        output[charger] = {}
        for key, values in data.items():
            kind,body = key.rsplit('_',1)
            source_count = source_fleet*source_shares[kind]*BODY_SHARES[body]
            target_count = target_counts[kind]*BODY_SHARES[body]
            output[charger][key] = values/source_count*target_count
    return output


def total(parts, prefix=None):
    return sum((x for data in parts.values() for k,x in data.items()
                if prefix is None or k.startswith(prefix)),start=np.zeros(96))


def hourly(values):
    return values.reshape(24,4).mean(axis=1)


def curve_metrics(name, kw, fleet):
    x=hourly(kw)
    return dict(scenario=name,fleet=fleet,daily_MWh=float(x.sum()/1000),
        daily_kWh_per_vehicle=float(x.sum()/fleet),peak_MW=float(x.max()/1000),
        peak_hour=int(x.argmax()),peak_kW_per1000=float(x.max()*1000/fleet),
        evening18_24_energy_share=float(x[18:].sum()/x.sum()))


def fetch(endpoint, inputs, name):
    path = OUT/name
    if path.exists():
        record = json.loads(path.read_text())
        if record['endpoint'] != endpoint or record['inputs'] != inputs:
            raise ValueError('Cache inputs mismatch')
    else:
        r = requests.get(endpoint,params={**inputs,'api_key':'DEMO_KEY'},timeout=45)
        try: payload = r.json()
        except ValueError: payload = {'raw_text':r.text}
        record = dict(endpoint=endpoint,inputs=inputs,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      http_status=r.status_code,response=payload)
        path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    if record['http_status'] != 200 or record['response'].get('errors'):
        raise ValueError(f"EVI request failed: {record['http_status']}; see {path}")
    for key,value in inputs.items():
        if str(record['response']['metadata']['inputs'].get(key)) != str(value):
            raise ValueError(f'EVI did not echo requested {key}')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-only',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protected = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*')
                 if p.is_file() and OUT not in p.parents}
    summary = json.loads((BASE/'charging_code_summary.json').read_text())
    inputs = summary['EVI_inputs']
    spec_file = OUT/'official_api_spec.yml'
    if not spec_file.exists():
        r=requests.get(SPEC,timeout=30);r.raise_for_status();spec_file.write_text(r.text,encoding='utf-8')
    record = fetch(API+'/daily-load-profile/breakdown.json',inputs,'evi_breakdown_BEV_30000.json')
    aggregate = fetch(API+'/daily-load-profile.json',inputs,'evi_aggregate_BEV_30000.json')
    result = record['response']['results']['weekday_load_profile']
    print('Breakdown charger keys:',list(result))
    for k,v in result.items():
        print(k, type(v).__name__,list(v)[:10] if isinstance(v,dict) else len(v))
    if args.fetch_only:return
    parts = unpack(record['response'])
    full = aggregate['response']['results']['weekday_load_profile']
    archived = json.loads(Path(summary['EVI_reference']).read_text())['response']['results']['weekday_load_profile']
    max_error = {}
    for c in CHARGERS:
        np.testing.assert_array_equal(full[c],archived[c])
        max_error[c]=float(np.max(np.abs(sum(parts[c].values())-np.asarray(full[c]))))
        # Eight rounded components + one rounded total: <=0.045 kW.
        np.testing.assert_allclose(sum(parts[c].values()),full[c],atol=.046,rtol=0)
    equal_record = fetch(API+'/daily-load-profile/breakdown.json',{**inputs,'pev_dist':'EQUAL'},
                         'evi_breakdown_EQUAL_30000.json')
    equal_parts = unpack(equal_record['response'])
    equal_shares = dict(phev20=.15,phev50=.35,bev100=.15,bev250=.35)
    inventory = pd.read_parquet(BASE/'existing_ev_inventory.parquet')
    metadata = pd.read_csv(BASE/'matched_vehicle_types.csv')
    vehicles = inventory.merge(metadata,left_on='vehicleType',right_on='vehicleTypeId',validate='many_to_one')
    if vehicles.battery_kWh.isna().any():raise ValueError('Missing capacity')
    vehicles['native_electric_kWh_per_km'] = vehicles.primaryFuelConsumptionInJoulePerMeter/3600
    # Capacity/rate gives a full-capacity range proxy, not certified usable range.
    vehicles['range_proxy_miles'] = vehicles.battery_kWh/vehicles.native_electric_kWh_per_km/1.609344
    if not np.isfinite(vehicles.range_proxy_miles).all() or vehicles.range_proxy_miles.le(0).any():
        raise ValueError('Invalid nominal range proxy')
    vehicles['evi_range_proxy_class'] = [
        min((('phev20',20),('phev50',50)) if kind=='PHEV' else (('bev100',100),('bev250',250)),
            key=lambda v:abs(v[1]-r))[0] for kind,r in zip(vehicles.powertrain,vehicles.range_proxy_miles)]
    mapping = vehicles.groupby(['powertrain','battery_kWh','native_electric_kWh_per_km',
        'range_proxy_miles','evi_range_proxy_class']).agg(sample_vehicles=('vehicle','nunique')).reset_index()
    mapping['expanded_vehicles'] = mapping.sample_vehicles*10
    mapping.to_csv(OUT/'vehicle_range_proxy_mapping.csv',index=False)
    vehicles[['vehicle','vehicleType','powertrain','battery_kWh','native_electric_kWh_per_km',
              'range_proxy_miles','evi_range_proxy_class']].to_parquet(OUT/'vehicle_mapping.parquet',index=False)
    fleet = len(vehicles)*10
    counts = vehicles.powertrain.value_counts()*10
    type_targets = {t:float(counts['PHEV' if t.startswith('phev') else 'BEV'])*share/
                    (.25 if t.startswith('phev') else .75) for t,share in TYPE_SHARES.items()}
    range_targets = {t:int(vehicles.evi_range_proxy_class.eq(t).sum())*10 for t in TYPE_SHARES}
    np.testing.assert_allclose(sum(type_targets.values()),fleet)
    assert sum(range_targets.values())==fleet
    matched_type = reweight(parts,type_targets)
    matched_range = reweight(parts,range_targets)
    equal_type = reweight(equal_parts,type_targets,source_shares=equal_shares)
    equal_range = reweight(equal_parts,range_targets,source_shares=equal_shares)
    crosscheck = dict(powertrain_only_daily_ratio_EQUAL_to_BEV=float(total(equal_type).sum()/total(matched_type).sum()),
                     range_proxy_daily_ratio_EQUAL_to_BEV=float(total(equal_range).sum()/total(matched_range).sum()))
    # Different source scenario draws need not have identical component shapes.
    if any(abs(x-1)>.05 for x in crosscheck.values()):
        raise ValueError(f'Unexpected source-mix dependence: {crosscheck}')
    pd.DataFrame([dict(evi_type=t,original_share=TYPE_SHARES[t],
        matched_powertrain_count=type_targets[t],matched_range_proxy_count=range_targets[t])
        for t in TYPE_SHARES]).to_csv(OUT/'target_fleet_weights.csv',index=False)
    old = pd.read_csv(BASE/'existing_ev_hourly_charging_curve_10x.csv')
    new = pd.read_csv(BASE/'phev20-sensitivity-20261012/phev20/existing_ev_hourly_charging_curve_10x.csv')
    original_kw = sum(np.asarray(x) for x in full.values())*fleet/inputs['fleet_size']
    curves = dict(charging_code_phev10=np.repeat(old.total_MW_10x.to_numpy()*1000,4),
                  charging_code_phev20=np.repeat(new.total_MW_10x.to_numpy()*1000,4),
                  evi_original=original_kw,evi_powertrain_matched=total(matched_type),
                  evi_powertrain_range_proxy_matched=total(matched_range))
    metrics = [curve_metrics(k,v,fleet) for k,v in curves.items()]
    pd.DataFrame(metrics).to_csv(OUT/'comparison_metrics.csv',index=False)
    table = pd.DataFrame({'hour':range(24)})
    for k,v in curves.items():
        table[k+'_MW'] = hourly(v)/1000
        table[k+'_kW_per1000'] = hourly(v)*1000/fleet
        table[k+'_energy_share_pct'] = hourly(v)/hourly(v).sum()*100
    table.to_csv(OUT/'comparison_hourly.csv',index=False)
    quarter = pd.DataFrame({'hour_start':np.arange(96)/4})
    for label,data in [('powertrain_matched',matched_type),('range_proxy_matched',matched_range)]:
        for charger,subtypes in data.items():
            for kind,x in subtypes.items():quarter[label+'_'+charger+'_'+kind+'_kW']=x
    quarter.to_csv(OUT/'evi_matched_15min_components.csv',index=False)
    refcounts = pd.DataFrame([dict(subtype=k,source_count=inputs['fleet_size']*TYPE_SHARES[k.rsplit('_',1)[0]]*
        BODY_SHARES[k.rsplit('_',1)[1]]) for k in parts['home_l1']])
    refcounts.to_csv(OUT/'source_component_vehicle_counts.csv',index=False)
    # Diagnostic powertrain comparison uses each powertrain's own denominator.
    plt.rcParams.update({'font.size':11})
    settings = [('charging_code_phev10','Charging code: PHEV factor 10%','#2166ac','--'),
                ('charging_code_phev20','Charging code: PHEV factor 20%','#00876c','-'),
                ('evi_original','EVI original: 75% BEV','#888888',':'),
                ('evi_powertrain_matched','EVI: BEV/PHEV shares matched','#984ea3','-.'),
                ('evi_powertrain_range_proxy_matched','EVI: shares + range proxy matched','#d95f02','-')]
    fig,axes=plt.subplots(3,1,figsize=(13,13),layout='constrained')
    for key,label,color,style in settings:
        x=hourly(curves[key])
        for ax,y in zip(axes,[x*1000/fleet,x/1000,x/x.sum()*100]):
            ax.plot(range(24),y,label=label,color=color,ls=style,lw=2.2)
    for ax,title,unit in zip(axes,['Per 1,000 PEVs','Total load: 15,440 PEVs for every curve',
                                  'Shape only: each daily energy total = 100%'],
                            ['Hourly-average kW / 1,000 PEVs','Hourly-average MW','Share of daily energy (%)']):
        ax.set(title=title,ylabel=unit,xlabel='Hour (native clock; no time shift)',xlim=(0,23),xticks=range(24),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2)
    fig.suptitle('Baseline charging vs fleet-matched EVI-Pro Lite references',fontsize=17)
    fig.supxlabel('Model fleet: 607 BEVs + 937 PHEVs, expanded 10x. EVI subtype profiles reweighted, not a new travel simulation.\n'
        'Range proxy: 602 BEV100 + 5 BEV250 + 937 PHEV50 (sample). Sedan/SUV retained at 80/20; not verified locally.\n'
        'Weekday; 35 mi/day, 10 C, Home80, immediate charging unchanged. SOC drift remains; no curve fitting.',fontsize=9)
    fig.savefig(OUT/'charging_vs_matched_evi.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    ptable=pd.DataFrame({'hour':range(24)})
    for ax,kind in zip(axes,['BEV','PHEV']):
        pop=counts[kind]
        for data,label,color,style in [(old,'Charging code 10%','#2166ac','--'),(new,'Charging code 20%','#00876c','-')]:
            y=data[kind+'_MW_10x']*1e6/pop
            ax.plot(range(24),y,label=label,color=color,ls=style,lw=2)
            ptable[kind+'_'+label.replace(' ','_')+'_kW_per1000']=y
        y=hourly(total(matched_range,kind.lower()))*1000/pop
        ax.plot(range(24),y,color='#d95f02',lw=2,label='EVI range proxy matched')
        ptable[kind+'_EVI_kW_per1000']=y
        ax.set(title=f'{kind}: {int(pop):,} expanded vehicles',ylabel=f'kW / 1,000 {kind}s',
               xlabel='Hour',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Separate powertrain comparison (own-fleet denominators)',fontsize=15)
    fig.supxlabel('EVI body mix 80/20 Sedan/SUV retained; nominal range mapping is an approximation, not validation.',fontsize=9)
    fig.savefig(OUT/'bev_phev_matched_evi.png',dpi=180);plt.close(fig)
    ptable.to_csv(OUT/'powertrain_comparison_hourly.csv',index=False)
    trips=pd.read_parquet(BASE/'existing_ev_trips.parquet')
    dvmt=float(trips.loc[trips.departureTime.lt(86400),'length'].sum()/1609.344/len(vehicles))
    manifest = dict(status='complete',input_baseline=str(BASE),output=str(OUT),
        EVI_inputs=inputs,expanded_fleet=fleet,sample_powertrain_counts=inventory.powertrain.value_counts().to_dict(),
        daily_miles_per_vehicle_departures_before24h=dvmt,mean_dvmt_kept=35,
        body_mix_retained=BODY_SHARES,powertrain_only_targets=type_targets,range_proxy_targets=range_targets,
        nominal_range_formula='battery_kWh / (primaryFuelConsumptionInJoulePerMeter / 3600) / 1.609344',
        mapping_method='Nearest same-powertrain EVI nominal range; not certified range. No 0.1/0.2 factor applied to range.',
        max_breakdown_sum_error_kW=max_error,current_aggregate_equals_archived=True,
        source_scenario_crosscheck=crosscheck,metrics=metrics,
        limitations=['This is post hoc linear reweighting of scenario component loads, not a fresh EVI travel simulation.',
            'Range uses full reported capacity and nominal energy rate, not a verified usable battery window.',
            'Sedan/SUV split 80/20 retained; vehicleCategory Car does not uniquely identify body style.',
            '10 C and Home80 retained as scenario assumptions; no local weather/access calibration performed.',
            'No direct SF-Oakland geography argument exists in this load-profile request.',
            'Different EVI source mixes have small profile differences; crosscheck reported, not assumed exact invariance.',
            'Baseline charging SOC drift and exclusions remain; no air-quality or power-generation calculation.'])
    for p,sha in protected.items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError(f'Prior result changed: {p}')
    manifest['all_existing_results_preserved']=True
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    (OUT/'matching_summary.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
