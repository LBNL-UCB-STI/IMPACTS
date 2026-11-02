"""Use native EVI composition, equal residential L1/L2, with fleet scaling only."""
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from match_baseline_evi_profiles import unpack, total, hourly, CHARGERS
from plot_latest_home_charging_starts import BASE, PARENT
from plot_charging_load_closed_day import close_day


def main():
    reference = BASE / 'evi-home-l1-l2-sensitivity-20261012'
    sources = [reference / 'Equal_breakdown.json', reference / 'Equal_aggregate.json',
               PARENT / 'comparison_hourly.csv']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    breakdown, aggregate = [json.loads(p.read_text()) for p in sources[:2]]
    assert breakdown['http_status'] == aggregate['http_status'] == 200
    inputs = breakdown['inputs']
    assert inputs == aggregate['inputs']
    assert inputs['home_power_dist'] == 'Equal' and inputs['pev_dist'] == 'BEV'
    assert inputs['pref_dist'] == 'Home80' and inputs['temp_c'] == 20 and inputs['mean_dvmt'] == 35
    assert inputs['fleet_size'] == 30000
    for record in [breakdown, aggregate]:
        for key, value in inputs.items():
            assert str(record['response']['metadata']['inputs'][key]) == str(value)
    parts = unpack(breakdown['response'])
    aggregate_parts = aggregate['response']['results']['weekday_load_profile']
    errors = {}
    for charger in CHARGERS:
        component_sum = sum(parts[charger].values())
        np.testing.assert_allclose(component_sum, aggregate_parts[charger], atol=.046, rtol=0)
        errors[charger] = float(np.max(np.abs(component_sum - aggregate_parts[charger])))
    fleet = 15440
    factor = fleet / inputs['fleet_size']
    # Apply one uniform fleet-size factor, with no subtype reweighting.
    refs = {kind: hourly(total(parts, None if kind == 'total' else kind.lower())) * factor / 1000
            for kind in ['total', 'BEV', 'PHEV']}
    np.testing.assert_allclose(refs['total'], refs['BEV'] + refs['PHEV'], atol=1e-12)
    saved = pd.read_csv(sources[2]).sort_values('hour_start')
    np.testing.assert_array_equal(saved.hour_start, np.arange(24))
    code = {'total': saved.BEV80_PHEV_Home80_MW.to_numpy(),
            'BEV': saved.BEV_Home80_MW.to_numpy(), 'PHEV': saved.PHEV_Home80_MW.to_numpy()}
    np.testing.assert_allclose(code['total'], code['BEV'] + code['PHEV'], atol=1e-12)
    out = PARENT / 'native-evi-l1-l2-50-50-closed-24h'
    if out.exists():
        raise FileExistsError(f'Preserve prior results: {out}')
    out.mkdir()
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})
    h = np.arange(25)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, layout='constrained')
    for ax, unit_factor, title, unit in zip(axes, [1, 1e6 / fleet],
        ['Total: same 15,440 PEVs', 'Per 1,000 input PEVs'],
        ['Hourly-average grid power (MW)', 'Hourly-average grid power (kW / 1,000 PEVs)']):
        ax.plot(h, close_day(code['total']) * unit_factor, color='#2166ac', lw=2.5,
                label='Charging code: BEV Home80 + PHEV Home80')
        ax.plot(h, close_day(refs['total']) * unit_factor, color='#d95f02', lw=2.5, ls='--',
                label='EVI native: home L1/L2 50:50 (75% BEV / 25% PHEV)')
        ax.set(title=title, ylabel=unit, xlim=(0, 24), ylim=(0, None), xticks=h)
        ax.grid(alpha=.22)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[1].set_xlabel('Hour of day (24:00 repeats 00:00 for display)')
    fig.suptitle('Charging load vs native EVI-Pro Lite', fontsize=17)
    fig.supxlabel('EVI: weekday, 20 C, 35 mi/day, Home80; residential L1/L2 50:50; workplace L1/L2 20:80.\n'
                  'Native EVI 30,000-vehicle profile scaled to 15,440; no composition reweighting. Code remains L1/L2 20:80.\n'
                  '24:00 is display-only; daily energy uses 24 bins. Charging model and events unchanged.', fontsize=10)
    fig.savefig(out / 'charging_load_native_evi_0_24h.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, kind, code_count, evi_count in zip(axes, ['BEV', 'PHEV'], [6070, 9370], [11580, 3860]):
        ax.plot(h, close_day(code[kind]) * 1e6 / code_count, color='#2166ac', lw=2.5,
                label=f'Charging code: {kind} Home80')
        ax.plot(h, close_day(refs[kind]) * 1e6 / evi_count, color='#d95f02', ls='--', lw=2.5,
                label='EVI native: home L1/L2 50:50')
        ax.set(title=kind, xlabel='Hour (24:00 repeats 00:00)', ylabel=f'Grid kW / 1,000 {kind}s',
               xlim=(0, 24), ylim=(0, None), xticks=np.arange(0, 25, 2))
        ax.grid(alpha=.22)
        ax.legend(fontsize=10)
    fig.suptitle('BEV / PHEV Home80 vs native EVI-Pro Lite', fontsize=16)
    fig.supxlabel('Own powertrain denominators: code 6,070 BEVs / 9,370 PHEVs; EVI 11,580 BEVs / 3,860 PHEVs.\n'
                  'Native EVI range-class mix retained. Home L1/L2: EVI 50:50; code 20:80. No time shift.', fontsize=10)
    fig.savefig(out / 'bev_phev_native_evi_0_24h.png', dpi=180)
    plt.close(fig)
    metrics = {name: dict(daily_MWh=float(x.sum()), peak_MW=float(x.max()), peak_hour=int(x.argmax()))
               for name, x in [('charging_code', code['total']), ('EVI_native', refs['total'])]}
    for p in sources:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
    report = dict(EVI_inputs=inputs, source_hashes=hashes, source_fleet=30000, comparison_fleet=fleet,
                  fleet_size_factor=factor, subtype_reweighting=False,
                  EVI_native_subtype_shares={'phev20': .10, 'phev50': .15, 'bev100': .25, 'bev250': .50},
                  aggregate_crosscheck_errors_kW=errors, metrics=metrics,
                  hourly_MW={name: {k: v.tolist() for k, v in group.items()} for name, group in [('code', code), ('EVI', refs)]},
                  display_hour24='Repeat hour 0; excluded from energy sums',
                  charging_code_rerun=False, air_quality_rerun=False)
    (out / 'comparison_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
