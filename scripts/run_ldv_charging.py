"""Preferred LDV charging entry point: unchanged Cristian notebook algorithms.

Supply the original prepared hourly activity table, or explicitly select stops
for an audited one-row-per-stop adapter table. No EVI inputs are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from impacts.pipeline.workflow.charging.ldv_charging import run_ldv_charging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activity-file', type=Path, required=True)
    parser.add_argument('--notebooks', type=Path, required=True)
    parser.add_argument('--input-layout', choices=['hourly_bins', 'stops'], default='hourly_bins')
    parser.add_argument('--fleet-scale', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annualize-representative-day', action='store_true',
                        help='Explicitly export 365 repetitions; does not impose periodic SOC.')
    args = parser.parse_args()
    if not np.isfinite(args.fleet_scale) or args.fleet_scale <= 0:
        parser.error('--fleet-scale must be finite and positive')
    if args.output.exists():
        raise FileExistsError('Choose a new output directory: ' + str(args.output))
    data = (pd.read_parquet(args.activity_file) if args.activity_file.suffix == '.parquet'
            else pd.read_csv(args.activity_file, low_memory=False))
    args.output.mkdir(parents=True)
    result, vehicles, daily, settings, provenance = run_ldv_charging(
        data, notebooks=args.notebooks, seed=args.seed, input_layout=args.input_layout)
    result.to_parquet(args.output / 'charging_activity_audit.parquet', index=False)
    result.loc[result.retained].to_parquet(args.output / 'charging_profiles.parquet', index=False)
    vehicles.to_csv(args.output / 'vehicle_soc_audit.csv', index=False)
    for kind in ('BEV', 'PHEV'):
        daily[kind + '_MW'] = daily[kind + '_sample_kWh'] * args.fleet_scale / 1000
    daily['charging_demand_MWh'] = daily.total_sample_kWh * args.fleet_scale / 1000
    daily.to_csv(args.output / 'charging_curve.csv', index=False)
    daily.to_parquet(args.output / 'representative_day_charging_demand.parquet', index=False)
    kept = vehicles.loc[vehicles.retained]
    gain = float((kept.final_soc_kWh - kept.initial_soc_kWh).sum())
    summary = dict(algorithm='cristian_original_notebook', notebook_parameters=settings,
        notebook_provenance=provenance, input_file=str(args.activity_file.resolve()),
        input_sha256=hashlib.sha256(args.activity_file.read_bytes()).hexdigest(),
        input_layout=args.input_layout, seed=args.seed, fleet_scale_factor=args.fleet_scale,
        modeled_vehicles=len(vehicles), retained_vehicles=len(kept),
        discarded_vehicles=int((~vehicles.retained).sum()),
        daily_energy_MWh=float(daily.charging_demand_MWh.sum()),
        peak_MW=float(daily.charging_demand_MWh.max()), peak_hour=int(daily.charging_demand_MWh.idxmax()),
        sample_SOC_gain_kWh=gain, sample_traction_kWh=float(kept.traction_kWh.sum()),
        periodic_SOC_enforced=False, electric_first_fuel_fallback=False, extra_charger_losses=False,
        evi_used_for_demand=False, input_attributes_inferred=False,
        annualization_days=365 if args.annualize_representative_day else None,
        curve_scope='Original single-day charging; retained vehicles; hourly labels folded modulo 24',
        ready_for_full_electrification_pipeline=bool(len(kept)==len(vehicles)
            and abs(gain) <= .01 * max(float(kept.traction_kWh.sum()), 1e-12)),
        compatibility='Original functions; pandas 2 weighted-sampling semantics restored during serial call',
        caveat='Input activity, charger availability and grid-load provenance must be audited separately. '
               'Postcharge filtering matches the notebook; it does not ensure precharge feasibility. '
               'Single-day SOC drift is reported, not corrected. No traffic or power results are overwritten.')
    if args.annualize_representative_day:
        pd.DataFrame(dict(hour_of_year=np.arange(8760),
            charging_demand_MWh=np.tile(daily.charging_demand_MWh.to_numpy(),365))).to_parquet(
                args.output / 'sfbay_hourly_charging_demand.parquet', index=False)
    (args.output / 'beam_charging_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11,5), layout='constrained')
    for col, label in [('BEV_MW','BEV proxy'),('PHEV_MW','PHEV proxy'),('charging_demand_MWh','Total')]:
        y = daily[col].to_numpy()
        ax.plot(np.arange(25), np.r_[y,y[0]], label=label, linewidth=2)
    ax.set(xlim=(0,24), xticks=np.arange(0,25,2), ylim=(0,None), xlabel='Hour of day',
           ylabel='Hourly-average charging power (MW)',
           title=f'Cristian original charging model | {args.fleet_scale:g}x fleet expansion')
    ax.legend(); ax.grid(alpha=.25)
    fig.savefig(args.output / 'charging_curve.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
