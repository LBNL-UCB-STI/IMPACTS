"""Run hourly plant allocation and ISRM using user-supplied local assets.

Requires charging_demand/sfbay_hourly_charging_demand.parquet (8760 hour_of_year,
charging_demand_MWh rows) and beam_charging_summary.json. Demand must already
contain the intended fleet expansion and grid-side charging losses.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demand-dir', type=Path, required=True)
    p.add_argument('--baseline-dir', type=Path, required=True,
                   help='Contains preprocess/inmap_study_area.parquet and county_boundaries.gpkg')
    p.add_argument('--breathe-root', type=Path, required=True)
    p.add_argument('--isrm-store', required=True)
    p.add_argument('--source-layer-index', type=int, choices=[0,1], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--allow-incomplete-charging-scenario', action='store_true')
    p.add_argument('--emissions-only', action='store_true',
                   help='Write complete hourly/annual emissions without computing concentrations.')
    p.add_argument('--allocation-method', choices=['historical_generation_share', 'cambium_marginal_technology'], default='historical_generation_share')
    p.add_argument('--electricity-year', type=int, default=2018)
    p.add_argument('--balancing-area', default='p9')
    p.add_argument('--cambium-scenario', default='LowRECost')
    p.add_argument('--egrid-year', type=int)
    p.add_argument('--grid-loss-region', default='Western')
    p.add_argument('--missing-rate-policy', choices=['error', 'subregion'], default='error')
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory: ' + str(args.output_dir))
    from impacts.pipeline.workflow.hourly_power import run_hourly_power
    run_hourly_power(config=dict(enabled=True,
        hourly_demand_file=str(args.demand_dir.resolve() / 'sfbay_hourly_charging_demand.parquet'),
        charging_summary_file=str(args.demand_dir.resolve() / 'beam_charging_summary.json'),
        breathe_root=str(args.breathe_root.resolve()),
        county_boundaries_file=str(args.baseline_dir.resolve() / 'preprocess/county_boundaries.gpkg'),
        source_layer_index=args.source_layer_index,
        allocation_method=args.allocation_method, electricity_year=args.electricity_year,
        balancing_area=args.balancing_area, cambium_scenario=args.cambium_scenario,
        egrid_year=args.egrid_year, grid_loss_region=args.grid_loss_region,
        missing_rate_policy=args.missing_rate_policy,
        allow_incomplete_charging_scenario=args.allow_incomplete_charging_scenario),
        isrm_url=args.isrm_store,
        inmap_grid_path=str(args.baseline_dir.resolve() / 'preprocess/inmap_study_area.parquet'),
        output_dir=args.output_dir.resolve() / 'power_generation',
        compute_concentrations=not args.emissions_only)


if __name__ == '__main__':
    main()
