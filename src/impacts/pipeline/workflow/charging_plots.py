"""Separate views of saved charging curves; never alter demand calculations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TOTAL = 'charging_demand_MWh'


def group_passenger_and_truck_curves(demand):
    """Aggregate existing load into four disjoint road-vehicle groups."""
    daily, components, _ = prepare_separate_curves(demand)
    prefixes = {
        'Passenger vehicles': ('cv_', 'ev_', 'hev_', 'phev_'),
        'LDT': ('ft_ld1_', 'ft_ld3_'),
        'MDT': ('ft_mdv_',),
        'HDT': ('ft_hdv_', 'ft_hdt_'),
    }
    mapping = {
        group: [c for c in components if c.removeprefix('charging_demand_').startswith(starts)]
        for group, starts in prefixes.items()
    }
    if any(not columns for columns in mapping.values()):
        raise ValueError('All four requested vehicle groups must have input columns.')
    selected = [c for columns in mapping.values() for c in columns]
    assert len(selected) == len(set(selected)), 'Vehicle groups must not overlap.'
    excluded = [c for c in components if c not in selected]
    grouped = pd.DataFrame({group: daily[columns].sum(axis=1) for group, columns in mapping.items()})
    np.testing.assert_allclose(grouped.sum(axis=1) + daily[excluded].sum(axis=1), daily[TOTAL], atol=1e-9)
    return grouped, mapping, excluded


def write_grouped_charging_curve(demand, output_dir, *, split=False):
    """Plot four curves together or split passenger and trucks; retain energy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grouped, mapping, excluded = group_passenger_and_truck_curves(demand)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    styles = dict(zip(grouped, [('C0', 'o', '-'), ('C1', 's', '--'), ('C2', '^', '-.'), ('C3', 'D', '-')]))
    panels = [(
        'passenger_ldt_mdt_hdt_charging_curve.png', list(grouped),
        'SF Bay representative-day charging demand',
        'LDT = ld1 + ld3; MDT = mdv; HDT = hdv + hdt. Passenger vehicles combine cv / ev / hev / phev.',
    )]
    if split:
        panels = [
            ('passenger_charging_curve.png', ['Passenger vehicles'],
             'SF Bay passenger-vehicle charging demand',
             'Representative day; passenger vehicles combine cv / ev / hev / phev.'),
            ('truck_ldt_mdt_hdt_charging_curve.png', ['LDT', 'MDT', 'HDT'],
             'SF Bay truck charging demand',
             'Representative day; LDT = ld1 + ld3; MDT = mdv; HDT = hdv + hdt.'),
        ]
    images = []
    for filename, groups, title, note in panels:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for group in groups:
            color, marker, linestyle = styles[group]
            ax.plot(grouped.index, grouped[group], label=group, color=color,
                    marker=marker, linestyle=linestyle, markersize=4.5, linewidth=2.6)
        peak = float(grouped[groups].to_numpy().max())
        ax.set(title=title, xlabel='Hour of day', ylabel='Charging demand (MW)',
               xlim=(0, 23), ylim=(0, peak * 1.15 if peak else 1), xticks=range(0, 24, 2))
        ax.grid(alpha=.25)
        ax.legend(loc='upper left', frameon=False, fontsize=12)
        fig.text(.5, .055, note, ha='center', fontsize=9)
        scope_note = ('Supplied load only; unmet truck energy excluded.' if 'HDT' in groups
                      else 'Existing all-electric passenger scenario; unchanged simulated load.')
        fig.text(.5, .025, scope_note + ' Transit, bicycle and other modes are not shown.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, 1))
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)
        images.append(filename)
    grouped.reset_index().to_parquet(output_dir / 'grouped_daily_charging_curve.parquet', index=False)
    manifest = dict(image=images[0], images=images, groups=mapping, excluded_columns=excluded,
                    daily_MWh={c: float(grouped[c].sum()) for c in grouped},
                    peak_MW={c: float(grouped[c].max()) for c in grouped},
                    hourly_daily_MWh=grouped.to_dict(orient='list'),
                    values_rescaled=False, axis=('Independent passenger/truck linear MW axes; truck classes share one axis'
                                                if split else 'Common linear MW axis'),
                    scope='Existing simulated supplied load; unmet truck energy excluded')
    (output_dir / 'grouping_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def prepare_separate_curves(demand):
    columns = [c for c in demand if c.startswith('charging_demand_') and c.endswith('_MWh')]
    if TOTAL not in columns or 'hour_of_year' not in demand:
        raise ValueError('Hourly charging demand and hour_of_year are required.')
    values = demand[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Charging curves must be finite and nonnegative.')
    components = [c for c in columns if c != TOTAL]
    np.testing.assert_allclose(demand[components].sum(axis=1), demand[TOTAL], atol=1e-9)
    ranked = sorted(components, key=lambda c: float(demand[c].sum()), reverse=True)
    daily = demand.assign(hour_of_day=demand.hour_of_year % 24).groupby('hour_of_day')[columns].mean()
    if daily.index.tolist() != list(range(24)):
        raise ValueError('All 24 clock hours must be present.')
    daily['Other'] = daily[ranked[8:]].sum(axis=1)
    displayed = ranked[:8] + (['Other'] if len(ranked) > 8 else []) + [TOTAL]
    return daily, ranked, displayed


def write_separate_charging_curves(demand, output_dir, *, labels=None,
                                    scenario_note='Supplied load only; unmet truck energy excluded'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    daily, ranked, displayed = prepare_separate_curves(demand)
    labels = labels or {}
    colors = {c: f'C{i}' for i, c in enumerate(ranked[:8])}
    colors.update({'Other': 'C8', TOTAL: 'black'})

    def label(column):
        if column == TOTAL:
            return 'Total fleet'
        if column == 'Other':
            return f'Other ({len(ranked[8:])} types combined)'
        token = column.removeprefix('charging_demand_').removesuffix('_MWh')
        return labels.get(token, token)

    def draw(ax, column, compact=False):
        series = daily[column]
        peak = float(series.max())
        ax.plot(daily.index, series, color=colors.get(column, 'C0'), linewidth=2.4,
                marker='o', markersize=3, linestyle='--' if column == 'Other' else '-')
        ax.set(xlim=(0, 23), ylim=(0, peak * 1.17 if peak else 1),
               xticks=range(0, 24, 3), xlabel='Hour of day', ylabel='Charging demand (MW)')
        ax.set_title(label(column), fontsize=11 if compact else 14)
        ax.grid(alpha=.25)
        ax.text(.02, .94, f'Peak: {peak:.4g} MW | Daily energy: {series.sum():.4g} MWh',
                transform=ax.transAxes, va='top', fontsize=8 if compact else 10)
        if 0 < peak < .01:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    products = []
    for column in ranked + (['Other'] if len(ranked) > 8 else []) + [TOTAL]:
        token = column.removeprefix('charging_demand_').removesuffix('_MWh')
        if column == TOTAL:
            token = 'total_fleet'
        elif column == 'Other':
            token = 'other_44_types' if len(ranked[8:]) == 44 else 'other_types'
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        draw(ax, column)
        fig.text(.5, .025, f'{scenario_note}\nIndependent y-axis range; original values retained.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, 1))
        path = output_dir / f'{token}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(dict(column=column, label=label(column), file=path.name,
                             peak_MW=float(daily[column].max()), daily_MWh=float(daily[column].sum())))
    rows = int(np.ceil(len(displayed) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, rows * 2.9), squeeze=False)
    for ax, column in zip(axes.ravel(), displayed):
        draw(ax, column, compact=True)
    for ax in axes.ravel()[len(displayed):]:
        ax.set_visible(False)
    fig.suptitle('Original legend series - separate charging curves\nIndependent y-axis ranges; compare axis values, not line heights', fontsize=15)
    fig.text(.5, .007, scenario_note, ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .02, 1, .95), h_pad=2.5)
    overview = output_dir / 'original_series_separate_panels.png'
    fig.savefig(overview, dpi=180)
    plt.close(fig)
    manifest = dict(component_count=len(ranked), original_series_count=len(displayed),
                    scenario_note=scenario_note, overview=overview.name, curves=products,
                    values_changed=False, independent_zero_based_y_axes=True)
    (output_dir / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    lines = ['# Separate charging curves', '', scenario_note, '',
             'These images retain the existing calculated loads. Each y-axis starts at zero',
             'but has its own upper limit. Line heights are not directly comparable across panels.', '',
             '[Original legend: separate panels](original_series_separate_panels.png)', '',
             'The original blue curve is `cv-L1-17000-2019`: gasoline passenger vehicles',
             'in the source events (car/car_hov2/car_hov3), treated as electric in this scenario.',
             'It is not a freight truck category. Nontruck demand uses the unchanged legacy model.', '',
             '| Series | Peak MW | Daily MWh | Image |', '|---|---:|---:|---|']
    lines.extend(f"| {p['label']} | {p['peak_MW']:.6g} | {p['daily_MWh']:.6g} | [PNG]({p['file']}) |" for p in products)
    (output_dir / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return manifest
