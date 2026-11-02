"""Replace diagnostic step plots with lines using saved hourly tables only."""
from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_collaborator_truck_charging import ROOT, COLORS, CLASSES, axis_style, savefig, sha


def main():
    out = ROOT/'OUTPUT/collaborator-truck-charging-curves-20261026-final'
    inputs = [out/'hourly_charging_diagnostics.csv', out/'hourly_charging_by_class.csv',
              out/'scenario_summary.csv', out/'flagged_events_preserved.csv', out/'analysis_manifest.json']
    hashes = {str(p): sha(p) for p in inputs}
    h = pd.read_csv(inputs[0])
    c = pd.read_csv(inputs[1])
    summary = pd.read_csv(inputs[2]).set_index(['scenario','powertrain'])
    scenarios = list(h.scenario.unique())
    profiles = {(s,p): d.sort_values('elapsed_hour_start') for (s,p),d in h.groupby(['scenario','powertrain'])}
    names = ['BE_hourly_first24_comparison', 'full_timeline_BE_and_H2FC',
             'BE_hourly_by_vehicle_class', 'BE_hourly_event_counts', 'BE_five_minute_timing_sensitivity']
    backup = out/('archive_step_plots_'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    backup.mkdir(exist_ok=False)
    for name in names:
        shutil.copy2(out/(name+'.png'), backup/(name+'.png'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    note = ('Diagnostic only; BE records; scale = 1; no loss correction; flagged events retained.\n'
            'Points at h represent averages over [h, h+1). Lines join hourly values; no smoothing or midnight wrapping.')

    def line(ax, frame, column='mean_power_MW', first24=True, **kwargs):
        d = frame.loc[frame.elapsed_hour_start.lt(24)] if first24 else frame
        assert d.elapsed_hour_start.is_monotonic_increasing
        ax.plot(d.elapsed_hour_start, d[column], **kwargs)

    def finish(fig, name, footer=note):
        savefig(fig,out,name,footer)

    fig,ax=plt.subplots(figsize=(12,6.5))
    for s,color in zip(scenarios,COLORS):
        n=int(summary.loc[(s,'BE'),'vehicles'])
        line(ax,profiles[s,'BE'],color=color,lw=2.3,label=f'{s} ({n:,} BE vehicles in file)')
    axis_style(ax);ax.set_ylabel('Hourly-average equivalent charging power (MW)');ax.legend()
    fig.suptitle('Collaborator truck charging: first 24 simulation hours\nBE only; unscaled input records',fontsize=16)
    finish(fig,names[0])

    horizon=int(h.elapsed_hour_end.max())
    fig,axes=plt.subplots(2,1,figsize=(12,8.5),sharex=True,sharey=True)
    for ax,pt in zip(axes,['BE','H2FC']):
        for s,color in zip(scenarios,COLORS):
            line(ax,profiles[s,pt],first24=False,color=color,lw=2,label=s)
        for x in [24,48]:
            if x<horizon:ax.axvline(x,color='.4',ls=':',lw=1)
        ax.set_title('BE: battery-electric records' if pt=='BE' else 'H2FC: reported-energy equivalent only; NOT electric-grid demand')
        axis_style(ax,horizon);ax.set_ylabel('Hourly-average equivalent power (MW)');ax.legend()
    axes[0].set_ylim(0,h.mean_power_MW.max()*1.1)
    fig.suptitle('Full elapsed-time charging reconstruction',fontsize=16)
    finish(fig,names[1],'Hourly averages plotted at interval starts; scale = 1; no timezone conversion, folding or added losses.\n'
           'H2FC is isolated, not added to BE. No activity is inferred beyond the supplied records; flagged events retained.')

    ymax=max(profiles[s,'BE'].mean_power_MW.max() for s in scenarios)*1.12
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True,sharey=True)
    for ax,s in zip(axes,scenarios):
        for k,color in zip(CLASSES,['#0072B2','#56B4E9','#009E73','#D55E00','#CC79A7']):
            d=c.loc[c.scenario.eq(s)&c.powertrain.eq('BE')&c.vehicle_class.eq(k)].sort_values('elapsed_hour_start')
            line(ax,d,color=color,lw=1.8,label=k)
        line(ax,profiles[s,'BE'],color='black',lw=2.5,label='BE total')
        axis_style(ax);ax.set_title(s);ax.set_ylabel('Hourly-average power (MW)');ax.legend(ncol=6,fontsize=9)
    axes[0].set_ylim(0,ymax)
    fig.suptitle('BE charging by vehicle class: first 24 hours',fontsize=16)
    finish(fig,names[2])

    fig,axes=plt.subplots(2,1,figsize=(12,8.5),sharex=True)
    for s,color in zip(scenarios,COLORS):
        line(axes[0],profiles[s,'BE'],column='event_starts',color=color,lw=2,label=s)
        line(axes[1],profiles[s,'BE'],column='unique_active_vehicles',color=color,lw=2,label=s)
    for ax in axes:axis_style(ax);ax.legend()
    axes[0].set_ylabel('Charging-event starts per hour')
    axes[1].set_ylabel('Unique vehicles charging during hour')
    fig.suptitle('BE charging activity: starts and participating vehicles',fontsize=16)
    finish(fig,names[3],'Hourly counts plotted at interval starts; unscaled. Unique vehicles counted once per hour despite overlapping sessions.\n'
           'Lines connect hourly counts, not continuous observations. Counts are not simultaneous occupancy or total fleet size.')

    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,sharey=True)
    for ax,s in zip(axes,scenarios):
        line(ax,profiles[s,'BE'],color='#0072B2',lw=2,label='Start at time')
        line(ax,profiles[s,'BE'],column='setup5min_mean_power_MW',color='#D55E00',ls='--',lw=1.8,label='Start after 5-min setup')
        axis_style(ax);ax.set_title(s);ax.set_ylabel('Hourly-average power (MW)');ax.legend()
    axes[0].set_ylim(0,max(h.loc[h.powertrain.eq('BE'),'setup5min_mean_power_MW'].max()*1.12,ymax))
    fig.suptitle('Timing sensitivity: placement of the extra 5 minutes',fontsize=16)
    finish(fig,names[4],'Hourly averages plotted at interval starts. Both alternatives preserve event energy; the extra 5 minutes draw no energy.\n'
           'Timing sensitivity only; not a correction of SOC, overlapping sessions or missing demand. Scale = 1.')
    assert all(sha(Path(p))==v for p,v in hashes.items()), 'Numerical data changed'
    report=dict(plot_style='linear segments connecting hourly bin-start values',
        no_recalculation=True,no_smoothing=True,no_midnight_closure=True,
        input_hashes=hashes,input_files_unchanged=True,previous_figures=str(backup),
        updated_figures=[str(out/(n+'.png')) for n in names])
    (out/('line_plot_update_'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
