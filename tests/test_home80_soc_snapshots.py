import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('soc_analysis', Path(__file__).resolve().parents[1]/'scripts/plot_home80_soc_snapshots.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
snapshot = module.snapshot


def test_overnight_charge_and_clock_wrap():
    # Start at 08:00 with 10 kWh; finish driving at 18:00 with 2 kWh;
    # replenish 8 kWh over 8 hours, ending at 02:00 next day.
    args=([8*3600],[18*3600],[8],[8],10,[8],[8*3600],10)
    assert snapshot(0,*args)[0] == pytest.approx(8)
    assert snapshot(8*3600,*args)[0] == pytest.approx(10)
    assert snapshot(12*3600,*args)[0] == pytest.approx(6.8)
    assert snapshot(18*3600,*args)[0] == pytest.approx(2)
    assert snapshot(24*3600,*args)[0] == snapshot(0,*args)[0]


def test_phev_exhausts_electricity_before_arrival():
    args=([8*3600],[10*3600],[16],[8],10,[8],[4*3600],10)
    assert snapshot(8.5*3600,*args)[0] == pytest.approx(6)
    assert snapshot(9.5*3600,*args)[0] == pytest.approx(2)
    assert snapshot(9.5*3600,*args)[1] == 'driving_after_electric_energy_exhausted'
    assert snapshot(9.5*3600,*args,arrival_debit=True)[0] == pytest.approx(10)
    assert snapshot(12*3600,*args)[0] == pytest.approx(6)


def test_zero_duration_trip_has_instant_debit():
    args=([8*3600],[8*3600],[2],[2],10,[2],[3600],10)
    assert snapshot(8*3600,*args)[0] == pytest.approx(8)
    assert snapshot(8.5*3600,*args)[0] == pytest.approx(9)


def test_consecutive_trips_no_fictitious_charge():
    args=([8*3600,9*3600],[9*3600,10*3600],[2,3],[2,3],10,[0,5],[0,5*3600],10)
    assert snapshot(9*3600,*args)[0] == pytest.approx(8)
    assert snapshot(9.5*3600,*args)[0] == pytest.approx(6.5)
    assert snapshot(15*3600,*args)[0] == pytest.approx(10)


def test_invalid_cycles_rejected():
    with pytest.raises(ValueError,match='balance'):
        snapshot(0,[3600],[7200],[2],[2],10,[1],[3600],10)
    with pytest.raises(ValueError,match='parking'):
        snapshot(0,[3600],[7200],[2],[2],10,[2],[86400],10)


def test_periodicity_at_every_hour():
    args=([8*3600],[18*3600],[8],[8],10,[8],[8*3600],10)
    for t in np.arange(0,86400,900):
        assert snapshot(t,*args)[0] == pytest.approx(snapshot(t+86400,*args)[0])
