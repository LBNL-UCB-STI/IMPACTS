"""Identify layer-0 source rows needed by the new dispatch, without omitting any."""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from impacts.pipeline.workflow.power_generation import POWER_COMPONENTS, _map_plants_to_source_cells

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'examples/pilates/impacts/impacts_output/beam-charging-fc07-v260419-it3--20260923'
STORE = Path(ISRM_PATH)
sr = zarr.open_group(str(STORE), mode='r')
plants = pd.read_csv(OUT / 'power_generation/plant_allocation/plant_emissions_approx.csv')
mapped = _map_plants_to_source_cells(plants, sr)
mapped['emitting'] = mapped[list(POWER_COMPONENTS)].gt(0).any(axis=1)
mapped['matrix_source_available'] = mapped.isrm_source_cell.isin(np.asarray(sr['source_ids'][:]))
missing = mapped.emitting & ~mapped.matrix_source_available
audit = OUT / 'power_generation/source_coverage'
audit.mkdir(exist_ok=True)
mapped.to_csv(audit / 'plant_source_coverage.csv', index=False)
mapped.loc[missing].to_csv(audit / 'missing_emitting_plants.csv', index=False)
pd.DataFrame({'isrm_source_cell': sorted(mapped.loc[missing, 'isrm_source_cell'].unique())}).to_csv(audit / 'missing_source_ids.csv', index=False)
report = {'matrix': str(STORE), 'source_layer_index': 0, 'receptor_domain': 'SFBAY',
          'emitting_plants': int(mapped.emitting.sum()), 'missing_emitting_plants': int(missing.sum()),
          'required_emitting_source_cells': int(mapped.loc[mapped.emitting, 'isrm_source_cell'].nunique()),
          'missing_emitting_source_cells': int(mapped.loc[missing, 'isrm_source_cell'].nunique()),
          'missing_emissions_fraction': {c: float(mapped.loc[missing, c].sum() / mapped[c].sum()) for c in POWER_COMPONENTS}}
(audit / 'coverage_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
