# Consist For IMPACTS

IMPACTS reads the exact source paths selected for its run and records them in its preprocessing manifest. It may register inputs with Consist for provenance, but it does not query Consist to discover or substitute a newer upstream file. PILATES passes run-specific BEAM paths into the mounted workspace; standalone runs use the paths configured in settings.

## Upstream inputs

- `network`: `network.csv.gz` or `network.parquet` under `beam.local_output_folder`. Duplicate candidates are an error.
- `emissions_skims_input` and optional `events_input`: selected from the single `ITERS` tree under `beam.local_output_folder`. If multiple `ITERS` trees are present, IMPACTS asks for a narrower output root. Within that tree, it selects the highest numbered iteration.
- `osm_network`: PILATES supplies the exact PBF path and optional Consist artifact ID in `impacts.emissions.osm_network_file` and `osm_network_artifact_id`. Standalone IMPACTS uses the first top-level `.pbf` by filename in `beam.router_directory`, matching PILATES's R5 choice; `.vex` files are logged and ignored.
- Exposure population: read the raw year-scoped UrbanSim HDF5 selected by PILATES. IMPACTS performs resident-to-grid assignment; `impacts.population.passenger_folder` remains for fleet generation and is not an exposure fallback.
- Other configured inputs, including vehicle type tables, EMFAC data, dispersion grids, and ISRM, are resolved from their declared settings and recorded in the manifest.

IMPACTS generates its EMFAC activity and fleet tables from the configured EMFAC source archive. Generated intermediates and outputs remain under the configured IMPACTS workspace.
