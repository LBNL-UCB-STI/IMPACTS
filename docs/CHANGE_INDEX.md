# File-level change index

Current November 2 release relative to upstream commit `159a5598210905471a0412609d8414ccf97f355f`. See [FULL_VERSION_REVIEW.md](FULL_VERSION_REVIEW.md) and the latest [charging compatibility update](CRISTIAN_COMPATIBILITY.md). All files and hashes are recorded in `source_inventory.json`.

Test names identify coverage, not complete real-data replay of every script.

## Core model and configuration

| Status | File |
|---|---|
| Modified | [src/impacts/common.py](../src/impacts/common.py) |
| Modified | [src/impacts/config/defaults.py](../src/impacts/config/defaults.py) |
| Modified | [src/impacts/config/settings.py](../src/impacts/config/settings.py) |
| Modified | [src/impacts/config/settings.yaml](../src/impacts/config/settings.yaml) |
| Modified | [src/impacts/manifest/file_ops.py](../src/impacts/manifest/file_ops.py) |
| Modified | [src/impacts/manifest/schema.py](../src/impacts/manifest/schema.py) |
| Modified | [src/impacts/pipeline/emfac/_common.py](../src/impacts/pipeline/emfac/_common.py) |
| Modified | [src/impacts/pipeline/emfac/fleet/step3_map_emfac_atlas.py](../src/impacts/pipeline/emfac/fleet/step3_map_emfac_atlas.py) |
| Modified | [src/impacts/pipeline/postprocess/__init__.py](../src/impacts/pipeline/postprocess/__init__.py) |
| Modified | [src/impacts/pipeline/postprocess/_common.py](../src/impacts/pipeline/postprocess/_common.py) |
| Added | [src/impacts/pipeline/postprocess/assets/human_exposure_webmap.html](../src/impacts/pipeline/postprocess/assets/human_exposure_webmap.html) |
| Modified | [src/impacts/pipeline/postprocess/step4_plot_concentrations.py](../src/impacts/pipeline/postprocess/step4_plot_concentrations.py) |
| Added | [src/impacts/pipeline/postprocess/step8_human_exposure.py](../src/impacts/pipeline/postprocess/step8_human_exposure.py) |
| Added | [src/impacts/pipeline/postprocess/step9_human_exposure_webmap.py](../src/impacts/pipeline/postprocess/step9_human_exposure_webmap.py) |
| Modified | [src/impacts/pipeline/preprocessing/step1_collect_inputs.py](../src/impacts/pipeline/preprocessing/step1_collect_inputs.py) |
| Added | [src/impacts/pipeline/workflow/charging_plots.py](../src/impacts/pipeline/workflow/charging_plots.py) |
| Added | [src/impacts/pipeline/workflow/cristian_charging.py](../src/impacts/pipeline/workflow/cristian_charging.py) |
| Added | [src/impacts/pipeline/workflow/existing_ev_charging.py](../src/impacts/pipeline/workflow/existing_ev_charging.py) |
| Added | [src/impacts/pipeline/workflow/home_priority_charging.py](../src/impacts/pipeline/workflow/home_priority_charging.py) |
| Added | [src/impacts/pipeline/workflow/hourly_power.py](../src/impacts/pipeline/workflow/hourly_power.py) |
| Added | [src/impacts/pipeline/workflow/isrm_near_source.py](../src/impacts/pipeline/workflow/isrm_near_source.py) |
| Added | [src/impacts/pipeline/workflow/native_ev_charging.py](../src/impacts/pipeline/workflow/native_ev_charging.py) |
| Added | [src/impacts/pipeline/workflow/notebook_charging.py](../src/impacts/pipeline/workflow/notebook_charging.py) |
| Added | [src/impacts/pipeline/workflow/phev_home_charging.py](../src/impacts/pipeline/workflow/phev_home_charging.py) |
| Added | [src/impacts/pipeline/workflow/power_generation.py](../src/impacts/pipeline/workflow/power_generation.py) |
| Modified | [src/impacts/pipeline/workflow/prepare_emissions/annualization.py](../src/impacts/pipeline/workflow/prepare_emissions/annualization.py) |
| Added | [src/impacts/pipeline/workflow/prepare_emissions/event_emfac.py](../src/impacts/pipeline/workflow/prepare_emissions/event_emfac.py) |
| Modified | [src/impacts/pipeline/workflow/prepare_emissions/from_events.py](../src/impacts/pipeline/workflow/prepare_emissions/from_events.py) |
| Modified | [src/impacts/pipeline/workflow/prepare_emissions/from_skims.py](../src/impacts/pipeline/workflow/prepare_emissions/from_skims.py) |
| Modified | [src/impacts/pipeline/workflow/step2_compute_inmap_concentrations.py](../src/impacts/pipeline/workflow/step2_compute_inmap_concentrations.py) |
| Modified | [src/impacts/pipeline/workflow/step3_compute_aermod_concentrations.py](../src/impacts/pipeline/workflow/step3_compute_aermod_concentrations.py) |
| Modified | [src/impacts/pipeline/workflow/step4_prepare_exposure.py](../src/impacts/pipeline/workflow/step4_prepare_exposure.py) |
| Added | [src/impacts/pipeline/workflow/truck_charging.py](../src/impacts/pipeline/workflow/truck_charging.py) |
| Modified | [src/impacts/postprocessor.py](../src/impacts/postprocessor.py) |
| Modified | [src/impacts/preprocessor.py](../src/impacts/preprocessor.py) |
| Modified | [src/impacts/runner.py](../src/impacts/runner.py) |

## Research and replay entry points

| Status | File |
|---|---|
| Added | [scripts/add_evi_bev25_reference.py](../scripts/add_evi_bev25_reference.py) |
| Added | [scripts/analyze_collaborator_truck_charging.py](../scripts/analyze_collaborator_truck_charging.py) |
| Added | [scripts/analyze_home80_charging_results.py](../scripts/analyze_home80_charging_results.py) |
| Added | [scripts/analyze_power_vs_road.py](../scripts/analyze_power_vs_road.py) |
| Added | [scripts/audit_baseline_ev_arrivals.py](../scripts/audit_baseline_ev_arrivals.py) |
| Added | [scripts/audit_future_scenario_inputs.py](../scripts/audit_future_scenario_inputs.py) |
| Added | [scripts/audit_original_skims_3km_test.py](../scripts/audit_original_skims_3km_test.py) |
| Added | [scripts/audit_power_source_coverage.py](../scripts/audit_power_source_coverage.py) |
| Added | [scripts/audit_reuse_charging_code.py](../scripts/audit_reuse_charging_code.py) |
| Added | [scripts/build_original_style_exposure_webmap.py](../scripts/build_original_style_exposure_webmap.py) |
| Added | [scripts/combine_ldv_freight_power.py](../scripts/combine_ldv_freight_power.py) |
| Added | [scripts/compare_both_home80_evi20c.py](../scripts/compare_both_home80_evi20c.py) |
| Added | [scripts/compare_charging_0224_evi20c.py](../scripts/compare_charging_0224_evi20c.py) |
| Added | [scripts/compare_charging_code_evi_15k.py](../scripts/compare_charging_code_evi_15k.py) |
| Added | [scripts/compare_collaborator_truck_charging_rmi.py](../scripts/compare_collaborator_truck_charging_rmi.py) |
| Added | [scripts/compare_evi_home_charger_mix.py](../scripts/compare_evi_home_charger_mix.py) |
| Added | [scripts/compare_home_priority_fleet20x.py](../scripts/compare_home_priority_fleet20x.py) |
| Added | [scripts/compare_native_evi_15k_weekday.py](../scripts/compare_native_evi_15k_weekday.py) |
| Added | [scripts/compare_phev_home90_evi_equal.py](../scripts/compare_phev_home90_evi_equal.py) |
| Added | [scripts/compare_phev_home_soc_evi20c.py](../scripts/compare_phev_home_soc_evi20c.py) |
| Added | [scripts/compare_sf_evi_410k.py](../scripts/compare_sf_evi_410k.py) |
| Added | [scripts/compare_unified_ev_energy_evi20c.py](../scripts/compare_unified_ev_energy_evi20c.py) |
| Added | [scripts/continue_beam_charging_pipeline.py](../scripts/continue_beam_charging_pipeline.py) |
| Added | [scripts/continue_events_partial_isrm.py](../scripts/continue_events_partial_isrm.py) |
| Added | [scripts/download_rmi_bay_area_forecast.py](../scripts/download_rmi_bay_area_forecast.py) |
| Added | [scripts/exclude_hf1ep_invalid_trips.py](../scripts/exclude_hf1ep_invalid_trips.py) |
| Added | [scripts/export_original_style_human_exposure_tables.py](../scripts/export_original_style_human_exposure_tables.py) |
| Added | [scripts/gateway_aermod_crop.py](../scripts/gateway_aermod_crop.py) |
| Added | [scripts/inspect_baseline_event_difference.py](../scripts/inspect_baseline_event_difference.py) |
| Added | [scripts/inspect_existing_emission_skims.py](../scripts/inspect_existing_emission_skims.py) |
| Added | [scripts/match_baseline_evi_profiles.py](../scripts/match_baseline_evi_profiles.py) |
| Added | [scripts/plot_charging_load_closed_day.py](../scripts/plot_charging_load_closed_day.py) |
| Added | [scripts/plot_freight_power_bay_outline.py](../scripts/plot_freight_power_bay_outline.py) |
| Added | [scripts/plot_future_fixed_factor_scenarios.py](../scripts/plot_future_fixed_factor_scenarios.py) |
| Added | [scripts/plot_gateway_products.py](../scripts/plot_gateway_products.py) |
| Added | [scripts/plot_grouped_charging_curve.py](../scripts/plot_grouped_charging_curve.py) |
| Added | [scripts/plot_home80_soc_snapshots.py](../scripts/plot_home80_soc_snapshots.py) |
| Added | [scripts/plot_initial_ev_soc.py](../scripts/plot_initial_ev_soc.py) |
| Added | [scripts/plot_latest_home_charging_starts.py](../scripts/plot_latest_home_charging_starts.py) |
| Added | [scripts/plot_native_ev_ldv_truck_evi.py](../scripts/plot_native_ev_ldv_truck_evi.py) |
| Added | [scripts/plot_native_evi_equal_home80.py](../scripts/plot_native_evi_equal_home80.py) |
| Added | [scripts/plot_original_baseline_bc_no2_components.py](../scripts/plot_original_baseline_bc_no2_components.py) |
| Added | [scripts/plot_original_vs_current_home_starts.py](../scripts/plot_original_vs_current_home_starts.py) |
| Added | [scripts/plot_separate_charging_curves.py](../scripts/plot_separate_charging_curves.py) |
| Added | [scripts/rebuild_truck_charging.py](../scripts/rebuild_truck_charging.py) |
| Added | [scripts/render_ces_pwc_manuscript_table.py](../scripts/render_ces_pwc_manuscript_table.py) |
| Added | [scripts/replay_charging_power.py](../scripts/replay_charging_power.py) |
| Added | [scripts/replay_human_exposure.py](../scripts/replay_human_exposure.py) |
| Added | [scripts/replay_reference_case.py](../scripts/replay_reference_case.py) |
| Added | [scripts/replay_skims_emissions.py](../scripts/replay_skims_emissions.py) |
| Added | [scripts/replot_collaborator_truck_charging_lines.py](../scripts/replot_collaborator_truck_charging_lines.py) |
| Added | [scripts/replot_ldv_figures_24h.py](../scripts/replot_ldv_figures_24h.py) |
| Added | [scripts/research_paths.py](../scripts/research_paths.py) |
| Added | [scripts/run_baseline0_collaborator_pipeline.py](../scripts/run_baseline0_collaborator_pipeline.py) |
| Added | [scripts/run_cristian_charging.py](../scripts/run_cristian_charging.py) |
| Added | [scripts/run_events_traffic_pipeline.py](../scripts/run_events_traffic_pipeline.py) |
| Added | [scripts/run_existing_ev_charging_code.py](../scripts/run_existing_ev_charging_code.py) |
| Added | [scripts/run_future_detailed_products.py](../scripts/run_future_detailed_products.py) |
| Added | [scripts/run_future_fixed_factor_scenarios.py](../scripts/run_future_fixed_factor_scenarios.py) |
| Added | [scripts/run_gateway_freight_products.py](../scripts/run_gateway_freight_products.py) |
| Added | [scripts/run_hourly_power.py](../scripts/run_hourly_power.py) |
| Added | [scripts/run_ldv_freight_products.py](../scripts/run_ldv_freight_products.py) |
| Added | [scripts/run_native_ev_baseline.py](../scripts/run_native_ev_baseline.py) |
| Added | [scripts/run_original_skims_3km_test.py](../scripts/run_original_skims_3km_test.py) |
| Added | [scripts/run_phev20_sensitivity.py](../scripts/run_phev20_sensitivity.py) |
| Added | [scripts/run_port_freight_products.py](../scripts/run_port_freight_products.py) |
| Added | [scripts/scale_events_traffic.py](../scripts/scale_events_traffic.py) |
| Added | [scripts/test_home_priority_scenarios.py](../scripts/test_home_priority_scenarios.py) |

## Tests

| Status | File |
|---|---|
| Modified | [tests/conftest.py](../tests/conftest.py) |
| Modified | [tests/test_aermod_concentrations.py](../tests/test_aermod_concentrations.py) |
| Added | [tests/test_charging_continuation.py](../tests/test_charging_continuation.py) |
| Added | [tests/test_charging_plots.py](../tests/test_charging_plots.py) |
| Added | [tests/test_cristian_charging.py](../tests/test_cristian_charging.py) |
| Added | [tests/test_events_emfac.py](../tests/test_events_emfac.py) |
| Added | [tests/test_evi_15k_comparison.py](../tests/test_evi_15k_comparison.py) |
| Added | [tests/test_evi_fleet_matching.py](../tests/test_evi_fleet_matching.py) |
| Added | [tests/test_evi_profile_comparison.py](../tests/test_evi_profile_comparison.py) |
| Added | [tests/test_existing_ev_charging.py](../tests/test_existing_ev_charging.py) |
| Added | [tests/test_future_detailed_products.py](../tests/test_future_detailed_products.py) |
| Added | [tests/test_future_fixed_factor_scenarios.py](../tests/test_future_fixed_factor_scenarios.py) |
| Added | [tests/test_gateway_figures.py](../tests/test_gateway_figures.py) |
| Added | [tests/test_gateway_freight_products.py](../tests/test_gateway_freight_products.py) |
| Added | [tests/test_home80_soc_snapshots.py](../tests/test_home80_soc_snapshots.py) |
| Added | [tests/test_home_priority_charging.py](../tests/test_home_priority_charging.py) |
| Added | [tests/test_hourly_power.py](../tests/test_hourly_power.py) |
| Added | [tests/test_human_exposure_webmap.py](../tests/test_human_exposure_webmap.py) |
| Modified | [tests/test_inmap_concentrations.py](../tests/test_inmap_concentrations.py) |
| Added | [tests/test_isrm_near_source.py](../tests/test_isrm_near_source.py) |
| Added | [tests/test_ldv_freight_products.py](../tests/test_ldv_freight_products.py) |
| Added | [tests/test_native_ev_baseline.py](../tests/test_native_ev_baseline.py) |
| Added | [tests/test_native_ev_charging.py](../tests/test_native_ev_charging.py) |
| Added | [tests/test_native_ev_charging_plots.py](../tests/test_native_ev_charging_plots.py) |
| Added | [tests/test_notebook_charging.py](../tests/test_notebook_charging.py) |
| Added | [tests/test_original_exposure_tables.py](../tests/test_original_exposure_tables.py) |
| Added | [tests/test_phev_home80.py](../tests/test_phev_home80.py) |
| Added | [tests/test_phev_home_charging.py](../tests/test_phev_home_charging.py) |
| Added | [tests/test_plot_concentration_components.py](../tests/test_plot_concentration_components.py) |
| Added | [tests/test_port_freight_products.py](../tests/test_port_freight_products.py) |
| Added | [tests/test_power_generation.py](../tests/test_power_generation.py) |
| Modified | [tests/test_prepare_emissions_from_skims.py](../tests/test_prepare_emissions_from_skims.py) |
| Modified | [tests/test_prepare_exposure.py](../tests/test_prepare_exposure.py) |
| Added | [tests/test_release_regressions.py](../tests/test_release_regressions.py) |
| Modified | [tests/test_terminal_contract.py](../tests/test_terminal_contract.py) |
| Added | [tests/test_traffic_expansion.py](../tests/test_traffic_expansion.py) |
| Added | [tests/test_truck_charging.py](../tests/test_truck_charging.py) |
| Added | [tests/test_unified_ev_energy.py](../tests/test_unified_ev_energy.py) |

## Documentation

| Status | File |
|---|---|
| Added | [docs/CHANGE_INDEX.md](../docs/CHANGE_INDEX.md) |
| Added | [docs/CHANGE_SUMMARY.md](../docs/CHANGE_SUMMARY.md) |
| Added | [docs/CRISTIAN_COMPATIBILITY.md](../docs/CRISTIAN_COMPATIBILITY.md) |
| Added | [docs/FULL_VERSION_REVIEW.md](../docs/FULL_VERSION_REVIEW.md) |
| Added | [docs/HOURLY_POWER_UPDATE.md](../docs/HOURLY_POWER_UPDATE.md) |
| Added | [docs/RELEASE_REVIEW.md](../docs/RELEASE_REVIEW.md) |
| Added | [docs/REPRODUCING.md](../docs/REPRODUCING.md) |
| Added | [docs/VALIDATION.md](../docs/VALIDATION.md) |
| Added | [docs/cristian_compatibility_validation.json](../docs/cristian_compatibility_validation.json) |
| Added | [docs/events_emfac.md](../docs/events_emfac.md) |
| Added | [docs/fc07_events_traffic_scenario.md](../docs/fc07_events_traffic_scenario.md) |
| Added | [docs/hourly_power_validation.json](../docs/hourly_power_validation.json) |
| Added | [docs/source_inventory.json](../docs/source_inventory.json) |
| Added | [docs/validation_environment.json](../docs/validation_environment.json) |
| Added | [docs/validation_results.json](../docs/validation_results.json) |

## Build and repository settings

| Status | File |
|---|---|
| Modified | [.gitignore](../.gitignore) |
| Modified | [README.md](../README.md) |
| Modified | [examples/pilates/.gitignore](../examples/pilates/.gitignore) |
| Modified | [setup.cfg](../setup.cfg) |
