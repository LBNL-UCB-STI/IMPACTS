from __future__ import annotations

from pathlib import Path
from typing import Sequence


class PathRegistry:
    """Locates artifacts by searching registered root directories in order."""

    def __init__(self, roots: Sequence[Path]):
        self._roots = [Path(r).resolve() for r in roots if r]

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def locate(self, name: str) -> Path | None:
        """Return the first existing match for name under the roots, or None."""
        raw = Path(str(name)).expanduser()
        if raw.is_absolute():
            return raw.resolve() if raw.exists() else None
        for root in self._roots:
            candidate = (root / raw).resolve()
            if candidate.exists():
                return candidate
        return None

    def locate_required(self, name: str, *, label: str = "") -> Path:
        """Like locate but raises FileNotFoundError if not found."""
        result = self.locate(name)
        if result is None:
            tag = f" '{label}'" if label else ""
            raise FileNotFoundError(
                f"Artifact{tag} not found: '{name}'. "
                f"Searched: {', '.join(str(r) for r in self._roots)}"
            )
        return result


def build_registry(settings, config_path: Path) -> PathRegistry:
    """Build a PathRegistry from an ImpactsSettings object."""
    from ..manifest.file_ops import resolve_path

    roots: list[Path] = []

    beam_input = resolve_path(settings.beam.local_input_folder, config_path)
    if beam_input:
        roots.append(Path(beam_input))
        region = settings.run.region
        if region:
            roots.append(Path(beam_input) / region / "vehicle-tech" / "emissions")
        vehicle_folder = settings.impacts.population.vehicle_folder
        if vehicle_folder:
            roots.append(Path(beam_input) / vehicle_folder / "emissions")

    beam_output = resolve_path(settings.beam.local_output_folder, config_path)
    if beam_output:
        roots.append(Path(beam_output))

    impacts_input = resolve_path(settings.impacts.local_input_folder, config_path)
    if impacts_input:
        roots.append(Path(impacts_input))

    return PathRegistry(roots)
