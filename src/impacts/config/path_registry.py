from __future__ import annotations

from pathlib import Path
from typing import Optional
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


class SettingsPathResolver:
    """Parses canonical root paths from settings for use by build_registry."""

    def __init__(
        self,
        *,
        beam_input: Optional[Path] = None,
        beam_output: Optional[Path] = None,
        impacts_input: Optional[Path] = None,
        impacts_output: Optional[Path] = None,
    ) -> None:
        self.beam_input = beam_input
        self.beam_output = beam_output
        self.impacts_input = impacts_input
        self.impacts_output = impacts_output

    @classmethod
    def from_settings(
        cls,
        settings,
        config_path: Path | str,
    ) -> "SettingsPathResolver":
        from ..manifest.file_ops import resolve_path

        def _r(raw: Optional[str]) -> Optional[Path]:
            v = resolve_path(raw, config_path) if raw else None
            return Path(v).resolve() if v else None

        beam = getattr(settings, "beam", None)
        return cls(
            beam_input=_r(getattr(beam, "local_input_folder", None)),
            beam_output=_r(getattr(beam, "local_output_folder", None)),
            impacts_input=_r(getattr(settings.impacts, "local_input_folder", None)),
            impacts_output=_r(getattr(settings.impacts, "local_output_folder", None)),
        )


def build_registry(
    settings,
    config_path: Path,
    *,
    extra_roots: Sequence[Path] = (),
) -> PathRegistry:
    """Build a PathRegistry for artifact staging/provisioning.

    Uses SettingsPathResolver for the canonical base roots, then adds
    provisioner-specific region and vehicle-folder subdirectories on top.

    Search order:
      1. beam_input, beam_input/region, beam_input/region/vehicle-tech/emissions
      2. beam_input/vehicle_folder/emissions, beam_input/vehicle_folder/dispersions
      3. beam_output
      4. impacts_input
      5. config_parent, cwd
    """
    resolver = SettingsPathResolver.from_settings(settings, config_path)
    roots: list[Path] = []

    if resolver.beam_input:
        bi = resolver.beam_input
        roots.append(bi)
        region = settings.run.region
        if region:
            roots.append(bi / region)
            roots.append(bi / region / "vehicle-tech" / "emissions")
        vehicle_folder = settings.impacts.population.vehicle_folder
        if vehicle_folder:
            roots.append(bi / vehicle_folder / "emissions")
            roots.append(bi / vehicle_folder / "dispersions")

    if resolver.beam_output:
        roots.append(resolver.beam_output)

    if resolver.impacts_input:
        roots.append(resolver.impacts_input)

    config_parent = Path(config_path).resolve().parent
    roots.append(config_parent)
    cwd = Path.cwd().resolve()
    if cwd != config_parent:
        roots.append(cwd)

    roots.extend(extra_roots)

    return PathRegistry(roots)
