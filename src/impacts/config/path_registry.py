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
    """Canonical entry point for resolving all IMPACTS + BEAM paths.

    Knows which root owns which namespace so callers declare intent explicitly:

      beam_input    → vehicle-tech/, r5/, urbansim/, freight/, dispersions/
      beam_output   → BEAM simulation outputs
      impacts_input → impacts-specific inputs (not shared with beam)
      impacts_output→ impacts run outputs

    Use the named resolve_* methods when the path namespace is known.
    Use resolve() for a general search across all input roots in priority order.
    """

    def __init__(
        self,
        *,
        beam_input: Optional[Path] = None,
        beam_output: Optional[Path] = None,
        impacts_input: Optional[Path] = None,
        impacts_output: Optional[Path] = None,
        extra_roots: tuple[Path, ...] = (),
    ) -> None:
        self.beam_input = beam_input
        self.beam_output = beam_output
        self.impacts_input = impacts_input
        self.impacts_output = impacts_output
        self._extra_roots = extra_roots

    @classmethod
    def from_settings(
        cls,
        settings,
        config_path: Path | str,
        *,
        extra_roots: tuple[str | Path, ...] = (),
    ) -> "SettingsPathResolver":
        """Build a resolver from an ImpactsSettings object and the config file path."""
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
            extra_roots=tuple(Path(r).expanduser().resolve() for r in extra_roots if r),
        )

    @property
    def all_input_roots(self) -> tuple[Path, ...]:
        """All input search roots in priority order (beam_input first)."""
        return tuple(
            r for r in (self.beam_input, self.impacts_input, *self._extra_roots)
            if r is not None
        )

    @property
    def all_roots(self) -> tuple[Path, ...]:
        """All roots including outputs, in priority order."""
        return tuple(
            r for r in (
                self.beam_input,
                self.beam_output,
                self.impacts_input,
                self.impacts_output,
                *self._extra_roots,
            )
            if r is not None
        )

    def _lookup(self, raw: str, roots: Sequence[Optional[Path]]) -> Optional[Path]:
        candidate = Path(str(raw)).expanduser()
        if candidate.is_absolute():
            return candidate.resolve() if candidate.exists() else None
        for root in roots:
            if root is None:
                continue
            localized = (root / candidate).resolve()
            if localized.exists():
                return localized
        return None

    def resolve_beam_input(self, raw: str) -> Optional[Path]:
        """Resolve a path whose root is beam.local_input_folder."""
        return self._lookup(raw, [self.beam_input])

    def resolve_impacts_input(self, raw: str) -> Optional[Path]:
        """Resolve a path whose root is impacts.local_input_folder."""
        return self._lookup(raw, [self.impacts_input])

    def resolve_output(self, raw: str) -> Optional[Path]:
        """Resolve a path whose root is beam.local_output_folder or impacts.local_output_folder."""
        return self._lookup(raw, [self.beam_output, self.impacts_output])

    def resolve(self, raw: str) -> Optional[Path]:
        """Search all input roots in priority order (beam_input first)."""
        return self._lookup(raw, list(self.all_input_roots))

    def resolve_required(self, raw: str, *, label: str = "") -> Path:
        """Like resolve() but raises FileNotFoundError if not found."""
        result = self.resolve(raw)
        if result is None:
            tag = f" '{label}'" if label else ""
            raise FileNotFoundError(
                f"Artifact{tag} not found: '{raw}'. "
                f"Searched: {', '.join(str(r) for r in self.all_input_roots)}"
            )
        return result


def build_registry(settings, config_path: Path) -> PathRegistry:
    """Build a PathRegistry from an ImpactsSettings object.

    Search order:
      1. beam_input, beam_input/region, beam_input/region/vehicle-tech/emissions
      2. beam_input/vehicle_folder/emissions, beam_input/vehicle_folder/dispersions
      3. beam_output
      4. impacts_inputs
      5. config_parent, cwd
    """
    from ..manifest.file_ops import resolve_path

    roots: list[Path] = []

    beam_input = resolve_path(settings.beam.local_input_folder, config_path)
    if beam_input:
        bi = Path(beam_input)
        roots.append(bi)
        region = settings.run.region
        if region:
            roots.append(bi / region)
            roots.append(bi / region / "vehicle-tech" / "emissions")
        vehicle_folder = settings.impacts.population.vehicle_folder
        if vehicle_folder:
            roots.append(bi / vehicle_folder / "emissions")
            roots.append(bi / vehicle_folder / "dispersions")

    beam_output = resolve_path(settings.beam.local_output_folder, config_path)
    if beam_output:
        roots.append(Path(beam_output))

    impacts_inputs = resolve_path(settings.impacts.local_input_folder, config_path)
    if impacts_inputs:
        roots.append(Path(impacts_inputs))

    config_parent = Path(config_path).resolve().parent
    roots.append(config_parent)
    cwd = Path.cwd().resolve()
    if cwd != config_parent:
        roots.append(cwd)

    return PathRegistry(roots)
