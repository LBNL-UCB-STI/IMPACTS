from pathlib import Path

import pytest
import yaml

from impacts.manifest.file_ops import load_structured_file


def test_load_structured_file_uses_yaml_semantics(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("values: [one, two]\nquoted: \"001\"\n", encoding="utf-8")

    assert load_structured_file(manifest) == {"values": ["one", "two"], "quoted": "001"}


def test_load_structured_file_rejects_malformed_yaml(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("values: [one, two\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_structured_file(manifest)
