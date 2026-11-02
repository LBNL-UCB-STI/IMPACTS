"""Build the current exposure map using the original final web-map template."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


def assignment(source: str, name: str, next_name: str) -> str:
    match = re.search(
        rf"const {name}\s*=\s*(.*?);\s*const {next_name}\s*=",
        source,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not find const {name} before const {next_name}")
    return match.group(1)


def load_emitting_plants(path: Path) -> list[dict[str, object]]:
    emissions = ("PM2_5_kg", "NOx_kg", "SOx_kg", "NH3_kg", "VOC_kg")
    plants: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                values = {key: float(row.get(key) or 0) for key in emissions}
                lon, lat = float(row["LON"]), float(row["LAT"])
            except (KeyError, TypeError, ValueError):
                continue
            if not any(value > 0 for value in values.values()):
                continue
            plants.append(
                {
                    "id": row.get("ORISPL", ""),
                    "name": row.get("PNAME", "Power plant"),
                    "lon": lon,
                    "lat": lat,
                    "generation": float(row.get("added_generation_MWh") or 0),
                    "technology": row.get("marginal_technologies", ""),
                    **values,
                }
            )
    return plants


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise ValueError(f"Original template insertion point not found: {label}")
    return text.replace(old, new, 1)


def build(template: Path, data_fragment: Path, plant_csv: Path, output: Path) -> int:
    outer = template.read_text(encoding="utf-8")
    fragment = data_fragment.read_text(encoding="utf-8")
    fragment_srcdoc = re.search(r'srcdoc="(.*?)"', fragment, flags=re.DOTALL)
    if fragment_srcdoc:
        fragment = html.unescape(fragment_srcdoc.group(1))
    srcdoc_match = re.search(r'srcdoc="(.*?)"', outer, flags=re.DOTALL)
    if not srcdoc_match:
        raise ValueError("The original HTML does not contain an iframe srcdoc")
    inner = html.unescape(srcdoc_match.group(1))

    new_data = assignment(fragment, "DATA", "RAW")
    raw_successor = "POWER_PLANTS" if re.search(r"const POWER_PLANTS\s*=", fragment) else "BASEMAP"
    new_raw = assignment(fragment, "RAW", raw_successor)
    inner = re.sub(
        r"const DATA\s*=\s*.*?;\s*const RAW\s*=",
        f"const DATA = {new_data};\n    const RAW =",
        inner,
        count=1,
        flags=re.DOTALL,
    )
    inner = re.sub(
        r"const RAW\s*=\s*.*?;\s*const BASEMAP\s*=",
        f"const RAW = {new_raw};\n    const BASEMAP =",
        inner,
        count=1,
        flags=re.DOTALL,
    )

    plants = load_emitting_plants(plant_csv)
    plant_json = json.dumps(plants, ensure_ascii=False, separators=(",", ":"))
    inner = replace_once(
        inner,
        '    const BASEMAP =',
        f'    const POWER_PLANTS = {plant_json};\n    const BASEMAP =',
        "power-plant data",
    )
    inner = replace_once(
        inner,
        '      <g id="bay-map-school-layer"></g>',
        '      <g id="bay-map-power-plant-layer"></g>\n      <g id="bay-map-school-layer"></g>',
        "SVG plant layer",
    )
    school_control = '''    <label class="form-check">
      <input class="form-check-input" type="checkbox" id="bay-map-schools" checked>'''
    plant_control = '''    <label class="form-check">
      <input class="form-check-input" type="checkbox" id="bay-map-power-plants" checked>
      <span class="form-check-label">Power plants</span>
    </label>
'''
    inner = replace_once(inner, school_control, plant_control + school_control, "plant toggle")
    inner = replace_once(
        inner,
        '    const schoolLayer = svg.select("#bay-map-school-layer");',
        '    const plantLayer = svg.select("#bay-map-power-plant-layer");\n'
        '    const schoolLayer = svg.select("#bay-map-school-layer");',
        "plant layer binding",
    )
    inner = replace_once(
        inner,
        '    const schoolToggle = root.querySelector("#bay-map-schools");',
        '    const plantToggle = root.querySelector("#bay-map-power-plants");\n'
        '    const schoolToggle = root.querySelector("#bay-map-schools");',
        "plant toggle binding",
    )
    marker_code = '''
    const plantSymbol = d3.symbol().type(d3.symbolTriangle).size(70);
    plantLayer
      .selectAll("path")
      .data(POWER_PLANTS.filter(item => {
        const point = projection([item.lon, item.lat]);
        return Number.isFinite(point?.[0]) && Number.isFinite(point?.[1]);
      }))
      .join("path")
      .attr("d", plantSymbol)
      .attr("transform", item => `translate(${projection([item.lon, item.lat]).join(",")})`)
      .attr("fill", "#2b83ba")
      .attr("stroke", "var(--background)")
      .attr("stroke-width", 0.9)
      .attr("aria-label", item => `Emitting power plant: ${item.name}`)
      .on("mousemove", function(event, item) {
        showTooltip(
          event,
          `<strong>${escapeHtml(item.name)}</strong><br>` +
          `Emitting power plant · ORIS ${escapeHtml(item.id)}<br>` +
          `Added generation: ${Number(item.generation).toLocaleString(undefined, {maximumFractionDigits: 2})} MWh<br>` +
          `PM₂.₅ emissions: ${formatValue(Number(item.PM2_5_kg))} kg`,
          `${item.name} · emitting power plant · added generation ${Number(item.generation).toLocaleString()} MWh`
        );
      })
      .on("mouseleave", hideTooltip);
'''
    inner = replace_once(
        inner,
        '    function deltaValue(item, spec) {',
        marker_code + '\n    function deltaValue(item, spec) {',
        "plant marker rendering",
    )
    inner = replace_once(
        inner,
        '''    function applyDeltaMarkerScreenSize() {
      deltaSchoolLayer.selectAll("circle")''',
        '''    function applyDeltaMarkerScreenSize() {
      schoolLayer.selectAll("circle")
        .attr("r", function() {
          return Number(this.dataset.baseRadius || 0) / currentZoomScale;
        });
      ltcLayer.selectAll("rect")
        .each(function() {
          const side = Number(this.dataset.baseSide || 0) / currentZoomScale;
          d3.select(this)
            .attr("x", -side / 2)
            .attr("y", -side / 2)
            .attr("width", side)
            .attr("height", side);
        });
      plantLayer.selectAll("path")
        .attr("transform", item => {
          const point = projection([item.lon, item.lat]);
          return `translate(${point.join(",")}) scale(${1 / currentZoomScale})`;
        });
      deltaSchoolLayer.selectAll("circle")''',
        "fixed-screen-size facility markers",
    )
    inner = replace_once(
        inner,
        '''      schoolLayer.selectAll("circle")
        .attr("r", item => schoolRadius(Number(item[metric]) * Number(item.weight)))''',
        '''      schoolLayer.selectAll("circle")
        .attr("data-base-radius", item => schoolRadius(Number(item[metric]) * Number(item.weight)))
        .attr("r", item => schoolRadius(Number(item[metric]) * Number(item.weight)))''',
        "school base marker size",
    )
    inner = replace_once(
        inner,
        '''      ltcLayer.selectAll("rect")
        .attr("x", item => -ltcSide(Number(item[metric]) * Number(item.weight)) / 2)''',
        '''      ltcLayer.selectAll("rect")
        .attr("data-base-side", item => ltcSide(Number(item[metric]) * Number(item.weight)))
        .attr("x", item => -ltcSide(Number(item[metric]) * Number(item.weight)) / 2)''',
        "LTC base marker size",
    )
    inner = replace_once(
        inner,
        '''      ltcLayer.style("display", ltcToggle.checked ? null : "none");
      updateDeltaLayer(metric);''',
        '''      ltcLayer.style("display", ltcToggle.checked ? null : "none");
      updateDeltaLayer(metric);
      applyDeltaMarkerScreenSize();''',
        "refresh fixed marker sizes",
    )
    inner = replace_once(
        inner,
        '      schoolLayer.style("display", schoolToggle.checked ? null : "none");',
        '      plantLayer.style("display", plantToggle.checked ? null : "none");\n'
        '      schoolLayer.style("display", schoolToggle.checked ? null : "none");',
        "plant display state",
    )
    inner = replace_once(
        inner,
        '    schoolToggle.addEventListener("change", () => {',
        '    plantToggle.addEventListener("change", () => {\n'
        '      plantLayer.style("display", plantToggle.checked ? null : "none");\n'
        '    });\n'
        '    schoolToggle.addEventListener("change", () => {',
        "plant toggle event",
    )
    inner = replace_once(
        inner,
        '    <span id="bay-map-concentration-facility-legend">',
        '    <span><span style="display:inline-block;color:#2b83ba;font-size:16px;line-height:0;transform:translateY(1px)">▲</span> Emitting power plant</span>\n'
        '    <span id="bay-map-concentration-facility-legend">',
        "plant legend",
    )

    escaped_inner = html.escape(inner, quote=True)
    result = outer[: srcdoc_match.start(1)] + escaped_inner + outer[srcdoc_match.end(1) :]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")
    return len(plants)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--data-fragment", required=True, type=Path)
    parser.add_argument("--plant-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = build(args.template, args.data_fragment, args.plant_csv, args.output)
    print(f"Wrote {args.output} with {count} emitting power plants")


if __name__ == "__main__":
    main()
