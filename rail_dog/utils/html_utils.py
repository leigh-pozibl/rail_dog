PLOTLY_SRC = "https://cdn.plot.ly/plotly-2.27.0.min.js"
# For offline use: download the file above, place it next to the HTML, and set:
# PLOTLY_SRC = "plotly-2.27.0.min.js"


def tqi_trend_html(data_json: str, line_id_json: str, mapbox_token: str = "") -> str:
    """Return a self-contained HTML string for the TQI trend explorer.

    Args:
        mapbox_token: Optional Mapbox public token. When provided, uses
                      'outdoors-v12' style (better for rail/terrain). Without
                      a token, falls back to open-street-map tiles.
    """
    _mapbox_style = "mapbox://styles/mapbox/outdoors-v12" if mapbox_token else "open-street-map"
    _mapbox_token_js = f'Plotly.setPlotConfig({{mapboxAccessToken: "{mapbox_token}"}});' if mapbox_token else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TQI Trend Explorer</title>
<script src="{PLOTLY_SRC}"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f5f6fa; color: #2c3e50; }}
  header {{ background: #2c3e50; color: #fff; padding: 16px 24px; }}
  header h1 {{ font-size: 1.4rem; font-weight: 600; }}
  .controls {{ display: flex; gap: 16px; padding: 16px 24px;
               background: #fff; border-bottom: 1px solid #dde; flex-wrap: wrap; }}
  .control-group {{ display: flex; flex-direction: column; gap: 4px; min-width: 200px; }}
  label {{ font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
           letter-spacing: 0.05em; color: #7f8c8d; }}
  select {{ padding: 8px 10px; border: 1px solid #ccd; border-radius: 6px;
            font-size: 0.9rem; background: #fff; cursor: pointer; min-width: 220px; }}
  select[multiple] {{ height: 120px; }}
  .summary-bar {{ display: flex; gap: 10px; padding: 10px 24px; background: #fff;
                  border-bottom: 1px solid #dde; flex-wrap: wrap; align-items: center; }}
  .summary-bar .total {{ font-size: 0.8rem; font-weight: 600; color: #7f8c8d; margin-right: 4px; }}
  .summary-pill {{ padding: 3px 12px; border-radius: 14px; font-size: 0.78rem;
                   font-weight: 600; border: 1px solid transparent; }}
  .summary-divider {{ font-size: 0.75rem; font-weight: 600; color: #7f8c8d;
                      margin-left: 8px; white-space: nowrap; }}
  /* Map + chart side by side */
  .viz-row {{ display: flex; gap: 16px; padding: 20px 24px; align-items: stretch; }}
  #map {{ flex: 3; height: 520px; background: #fff; border-radius: 8px;
          box-shadow: 0 1px 4px #0001; display: none; }}
  #chart-wrap {{ flex: 2; min-width: 0; height: 520px; background: #fff;
                 border-radius: 8px; box-shadow: 0 1px 4px #0001; padding: 12px; }}
  /* When map is hidden, chart expands to full width */
  #map[style*="display: none"] + #chart-wrap,
  #map:not([style]) + #chart-wrap {{ flex: 1; }}
  .main {{ padding: 0 24px 20px 24px; }}
  .metrics {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
  .metric-card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px #0001;
                  padding: 14px 20px; min-width: 180px; }}
  .metric-card .chainage {{ font-size: 0.75rem; color: #7f8c8d; margin-bottom: 4px;
                             white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                             max-width: 220px; }}
  .metric-card .tqi-val {{ font-size: 1.8rem; font-weight: 700; }}
  .metric-card .meta {{ font-size: 0.8rem; margin-top: 4px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.75rem; font-weight: 600; margin-right: 4px; }}
  .badge-good         {{ background: #d5f5e3; color: #1e8449; }}
  .badge-satisfactory {{ background: #fef9e7; color: #9a7d0a; }}
  .badge-fair,
  .badge-moderate     {{ background: #fdebd0; color: #a04000; }}
  .badge-poor         {{ background: #fadbd8; color: #922b21; }}
  .badge-critical     {{ background: #c0392b; color: #fff; }}
  .badge-failure      {{ background: #fadbd8; color: #922b21; }}
  .badge-repair       {{ background: #d5f5e3; color: #1e8449; }}
  .trend-up   {{ color: #27ae60; font-weight: 700; }}
  .trend-down {{ color: #e74c3c; font-weight: 700; }}
  .trend-flat {{ color: #7f8c8d; font-weight: 700; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px #0001; }}
  th {{ background: #2c3e50; color: #fff; padding: 10px 14px;
        text-align: left; font-size: 0.8rem; letter-spacing: 0.04em; }}
  td {{ padding: 8px 14px; font-size: 0.85rem; border-bottom: 1px solid #eee; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f4ff; }}
</style>
</head>
<body>
<header><h1>TQI Trend Explorer</h1></header>

<div class="controls">
  <div class="control-group">
    <label>Line</label>
    <select id="lineSelect"></select>
  </div>
  <div class="control-group">
    <label>Chainage ID (hold Ctrl / ⌘ for multiple)</label>
    <select id="idSelect" multiple></select>
  </div>
</div>

<div class="summary-bar" id="summary"></div>

<div class="viz-row">
  <div id="map"></div>
  <div id="chart-wrap"></div>
</div>

<div class="main">
  <div class="metrics" id="metrics"></div>
  <table id="table">
    <thead>
      <tr>
        <th>Chainage ID</th><th>Date</th><th>TQI</th>
        <th>Status</th><th>Trend</th><th>Slope</th><th>R²</th>
        <th>Event</th><th>Event Date</th><th>Trend From</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
</div>

<script>
{_mapbox_token_js}
const ALL_DATA   = {data_json};
const LINE_IDS   = {line_id_json};

const STATUS_COLOURS = {{
  good: "#2ecc71", satisfactory: "#f1c40f", fair: "#f39c12", moderate: "#f39c12",
  poor: "#e74c3c", critical: "#8e0000"
}};

const BREAKPOINT_COLOURS = {{ failure: "#e74c3c", repair: "#27ae60" }};

// Pre-compute latest snapshot per chainage_id for map colouring
const ALL_LATEST = (function() {{
  const m = {{}};
  ALL_DATA.forEach(r => {{
    if (!m[r.chainage_id] || r.collection_date > m[r.chainage_id].collection_date)
      m[r.chainage_id] = r;
  }});
  return Object.values(m);
}})();

const HAS_COORDS = ALL_DATA.some(r => r.mid_coord_lat != null && r.mid_coord_lng != null);
let mapListenerAdded = false;

window.onload = function() {{
  const lineSelect = document.getElementById("lineSelect");
  const idSelect   = document.getElementById("idSelect");

  Object.keys(LINE_IDS).sort().forEach(lc => {{
    const o = document.createElement("option");
    o.value = o.textContent = lc;
    lineSelect.appendChild(o);
  }});

  lineSelect.addEventListener("change", onLineChange);
  idSelect.addEventListener("change", render);
  onLineChange();

  function onLineChange() {{
    const lc = lineSelect.value;
    idSelect.innerHTML = "";
    (LINE_IDS[lc] || []).forEach((cid, i) => {{
      const o = document.createElement("option");
      o.value = o.textContent = cid;
      if (i === 0) o.selected = true;
      idSelect.appendChild(o);
    }});
    renderSummary();
    render();
  }}

  function renderSummary() {{
    const lc = lineSelect.value;
    const linePoints = ALL_LATEST.filter(r => r.line_code === lc);
    const total = linePoints.length;

    const statusCounts = {{}};
    const trendCounts  = {{}};
    linePoints.forEach(r => {{
      const s = (r.status || "unknown").toLowerCase();
      const t = (r.trend  || "unknown").toLowerCase();
      statusCounts[s] = (statusCounts[s] || 0) + 1;
      trendCounts[t]  = (trendCounts[t]  || 0) + 1;
    }});

    const STATUS_ORDER = ["good", "satisfactory", "fair", "moderate", "poor", "critical", "unknown"];
    const TREND_COLOURS = {{ improving: "#27ae60", stable: "#7f8c8d", degrading: "#e74c3c", unknown: "#bdc3c7" }};
    const TREND_ICONS   = {{ improving: "↑", stable: "→", degrading: "↓" }};
    const TREND_ORDER   = ["improving", "stable", "degrading", "unknown"];

    const pill = (label, count, colour) => {{
      const pct = total ? Math.round(count / total * 100) : 0;
      return `<span class="summary-pill" style="background:${{colour}}22;color:${{colour}};border-color:${{colour}}88">`+
             `${{label}}: ${{count}} (${{pct}}%)</span>`;
    }};

    const el = document.getElementById("summary");
    el.innerHTML =
      `<span class="total">Total: ${{total}}</span>` +
      `<span class="summary-divider">Status —</span>` +
      STATUS_ORDER.filter(s => statusCounts[s]).map(s =>
        pill(s.charAt(0).toUpperCase() + s.slice(1), statusCounts[s], STATUS_COLOURS[s] || "#95a5a6")
      ).join("") +
      `<span class="summary-divider">Trend —</span>` +
      TREND_ORDER.filter(t => trendCounts[t]).map(t => {{
        const icon = TREND_ICONS[t] || "";
        return pill(`${{icon}} ${{t.charAt(0).toUpperCase() + t.slice(1)}}`, trendCounts[t], TREND_COLOURS[t]);
      }}).join("");
  }}

  function render() {{
    const selected = Array.from(idSelect.selectedOptions).map(o => o.value);
    if (!selected.length) return;
    const rows = ALL_DATA.filter(r => selected.includes(r.chainage_id))
                         .sort((a, b) => a.collection_date.localeCompare(b.collection_date));
    renderMetrics(rows, selected);
    renderChart(rows, selected);
    renderTable(rows);
    renderMap(selected);
  }}

  function renderMetrics(rows, selected) {{
    const el = document.getElementById("metrics");
    el.innerHTML = "";
    selected.forEach(cid => {{
      const cidRows = rows.filter(r => r.chainage_id === cid);
      if (!cidRows.length) return;
      const last = cidRows[cidRows.length - 1];
      const statusKey = (last.status || "").toLowerCase();
      const trendIcon = {{improving:"↑", degrading:"↓", stable:"→"}}[last.trend?.toLowerCase()] || "";
      const trendCls  = {{improving:"trend-up", degrading:"trend-down", stable:"trend-flat"}}[last.trend?.toLowerCase()] || "";
      el.innerHTML += `
        <div class="metric-card">
          <div class="chainage" title="${{cid}}">${{cid}}</div>
          <div class="tqi-val">${{last.tqi != null ? last.tqi.toFixed(1) : "—"}}</div>
          <div class="meta">
            <span class="badge badge-${{statusKey}}">${{last.status || "—"}}</span>
            <span class="${{trendCls}}">${{trendIcon}} ${{last.trend || "—"}}</span>
          </div>
          <div class="meta" style="color:#999;font-size:0.75rem;margin-top:4px">${{last.collection_date}}</div>
        </div>`;
    }});
  }}

  // Collect all unique breakpoints for the selected chainage IDs.
  // Each row stores the most-recently-detected step change as of that collection date,
  // so scanning all rows surfaces every event that was ever the "latest" breakpoint.
  function collectBreakpoints(rows, selected) {{
    const seen = new Set();
    const bps  = [];
    rows.forEach(r => {{
      if (!r.step_change_date || !r.step_change_type) return;
      const key = `${{r.chainage_id}}|${{r.step_change_date}}|${{r.step_change_type}}`;
      if (seen.has(key)) return;
      seen.add(key);
      bps.push({{
        cid:   r.chainage_id,
        date:  r.step_change_date,
        type:  r.step_change_type,
        label: r.step_change_type === "failure" ? "⚠ Failure" : "✔ Repair",
      }});
    }});
    return bps;
  }}

  function renderChart(rows, selected) {{
    const traces = selected.map(cid => {{
      const cidRows = rows.filter(r => r.chainage_id === cid);
      return {{
        type: "scatter", mode: "markers",
        name: cid,
        x: cidRows.map(r => r.collection_date),
        y: cidRows.map(r => r.tqi),
        text: cidRows.map(r =>
          `${{r.chainage_id}}<br>Date: ${{r.collection_date}}<br>TQI: ${{r.tqi?.toFixed(2)}}<br>`+
          `Status: ${{r.status || "—"}}<br>Trend: ${{r.trend || "—"}}`),
        hoverinfo: "text",
        marker: {{
          size: 10,
          color: selected.length === 1
            ? cidRows.map(r => STATUS_COLOURS[(r.status || "").toLowerCase()] || "#3498db")
            : undefined,
        }},
      }};
    }});

    // Build breakpoint vertical lines and annotation traces
    const breakpoints = collectBreakpoints(rows, selected);
    const shapes      = [];
    const bpTraces    = [];
    const labelledTypes = new Set();

    breakpoints.forEach(bp => {{
      const colour = BREAKPOINT_COLOURS[bp.type] || "#95a5a6";
      shapes.push({{
        type: "line",
        x0: bp.date, x1: bp.date,
        y0: 0, y1: 1, yref: "paper",
        line: {{ color: colour, width: 2, dash: "dash" }},
      }});
      // One invisible scatter trace per unique type for legend entry
      if (!labelledTypes.has(bp.type)) {{
        labelledTypes.add(bp.type);
        bpTraces.push({{
          type: "scatter", mode: "markers",
          name: bp.type === "failure" ? "⚠ Failure event" : "✔ Repair event",
          x: [bp.date], y: [null],
          marker: {{ color: colour, size: 10, symbol: "line-ns", line: {{ color: colour, width: 2 }} }},
          showlegend: true,
        }});
      }}
    }});

    Plotly.react("chart-wrap", [...traces, ...bpTraces], {{
      xaxis: {{ title: "Date", tickformat: "%Y-%m", type: "date" }},
      yaxis: {{ title: "TQI" }},
      shapes,
      legend: {{ orientation: "h", y: -0.15 }},
      margin: {{ t: 20, r: 20, b: 60, l: 60 }},
      hovermode: "closest",
      autosize: true,
    }}, {{responsive: true}});
  }}

  function renderTable(rows) {{
    document.getElementById("tableBody").innerHTML = rows.map(r => {{
      const statusKey  = (r.status || "").toLowerCase();
      const eventKey   = (r.step_change_type || "").toLowerCase();
      const trendIcon  = {{improving:"↑", degrading:"↓", stable:"→"}}[r.trend?.toLowerCase()] || "";
      const eventLabel = r.step_change_type
        ? `<span class="badge badge-${{eventKey}}">${{r.step_change_type}}</span>` : "—";
      return `<tr>
        <td style="font-size:0.8rem">${{r.chainage_id}}</td>
        <td>${{r.collection_date}}</td>
        <td>${{r.tqi != null ? r.tqi.toFixed(2) : "—"}}</td>
        <td><span class="badge badge-${{statusKey}}">${{r.status || "—"}}</span></td>
        <td>${{trendIcon}} ${{r.trend || "—"}}</td>
        <td>${{r.trend_slope != null ? r.trend_slope.toFixed(3) : "—"}}</td>
        <td>${{r.trend_r_squared != null ? r.trend_r_squared.toFixed(3) : "—"}}</td>
        <td>${{eventLabel}}</td>
        <td>${{r.step_change_date || "—"}}</td>
        <td>${{r.trend_segment_start || "—"}}</td>
      </tr>`;
    }}).join("");
  }}

  function renderMap(selected) {{
    if (!HAS_COORDS) return;
    const lc = lineSelect.value;
    const linePoints = ALL_LATEST.filter(r =>
      r.line_code === lc && r.mid_coord_lat != null && r.mid_coord_lng != null);
    const mapEl = document.getElementById("map");
    if (!linePoints.length) {{ mapEl.style.display = "none"; return; }}
    mapEl.style.display = "block";

    const byStatus = {{}};
    linePoints.forEach(r => {{
      const s = (r.status || "unknown").toLowerCase();
      (byStatus[s] = byStatus[s] || []).push(r);
    }});

    const traces = [];
    Object.entries(byStatus).forEach(([status, pts]) => {{
      const colour = STATUS_COLOURS[status] || "#95a5a6";
      const unsel = pts.filter(r => !selected.includes(r.chainage_id));
      const sel   = pts.filter(r =>  selected.includes(r.chainage_id));
      if (unsel.length) traces.push({{
        type: "scattermapbox",
        lat: unsel.map(r => r.mid_coord_lat),
        lon: unsel.map(r => r.mid_coord_lng),
        mode: "markers",
        marker: {{size: 8, color: colour, opacity: 0.7}},
        customdata: unsel.map(r => [r.chainage_id, r.tqi, r.status, r.trend]),
        hovertemplate: "<b>%{{customdata[0]}}</b><br>TQI: %{{customdata[1]:.2f}}<br>Status: %{{customdata[2]}}<br>Trend: %{{customdata[3]}}<extra></extra>",
        name: status, showlegend: true,
      }});
      if (sel.length) traces.push({{
        type: "scattermapbox",
        lat: sel.map(r => r.mid_coord_lat),
        lon: sel.map(r => r.mid_coord_lng),
        mode: "markers",
        marker: {{size: 14, color: colour, opacity: 1.0}},
        customdata: sel.map(r => [r.chainage_id, r.tqi, r.status, r.trend]),
        hovertemplate: "<b>%{{customdata[0]}}</b><br>TQI: %{{customdata[1]:.2f}}<br>Status: %{{customdata[2]}}<br>Trend: %{{customdata[3]}}<extra></extra>",
        name: `${{status}} (selected)`, showlegend: false,
      }});
    }});

    // Preserve current pan/zoom; only use computed centre on first render
    const existingMapbox = mapEl.layout?.mapbox;
    const centreLat = linePoints.reduce((s, r) => s + r.mid_coord_lat, 0) / linePoints.length;
    const centreLon = linePoints.reduce((s, r) => s + r.mid_coord_lng, 0) / linePoints.length;
    const mapCenter = existingMapbox ? existingMapbox.center : {{lat: centreLat, lon: centreLon}};
    const mapZoom   = existingMapbox ? existingMapbox.zoom   : 8;
    Plotly.react("map", traces, {{
      mapbox: {{style: "{_mapbox_style}", center: mapCenter, zoom: mapZoom}},
      margin: {{t: 0, r: 0, b: 0, l: 0}},
      autosize: true,
      legend: {{orientation: "h", y: -0.05}},
    }}, {{responsive: true}});

    if (!mapListenerAdded) {{
      mapEl.on("plotly_click", function(data) {{
        if (!data.points.length) return;
        const cid = data.points[0].customdata[0];
        Array.from(idSelect.options).forEach(o => {{ o.selected = o.value === cid; }});
        render();
      }});
      mapListenerAdded = true;
    }}
  }}
}};
</script>
</body>
</html>"""
