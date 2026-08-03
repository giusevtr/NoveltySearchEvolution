"""Renders a self-contained static HTML page for browsing a population snapshot.

The returned page embeds the sample list inline as JSON and does all filtering/
navigation client-side in vanilla JS, so it works when opened directly via
`file://` — no server, no fetch/CORS issues, no external resources.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

_SAMPLES_PLACEHOLDER = "__SAMPLES_JSON__"

_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Population Snapshot Viewer</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem; background: #111; color: #eee; }
  h1 { font-size: 1.1rem; margin: 0 0 1rem; }
  .controls { display: flex; gap: 1rem; margin-bottom: 1rem; }
  select { background: #222; color: #eee; border: 1px solid #444; padding: 0.3rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th, td { border-bottom: 1px solid #333; padding: 0.4rem 0.6rem; text-align: left; }
  tr.sample-row { cursor: pointer; }
  tr.sample-row:hover { background: #1d1d1d; }
  .badge { padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.75rem; }
  .active { background: #1f5c2b; }
  .inactive { background: #444; }
  .stale { background: #5c4a1f; }
  .rejected { background: #5c1f1f; }
  #detail h2 { font-size: 1rem; }
  pre { background: #1a1a1a; padding: 0.75rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
  .links button { margin: 0.15rem; background: #222; color: #eee; border: 1px solid #444; padding: 0.2rem 0.5rem; cursor: pointer; }
  .links button:hover { background: #2a2a2a; }
  .section-label { font-weight: 600; margin-top: 0.75rem; }
  #layout { display: flex; gap: 1.5rem; align-items: flex-start; }
  #table-col { flex: 1; min-width: 0; }
  #detail { flex: 1; min-width: 0; }
</style>
</head>
<body>
<h1>Population Snapshot Viewer</h1>
<div class="controls">
  <label>Generation: <select id="gen-filter"></select></label>
  <label>Status: <select id="status-filter">
    <option value="">all</option>
    <option value="active">active</option>
    <option value="inactive">inactive</option>
    <option value="stale">stale</option>
    <option value="rejected">rejected</option>
  </select></label>
</div>
<div id="layout">
  <div id="table-col">
    <table>
      <thead><tr><th>id</th><th>status</th><th>gen</th><th>depth</th><th>preview</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <div id="detail"><em>Select a sample to see details.</em></div>
</div>

<script>
const SAMPLES = __SAMPLES_JSON__;
const BY_ID = {};
for (const s of SAMPLES) { BY_ID[s.id] = s; }

function preview(data) {
  const s = (typeof data === "object") ? JSON.stringify(data) : String(data);
  return s.length > 60 ? s.slice(0, 60) + "..." : s;
}

function populateFilters() {
  const gens = [...new Set(SAMPLES.map(s => s.generation))].sort((a, b) => a - b);
  const genSelect = document.getElementById("gen-filter");
  genSelect.innerHTML = '<option value="">all</option>' +
    gens.map(g => `<option value="${g}">${g}</option>`).join("");
}

function renderTable() {
  const gen = document.getElementById("gen-filter").value;
  const status = document.getElementById("status-filter").value;
  const rows = document.getElementById("rows");
  rows.innerHTML = "";
  for (const s of SAMPLES) {
    if (gen !== "" && String(s.generation) !== gen) continue;
    if (status !== "" && s.status !== status) continue;
    const tr = document.createElement("tr");
    tr.className = "sample-row";
    tr.onclick = () => showDetail(s.id);
    tr.innerHTML = `
      <td>${s.id.slice(0, 8)}</td>
      <td><span class="badge ${s.status}">${s.status}</span></td>
      <td>${s.generation}</td>
      <td>${s.depth}</td>
      <td>${preview(s.data)}</td>
    `;
    rows.appendChild(tr);
  }
}

function linkButtons(ids) {
  if (!ids.length) return "<em>none</em>";
  return ids.map(id => {
    const known = BY_ID[id] !== undefined;
    const label = id.slice(0, 8);
    return known
      ? `<button onclick="showDetail('${id}')">${label}</button>`
      : `<button disabled title="not in this snapshot">${label}</button>`;
  }).join(" ");
}

function showDetail(id) {
  const s = BY_ID[id];
  const detail = document.getElementById("detail");
  if (!s) { detail.innerHTML = "<em>Sample not found.</em>"; return; }

  let dataForDump = s.data;
  let promptBlock = "";
  let solutionBlock = "";
  if (s.data && typeof s.data === "object" && !Array.isArray(s.data)) {
    const rest = {...s.data};
    if (rest.prompt !== undefined) {
      promptBlock = `<div class="section-label">Prompt</div><pre>${escapeHtml(rest.prompt)}</pre>`;
      delete rest.prompt;
    }
    if (rest.solution !== undefined) {
      solutionBlock = `<div class="section-label">Solution</div><pre>${escapeHtml(rest.solution)}</pre>`;
      delete rest.solution;
    }
    dataForDump = rest;
  }

  detail.innerHTML = `
    <h2>${s.id}</h2>
    <div><b>status:</b> ${s.status} &nbsp; <b>generation:</b> ${s.generation} &nbsp; <b>depth:</b> ${s.depth}</div>
    ${promptBlock}
    ${solutionBlock}
    <div class="section-label">Data</div>
    <pre>${escapeHtml(JSON.stringify(dataForDump, null, 2))}</pre>
    <div class="section-label">Feedback</div>
    <pre>${s.feedback.length ? escapeHtml(JSON.stringify(s.feedback, null, 2)) : "none"}</pre>
    <div class="section-label">Parents</div>
    <div class="links">${linkButtons(s.parent_ids)}</div>
    <div class="section-label">Children</div>
    <div class="links">${linkButtons(s.child_ids)}</div>
  `;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

document.getElementById("gen-filter").addEventListener("change", renderTable);
document.getElementById("status-filter").addEventListener("change", renderTable);
populateFilters();
renderTable();
</script>
</body>
</html>
"""


def render_population_html(samples: List[Dict[str, Any]]) -> str:
    """Render a self-contained HTML page for browsing `samples`.

    `samples` is the same list of per-sample dicts `_snapshot_population()` builds:
    id/data/status/generation/depth/parent_ids/child_ids/feedback.
    """
    samples_json = json.dumps(samples, default=str)
    return _TEMPLATE.replace(_SAMPLES_PLACEHOLDER, samples_json)
