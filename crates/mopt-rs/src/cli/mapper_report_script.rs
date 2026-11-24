//! The script and the style of the report of `run_mapper` (plain JavaScript, nothing loaded
//! from elsewhere). The map and the graph are drawn on a canvas with every node shown (one
//! path per colour of the energy scale, so tens of thousands of nodes draw in a frame), with
//! zoom (Ctrl + wheel, pinch, buttons, keys), panning by dragging, fitting to the window, and
//! the label of the node under the pointer; the disconnectivity graph and the energy diagram
//! are the SVG figures of the page. A node chosen in any figure or table is marked in all of
//! them and shown with its numbers, its links, its structure (the viewer of the AutoTS report)
//! and, for a transition state, the energy along its IRC; the lowest-barrier path between two
//! minima (the path of the minimum spanning tree) is marked and drawn along the IRCs of its
//! steps; the tables page through all nodes and can be filtered.
//!
//! Where scripts do not run, the page keeps the figures drawn by mopt-rs, the tables of the
//! lowest nodes, and the structure of the lowest minimum.
//!
//! MultiOptPy: no counterpart (new in mopt-rs).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

/// The style of the parts of the page that the AutoTS report does not have.
pub const NETWORK_STYLE: &str = r##"
.js .view-panel:not(.active) { display: none; }
.view-panel > h3 { margin: 10px 0 4px; font-size: 1rem; }
.js .view-panel > h3, .js .no-js-only { display: none; }
.needs-js, .needs-js-inline { display: none; }
.js .needs-js { display: block; }
.js .needs-js-inline { display: inline; }
.js .toolrow.needs-js, .js .pathbar.needs-js, .js .pager.needs-js, .js .axisbar.needs-js { display: flex; }
.axisbar { flex-wrap: wrap; gap: 8px 14px; align-items: center; margin: 6px 0 0; font-size: .86rem; color: var(--muted); }
.axisbar select { font: inherit; padding: 3px 8px; border-radius: 9px; border: 1px solid var(--line); background: var(--glass-strong); color: var(--fg); }
.netfig { width: 100%; height: auto; max-height: calc(100vh - 150px); display: block; }
.netfig .frame { fill: none; stroke: var(--line-strong); stroke-width: 1; }
.netfig .frame-line { stroke: var(--line-strong); stroke-width: 1; }
.netfig .grid { stroke: var(--line); stroke-width: 1; }
.netfig text { fill: var(--fg); font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; }
.netfig .tick { fill: var(--muted); font-size: 10px; }
.netfig .axis-title { fill: var(--muted); font-size: 11px; }
.netfig .lbl { font-size: 10px; pointer-events: none; }
.netfig .edge { fill: none; stroke: var(--muted); stroke-width: 1.1; opacity: .6; }
.netfig .edge.crossing { stroke-dasharray: 4 3; }
button.get { background: none; border: 0; padding: 0; margin: 0; font: inherit; color: var(--accent); cursor: pointer; text-decoration: underline; }
button.get:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.is-full { position: fixed; inset: 0; width: 100vw; height: 100vh; z-index: 1000; background: var(--bg); }
.explore.is-full { grid-template-columns: minmax(0, 1fr) min(430px, 36vw); grid-template-rows: 100vh; gap: 0; align-items: stretch; overflow: hidden; }
.explore.is-full > div { height: 100vh; overflow: hidden; }
.explore.is-full .view-panel > :not(.canvas-wrap) { display: none; }
.explore.is-full .canvas-wrap { border-radius: 0; border: 0; margin: 0; }
.explore.is-full #details { position: static; height: 100vh; max-height: none; border-radius: 0; margin: 0; overflow-y: auto; }
.explore.is-full #details .viewer { height: 42vh; }
@media (max-width: 920px) { .explore.is-full { grid-template-columns: minmax(0, 1fr); grid-template-rows: 55vh 45vh; } .explore.is-full > div { height: 55vh; } .explore.is-full #details { height: 45vh; } .explore.is-full #details .viewer { height: 220px; } }
.species-wrap { position: relative; }
.species-wrap button.full { position: absolute; right: 8px; top: 8px; z-index: 1; }
.species-wrap.is-full { display: flex; align-items: center; justify-content: center; overflow: auto; }
.species-wrap.is-full svg.species-graph { width: 100vw; max-height: 100vh; }
select[data-species-select] { font: inherit; font-size: .86rem; padding: 3px 8px; border-radius: 9px; border: 1px solid var(--line); background: var(--glass-strong); color: var(--fg); max-width: 22rem; }
svg.species-graph { display: block; width: 100%; height: auto; min-width: 520px; }
svg.species-graph line { stroke: var(--line-strong); stroke-width: 1.5; }
svg.species-graph text { fill: var(--fg); font-size: 11px; }
svg.species-graph .edge-label { fill: var(--muted); font-size: 10px; }
svg.species-graph { cursor: move; touch-action: none; }
svg.species-graph g.species-node { cursor: grab; touch-action: none; }
svg.species-graph g.species-node.dragged { cursor: grabbing; }
svg.species-graph .species-name { font-weight: 700; fill: #111; paint-order: stroke; stroke: rgba(255, 255, 255, .7); stroke-width: 2.5px; }
svg.species-graph circle { stroke: var(--line-strong); stroke-width: 1; }
svg.species-graph g[data-node] { cursor: pointer; }
svg.species-graph g[data-node]:focus-visible circle, svg.species-graph g[data-node]:hover circle { stroke: var(--accent); stroke-width: 2.5; }
svg.species-graph g.species-edge:hover line, svg.species-graph g.species-edge:focus-visible line { stroke: var(--accent); stroke-width: 2.5; }
.netfig .dg-line { stroke: var(--fg); stroke-width: 1.3; opacity: .8; }
.netfig .level { stroke: var(--s-green); stroke-width: 3; cursor: pointer; }
.netfig .node { stroke: var(--fg); stroke-width: .8; cursor: pointer; }
.netfig .node:hover, .netfig .node:focus-visible, .netfig .level:hover { stroke-width: 2.4; outline: none; }
.netfig .lowest { fill: none; stroke: var(--accent); stroke-width: 2; }
.netfig .curve { fill: none; stroke: var(--accent); stroke-width: 2; }
.netfig .curve.guess { stroke-dasharray: 5 4; }
.netfig .end { fill: var(--accent); }
.netfig .edge.on-path, .netfig .dg-line.on-path { stroke: var(--s-amber); stroke-width: 3.2; opacity: 1; }
.netfig .node.on-path { stroke: var(--s-amber); stroke-width: 2.8; }
.netfig .level.on-path { stroke: var(--s-amber); }
.netfig .node.sel { stroke: var(--s-red); stroke-width: 3.2; }
.netfig .level.sel, .netfig .dg-line.sel { stroke: var(--s-red); stroke-width: 4; opacity: 1; }
.canvas-wrap { position: relative; width: 100%; height: 420px; margin: 8px 0; border-radius: 14px; overflow: hidden; border: 1px solid var(--line); background: var(--viewer-bg); }
.canvas-wrap canvas { display: block; touch-action: none; cursor: grab; }
.canvas-wrap canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.canvas-wrap .tip { position: absolute; pointer-events: none; padding: 2px 8px; border-radius: 8px; font-size: .78rem; background: var(--glass-strong); color: var(--fg); border: 1px solid var(--line); white-space: nowrap; }
.canvas-wrap .zoombar { position: absolute; left: 8px; top: 8px; display: flex; gap: 4px; }
.canvas-wrap .zoombar button { font: inherit; font-size: .8rem; min-width: 28px; padding: 2px 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--glass-strong); color: var(--fg); cursor: pointer; }
.explore { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(280px, 1fr); gap: 18px; align-items: start; }
.explore > div { min-width: 0; }
#details { position: sticky; top: 76px; min-width: 0; max-height: calc(100vh - 90px); overflow-y: auto; }
@media (max-width: 920px) { .explore { grid-template-columns: minmax(0, 1fr); } #details { position: static; max-height: none; } }
#details h3 { margin: 0 0 8px; }
#details h4 { margin: 12px 0 4px; font-size: .92rem; }
#details .viewer { height: 260px; min-height: 200px; }
dl.facts { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 12px; margin: 0 0 10px; font-size: .88rem; }
dl.facts dt { color: var(--muted); }
dl.facts dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
ul.links, ol.steps { margin: 4px 0 10px; padding-left: 20px; font-size: .88rem; }
ul.links li, ol.steps li { margin: 4px 0; }
.more { font-size: .82rem; color: var(--muted); }
button.chip { font: inherit; font-size: .8rem; font-weight: 600; border: 1px solid var(--line); background: var(--glass-strong); color: var(--accent); border-radius: 999px; padding: 0 8px; cursor: pointer; }
button.chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
tr[data-node] { cursor: pointer; }
tr[data-node]:hover td, tr[data-node]:focus-visible td { background: var(--accent-soft); outline: none; }
tr.sel td { background: var(--accent-soft); font-weight: 600; }
tr.on-path td:first-child { box-shadow: inset 3px 0 0 var(--s-amber); }
.pathbar, .pager { flex-wrap: wrap; gap: 8px 12px; align-items: center; margin: 6px 0 10px; }
.pathbar input, .pager input { font: inherit; font-size: .86rem; padding: 4px 8px; border-radius: 9px; border: 1px solid var(--line); background: var(--glass-strong); color: var(--fg); max-width: 100%; }
.pager .pageinfo { font-size: .84rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.netfig.kinetics { max-width: 640px; }
.profiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); gap: 12px; }
"##;

/// The script of the page; it runs after the structure viewer of the AutoTS report, whose
/// `window.moptViewer.setup` it uses.
pub const NETWORK_SCRIPT: &str = r##"<script>
(function () {
  var root = document.documentElement;
  root.classList.add('js');
  var D = JSON.parse(document.getElementById('network-data').textContent);
  var N = D.id.length, M = D.minima, KIND = D.kinds, E = D.edges, L = E.node.length;
  var unit = D.unit, digits = D.digits;
  function f(v) { return (typeof v === 'number' && isFinite(v)) ? v.toFixed(digits) : '—'; }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  function pad6(n) { var s = String(n); while (s.length < 6) s = '0' + s; return s; }
  // Labels and files by the number of a node within its kind (EQ0, EQ1, ... and TS0, TS1, ...).
  function label(i) { return KIND[D.kind[i]] + D.num[i]; }
  var EXT = D.ext || 'xyz';
  // What the page holds (a structure, the frames of a path) is saved from it by a button; the
  // page has no links to the files of the run directory, so that it can be moved on its own.
  function get(key, text) {
    return held(key) ? '<button type="button" class="get" data-get="' + key + '">' + text + '</button>'
      : '<span class="note" title="not in this page">—</span>';
  }
  // The points of frames of plain variables (the number of variables, a comment, name value).
  function frames(text) {
    var out = [], lines = String(text || '').split('\n'), j = 0;
    while (j < lines.length) {
      var n = parseInt(lines[j], 10);
      if (!(n > 0)) { j++; continue; }
      var names = [], values = [];
      for (var q = 0; q < n; q++) { var parts = (lines[j + 2 + q] || '').trim().split(/\s+/); names.push(parts[0]); values.push(+parts[1]); }
      out.push({ comment: lines[j + 1] || '', names: names, values: values });
      j += 2 + n;
    }
    return out;
  }
  var ircPoints = {};
  function ircPath(k) {
    if (ircPoints[k] !== undefined) return ircPoints[k];
    var p = D.variables && D.irc_xyz[k] ? frames(D.irc_xyz[k]).map(function (fr) { return fr.values; }) : null;
    ircPoints[k] = p && p.length > 1 ? p : null;
    return ircPoints[k];
  }
  function chip(i) { return '<button type="button" class="chip" data-node="' + D.id[i] + '">' + esc(label(i)) + '</button>'; }
  function wave(v) { return v < 0 ? (-v).toFixed(1) + 'i' : v.toFixed(1); }
  // The imaginary modes of a saddle point: their number and wavenumbers (those of its node, or
  // the one of the IRC of its link `k`).
  function imaginary(id, k) {
    if (D.imag && D.imag[id]) return D.imag[id].length + ': ' + D.imag[id].map(wave).join(', ');
    return (k >= 0 && E.wave[k] !== null) ? '1: ' + wave(E.wave[k]) : '—';
  }
  function pct(p) { return p >= 1e-5 ? (100 * p).toFixed(1) + ' %' : (p > 0 ? (100 * p).toExponential(1) + ' %' : '0 %'); }
  function still() { return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches); }
  function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* no storage */ } }
  function stored(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
  function css(name, d) { var v = getComputedStyle(root).getPropertyValue(name).trim(); return v || d; }
  var at = {};
  for (var i = 0; i < N; i++) at[D.id[i]] = i;
  var eN = new Int32Array(L), eA = new Int32Array(L), eB = new Int32Array(L), firstLink = new Int32Array(N).fill(-1), linksOf = [];
  for (var k = 0; k < L; k++) {
    eN[k] = at[E.node[k]]; eA[k] = at[E.a[k]]; eB[k] = at[E.b[k]];
    if (firstLink[eN[k]] < 0) firstLink[eN[k]] = k;
    [eA[k], eB[k], eN[k]].forEach(function (j, n, all) { if (all.indexOf(j) !== n) return; (linksOf[j] = linksOf[j] || []).push(k); });
  }
  function profile(k) { var p = D.profiles[k], out = []; if (!p) return out; for (var j = 0; j + 1 < p.length; j += 2) out.push([p[j], p[j + 1]]); return out; }
  // Files the page holds (structures, IRC paths) are saved from it; the others are links to the
  // files of the run directory.
  function held(get) {
    var p = String(get).split(':'), i = at[p[1]];
    // (A saddle point of higher order is not among the nodes of the figures.)
    var hk = p[0] === 'node' && i === undefined && D.hs ? D.hs.id.indexOf(+p[1]) : -1;
    if (p[0] === 'node' && (i !== undefined || hk >= 0)) {
      var nodeFile = (i !== undefined ? label(i) : D.hs.tag[hk]) + '.' + EXT;
      if (D.model[p[1]]) return { name: nodeFile, text: D.model[p[1]] };
      if (D.model_z && D.model_z[p[1]]) return { name: nodeFile, packedModel: p[1] };
    }
    if (p[0] === 'irc' && D.irc_xyz[p[1]]) return { name: label(eN[+p[1]]) + (D.kind[eN[+p[1]]] >= 2 ? '_path.' : '_irc.') + EXT, text: D.irc_xyz[p[1]] };
    if (p[0] === 'irc' && D.irc_z && D.irc_z[p[1]]) return { name: label(eN[+p[1]]) + (D.kind[eN[+p[1]]] >= 2 ? '_path.' : '_irc.') + EXT, packed: +p[1] };
    return null;
  }
  // The paths beyond those the page holds as text are packed (a raw DEFLATE stream in
  // base64) and unpacked by the browser when one is opened or saved.
  // What is packed is a raw DEFLATE stream of either a text or of xyz frames written compactly
  // (MAGIC, the number of atoms, the elements, the number of frames and their titles as lines,
  // a byte 0, then the coordinates as whole numbers of 1e-5 Angstrom: those of the first frame,
  // and for every later frame the differences from the frame before; each number zigzag, seven
  // bits a byte, the lowest first).
  var MAGIC = 'MOPTXYZ1\n';
  function expanded(bytes) {
    for (var q = 0; q < MAGIC.length; q++) if (bytes[q] !== MAGIC.charCodeAt(q)) return new TextDecoder().decode(bytes);
    var zero = bytes.indexOf(0, MAGIC.length);
    var head = new TextDecoder().decode(bytes.subarray(MAGIC.length, zero)).split('\n');
    var n = +head[0], elements = head[1].split(' '), count = +head[2], pos = zero + 1;
    var frame = new Float64Array(3 * n), out = [];
    for (var f = 0; f < count; f++) {
      out.push(String(n), head[3 + f]);
      for (var a = 0; a < n; a++) {
        for (var c = 0; c < 3; c++) {
          var v = 0, mul = 1, byte;
          do { byte = bytes[pos++]; v += (byte & 127) * mul; mul *= 128; } while (byte & 128);
          frame[3 * a + c] += v % 2 ? -(v + 1) / 2 : v / 2;
        }
        out.push(elements[a] + ' ' + (frame[3 * a] / 1e5).toFixed(5) + ' ' + (frame[3 * a + 1] / 1e5).toFixed(5) + ' ' + (frame[3 * a + 2] / 1e5).toFixed(5));
      }
    }
    return out.join('\n') + '\n';
  }
  function inflate(b64) {
    if (!window.DecompressionStream) return Promise.reject(new Error('This browser cannot unpack what the page holds packed (it has no DecompressionStream).'));
    var bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (var j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
    return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer()
      .then(function (buffer) { return expanded(new Uint8Array(buffer)); });
  }
  var unpacked = {}, unpackedModels = {};
  // The structure of node `id` as a promise of its text; null when the page does not hold it.
  function modelText(id) {
    if (D.model[id]) return Promise.resolve(D.model[id]);
    if (D.model_z && D.model_z[id]) { if (!unpackedModels[id]) unpackedModels[id] = inflate(D.model_z[id]); return unpackedModels[id]; }
    return null;
  }
  // The structure of node `id` in `slot`: at once where the page has its text, after it is
  // unpacked otherwise.
  function showNode(slot, id, name) {
    if (D.model[id] || !(D.model_z && D.model_z[id])) { showModel(slot, D.model[id], name); return; }
    slot.innerHTML = '<p class="note">Unpacking the structure…</p>';
    modelText(id).then(function (text) { if (slot.isConnected) { slot.innerHTML = ''; showModel(slot, text, name); } },
      function (e) { if (slot.isConnected) slot.innerHTML = '<p class="warn">' + esc(e.message) + '</p>'; });
  }
  function inPage(id) { return !!(D.model[id] || (D.model_z && D.model_z[id])); }
  // The frames of the path of link k as a promise of their text; null when the page does not
  // hold them.
  function ircText(k) {
    if (D.irc_xyz[k]) return Promise.resolve(D.irc_xyz[k]);
    if (D.irc_z && D.irc_z[k]) { if (!unpacked[k]) unpacked[k] = inflate(D.irc_z[k]); return unpacked[k]; }
    return null;
  }
  // An element over the whole screen (`is-full`): by the browser where it can, and over the
  // whole window where it cannot (a page inside another, a phone). The figures of the
  // network go there with the panel of the chosen node at their right.
  function setFull(el, on) {
    document.querySelectorAll('.is-full').forEach(function (other) { if (other !== el || !on) other.classList.remove('is-full'); });
    if (on) el.classList.add('is-full');
    document.querySelectorAll('[data-z="full"], button.full').forEach(function (b) { b.textContent = on ? 'Exit full screen' : 'Full screen'; });
    window.dispatchEvent(new Event('resize'));
  }
  function toggleFull(el) {
    if (document.fullscreenElement) { if (document.exitFullscreen) document.exitFullscreen(); return; }
    if (el.classList.contains('is-full')) { setFull(el, false); return; }
    if (!el.requestFullscreen) { setFull(el, true); return; }
    var asked = el.requestFullscreen();
    if (asked && asked.catch) asked.catch(function () { setFull(el, true); });
  }
  document.addEventListener('fullscreenchange', function () {
    var el = document.fullscreenElement;
    if (el) setFull(el, true); else { var was = document.querySelector('.is-full'); if (was) setFull(was, false); }
  });
  document.addEventListener('keydown', function (e) {
    var was = document.querySelector('.is-full');
    if (e.key === 'Escape' && was && !document.fullscreenElement) setFull(was, false);
  });
  function save(name, text) {
    var url = URL.createObjectURL(new Blob([text], { type: 'text/plain' })), a = document.createElement('a');
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }
  var eTop = 1e-12;
  for (var i2 = 0; i2 < N; i2++) if (isFinite(D.e[i2]) && D.e[i2] > eTop) eTop = D.e[i2];

  // The state: the chosen node and the path (node and link indices).
  var chosen = M > 0 ? 0 : -1, pathNodes = new Uint8Array(N), pathEdges = new Uint8Array(L), pathList = { nodes: [], edges: [] };
  var figures = [], pagers = [];
  // The surface shown in the figures of a run of several states: -1 all, -2 all in layers,
  // k the state k (PES k + 1), kept between visits.
  var SURF = -1;
  if (D.pes) { var savedSurface = +(stored('mopt-mapper-surface') || '-1'); if (savedSurface >= -2 && savedSurface < D.states.length) SURF = savedSurface; }
  function onSurface(i) { return !D.pes || SURF < 0 || D.pes[i] === SURF || D.pes2[i] === SURF; }
  // The species shown in the figures: -1 all, k the conformers of the species k with the
  // links between them.
  var SPEC = -1;
  function inSpecies(i) {
    if (SPEC < 0 || !D.species_of) return true;
    if (i < M) return D.species_of[D.id[i]] === SPEC;
    var k = firstLink[i];
    return k >= 0 && D.species_of[D.id[eA[k]]] === SPEC && D.species_of[D.id[eB[k]]] === SPEC;
  }
  // The surfaces of node i (`PES2`, `PES1/PES2` for a crossing point), empty for one state.
  function surfaceOf(i) { return D.pes && D.pes[i] >= 0 ? 'PES' + (D.pes[i] + 1) + (D.pes2[i] >= 0 ? '/PES' + (D.pes2[i] + 1) : '') : ''; }
  function named(i) { var p = surfaceOf(i); return label(i) + (p ? ' <span class="muted">· ' + p + '</span>' : ''); }
  function refresh() {
    document.querySelectorAll('.sel, .on-path').forEach(function (el) { el.classList.remove('sel'); el.classList.remove('on-path'); });
    function on(cls, sel) { document.querySelectorAll(sel).forEach(function (el) { if (!el.classList.contains('chip') && el.tagName !== 'TR') el.classList.add(cls); }); }
    if (chosen >= 0) on('sel', '[data-node="' + D.id[chosen] + '"]');
    pathList.nodes.forEach(function (j) { on('on-path', '[data-node="' + D.id[j] + '"]'); });
    pathList.edges.forEach(function (k) { on('on-path', '[data-edge="' + k + '"]'); });
    if (circleNew) circleNew();
    figures.forEach(function (fig) { fig.schedule(); });
    pagers.forEach(function (p) { p.show(); });
  }
  // Set by the circle of the minima: places them again when the path chosen has changed.
  var circleNew = null;

  // The energy scale (viridis), 32 colours.
  var STOPS = [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]];
  function scale(t) {
    var s = 0;
    while (s < 3 && t > STOPS[s + 1][0]) s++;
    var a = STOPS[s], c = STOPS[s + 1], u = Math.max(0, Math.min(1, (t - a[0]) / (c[0] - a[0])));
    return 'rgb(' + [0, 1, 2].map(function (j) { return Math.round(a[1][j] + u * (c[1][j] - a[1][j])); }).join(',') + ')';
  }
  var COLORS = [];
  for (var b = 0; b < 32; b++) COLORS.push(scale((b + 0.5) / 32));
  var bucket = new Uint8Array(N);
  for (var i3 = 0; i3 < N; i3++) bucket[i3] = isFinite(D.e[i3]) ? Math.max(0, Math.min(31, Math.floor(D.e[i3] / eTop * 32))) : 31;

  function ticks(lo, hi, n) {
    var span = hi - lo;
    if (!(span > 0)) return { values: [lo], digits: 2 };
    var raw = span / n, p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10 + 1e-9)), r = raw / p - 1e-9;
    var step = p * (r <= 1 ? 1 : r <= 2 ? 2 : r <= 5 ? 5 : 10), out = [];
    for (var j = Math.ceil(lo / step - 1e-9); j * step <= hi + 1e-9 * step; j++) out.push(j * step);
    return { values: out, digits: Math.max(0, Math.min(8, -Math.floor(Math.log(step) / Math.LN10 + 1e-9))) };
  }

  // A canvas figure of every node at (xs, ys); a graph draws its links straight between minima.
  // `under` draws what lies behind the nodes (with the place on the canvas of a point of the
  // map); `under.skip(k)` leaves out the straight line of a link it draws itself.
  function Figure(wrap, xs, ys, graph, names, under) {
    var canvas = document.createElement('canvas'), tip = document.createElement('div'), bar = document.createElement('div');
    canvas.tabIndex = 0; canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', graph === 'circle' ? 'The minima on a circle with their links as chords' : graph ? 'The network as a graph of its minima and links' : 'Map of the network');
    tip.className = 'tip'; tip.hidden = true; bar.className = 'zoombar';
    bar.innerHTML = '<button type="button" data-z="in" aria-label="Zoom in">+</button><button type="button" data-z="out" aria-label="Zoom out">−</button><button type="button" data-z="fit">Fit</button><button type="button" data-z="full">Full screen</button>';
    wrap.appendChild(canvas); wrap.appendChild(tip); wrap.appendChild(bar);
    var ctx = canvas.getContext('2d'), has, idx, groups, x0, x1, y0, y1, many, labelsUpTo = 80;
    var W = 0, H = 0, s = 1, ox = 0, oy = 0, sFit = 1, pending = false, hsAt = [], bx = xs, by = ys;
    // The places of the surfaces in layers: each figure scaled into a band of its own, PES1 at
    // the bottom, a crossing point between the bands of its two states.
    var BAND = 0.6, WIDE = 1.6;
    function layered(nx, ny) {
      var lx = [], ly = [], a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
      for (var j = 0; j < N; j++) {
        if (nx[j] === null || ny[j] === null || !isFinite(nx[j]) || !isFinite(ny[j])) continue;
        a0 = Math.min(a0, nx[j]); a1 = Math.max(a1, nx[j]); b0 = Math.min(b0, ny[j]); b1 = Math.max(b1, ny[j]);
      }
      var w = Math.max(a1 - a0, 1e-9), h = Math.max(b1 - b0, 1e-9);
      for (var k = 0; k < N; k++) {
        if (nx[k] === null || ny[k] === null || !isFinite(nx[k]) || !isFinite(ny[k]) || D.pes[k] < 0) { lx.push(null); ly.push(null); continue; }
        lx.push((nx[k] - a0) / w * WIDE);
        var y = (ny[k] - b0) / h * BAND;
        ly.push(D.pes2[k] >= 0 ? (D.pes[k] + D.pes2[k] + BAND) / 2 : D.pes[k] + y);
      }
      return [lx, ly];
    }
    // The places of the nodes (other axes of the map are loaded the same way), on the surface
    // chosen.
    function load(nx, ny, nn) {
      bx = nx; by = ny; names = nn || names;
      if (D.pes && SURF === -2) { var l = layered(nx, ny); xs = l[0]; ys = l[1]; } else { xs = nx; ys = ny; }
      has = new Uint8Array(N); idx = []; groups = [];
      x0 = Infinity; x1 = -Infinity; y0 = Infinity; y1 = -Infinity;
      for (var b = 0; b < 64; b++) groups.push([]);
      for (var i = 0; i < N; i++) {
        if (xs[i] === null || ys[i] === null || !isFinite(xs[i]) || !isFinite(ys[i]) || !onSurface(i) || !inSpecies(i)) continue;
        has[i] = 1; idx.push(i); groups[bucket[i] + (i < M ? 0 : 32)].push(i);
        x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]); y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]);
      }
      many = idx.length > 3000;
      if (W) fit();
    }
    // In the graph a minimum can be dragged; its saddles keep their place relative to the middle
    // of their two minima.
    var offsets = {};
    function middle(t) {
      var k = firstLink[t];
      return k < 0 ? null : [(xs[eA[k]] + xs[eB[k]]) / 2, (ys[eA[k]] + ys[eB[k]]) / 2];
    }
    if (graph === true) for (var t0 = M; t0 < N; t0++) { var m0 = middle(t0); if (m0) offsets[t0] = [xs[t0] - m0[0], ys[t0] - m0[1]]; }
    function moveMinimum(i, x, y) {
      xs[i] = x; ys[i] = y;
      (linksOf[i] || []).forEach(function (k) {
        var t = eN[k], m = middle(t);
        if (m && offsets[t]) { xs[t] = m[0] + offsets[t][0]; ys[t] = m[1] + offsets[t][1]; }
      });
    }
    function fit() {
      x0 = Infinity; x1 = -Infinity; y0 = Infinity; y1 = -Infinity;
      idx.forEach(function (i) { x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]); y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]); });
      // In layers every band and its name are in view.
      if (D.pes && SURF === -2) { x0 = Math.min(x0, -0.3); x1 = Math.max(x1, WIDE); y0 = Math.min(y0, -0.05); y1 = Math.max(y1, D.states.length - 1 + BAND + 0.05); }
      var pad = 30, right = 64, sx = (W - pad - right) / Math.max(x1 - x0, 1e-9), sy = (H - 2 * pad) / Math.max(y1 - y0, 1e-9);
      s = sFit = Math.min(sx, sy);
      ox = pad + (W - pad - right - s * (x1 - x0)) / 2 - s * x0;
      oy = pad + (H - 2 * pad - s * (y1 - y0)) / 2 + s * y1;
    }
    load(xs, ys, names);
    function size() {
      // Over the whole screen the figure has the height of its column, beside the panel of
      // the chosen node.
      var host = wrap.closest('.is-full'), w = wrap.clientWidth;
      if (!w || wrap.offsetParent === null) return false;
      var h = host ? host.firstElementChild.clientHeight : Math.round(Math.max(300, Math.min(w * 0.72, window.innerHeight - 190)));
      if (wrap.style.height !== h + 'px') wrap.style.height = h + 'px';
      var dpr = window.devicePixelRatio || 1;
      if (w !== W || h !== H) {
        W = w; H = h; canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
        canvas.style.width = w + 'px'; canvas.style.height = h + 'px'; fit();
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return true;
    }
    function X(i) { return ox + s * xs[i]; }
    function Y(i) { return oy - s * ys[i]; }
    function linkPath(k) {
      var a = eA[k], c = eB[k], t = eN[k];
      if (!has[a] || !has[c]) {
        // One surface of several: a crossing point leads to the minimum of this surface.
        if (D.pes && has[t] && (has[a] || has[c])) { var shownEnd = has[a] ? a : c; ctx.moveTo(X(t), Y(t)); ctx.lineTo(X(shownEnd), Y(shownEnd)); }
        return;
      }
      // Through the saddle: in the graph, parallel links of one pair bend apart.
      if (has[t]) { ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(t), Y(t)); ctx.lineTo(X(c), Y(c)); }
      else if (graph) { ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(c), Y(c)); }
    }
    function shape(i, r) {
      var x = X(i), y = Y(i);
      if (x < -r || x > W + r || y < -r || y > H + r) return;
      if (i < M) { ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, 2 * Math.PI); }
      else if (D.kind[i] >= 2) { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y + 0.8 * r); ctx.lineTo(x - r, y + 0.8 * r); ctx.closePath(); }
      else { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); }
    }
    function axes(muted, line) {
      ctx.font = '10px system-ui, sans-serif'; ctx.fillStyle = muted; ctx.strokeStyle = line; ctx.lineWidth = 1;
      var tx = ticks(-ox / s, (W - ox) / s, 7), ty = ticks((oy - H) / s, oy / s, 6);
      ctx.beginPath();
      tx.values.forEach(function (v) { var x = ox + s * v; ctx.moveTo(x, 0); ctx.lineTo(x, H); });
      ty.values.forEach(function (v) { var y = oy - s * v; ctx.moveTo(0, y); ctx.lineTo(W, y); });
      ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      tx.values.forEach(function (v) { ctx.fillText(v.toFixed(tx.digits), ox + s * v, H - 3); });
      ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ty.values.forEach(function (v) { var y = oy - s * v; if (y > 40) ctx.fillText(v.toFixed(ty.digits), 4, y); });
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(names[0] + ' →, ' + names[1] + ' ↑' + (D.variables ? '' : ' (Å amu½)'), (W - 64) / 2, 6);
    }
    function colorBar(fg, muted) {
      var x = W - 46, top = 30, bottom = Math.min(H - 30, top + 240);
      var g = ctx.createLinearGradient(0, bottom, 0, top);
      STOPS.forEach(function (st) { g.addColorStop(st[0], scale(st[0])); });
      ctx.fillStyle = g; ctx.fillRect(x, top, 10, bottom - top);
      ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.strokeRect(x, top, 10, bottom - top);
      ctx.fillStyle = fg; ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      var t = ticks(0, eTop, 4);
      t.values.forEach(function (v) { ctx.fillText(v.toFixed(t.digits), x + 14, bottom - (bottom - top) * v / eTop); });
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(unit, x + 5, top - 4);
    }
    function draw() {
      pending = false;
      if (!size()) return;
      var bg = css('--viewer-bg', '#ffffff'), fg = css('--fg', '#111111'), muted = css('--muted', '#666666');
      var line = css('--line', 'rgba(0,0,0,.1)'), amber = css('--s-amber', '#d97706'), red = css('--s-red', '#dc2626'), accent = css('--accent', '#0a7cff');
      ctx.globalAlpha = 1; ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
      if (under) under(ctx, function (x, y) { return [ox + s * x, oy - s * y]; }, s);
      if (D.pes && SURF === -2) {
        ctx.save(); ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (var band = 0; band < D.states.length; band++) {
          var top = oy - s * (band + BAND + 0.04), bottom = oy - s * (band - 0.04);
          ctx.fillStyle = muted; ctx.globalAlpha = 0.08; ctx.fillRect(ox - s * 0.03, top, s * (WIDE + 0.06), bottom - top);
          ctx.globalAlpha = 0.5; ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.strokeRect(ox - s * 0.03, top, s * (WIDE + 0.06), bottom - top);
          ctx.globalAlpha = 1; ctx.fillStyle = muted; ctx.fillText('PES' + (band + 1), ox - s * 0.05, oy - s * (band + BAND / 2));
        }
        ctx.restore();
      } else if (!graph) axes(muted, line);
      var grow = Math.min(2.5, Math.sqrt(s / sFit)), r = (many ? 2.2 : idx.length > 600 ? 3.4 : 5.2) * grow;
      ctx.lineWidth = 1; ctx.strokeStyle = muted; ctx.globalAlpha = many ? 0.22 : 0.5;
      ctx.beginPath();
      for (var k = 0; k < L; k++) if (!(under && under.skip && under.skip(k))) linkPath(k);
      ctx.stroke();
      ctx.globalAlpha = 1;
      // The saddle points of higher order: a square above what their descents reach, joined
      // to it by dashed lines (one kept without descents: beside the node whose job found it,
      // joined by a dotted line).
      hsAt = [];
      if (D.hs && D.hs.id.length) {
        ctx.save(); ctx.strokeStyle = muted; ctx.lineWidth = 1;
        var beside = {};
        for (var hk = 0; hk < D.hs.id.length; hk++) {
          var to = D.hs.to[hk], sx = 0, sy = 0, cnt = 0;
          to.forEach(function (j) { if (has[j]) { sx += X(j); sy += Y(j); cnt++; } });
          if (!cnt) { hsAt.push(null); continue; }
          var down = !D.hs.down || D.hs.down[hk], hx = sx / cnt, hy = sy / cnt - 4 * r - 8;
          if (!down) {
            var key = to.join(','), nth = beside[key] = (beside[key] || 0) + 1;
            hx += (3 * r + 6) * nth; hy = sy / cnt - (3 * r + 6) * nth;
          }
          hsAt.push([hx, hy]);
          ctx.setLineDash(down ? [4, 3] : [1, 3]);
          ctx.beginPath();
          to.forEach(function (j) { if (has[j]) { ctx.moveTo(hx, hy); ctx.lineTo(X(j), Y(j)); } });
          ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        hsAt.forEach(function (p, hk2) {
          if (!p) return;
          var q = 1.1 * r;
          ctx.fillStyle = scale(Math.max(0, Math.min(1, D.hs.e[hk2] / eTop)));
          ctx.fillRect(p[0] - q, p[1] - q, 2 * q, 2 * q);
          ctx.strokeStyle = fg; ctx.lineWidth = 0.8; ctx.strokeRect(p[0] - q, p[1] - q, 2 * q, 2 * q);
          ctx.fillStyle = fg; ctx.fillText(D.hs.tag[hk2], p[0] + q + 2, p[1] - q);
        });
        ctx.restore();
      }
      for (var b = 0; b < 64; b++) {
        if (!groups[b].length) continue;
        var size_ = b < 32 ? r : 0.85 * r;
        ctx.beginPath();
        groups[b].forEach(function (i) { shape(i, size_); });
        ctx.fillStyle = COLORS[b % 32]; ctx.fill();
        if (!many) { ctx.globalAlpha = 0.6; ctx.strokeStyle = fg; ctx.lineWidth = 0.8; ctx.stroke(); ctx.globalAlpha = 1; }
      }
      if (M > 0 && has[0]) { ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(0), Y(0), r + 4, 0, 2 * Math.PI); ctx.stroke(); }
      if (pathList.edges.length) {
        ctx.strokeStyle = amber; ctx.lineWidth = 3; ctx.beginPath();
        pathList.edges.forEach(linkPath); ctx.stroke();
        ctx.lineWidth = 2.5; ctx.beginPath();
        pathList.nodes.forEach(function (i) { if (has[i]) { ctx.moveTo(X(i) + r + 2, Y(i)); ctx.arc(X(i), Y(i), r + 2, 0, 2 * Math.PI); } });
        ctx.stroke();
      }
      if (chosen >= 0 && has[chosen]) { ctx.strokeStyle = red; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(chosen), Y(chosen), r + 5, 0, 2 * Math.PI); ctx.stroke(); }
      // Labels of the minima in view, when there are few enough.
      var seen = [];
      for (var i = 0; i < M && seen.length <= labelsUpTo; i++) {
        if (has[i]) { var x = X(i), y = Y(i); if (x > 0 && x < W - 60 && y > 0 && y < H) seen.push(i); }
      }
      // Crossing points (triangles) get an outline of their own, apart from the transition
      // states (diamonds).
      ctx.save(); ctx.strokeStyle = css('--s-magenta', '#c026d3'); ctx.lineWidth = 1.6; ctx.beginPath();
      idx.forEach(function (i) { if (i >= M && D.kind[i] >= 2) shape(i, 1.35 * r); });
      ctx.stroke(); ctx.restore();
      if (seen.length <= labelsUpTo) {
        ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = fg;
        seen.forEach(function (i) { ctx.fillText(label(i), X(i) + r + 2, Y(i) - r); });
      }
      colorBar(fg, muted);
    }
    function schedule() { if (!pending) { pending = true; requestAnimationFrame(draw); } }
    // The square of a saddle point of higher order under the pointer (its index in D.hs).
    function nearestSquare(mx, my) {
      var best = -1, bd = 100;
      hsAt.forEach(function (p, k) {
        if (!p) return;
        var dx = p[0] - mx, dy = p[1] - my, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = k; }
      });
      return best;
    }
    // The square of the distance of (px, py) from the line between two points.
    function fromLine(px, py, x1, y1, x2, y2) {
      var dx = x2 - x1, dy = y2 - y1, len = dx * dx + dy * dy;
      var u = len > 0 ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / len)) : 0;
      var ex = x1 + u * dx - px, ey = y1 + u * dy - py;
      return ex * ex + ey * ey;
    }
    function nearest(mx, my) {
      var best = -1, bd = 196;
      for (var j = 0; j < idx.length; j++) {
        var i = idx[j], dx = X(i) - mx, dy = Y(i) - my, d = dx * dx + dy * dy;
        if (d < bd || (d === bd && i < M)) { bd = d; best = i; }
      }
      if (best >= 0) return best;
      // No node there: a link under the pointer stands for its transition state or crossing
      // point.
      var bl = 30;
      for (var k = 0; k < L; k++) {
        var a = eA[k], c = eB[k], t = eN[k];
        if (!has[t] || (under && under.skip && under.skip(k))) continue;
        var ends = has[a] && has[c] ? [a, c] : (D.pes && (has[a] || has[c]) ? [has[a] ? a : c] : []);
        for (var q = 0; q < ends.length; q++) {
          var dl = fromLine(mx, my, X(t), Y(t), X(ends[q]), Y(ends[q]));
          if (dl < bl) { bl = dl; best = t; }
        }
      }
      return best;
    }
    // The tooltip beside the pointer, kept inside the figure by its own width.
    function showTip(p) {
      tip.hidden = false;
      tip.style.left = Math.max(0, Math.min(W - tip.offsetWidth - 4, p[0] + 14)) + 'px';
      tip.style.top = Math.max(0, p[1] - 30) + 'px';
    }
    function zoomAt(mx, my, z) { ox = mx - (mx - ox) * z; oy = my - (my - oy) * z; s *= z; schedule(); }
    function point(e) { var rect = canvas.getBoundingClientRect(); return [e.clientX - rect.left, e.clientY - rect.top]; }
    var drag = null, moved = false, held = -1;
    canvas.addEventListener('pointerdown', function (e) {
      drag = [e.clientX, e.clientY]; moved = false; canvas.setPointerCapture(e.pointerId);
      var p = point(e), i = nearest(p[0], p[1]);
      held = graph === true && i >= 0 && i < M ? i : -1;
    });
    canvas.addEventListener('pointermove', function (e) {
      var p = point(e);
      if (drag) {
        var dx = e.clientX - drag[0], dy = e.clientY - drag[1];
        if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
        if (moved) {
          if (held >= 0) moveMinimum(held, (p[0] - ox) / s, (oy - p[1]) / s);
          else { ox += dx; oy += dy; }
          drag = [e.clientX, e.clientY]; tip.hidden = true; schedule();
        }
        return;
      }
      var sq = nearestSquare(p[0], p[1]);
      if (sq >= 0) {
        var im = D.hs.imag ? D.hs.imag[sq] : [];
        canvas.style.cursor = 'pointer';
        tip.textContent = D.hs.tag[sq] + ' · ' + f(D.hs.e[sq]) + ' ' + unit + ' · ' + (im.length ? im.length + ' imaginary: ' + im.map(wave).join(', ') : 'imaginary modes unknown');
        showTip(p);
        return;
      }
      var i = nearest(p[0], p[1]);
      if (i < 0) { tip.hidden = true; canvas.style.cursor = 'grab'; return; }
      canvas.style.cursor = 'pointer';
      tip.textContent = label(i) + (surfaceOf(i) ? ' · ' + surfaceOf(i) : '') + ' · ' + f(D.e[i]) + ' ' + unit + (i < M ? ' · ' + pct(D.pop[i]) : '');
      showTip(p);
    });
    canvas.addEventListener('pointerup', function (e) {
      var was = drag; drag = null; held = -1;
      if (was && !moved) {
        var p = point(e), sq = nearestSquare(p[0], p[1]);
        if (sq >= 0) {
          // A saddle point of higher order: in the panel, as a node.
          higher(sq, false);
          return;
        }
        var i = nearest(p[0], p[1]); if (i >= 0) select(i, false);
      }
    });
    canvas.addEventListener('pointerleave', function () { tip.hidden = true; });
    canvas.addEventListener('wheel', function (e) {
      // Over the whole screen the wheel zooms without a key: nothing else is to scroll.
      if (!e.ctrlKey && !e.metaKey && !wrap.closest('.is-full')) return;
      e.preventDefault(); var p = point(e); zoomAt(p[0], p[1], e.deltaY < 0 ? 1.2 : 1 / 1.2);
    }, { passive: false });
    canvas.addEventListener('dblclick', function () { fit(); schedule(); });
    canvas.addEventListener('keydown', function (e) {
      var step = 40, done = true;
      if (e.key === '+' || e.key === '=') zoomAt(W / 2, H / 2, 1.3);
      else if (e.key === '-') zoomAt(W / 2, H / 2, 1 / 1.3);
      else if (e.key === '0') { fit(); schedule(); }
      else if (e.key === 'ArrowLeft') { ox += step; schedule(); }
      else if (e.key === 'ArrowRight') { ox -= step; schedule(); }
      else if (e.key === 'ArrowUp') { oy += step; schedule(); }
      else if (e.key === 'ArrowDown') { oy -= step; schedule(); }
      else done = false;
      if (done) e.preventDefault();
    });
    bar.addEventListener('click', function (e) {
      var z = e.target.closest && e.target.closest('[data-z]');
      if (!z) return;
      if (z.dataset.z === 'full') { toggleFull(wrap.closest('.explore') || wrap); return; }
      if (z.dataset.z === 'fit') { fit(); schedule(); } else zoomAt(W / 2, H / 2, z.dataset.z === 'in' ? 1.5 : 1 / 1.5);
    });
    if (window.ResizeObserver) new ResizeObserver(schedule).observe(wrap);
    window.addEventListener('resize', schedule);
    document.addEventListener('fullscreenchange', schedule);
    return { schedule: schedule, load: function (nx, ny, nn) { load(nx, ny, nn); schedule(); }, surface: function () { load(bx, by); if (W) fit(); schedule(); }, focus: function (i) {
      if (!has[i] || !size()) return;
      var x = X(i), y = Y(i);
      if (x < 40 || x > W - 80 || y < 40 || y > H - 40) { ox += W / 2 - x; oy += H / 2 - y; }
      schedule();
    } };
  }
  var mapWrap = document.querySelector('[data-canvas="map"]'), graphWrap = document.querySelector('[data-canvas="graph"]');
  if (mapWrap && D.pc && D.pc.length) {
    // Any two principal coordinates: two cannot follow a wide network, the others show more.
    var axes = [0, Math.min(1, D.pc.length - 1)];
    var saved = (stored('mopt-mapper-axes') || '').split(',').map(Number);
    if (saved.length === 2 && saved.every(function (a) { return a >= 0 && a < D.pc.length; }) && saved[0] !== saved[1]) axes = saved;
    var axisName = function (a) { return D.map_axes ? String(D.variables[a]) : 'PCo' + (a + 1); };
    var axisNames = function () { return [axisName(axes[0]), axisName(axes[1])]; };
    // Plain variables: the surface itself behind the nodes (two variables), and the IRC paths.
    var behind = null;
    if (D.map_axes) {
      behind = function (ctx, P, sc) {
        var g = D.surface;
        if (g && axes[0] === 0 && axes[1] === 1) {
          var dx = (g.x[1] - g.x[0]) / (g.nx - 1), dy = (g.y[1] - g.y[0]) / (g.ny - 1);
          ctx.globalAlpha = 0.5;
          for (var j = 0; j < g.ny; j++) {
            for (var i = 0; i < g.nx; i++) {
              var v = g.e[j * g.nx + i];
              if (v === null || !isFinite(v) || v > eTop) continue;
              var c = P(g.x[0] + (i - 0.5) * dx, g.y[0] + (j + 0.5) * dy);
              ctx.fillStyle = scale(Math.max(0, v) / eTop);
              ctx.fillRect(c[0], c[1], sc * dx + 1, sc * dy + 1);
            }
          }
          ctx.globalAlpha = 1;
        }
        ctx.strokeStyle = css('--fg', '#111111'); ctx.lineWidth = 1.5; ctx.globalAlpha = 0.75;
        for (var k = 0; k < L; k++) {
          var pts = ircPath(k);
          if (!pts) continue;
          ctx.beginPath();
          pts.forEach(function (v, n) { var c = P(v[axes[0]], v[axes[1]]); if (n) ctx.lineTo(c[0], c[1]); else ctx.moveTo(c[0], c[1]); });
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      };
      behind.skip = function (k) { return !!ircPath(k); };
    }
    var mapFigure = Figure(mapWrap, D.pc[axes[0]], D.pc[axes[1]], false, axisNames(), behind);
    figures.push(mapFigure);
    var landmarks = String(D.map_method).indexOf('landmark') === 0 ? ' (of the landmarks)' : '';
    var quality = function () {
      var two = D.pc_share[axes[0]] + (axes[1] !== axes[0] ? D.pc_share[axes[1]] : 0);
      var warn = document.querySelector('[data-map-warning]');
      if (!warn) return;
      warn.hidden = two >= 0.5;
      warn.textContent = 'The two axes shown hold only ' + (100 * two).toFixed(0) + ' % of the spread of the distances' + landmarks + ': structures far apart can lie close on this map. Other axes, or the graph (which shows the links whatever the distances), show what this map cannot.';
    };
    document.querySelectorAll('[data-axis]').forEach(function (sel) {
      var n = +sel.dataset.axis;
      D.pc.forEach(function (_, a) { sel.appendChild(new Option(axisName(a) + ' (' + (100 * D.pc_share[a]).toFixed(0) + ' %)', String(a))); });
      sel.value = String(axes[n]);
      sel.addEventListener('change', function () {
        axes[n] = +sel.value;
        mapFigure.load(D.pc[axes[0]], D.pc[axes[1]], axisNames());
        store('mopt-mapper-axes', axes.join(','));
        quality();
      });
    });
    quality();
  }
  if (graphWrap) figures.push(Figure(graphWrap, D.gx, D.gy, true, ['', '']));
  // The minima on a circle, every link a chord through its saddle. With a path chosen its
  // minima are on the upper half in their order, the start at the left and the end at the
  // right, and the others on the lower half, each under the minimum of the path it is
  // joined to in the fewest steps; without one they follow round the circle in the order of
  // their steps from the lowest minimum shown.
  var circleWrap = document.querySelector('[data-canvas="circle"]');
  // One circle of all the minima shown, or a circle per species (its conformers round it),
  // the circles of the species round a ring: `circleRings` has what is drawn behind the
  // nodes, [x, y, radius, name] of every circle.
  var CIRCLES = stored('mopt-mapper-circles') === 'species' ? 'species' : 'one', circleRings = [];
  // The minima `members` from `start` on in the order of their steps over the links between
  // them; those that nothing joins to it follow.
  function inSteps(members, start) {
    var inside = {}, order = [start], seen = {};
    members.forEach(function (j) { inside[j] = 1; });
    seen[start] = 1;
    for (var head = 0; head < order.length; head++) {
      var here = order[head];
      (linksOf[here] || []).forEach(function (k) {
        var j = eA[k] === here ? eB[k] : (eB[k] === here ? eA[k] : -1);
        if (j < 0 || j >= M || !inside[j] || seen[j]) return;
        seen[j] = 1; order.push(j);
      });
    }
    members.forEach(function (j) { if (!seen[j]) order.push(j); });
    return order;
  }
  function saddlesBetween(xs, ys) {
    // A saddle at the middle of the line of its two minima; those of one pair apart.
    var pairs = {};
    for (var t = M; t < N; t++) {
      var k = firstLink[t];
      if (k < 0 || xs[eA[k]] === null || xs[eB[k]] === null) continue;
      var a = eA[k], b = eB[k], key = Math.min(a, b) + ':' + Math.max(a, b), m = pairs[key] = (pairs[key] || 0) + 1;
      if (a === b) { xs[t] = xs[a] * 0.9; ys[t] = ys[a] * 0.9; continue; }
      var mx = (xs[a] + xs[b]) / 2, my = (ys[a] + ys[b]) / 2, off = 0.03 * (m - 1) * (m % 2 ? 1 : -1);
      xs[t] = mx + off * (ys[b] - ys[a]); ys[t] = my - off * (xs[b] - xs[a]);
    }
  }
  function speciesCircles(xs, ys, shown) {
    var groups = [];
    D.species.forEach(function (_, k) {
      var members = shown.filter(function (j) { return D.species_of[D.id[j]] === k; });
      if (members.length) groups.push({ k: k, members: members });
    });
    var S = groups.length, most = 1;
    groups.forEach(function (g) { most = Math.max(most, g.members.length); });
    var base = S > 1 ? 0.85 * Math.sin(Math.PI / S) : 1;
    groups.forEach(function (g, n) {
      var angle = Math.PI - 2 * Math.PI * n / S, cx = S > 1 ? Math.cos(angle) : 0, cy = S > 1 ? Math.sin(angle) : 0;
      var r = g.members.length > 1 ? base * Math.max(0.25, Math.sqrt(g.members.length / most)) : 0;
      var order = inSteps(g.members, g.members[0]);
      // The first conformer of a circle looks towards the middle of the ring.
      order.forEach(function (j, m) {
        var phi = angle + Math.PI + 2 * Math.PI * m / order.length;
        xs[j] = cx + r * Math.cos(phi); ys[j] = cy + r * Math.sin(phi);
      });
      circleRings.push([cx, cy, Math.max(r, 0.04), 'S' + (g.k + 1)]);
    });
  }
  function circlePlaces() {
    var xs = [], ys = [], shown = [], isShown = new Uint8Array(N);
    circleRings = [];
    for (var q = 0; q < N; q++) { xs.push(null); ys.push(null); }
    for (var i = 0; i < M; i++) if (onSurface(i) && inSpecies(i)) { shown.push(i); isShown[i] = 1; }
    if (!shown.length) return [xs, ys];
    if (CIRCLES === 'species' && SPEC < 0 && D.species && D.species.length > 1) {
      speciesCircles(xs, ys, shown);
      saddlesBetween(xs, ys);
      return [xs, ys];
    }
    circleRings.push([0, 0, 1, '']);
    var taken = {}, upper = pathList.nodes.filter(function (j) {
      if (j >= M || !isShown[j] || taken[j]) return false;
      taken[j] = 1; return true;
    });
    var sources = upper.length >= 2 ? upper : [shown[0]];
    var depth = new Int32Array(N).fill(-1), source = new Int32Array(N).fill(-1), queue = [];
    sources.forEach(function (j, n) { depth[j] = 0; source[j] = n; queue.push(j); });
    for (var head = 0; head < queue.length; head++) {
      var at0 = queue[head];
      (linksOf[at0] || []).forEach(function (k) {
        var j = eA[k] === at0 ? eB[k] : (eB[k] === at0 ? eA[k] : -1);
        if (j < 0 || j >= M || !isShown[j] || depth[j] >= 0) return;
        depth[j] = depth[at0] + 1; source[j] = source[at0]; queue.push(j);
      });
    }
    var rest = shown.filter(function (j) { return depth[j] !== 0; });
    rest.sort(function (a, b) {
      var sa = source[a] < 0 ? 1e9 : source[a], sb = source[b] < 0 ? 1e9 : source[b];
      return sa - sb || depth[a] - depth[b] || a - b;
    });
    function put(j, angle) { xs[j] = Math.cos(angle); ys[j] = Math.sin(angle); }
    if (upper.length >= 2) {
      upper.forEach(function (j, n) { put(j, Math.PI * (1 - n / (upper.length - 1))); });
      rest.forEach(function (j, n) { put(j, Math.PI * (1 + (n + 1) / (rest.length + 1))); });
    } else {
      var all = [sources[0]].concat(rest);
      all.forEach(function (j, n) { put(j, Math.PI - 2 * Math.PI * n / all.length); });
    }
    saddlesBetween(xs, ys);
    return [xs, ys];
  }
  if (circleWrap) {
    var ring = function (ctx, P, sc) {
      ctx.save(); ctx.strokeStyle = css('--line-strong', '#888888'); ctx.lineWidth = 1;
      ctx.fillStyle = css('--fg', '#111111'); ctx.font = '600 12px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      circleRings.forEach(function (g) {
        var c = P(g[0], g[1]);
        ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.arc(c[0], c[1], sc * g[2], 0, 2 * Math.PI); ctx.stroke();
        if (g[3]) { ctx.globalAlpha = 0.9; ctx.fillText(g[3], c[0], c[1] - sc * g[2] - 4); }
      });
      ctx.restore();
    };
    var circleStart = circlePlaces(), circleFigure = Figure(circleWrap, circleStart[0], circleStart[1], 'circle', ['', ''], ring);
    var circleKey = '';
    var placeCircle = function () { var c = circlePlaces(); circleFigure.load(c[0], c[1]); };
    circleNew = function () {
      var key = pathList.nodes.join(',');
      if (key !== circleKey) { circleKey = key; placeCircle(); }
    };
    figures.push({ schedule: circleFigure.schedule, focus: circleFigure.focus, surface: placeCircle });
    var modeButtons = document.querySelectorAll('[data-circle-mode]');
    var pressMode = function () { modeButtons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.circleMode === CIRCLES)); }); };
    modeButtons.forEach(function (b) {
      b.addEventListener('click', function () { CIRCLES = b.dataset.circleMode; store('mopt-mapper-circles', CIRCLES); pressMode(); placeCircle(); });
    });
    pressMode();
  }
  // The species shown: all, or the conformers of one (the select above the figures, and the
  // buttons of the table of the species).
  var speciesSelect = document.querySelector('[data-species-select]');
  function showSpecies(k) {
    SPEC = k;
    if (speciesSelect) speciesSelect.value = String(k);
    figures.forEach(function (fig) { if (fig.surface) fig.surface(); });
    // The disconnectivity graph and the energy diagram of the species, where the page has
    // them; those of the whole network otherwise.
    document.querySelectorAll('[data-view="dg"], [data-view="diagram"]').forEach(function (panel) {
      var own = panel.querySelector('[data-species-figure="' + k + '"]');
      panel.querySelectorAll('[data-species-figure]').forEach(function (fig) { fig.hidden = fig !== (own || panel.querySelector('[data-species-figure="-1"]')); });
    });
    refresh();
  }
  if (speciesSelect) speciesSelect.addEventListener('change', function () { showSpecies(+speciesSelect.value); });
  document.querySelectorAll('[data-full]').forEach(function (b) {
    b.addEventListener('click', function (ev) { ev.stopPropagation(); toggleFull(b.parentNode); });
  });
  // The graph of the species: a node is dragged to another place, and the edges of its species
  // (their lines and the numbers at their middles) follow. A drag does not choose the node.
  document.querySelectorAll('svg.species-graph').forEach(function (svg) {
    var held = null, wasDragged = false;
    function at(ev) {
      var m = svg.getScreenCTM();
      if (!m) return null;
      var p = svg.createSVGPoint();
      p.x = ev.clientX; p.y = ev.clientY;
      return p.matrixTransform(m.inverse());
    }
    function place(g, x, y) {
      var circle = g.querySelector('circle'), cy = +circle.getAttribute('cy');
      g.querySelectorAll('text').forEach(function (t) {
        t.setAttribute('x', x.toFixed(1)); t.setAttribute('y', (+t.getAttribute('y') + y - cy).toFixed(1));
      });
      circle.setAttribute('cx', x.toFixed(1)); circle.setAttribute('cy', y.toFixed(1));
      var k = g.dataset.s;
      svg.querySelectorAll('g.species-edge').forEach(function (e) {
        var line = e.querySelector('line'), label = e.querySelector('text');
        if (!line || (e.dataset.a !== k && e.dataset.b !== k)) return;
        if (e.dataset.a === k) { line.setAttribute('x1', x.toFixed(1)); line.setAttribute('y1', y.toFixed(1)); }
        if (e.dataset.b === k) { line.setAttribute('x2', x.toFixed(1)); line.setAttribute('y2', y.toFixed(1)); }
        if (label) {
          label.setAttribute('x', ((+line.getAttribute('x1') + +line.getAttribute('x2')) / 2).toFixed(1));
          label.setAttribute('y', ((+line.getAttribute('y1') + +line.getAttribute('y2')) / 2 - 4).toFixed(1));
        }
      });
    }
    // A drag that starts on a node moves the node; one that starts anywhere else moves the
    // view (the part of the figure that is shown). The wheel zooms about the pointer with Ctrl,
    // and without it where the figure is over the whole screen; a double click shows all again.
    var whole = svg.getAttribute('viewBox');
    function shown() { var b = svg.viewBox.baseVal; return [b.x, b.y, b.width, b.height]; }
    function show(b) { svg.setAttribute('viewBox', b.map(function (v) { return v.toFixed(2); }).join(' ')); }
    svg.addEventListener('pointerdown', function (ev) {
      if (ev.target.closest && ev.target.closest('button')) return;
      var g = ev.target.closest && ev.target.closest('g.species-node');
      var p = at(ev);
      if (!p) return;
      if (g) {
        var circle = g.querySelector('circle');
        held = { g: g, dx: +circle.getAttribute('cx') - p.x, dy: +circle.getAttribute('cy') - p.y, x: ev.clientX, y: ev.clientY, moved: false };
      } else {
        held = { box: shown(), x: ev.clientX, y: ev.clientY, moved: false };
      }
    });
    window.addEventListener('pointermove', function (ev) {
      if (!held) return;
      if (!held.moved && Math.abs(ev.clientX - held.x) + Math.abs(ev.clientY - held.y) <= 3) return;
      held.moved = true;
      if (held.g) {
        var p = at(ev);
        if (!p) return;
        held.g.classList.add('dragged');
        place(held.g, p.x + held.dx, p.y + held.dy);
      } else {
        // The view follows the pointer: the same length on the screen is the same length of
        // the figure (the figure keeps its shape inside the element).
        var r = svg.getBoundingClientRect(), b = held.box;
        var unit = Math.max(b[2] / Math.max(1, r.width), b[3] / Math.max(1, r.height));
        show([b[0] - (ev.clientX - held.x) * unit, b[1] - (ev.clientY - held.y) * unit, b[2], b[3]]);
      }
      ev.preventDefault();
    });
    function release() {
      if (!held) return;
      if (held.g) held.g.classList.remove('dragged');
      wasDragged = held.moved; held = null;
      // The click that ends a drag comes right after: it is not a choice of a node.
      if (wasDragged) setTimeout(function () { wasDragged = false; }, 0);
    }
    // (On the window: the pointer may leave the figure while it is held.)
    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);
    svg.addEventListener('click', function (ev) { if (wasDragged) { ev.stopPropagation(); ev.preventDefault(); wasDragged = false; } }, true);
    svg.addEventListener('wheel', function (ev) {
      if (!ev.ctrlKey && !ev.metaKey && !svg.closest('.is-full')) return;
      var p = at(ev);
      if (!p) return;
      ev.preventDefault();
      var b = shown(), z = ev.deltaY < 0 ? 1 / 1.2 : 1.2;
      show([p.x - (p.x - b[0]) * z, p.y - (p.y - b[1]) * z, b[2] * z, b[3] * z]);
    }, { passive: false });
    svg.addEventListener('dblclick', function (ev) {
      if (ev.target.closest && ev.target.closest('g.species-node, g.species-edge')) return;
      svg.setAttribute('viewBox', whole);
    });
  });
  // The surface bar of a run of several states: one surface, all, or all in layers.
  var surfBar = document.querySelector('[data-surface-bar]');
  if (surfBar && D.pes) {
    var pressSurface = function () {
      surfBar.querySelectorAll('[data-surface]').forEach(function (b) { b.setAttribute('aria-pressed', +b.dataset.surface === SURF ? 'true' : 'false'); });
    };
    surfBar.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('[data-surface]');
      if (!b) return;
      SURF = +b.dataset.surface; store('mopt-mapper-surface', String(SURF)); pressSurface();
      figures.forEach(function (fig) { if (fig.surface) fig.surface(); });
    });
    pressSurface();
  }
  // Redraw on a change of the theme (the viewer script sets it).
  document.querySelectorAll('[data-theme-choice]').forEach(function (btn) { btn.addEventListener('click', function () { figures.forEach(function (fig) { fig.schedule(); }); }); });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var redraw = function () { figures.forEach(function (fig) { fig.schedule(); }); };
    if (mq.addEventListener) mq.addEventListener('change', redraw); else if (mq.addListener) mq.addListener(redraw);
  }

  // One figure at a time.
  var choices = document.querySelectorAll('[data-view-choice]');
  function showView(name) {
    var found = false;
    document.querySelectorAll('[data-view]').forEach(function (p) {
      var on = p.dataset.view === name; p.classList.toggle('active', on); if (on) found = true;
    });
    if (!found) return false;
    choices.forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.viewChoice === name)); });
    store('mopt-mapper-view', name);
    figures.forEach(function (fig) { fig.schedule(); });
    return true;
  }
  choices.forEach(function (b) { b.addEventListener('click', function () { showView(b.dataset.viewChoice); }); });
  if (!showView(stored('mopt-mapper-view') || 'map') && choices.length) showView(choices[0].dataset.viewChoice);

  // A chart of energies: series of [x, energy] points, marks with names.
  function chart(series, marks, xTitle) {
    var w = 620, h = 250, Lm = 64, R = 16, T = 22, B = 42, pw = w - Lm - R, ph = h - T - B;
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    series.forEach(function (sr) { sr.points.forEach(function (p) {
      x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
    }); });
    if (!(x1 > x0)) return '';
    var pad = 0.08 * Math.max(y1 - y0, 1e-9); y0 -= pad; y1 += 2 * pad;
    function X(v) { return (Lm + pw * (v - x0) / (x1 - x0)).toFixed(1); }
    function Y(v) { return (T + ph * (1 - (v - y0) / (y1 - y0))).toFixed(1); }
    var svg = '<svg class="netfig profile" viewBox="0 0 ' + w + ' ' + h + '" role="img" aria-label="Energy profile" xmlns="http://www.w3.org/2000/svg">';
    var ty = ticks(y0, y1, 5);
    ty.values.forEach(function (v) {
      svg += '<line class="grid" x1="' + Lm + '" x2="' + (Lm + pw) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/><text class="tick" x="' + (Lm - 5) + '" y="' + (+Y(v) + 3.5) + '" text-anchor="end">' + v.toFixed(ty.digits) + '</text>';
    });
    var tx = ticks(x0, x1, 6);
    tx.values.forEach(function (v) {
      svg += '<text class="tick" x="' + X(v) + '" y="' + (T + ph + 14) + '" text-anchor="middle">' + v.toFixed(tx.digits) + '</text>';
    });
    svg += '<text class="axis-title" transform="translate(14 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">Relative energy (' + esc(unit) + ')</text>';
    svg += '<text class="axis-title" x="' + (Lm + pw / 2) + '" y="' + (h - 6) + '" text-anchor="middle">' + xTitle + '</text>';
    series.forEach(function (sr) {
      svg += '<polyline class="curve' + (sr.dashed ? ' guess' : '') + '" points="' + sr.points.map(function (p) { return X(p[0]) + ',' + Y(p[1]); }).join(' ') + '"/>';
    });
    marks.forEach(function (m) {
      svg += '<circle class="end" cx="' + X(m.s) + '" cy="' + Y(m.e) + '" r="3.5"/>';
      if (m.text) svg += '<text class="lbl" x="' + X(m.s) + '" y="' + (+Y(m.e) - 7) + '" text-anchor="middle">' + esc(m.text) + '</text>';
    });
    return svg + '</svg>';
  }
  var ARC = 'IRC arc length (amu<tspan baseline-shift="super" font-size="8">1/2</tspan> Bohr)';

  // The chosen node.
  var box = document.getElementById('details');
  var NAMES = ['minimum', 'transition state', 'crossing point (MECP)', 'conical intersection (MECI)'];
  function fact(name, value) { return '<dt>' + name + '</dt><dd>' + value + '</dd>'; }
  // The structure `model` of a node in `slot`: a viewer, or the values of plain variables.
  function showModel(slot, model, name) {
    if (D.variables) {
      // A point of plain variables: its values.
      var fr = model ? frames(model)[0] : null;
      slot.innerHTML = fr ? '<div class="table-wrap"><table class="variables"><thead><tr><th>variable</th><th>value</th></tr></thead><tbody>' +
        fr.names.map(function (n, q) { return '<tr><td>' + esc(n) + '</td><td>' + fr.values[q].toFixed(8) + '</td></tr>'; }).join('') + '</tbody></table></div>'
        : '<p class="note">Its values are not in this page (-report_opt structures=N puts more in).</p>';
    } else if (model && window.moptViewer) {
      var fig = document.createElement('figure'), el = document.createElement('div');
      fig.className = 'viewer-box'; el.className = 'viewer';
      fig.appendChild(el); slot.appendChild(fig);
      window.moptViewer.setup(el, { xyz: model, name: name, animate: false });
    } else {
      slot.innerHTML = '<p class="note">Its structure is not in this page (-report_opt structures=N puts more in).</p>';
    }
  }
  // An end of a descent from a saddle point of higher order (a node id, -1: none): a node of
  // the figures, another saddle point of higher order, or a node that is not shown.
  function descentEnd(id) {
    if (id < 0) return '—';
    if (at[id] !== undefined) return chip(at[id]);
    var name = D.hs.names && D.hs.names[id] ? D.hs.names[id] : 'node ' + id;
    return D.hs.id.indexOf(id) >= 0 ? '<button type="button" class="chip" data-node="' + id + '">' + esc(name) + '</button>' : esc(name) + ' (not shown)';
  }
  // The panel of the saddle point of higher order `hk` (its index in D.hs): what the panel of
  // a node has, and the ends of the descents along its unstable modes.
  function higher(hk, jump) {
    var id = D.hs.id[hk], tag = D.hs.tag[hk], im = D.hs.imag ? D.hs.imag[hk] : [];
    box.querySelectorAll('.viewer').forEach(function (old) { if (old.moptViewer) old.moptViewer.remove(); });
    var html = '<h3>' + esc(tag) + ' <span class="pill">saddle point of order ' + (D.hs.order ? D.hs.order[hk] : im.length) + '</span></h3><dl class="facts">';
    html += fact('Energy', f(D.hs.e[hk]) + ' ' + esc(unit) + (M > 0 ? ' above ' + esc(label(0)) : ''));
    if (D.hs.abs) html += fact(D.variables ? 'Value of the objective' : 'Energy (Hartree)', D.hs.abs[hk].toFixed(8));
    html += fact('Imaginary modes (cm⁻¹)', im.length ? im.length + ': ' + im.map(wave).join(', ') : 'not known');
    if (D.hs.job) html += fact('Found by', D.hs.job[hk] < 0 ? 'an input structure' : 'job ' + D.hs.job[hk]);
    html += fact('Structure', inPage(id) ? get('node:' + id, esc(tag) + '.' + EXT) + ' (in this page)' : 'not in this page (-report_opt structure_packed_atoms=N puts more in)');
    html += '</dl>';
    var ds = D.hs.desc ? D.hs.desc[hk] : [];
    if (ds.length) {
      html += '<h4>Descents along its unstable modes (+ / −)</h4><ul class="links">';
      ds.forEach(function (d1) { html += '<li>mode ' + (d1[0] + 1) + ': ' + descentEnd(d1[1]) + ' / ' + descentEnd(d1[2]) + '</li>'; });
      html += '</ul>';
    } else {
      html += '<p class="note">No descents from it: it is kept as it was found (-map_opt higher_saddles=on follows its unstable modes).</p>';
    }
    html += '<p class="note">Not a state of the kinetics: the flux of the transition state whose IRC reached it is divided among what lies below it.</p><div class="viewer-slot"></div>';
    box.innerHTML = html;
    showNode(box.querySelector('.viewer-slot'), id, tag);
    if (jump) box.scrollIntoView({ behavior: still() ? 'auto' : 'smooth', block: 'start' });
  }
  function details(i) {
    var id = D.id[i];
    box.querySelectorAll('.viewer').forEach(function (old) { if (old.moptViewer) old.moptViewer.remove(); });
    var html = '<h3>' + esc(label(i)) + ' <span class="pill">' + NAMES[D.kind[i]] + '</span></h3><dl class="facts">';
    html += fact('Energy', f(D.e[i]) + ' ' + esc(unit) + (i === 0 && M > 0 ? ' (the lowest minimum)' : (M > 0 ? ' above ' + esc(label(0)) : '')));
    html += fact(D.variables ? 'Value of the objective' : 'Energy (Hartree)', D.abs[i].toFixed(8));
    if (D.g && isFinite(D.g[i])) html += fact('Free energy', f(D.g[i]) + ' ' + esc(unit) + ' above the lowest of the minima shown');
    if (i < M) {
      html += fact('Population at ' + D.temperature + ' K', pct(D.pop[i]) + ' (' + (D.free ? 'free energies' : 'energies') + ' of the minima shown)');
      if (i > 0) html += fact('Highest point on the way from ' + esc(label(0)), D.reach[i] !== null ? f(D.reach[i]) + ' ' + esc(unit) : 'nothing shown joins them');
      html += fact('Jobs from it', String(D.visits[i]));
    }
    if (D.pes && D.pes[i] >= 0) html += fact(D.pes2[i] >= 0 ? 'Surfaces' : 'Surface', esc(D.states[D.pes[i]]) + (D.pes2[i] >= 0 ? '<br>' + esc(D.states[D.pes2[i]]) : ''));
    if (D.imag && D.imag[id]) html += fact('Imaginary modes (cm⁻¹)', D.imag[id].length + ': ' + D.imag[id].map(wave).join(', '));
    else if (i >= M && D.kind[i] === 1) html += fact('Imaginary modes (cm⁻¹)', '1');
    if (D.freq[id]) html += fact('Lowest wavenumbers (cm⁻¹)', D.freq[id].map(wave).join(', '));
    if (D.followed && D.followed[id]) html += fact('Followed states here', D.followed[id].map(function (p) { return 'state ' + p[0] + ': excited state ' + (p[1] + 1) + ' in the order of the energies'; }).join('<br>'));
    if (D.species_of && D.species_of[id] !== undefined) html += fact('Species', 'S' + (D.species_of[id] + 1) + ' · <code>' + esc(D.species[D.species_of[id]]) + '</code>');
    html += fact('Found by', D.job[i] < 0 ? 'an input structure' : 'job ' + D.job[i]);
    if (D.excluded[id]) html += fact('Not explored', esc(D.excluded[id]));
    html += fact('Structure', inPage(id) ? get('node:' + id, esc(label(i)) + '.' + EXT) + ' (in this page)' : 'not in this page (-report_opt structure_packed_atoms=N puts more in)');
    html += '</dl>';
    var ks = (linksOf[i] || []).slice().sort(function (p, q) { return E.top[p] - E.top[q]; });
    var own = ks.filter(function (k) { return eN[k] === i; }), through = ks.filter(function (k) { return eN[k] !== i; });
    var most = 30;
    if (own.length) {
      html += '<h4>Between</h4><ul class="links">';
      own.slice(0, most).forEach(function (k) {
        html += '<li>' + chip(eA[k]) + ' ⇄ ' + chip(eB[k]) + ' · barriers ' + f(E.b0[k]) + ' and ' + f(E.b1[k]) + ' ' + esc(unit) + (E.wave[k] !== null ? ' · ' + wave(E.wave[k]) + ' cm⁻¹' : '') + '</li>';
      });
      html += '</ul>';
    }
    if (through.length) {
      html += '<h4>Links (' + through.length + ')</h4><ul class="links">';
      through.slice(0, most).forEach(function (k) {
        var here = eA[k] === i ? 0 : 1, other = here === 0 ? eB[k] : eA[k];
        html += '<li>' + chip(eN[k]) + ' to ' + (other === i ? 'itself' : chip(other)) + ' · barrier ' + f(here === 0 ? E.b0[k] : E.b1[k]) + ' ' + esc(unit) + '</li>';
      });
      html += '</ul>';
      if (through.length > most) html += '<p class="more">The ' + most + ' lowest of ' + through.length + '.</p>';
    }
    html += '<div class="viewer-slot"></div>';
    var k0 = own.length ? own[0] : -1, p = k0 >= 0 ? profile(k0) : [];
    if (p.length > 1) {
      html += '<h4>' + (D.kind[i] >= 2 ? 'Energy along the meta-IRC of each state' : 'Energy along the IRC') + '</h4>' + chart([{ points: p }], [
        { s: p[0][0], e: p[0][1], text: label(eA[k0]) }, { s: 0, e: E.top[k0], text: label(i) },
        { s: p[p.length - 1][0], e: p[p.length - 1][1], text: label(eB[k0]) }], ARC);
    }
    if (k0 >= 0 && D.irc_xyz[k0] && D.variables) html += '<p class="note">The IRC path is drawn on the map' + (D.map_axes ? '' : ' (with up to four variables)') + '.</p>';
    else if (k0 >= 0 && (D.irc_xyz[k0] || (D.irc_z && D.irc_z[k0]))) html += '<h4>' + (D.kind[i] >= 2 ? 'The path between the two minima through the crossing point (animated)' : 'The IRC path') + '</h4><div class="irc-slot"></div>';
    else if (k0 >= 0 && E.irc[k0]) html += '<p class="note">The structures along the ' + (D.kind[i] >= 2 ? 'path between the two minima' : 'IRC') + ' are not in this page (-report_opt irc_lines=N puts more in).</p>';
    if (i < M && i > 0) html += '<p><button type="button" class="dl" data-path-to="' + i + '">Lowest-barrier path from ' + esc(label(0)) + '</button></p>';
    box.innerHTML = html;
    showNode(box.querySelector('.viewer-slot'), id, label(i));
    var ircSlot = box.querySelector('.irc-slot');
    if (ircSlot && window.moptViewer) {
      var f2 = document.createElement('figure'), v2 = document.createElement('div');
      f2.className = 'viewer-box'; v2.className = 'viewer';
      f2.appendChild(v2); ircSlot.appendChild(f2);
      // The path of a crossing point plays from the minimum of its first state through the
      // crossing point to the minimum of its second state.
      ircText(k0).then(function (text) {
        if (chosen !== i || !v2.isConnected) return;
        // Shown in one orientation (every frame laid on the one before it).
        var shown = D.variables ? text : xyzText(inOneFrame(xyzFrames(text)));
        window.moptViewer.setup(v2, { xyz: shown, name: label(i) + (D.kind[i] >= 2 ? '_path' : '_irc'), animate: D.kind[i] >= 2, captions: D.kind[i] >= 2 });
      }, function (e) { ircSlot.innerHTML = '<p class="warn">' + esc(e.message) + '</p>'; });
    }
  }
  function select(i, jump) {
    if (i < 0 || i >= N) return;
    chosen = i;
    details(i);
    figures.forEach(function (fig) { fig.focus(i); });
    refresh();
    if (jump) box.scrollIntoView({ behavior: still() ? 'auto' : 'smooth', block: 'start' });
  }

  // The structures along the path chosen as one file: the start, then for every step the
  // frames of its IRC (turned round where the step goes against it) and the minimum it ends
  // at. Every piece is laid on the frame before it (the best rotation and translation), since
  // the paths of different jobs are in different frames.
  var lastPath = null;
  function xyzFrames(text) {
    var out = [], lines = String(text || '').split('\n'), j = 0;
    while (j < lines.length) {
      var n = parseInt(lines[j], 10);
      if (!(n > 0)) { j++; continue; }
      var atoms = [];
      for (var q = 0; q < n; q++) {
        var w = (lines[j + 2 + q] || '').trim().split(/\s+/);
        atoms.push([w[0], +w[1], +w[2], +w[3]]);
      }
      out.push({ comment: lines[j + 1] || '', atoms: atoms });
      j += 2 + n;
    }
    return out;
  }
  // The rotation (a 3 x 3 matrix by rows) and the two centres that lay the atoms `p` on the
  // atoms `q` with the least squared distances (Horn 1987: the largest eigenvector of a
  // symmetric 4 x 4 matrix, here by power iteration).
  function superpose(p, q) {
    var n = Math.min(p.length, q.length), cp = [0, 0, 0], cq = [0, 0, 0], S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    if (!n) return null;
    for (var i = 0; i < n; i++) for (var c = 0; c < 3; c++) { cp[c] += p[i][c + 1] / n; cq[c] += q[i][c + 1] / n; }
    for (var i2 = 0; i2 < n; i2++) for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) S[a][b] += (p[i2][a + 1] - cp[a]) * (q[i2][b + 1] - cq[b]);
    var K = [
      [S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]],
      [S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]],
      [S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]],
      [S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]]
    ];
    var shift = 0;
    K.forEach(function (row) { row.forEach(function (v) { shift += Math.abs(v); }); });
    var v = [1, 0, 0, 0];
    for (var it = 0; it < 500; it++) {
      var u = [0, 0, 0, 0], norm = 0;
      for (var r = 0; r < 4; r++) { for (var s2 = 0; s2 < 4; s2++) u[r] += K[r][s2] * v[s2]; u[r] += shift * v[r]; }
      u.forEach(function (x) { norm += x * x; });
      norm = Math.sqrt(norm);
      if (!(norm > 0)) break;
      v = u.map(function (x) { return x / norm; });
    }
    var w0 = v[0], x = v[1], y = v[2], z = v[3];
    var R = [
      [w0 * w0 + x * x - y * y - z * z, 2 * (x * y - w0 * z), 2 * (x * z + w0 * y)],
      [2 * (x * y + w0 * z), w0 * w0 - x * x + y * y - z * z, 2 * (y * z - w0 * x)],
      [2 * (x * z - w0 * y), 2 * (y * z + w0 * x), w0 * w0 - x * x - y * y + z * z]
    ];
    return { R: R, from: cp, to: cq };
  }
  function moved(fit, atoms) {
    return atoms.map(function (at1) {
      var d = [at1[1] - fit.from[0], at1[2] - fit.from[1], at1[3] - fit.from[2]];
      return [at1[0]].concat([0, 1, 2].map(function (c) { return fit.R[c][0] * d[0] + fit.R[c][1] * d[1] + fit.R[c][2] * d[2] + fit.to[c]; }));
    });
  }
  function apart(p, q) {
    var s = 0;
    for (var i = 0; i < p.length; i++) for (var c = 1; c < 4; c++) s += (p[i][c] - q[i][c]) * (p[i][c] - q[i][c]);
    return Math.sqrt(s / Math.max(1, p.length));
  }
  // The frames of one path, each laid on the one before it (a rotation and a translation):
  // the frames of a path file are not all in one orientation (the transition state of an IRC
  // file is the structure of its node, the sides are as the IRC ran).
  function inOneFrame(frames) {
    for (var k = 1; k < frames.length; k++) {
      // Not in a periodic cell, whose frames share the cell.
      if (frames[k].atoms.length !== frames[k - 1].atoms.length || /Lattice=/i.test(frames[k].comment)) continue;
      var fit = superpose(frames[k].atoms, frames[k - 1].atoms);
      if (fit) frames[k].atoms = moved(fit, frames[k].atoms);
    }
    return frames;
  }
  function xyzText(frames) {
    return frames.map(function (fr) {
      return fr.atoms.length + '\n' + fr.comment + '\n' + fr.atoms.map(function (at1) { return at1[0] + ' ' + at1[1].toFixed(5) + ' ' + at1[2].toFixed(5) + ' ' + at1[3].toFixed(5); }).join('\n') + '\n';
    }).join('');
  }
  // How the atoms `p` are laid on the atoms `q` when atoms of one element may have changed
  // places (two jobs can number equivalent atoms differently): superposition and the nearest
  // pairs of each element in turn, while the distance falls. `order[b]` is the atom of `p`
  // at the place of atom `b` of `q`.
  function matched(p, q) {
    var n = q.length, order = [], best = null;
    for (var i = 0; i < n; i++) order.push(i);
    for (var round = 0; round < 8; round++) {
      var pm = order.map(function (j) { return p[j]; }), fit = superpose(pm, q);
      if (!fit) break;
      var laid = moved(fit, pm), r = apart(laid, q);
      if (best && r >= best.r - 1e-9) break;
      best = { r: r, order: order.slice(), fit: fit };
      var pairs = [];
      for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) {
        if (laid[a][0] !== q[b][0]) continue;
        var d = 0;
        for (var c = 1; c < 4; c++) d += (laid[a][c] - q[b][c]) * (laid[a][c] - q[b][c]);
        pairs.push([d, a, b]);
      }
      pairs.sort(function (x, y) { return x[0] - y[0]; });
      var usedA = new Uint8Array(n), usedB = new Uint8Array(n), next = new Array(n), paired = 0;
      pairs.forEach(function (pr) {
        if (usedA[pr[1]] || usedB[pr[2]]) return;
        usedA[pr[1]] = usedB[pr[2]] = 1; next[pr[2]] = order[pr[1]]; paired++;
      });
      if (paired !== n) break;
      order = next;
    }
    return best;
  }
  // The structures of the path chosen as one run of frames (a promise of them): every piece
  // (the start, the path of every step, the minimum it ends at) laid on the frame before it,
  // with atoms of one element matched where two jobs numbered them differently. Where two
  // pieces do not meet (an IRC that stopped before its minimum in a path file without the
  // minimizations, one minimum as two jobs found it), frames interpolated on a straight line
  // are put between them, so that the path is one motion; their titles say they are not
  // computed.
  var JOIN = 0.12, JOIN_STEP = 0.08, JOIN_MOST = 16;
  function joinedPath() {
    if (!lastPath) { out.innerHTML = '<p class="warn">Find a path first.</p>'; return Promise.resolve(null); }
    if (D.variables) { out.insertAdjacentHTML('beforeend', '<p class="note">The path of a network of plain variables is drawn on the map.</p>'); return Promise.resolve(null); }
    var path = lastPath;
    var nodeText = function (i) { return modelText(D.id[i]) || Promise.resolve(null); };
    var wanted = [nodeText(path.start)];
    path.steps.forEach(function (st) { wanted.push(ircText(st.k) || Promise.resolve(null), nodeText(eN[st.k]), nodeText(st.to)); });
    return Promise.all(wanted).then(function (found) {
      var r = { frames: [], without: 0, renumbered: false, joins: 0, interpolated: 0, name: 'path_' + label(path.start) + '_' + label(path.end) };
      var last = null;
      function one(text) { return text ? xyzFrames(text).slice(0, 1) : []; }
      function add(piece) {
        if (!piece.length) return;
        inOneFrame(piece);
        var same = last && piece[0].atoms.length === last.atoms.length && !/Lattice=/i.test(piece[0].comment);
        var m = same ? matched(piece[0].atoms, last.atoms) : null;
        piece.forEach(function (fr) {
          if (m) {
            if (m.order.some(function (j, b) { return j !== b; })) r.renumbered = true;
            fr.atoms = moved(m.fit, m.order.map(function (j) { return fr.atoms[j]; }));
          }
        });
        if (same) {
          var first = piece[0].atoms, most = 0;
          for (var a = 0; a < first.length; a++) {
            var d = 0;
            for (var c = 1; c < 4; c++) d += (first[a][c] - last.atoms[a][c]) * (first[a][c] - last.atoms[a][c]);
            most = Math.max(most, Math.sqrt(d));
          }
          if (most > JOIN) {
            var steps = Math.min(JOIN_MOST, Math.ceil(most / JOIN_STEP));
            for (var t = 1; t < steps; t++) {
              r.frames.push({
                comment: 'interpolated (not computed) between two structures of the path, ' + t + ' of ' + (steps - 1),
                atoms: first.map(function (p, a2) { var q = last.atoms[a2]; return [p[0], q[1] + (p[1] - q[1]) * t / steps, q[2] + (p[2] - q[2]) * t / steps, q[3] + (p[3] - q[3]) * t / steps]; })
              });
            }
            r.joins++; r.interpolated += steps - 1;
          }
        }
        piece.forEach(function (fr) { r.frames.push(fr); });
        last = r.frames[r.frames.length - 1];
      }
      add(one(found[0]));
      path.steps.forEach(function (st, n) {
        var piece = found[1 + 3 * n] ? xyzFrames(found[1 + 3 * n]) : [];
        if (piece.length) { if (st.from !== eA[st.k]) piece.reverse(); }
        else { r.without++; piece = one(found[2 + 3 * n]); }
        add(piece); add(one(found[3 + 3 * n]));
      });
      if (!r.frames.length) { out.insertAdjacentHTML('beforeend', '<p class="warn">The page holds no structure of this path.</p>'); return null; }
      return r;
    }, function (e) { out.insertAdjacentHTML('beforeend', '<p class="warn">' + esc(e.message) + '</p>'); return null; });
  }
  // What is to be said about the joined path, once.
  function pathNotes(r) {
    out.querySelectorAll('.path-note').forEach(function (old) { old.remove(); });
    var notes = [];
    // Frames that no calculation gave are a warning, not a note.
    if (r.joins) notes.push(['warn', 'Warning: ' + r.interpolated + ' of the ' + r.frames.length + ' frames of this path were not computed. At ' + r.joins + ' place(s) two pieces of the path do not meet within ' + JOIN + ' Å (an IRC that stopped before its minimum, or one minimum as two jobs found it), and frames on a straight line between them make the path one motion; their titles say “interpolated (not computed)”. They are no part of a reaction path, and their energies are not known.']);
    if (r.renumbered) notes.push(['note', 'Atoms of one element are renumbered where two steps had equivalent atoms in other places, so that the structures follow each other without jumps; the numbers are those of the start.']);
    if (r.without) notes.push(['note', r.without + ' step(s) have their transition state alone: the page does not hold their path (-report_opt irc_packed_lines=N puts more in).']);
    notes.forEach(function (n) { out.insertAdjacentHTML('beforeend', '<p class="' + n[0] + ' path-note">' + esc(n[1]) + '</p>'); });
  }
  function savePath() {
    joinedPath().then(function (r) { if (!r) return; save(r.name + '.' + EXT, xyzText(r.frames)); pathNotes(r); });
  }
  // The joined path played in the page.
  function playPath() {
    joinedPath().then(function (r) {
      if (!r || !window.moptViewer) return;
      var slot = out.querySelector('.path-player');
      if (!slot) { out.insertAdjacentHTML('beforeend', '<div class="path-player"></div>'); slot = out.querySelector('.path-player'); }
      slot.querySelectorAll('.viewer').forEach(function (old) { if (old.moptViewer) old.moptViewer.remove(); });
      slot.innerHTML = '<h4>The path from ' + esc(label(lastPath.start)) + ' to ' + esc(label(lastPath.end)) + ' (' + r.frames.length + ' frames)</h4><figure class="viewer-box"><div class="viewer"></div></figure>';
      window.moptViewer.setup(slot.querySelector('.viewer'), { xyz: xyzText(r.frames), name: r.name, animate: true, captions: true });
      pathNotes(r);
    });
  }

  // The lowest-barrier path: the path of the minimum spanning tree.
  var from = document.getElementById('path-from'), to = document.getElementById('path-to'), out = document.getElementById('path-result');
  var adj = [];
  for (var k2 = 0; k2 < L; k2++) {
    if (!E.tree[k2]) continue;
    (adj[eA[k2]] = adj[eA[k2]] || []).push([eB[k2], k2]);
    (adj[eB[k2]] = adj[eB[k2]] || []).push([eA[k2], k2]);
  }
  function treePath(a, b) {
    if (a === b) return [];
    var came = new Int32Array(N).fill(-2), via = new Int32Array(N), queue = [a], head = 0;
    came[a] = -1;
    while (head < queue.length) {
      var i = queue[head++];
      if (i === b) break;
      (adj[i] || []).forEach(function (st) { if (came[st[0]] === -2) { came[st[0]] = i; via[st[0]] = st[1]; queue.push(st[0]); } });
    }
    if (came[b] === -2) return null;
    var steps = [], here = b;
    while (came[here] >= 0) { steps.push({ from: came[here], to: here, k: via[here] }); here = came[here]; }
    return steps.reverse();
  }
  var byLabel = {};
  for (var i4 = 0; i4 < M; i4++) byLabel[label(i4).toUpperCase()] = i4;
  function minimumOf(input) {
    var t = input.value.trim().toUpperCase();
    if (/^[0-9]+$/.test(t)) t = 'EQ' + t;
    return t in byLabel ? byLabel[t] : -1;
  }
  function findPath() {
    var a = minimumOf(from), b = minimumOf(to);
    lastPath = null;
    if (a < 0 || b < 0) { out.innerHTML = '<p class="warn">Give two minima shown, as EQ12 or 12.</p>'; return; }
    var steps = treePath(a, b);
    if (steps && steps.length) lastPath = { start: a, end: b, steps: steps };
    pathList = { nodes: [], edges: [] };
    if (steps === null) {
      refresh();
      out.innerHTML = '<p class="warn">Nothing in the network shown joins ' + esc(label(a)) + ' and ' + esc(label(b)) + '.</p>';
      return;
    }
    if (!steps.length) { pathList.nodes = [a]; refresh(); out.innerHTML = '<p class="note">That is one minimum.</p>'; return; }
    var high = D.e[a], highAt = a, series = [], marks = [], s = 0, guessed = false, rows = '', named = steps.length <= 8;
    pathList.nodes.push(a);
    marks.push({ s: 0, e: D.e[a], text: label(a) });
    steps.forEach(function (st, n) {
      var k = st.k, forward = st.from === eA[k], prof = profile(k), irc = prof.length > 1;
      pathList.nodes.push(eN[k], st.to); pathList.edges.push(k);
      if (E.top[k] > high) { high = E.top[k]; highAt = eN[k]; }
      var pts = irc ? (forward ? prof : prof.map(function (q) { return [-q[0], q[1]]; }).reverse())
        : [[0, D.e[st.from]], [1, E.top[k]], [2, D.e[st.to]]];
      if (!irc) guessed = true;
      var shift = s - pts[0][0];
      pts = pts.map(function (q) { return [q[0] + shift, q[1]]; });
      series.push({ points: pts, dashed: !irc });
      marks.push({ s: (irc ? 0 : 1) + shift, e: E.top[k], text: named ? label(eN[k]) : '', ts: eN[k] });
      s = pts[pts.length - 1][0];
      marks.push({ s: s, e: D.e[st.to], text: named || n === steps.length - 1 ? label(st.to) : '' });
      if (n < 200) rows += '<li>' + chip(st.from) + ' → ' + chip(eN[k]) + ' → ' + chip(st.to) + ' · barrier ' + f(forward ? E.b0[k] : E.b1[k]) + ', then ' + f(D.e[st.to] - D.e[st.from]) + ' ' + esc(unit) + '</li>';
    });
    if (!named) marks.forEach(function (m) { if (m.ts === highAt) m.text = label(highAt); });
    refresh();
    var html = '<p>Highest point: <b>' + f(high) + ' ' + esc(unit) + '</b> at ' + chip(highAt) + ', ' + f(high - D.e[a]) + ' ' + esc(unit) + ' above ' + esc(label(a)) + '; ' + steps.length + ' step(s).</p>';
    html += '<ol class="steps">' + rows + '</ol>' + (steps.length > 200 ? '<p class="more">The first 200 steps.</p>' : '') + chart(series, marks, ARC);
    if (guessed) html += '<p class="note">Dashed: a link without the energies of its IRC in this page, drawn one unit long.</p>';
    out.innerHTML = html;
  }
  if (from && to && out) {
    var list = document.getElementById('minima-list'), options = '';
    for (var i5 = 0; i5 < Math.min(M, 1000); i5++) options += '<option value="' + label(i5) + '">' + f(D.e[i5]) + ' ' + esc(unit) + '</option>';
    list.innerHTML = options;
    if (M > 1) { from.value = label(0); to.value = label(1); }
    document.getElementById('path-find').addEventListener('click', findPath);
    [from, to].forEach(function (inp) { inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') findPath(); }); });
    document.getElementById('path-swap').addEventListener('click', function () { var v = from.value; from.value = to.value; to.value = v; findPath(); });
    document.getElementById('path-use-from').addEventListener('click', function () { if (chosen >= 0 && chosen < M) { from.value = label(chosen); findPath(); } });
    document.getElementById('path-use-to').addEventListener('click', function () { if (chosen >= 0 && chosen < M) { to.value = label(chosen); findPath(); } });
    document.getElementById('path-clear').addEventListener('click', function () { pathList = { nodes: [], edges: [] }; lastPath = null; refresh(); out.innerHTML = ''; });
    var saveButton = document.getElementById('path-save');
    if (saveButton) saveButton.addEventListener('click', savePath);
    var playButton = document.getElementById('path-play');
    if (playButton) playButton.addEventListener('click', playPath);
  }

  // The tables, a page at a time.
  function Pager(name, rows, render) {
    var bar = document.querySelector('[data-pager="' + name + '"]'), tbody = document.querySelector('#' + name + '-table tbody');
    if (!bar || !tbody) return null;
    bar.innerHTML = '<input type="search" placeholder="Filter by label" aria-label="Filter the table by label"><button type="button" class="dl" data-p="-1">Previous</button><span class="pageinfo"></span><button type="button" class="dl" data-p="1">Next</button>';
    var input = bar.querySelector('input'), info = bar.querySelector('.pageinfo'), page = 0, size = 100, shown = rows;
    function show() {
      var pages = Math.max(1, Math.ceil(shown.length / size));
      page = Math.max(0, Math.min(page, pages - 1));
      var html = '', end = Math.min(shown.length, (page + 1) * size);
      for (var j = page * size; j < end; j++) html += render(shown[j]);
      tbody.innerHTML = html;
      info.textContent = (shown.length ? (page * size + 1) + '–' + end : '0') + ' of ' + shown.length;
    }
    input.addEventListener('input', function () {
      var q = input.value.trim().toUpperCase();
      shown = q ? rows.filter(function (i) { return label(i).toUpperCase().indexOf(q) >= 0; }) : rows;
      page = 0; show();
    });
    bar.addEventListener('click', function (e) { var b = e.target.closest && e.target.closest('[data-p]'); if (b) { page += +b.dataset.p; show(); } });
    return { show: show };
  }
  function cls(i) { return (i === chosen ? ' sel' : '') + (pathList.nodes.indexOf(i) >= 0 ? ' on-path' : ''); }
  var minimaRows = [], tsRows = [];
  for (var i6 = 0; i6 < N; i6++) { if (i6 < M) minimaRows.push(i6); else if (D.kind[i6] === 1) tsRows.push(i6); }
  var pm = Pager('minima', minimaRows, function (i) {
    var id = D.id[i], fr = D.freq[id] ? D.freq[id].slice(0, 3).map(wave).join(', ') : '—';
    return '<tr data-node="' + id + '" tabindex="0" class="' + cls(i) + '"><td>' + named(i) + '</td><td class="num">' + f(D.e[i]) + '</td><td class="num">' + D.abs[i].toFixed(8) + '</td><td class="num">' + pct(D.pop[i]) + '</td><td class="num">' + (i === 0 ? '—' : (D.reach[i] === null ? 'not joined' : f(D.reach[i]))) + '</td><td>' + fr + '</td><td class="num">' + D.visits[i] + '</td><td>' + (D.job[i] < 0 ? 'input' : 'job ' + D.job[i]) + '</td><td>' + (D.excluded[id] ? esc(D.excluded[id]) : '') + '</td><td>' + get('node:' + id, 'xyz') + '</td></tr>';
  });
  var pt = Pager('ts', tsRows, function (i) {
    var id = D.id[i], k = firstLink[i];
    if (k < 0) {
      var fr = D.freq[id] ? wave(D.freq[id][0]) : '—';
      return '<tr data-node="' + id + '" tabindex="0" class="' + cls(i) + '"><td>' + named(i) + '</td><td class="num">' + f(D.e[i]) + '</td><td class="num">' + D.abs[i].toFixed(8) + '</td><td>— (joins no two minima shown)</td><td class="num">—</td><td class="num">—</td><td class="num">' + fr + '</td><td></td><td>' + get('node:' + id, 'xyz') + '</td></tr>';
    }
    return '<tr data-node="' + id + '" tabindex="0" class="' + cls(i) + '"><td>' + named(i) + '</td><td class="num">' + f(D.e[i]) + '</td><td class="num">' + D.abs[i].toFixed(8) + '</td><td>' + label(eA[k]) + ' → ' + label(eB[k]) + '</td><td class="num">' + f(E.b0[k]) + '</td><td class="num">' + f(E.b1[k]) + '</td><td class="num">' + imaginary(id, k) + '</td><td>' + (E.tree[k] ? '✓' : '') + '</td><td>' + get('node:' + id, 'xyz') + (held('irc:' + k) ? ' · ' + get('irc:' + k, 'IRC') : '') + '</td></tr>';
  });
  [pm, pt].forEach(function (p) { if (p) pagers.push(p); });

  // Choosing: a click (or Enter) on anything with a node.
  document.addEventListener('click', function (ev) {
    var t = ev.target;
    var g = t.closest && t.closest('[data-get]');
    var sp = t.closest && t.closest('[data-species]');
    if (sp) {
      showSpecies(+sp.dataset.species);
      var net = document.getElementById('network');
      if (net) net.scrollIntoView({ behavior: still() ? 'auto' : 'smooth', block: 'start' });
      return;
    }
    if (g) {
      var h = held(g.dataset.get);
      if (h) {
        ev.preventDefault();
        if (h.packed !== undefined) ircText(h.packed).then(function (text) { save(h.name, text); }, function (e) { window.alert(e.message); });
        else if (h.packedModel !== undefined) modelText(h.packedModel).then(function (text) { save(h.name, text); }, function (e) { window.alert(e.message); });
        else save(h.name, h.text);
      }
      return;
    }
    if (!t.closest || t.closest('a')) return;
    var p = t.closest('[data-path-to]');
    if (p && from) {
      from.value = label(0); to.value = label(+p.dataset.pathTo); findPath();
      document.getElementById('paths').scrollIntoView({ behavior: still() ? 'auto' : 'smooth', block: 'start' });
      return;
    }
    var hit = t.closest('[data-node]');
    if (!hit) return;
    if (!(hit.getAttribute('data-node') in at)) {
      // A saddle point of higher order (a row of its table, an end of a descent).
      var hk = D.hs ? D.hs.id.indexOf(+hit.getAttribute('data-node')) : -1;
      if (hk >= 0) higher(hk, !box.contains(hit));
      return;
    }
    select(at[hit.getAttribute('data-node')], !box.contains(hit) && !hit.closest('svg'));
  });
  document.addEventListener('keydown', function (ev) {
    if (ev.key !== 'Enter' && ev.key !== ' ') return;
    var hit = ev.target.closest && ev.target.closest('[data-node]');
    if (!hit || hit.tagName === 'BUTTON') return;
    if (!(hit.getAttribute('data-node') in at)) {
      var hk = D.hs ? D.hs.id.indexOf(+hit.getAttribute('data-node')) : -1;
      if (hk >= 0) { ev.preventDefault(); higher(hk, true); }
      return;
    }
    ev.preventDefault();
    select(at[hit.getAttribute('data-node')], !hit.closest('svg'));
  });

  if (chosen >= 0) select(chosen, false); else refresh();
})();
</script>
"##;
