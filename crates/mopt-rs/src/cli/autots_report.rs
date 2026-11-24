//! The report of `run_autots`: one HTML page, `autots_report.html` in the working directory,
//! with every step of the run.
//!
//! - A summary of the optimizations of step 3 (energy, imaginary frequency, whether it is a
//!   transition state, the minima of its IRC, barriers, reaction energy), with a warning
//!   when some are not transition states; an energy diagram of all transition states on
//!   one scale, each between the two minima its IRC ends at (the same minimum has the same
//!   label, so transition states that share one connect), and the table of the minima.
//! - Step 1: the energy along the first path with and without its bias (AFIR or another
//!   bias potential, named in the heading), and the path itself; the given path with
//!   `--skip_step1`.
//! - Step 2: what the method of the path made (the chain, DMF, BITSS, ...): its energy
//!   profile with the candidates where the energies are known, the path, and the estimates
//!   of transition states.
//! - Step 3: every optimization with its lowest mode moving; one that did not converge or
//!   has the wrong number of imaginary frequencies is a "candidate" with a warning.
//! - Step 4: the energy along each IRC and the IRC itself, with the optimized ends.
//! - The settings of the run and the main files.
//!
//! The figures are SVG drawn by mopt-rs ([`crate::core_infra::viz`]) inside the page; the
//! structures are drawn by a small viewer written into the page ([`VIEWER`]: on a canvas,
//! turned with the mouse or a finger, zoomed with the wheel; ball and stick, sticks,
//! space-filling or wireframe, for one molecule or all), with the atoms labelled by their
//! numbers (1-based, as the options of mopt-rs count them), by their elements, by both, or not
//! at all (for one molecule or all), the frames of a path played or stepped through, the periodic
//! cell of extended xyz frames, and the ONIOM layers of the settings (the high layer in the
//! chosen style, a middle layer as sticks, the rest as faded lines). Every structure can be
//! saved as an xyz file (one frame or all). The page follows the light or dark theme of the
//! system, or the one chosen on it. It is one file: it loads nothing from elsewhere, and the
//! structures are in it, so it can be moved or sent without the run directory.
//!
//! MultiOptPy: no counterpart (new in mopt-rs; MultiOptPy's `run_autots` writes
//! `energy_profile.png` and `energy_profile.txt` per transition state).
//!
//! This file is part of mopt-rs.
//! Copyright (C) 2023-2026 ss0832
//! SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use faer::Col;

use super::autots::{AutotsConfig, AutotsResult, BandProfile, PathStage, TransitionState};
use crate::core_infra::arg_parser::parse_atom_indices_one_based;
use crate::core_infra::json::JsonValue;
use crate::core_infra::viz::color::Rgb;
use crate::core_infra::viz::figure::{PlotFigure, PlotSeries, PlotStyle};
use crate::domain::{SpinCharge, Units};
use crate::error::MoptResult;
use crate::io::xyz::read_xyz_trajectory_file;
use crate::path::band::{path_lengths, ChainKind};

/// The structure viewer of the page (plain JavaScript on a canvas, no library): display
/// styles (ball and stick, sticks, space-filling, wireframe), atom numbers, the periodic cell
/// of extended xyz frames (bonds to the next cells drawn as halves), the ONIOM layers, the
/// frames of a path, downloads of the structures as xyz files, and the light or dark theme.
pub const VIEWER: &str = r##"<script>
(function () {
  var root = document.documentElement;
  var models = JSON.parse(document.getElementById('models').textContent);
  var info = JSON.parse(document.getElementById('page-info').textContent);
  // Element colors after the CPK convention (as in Jmol), single-bond covalent radii of Pyykko
  // and Atsumi (2009), and van der Waals radii mostly of Bondi (1964) (Angstrom).
  var COLOR = { H: '#e8e8e8', N: '#3050f8', O: '#ff0d0d', F: '#90e050', S: '#ffd02a', Cl: '#1ff01f',
    Br: '#a62929', I: '#940094', P: '#ff8000', Si: '#f0c8a0', B: '#ffb5b5', Cu: '#c88033', Ag: '#c0c0c0', Au: '#ffd123',
    Pd: '#006985', Pt: '#d0d0e0', Ni: '#50d050', Fe: '#e06633', Al: '#bfa6a6' };
  var RADIUS = { H: 0.32, C: 0.75, N: 0.71, O: 0.63, F: 0.64, S: 1.03, Cl: 0.99, Br: 1.14, I: 1.33, P: 1.11, Si: 1.16,
    B: 0.85, Cu: 1.12, Ag: 1.28, Au: 1.24, Pd: 1.20, Pt: 1.23, Ni: 1.10, Fe: 1.16, Al: 1.26 };
  var VDW = { H: 1.20, C: 1.70, N: 1.55, O: 1.52, F: 1.47, S: 1.80, Cl: 1.75, Br: 1.85, I: 1.98, P: 1.80, Si: 2.10,
    B: 1.92, Cu: 1.40, Ag: 1.72, Au: 1.66, Pd: 1.63, Pt: 1.75, Ni: 1.63, Fe: 1.94, Al: 1.84 };
  // Display styles: bond width and atom radius in Angstrom (bond 0: no bonds, -1: thin lines).
  var STYLES = {
    ball: { name: 'Ball and stick', bond: 0.16, atom: function (e) { return 0.32 * (RADIUS[e] || 1.0); }, shade: true },
    stick: { name: 'Sticks', bond: 0.30, atom: function () { return 0.15; }, shade: false },
    spacefill: { name: 'Space-filling', bond: 0, atom: function (e) { return 0.9 * (VDW[e] || 1.8); }, shade: true },
    wire: { name: 'Wireframe', bond: -1, atom: null, shade: false }
  };
  var high = {}, middle = {};
  if (info.layers) {
    (info.layers.high || []).forEach(function (i) { high[i] = true; });
    (info.layers.middle || []).forEach(function (i) { middle[i] = true; });
  }
  // The links of the top bar scroll the page to their section themselves: where the page is
  // shown embedded (a preview, a data URL), following a "#name" link may not scroll.
  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      var target = document.getElementById(a.getAttribute('href').slice(1));
      if (!target) return;
      e.preventDefault();
      // Clear of the sticky top bar, whose height depends on the width of the page.
      var bar = document.querySelector('.topbar');
      target.style.scrollMarginTop = ((bar ? bar.offsetHeight : 0) + 12) + 'px';
      var still = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      target.scrollIntoView({ behavior: still ? 'auto' : 'smooth', block: 'start' });
      try { history.replaceState(null, '', a.getAttribute('href')); } catch (err) { /* not here */ }
    });
  });
  function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* no storage */ } }
  function stored(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }

  // Colors of the theme, read from the page's CSS variables.
  var palette = {};
  function readPalette() {
    var s = getComputedStyle(root);
    function v(name, d) { var x = s.getPropertyValue(name).trim(); return x || d; }
    palette = { bg: v('--viewer-bg', '#ffffff'), ink: v('--viewer-ink', '#111111'), halo: v('--viewer-halo', 'rgba(255,255,255,0.85)'),
      carbon: v('--viewer-carbon', '#505050'), cell: v('--viewer-cell', 'rgba(0,0,0,0.45)'), dim: v('--viewer-dim', 'rgba(120,120,120,0.55)') };
  }
  var viewers = [];
  function redrawAll() { readPalette(); viewers.forEach(function (v) { v.draw(); }); }
  function setTheme(t) {
    if (t === 'light' || t === 'dark') root.dataset.theme = t; else { delete root.dataset.theme; t = 'auto'; }
    document.querySelectorAll('[data-theme-choice]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.themeChoice === t)); });
    store('mopt-report-theme', t);
    redrawAll();
  }
  document.querySelectorAll('[data-theme-choice]').forEach(function (b) {
    b.addEventListener('click', function () { setTheme(b.dataset.themeChoice); });
  });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', redrawAll); else if (mq.addListener) mq.addListener(redrawAll);
  }
  setTheme(stored('mopt-report-theme') || 'auto');

  // Downloads of the embedded structures as xyz files (made in the page, nothing is fetched).
  function download(name, text) {
    var url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    var a = document.createElement('a');
    a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }

  function parse(text) {
    var lines = text.split('\n'), out = [], k = 0;
    while (k < lines.length) {
      if (lines[k].trim() === '') { k++; continue; }
      var n = parseInt(lines[k], 10);
      if (!(n > 0)) break;
      var comment = lines[k + 1] || '', cell = null;
      var lat = /Lattice="([^"]+)"/i.exec(comment);
      if (lat) {
        var v = lat[1].trim().split(/\s+/).map(Number);
        if (v.length === 9 && v.every(isFinite)) cell = [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]];
      }
      var atoms = [];
      for (var i = 0; i < n; i++) {
        var f = (lines[k + 2 + i] || '').trim().split(/\s+/);
        atoms.push({ el: f[0], x: +f[1], y: +f[2], z: +f[3] });
      }
      out.push({ atoms: atoms, cell: cell, comment: comment.trim(), text: lines.slice(k, k + 2 + n).join('\n') + '\n' });
      k += n + 2;
    }
    return out;
  }
  // Bonds by covalent radii; in a periodic cell also to the images in the next cells (t is the
  // translation of the second atom).
  function bonds(frame) {
    var atoms = frame.atoms, out = [], shifts = [[0, 0, 0]];
    if (frame.cell) {
      var c = frame.cell;
      shifts = [];
      for (var a = -1; a <= 1; a++) for (var b = -1; b <= 1; b++) for (var g = -1; g <= 1; g++)
        shifts.push([a * c[0][0] + b * c[1][0] + g * c[2][0], a * c[0][1] + b * c[1][1] + g * c[2][1], a * c[0][2] + b * c[1][2] + g * c[2][2]]);
    }
    for (var i = 0; i < atoms.length; i++) {
      for (var j = 0; j < i; j++) {
        var p = atoms[i], q = atoms[j];
        var r = 1.2 * ((RADIUS[p.el] || 1.0) + (RADIUS[q.el] || 1.0));
        for (var s = 0; s < shifts.length; s++) {
          var t = shifts[s], dx = p.x - q.x - t[0], dy = p.y - q.y - t[1], dz = p.z - q.z - t[2];
          if (dx * dx + dy * dy + dz * dz < r * r) out.push({ i: i, j: j, t: t, image: t[0] !== 0 || t[1] !== 0 || t[2] !== 0 });
        }
      }
    }
    return out;
  }
  function multiply(a, b) {
    var c = [];
    for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++)
      c[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    return c;
  }
  function rotation(ax, ay) {
    var cx = Math.cos(ax), sx = Math.sin(ax), cy = Math.cos(ay), sy = Math.sin(ay);
    return multiply([cy, 0, sy, 0, 1, 0, -sy, 0, cy], [1, 0, 0, 0, cx, -sx, 0, sx, cx]);
  }
  var globalStyle = STYLES[stored('mopt-report-style')] ? stored('mopt-report-style') : 'ball';
  // The labels of the atoms: their numbers (from 1), nothing, their elements, or both.
  var LABELS = { number: 'Atom numbers', none: 'No labels', element: 'Elements', both: 'Elements and numbers' };
  var globalLabels = LABELS[stored('mopt-report-labels')] ? stored('mopt-report-labels') : 'number';
  function labelOf(mode, a, k) { return mode === 'number' ? String(k + 1) : (mode === 'element' ? a.el : a.el + (k + 1)); }

  // The viewer of the model m ({xyz, name, animate, captions}) in the element el; its handle can draw,
  // restyle, and remove it.
  function setup(el, m) {
    var list = parse(m.xyz);
    if (!list.length) return null;
    // The drawing and the link that stand in where scripts do not run give way to the viewer.
    el.textContent = '';
    var standIn = el.parentNode.querySelector('.static-controls');
    if (standIn) standIn.remove();
    var canvas = document.createElement('canvas');
    el.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    // Centre and size from all frames (and the cell), so that the picture does not jump.
    var cx = 0, cy = 0, cz = 0, count = 0, periodic = list.some(function (f) { return f.cell; });
    list.forEach(function (f) { f.atoms.forEach(function (a) { cx += a.x; cy += a.y; cz += a.z; count++; }); });
    cx /= count; cy /= count; cz /= count;
    var extent = 1;
    list.forEach(function (f) {
      f.atoms.forEach(function (a) { extent = Math.max(extent, Math.hypot(a.x - cx, a.y - cy, a.z - cz) + 0.8); });
      if (f.cell) {
        var c = f.cell;
        for (var u = 0; u < 2; u++) for (var v = 0; v < 2; v++) for (var w = 0; w < 2; w++)
          extent = Math.max(extent, Math.hypot(u * c[0][0] + v * c[1][0] + w * c[2][0] - cx,
            u * c[0][1] + v * c[1][1] + w * c[2][1] - cy, u * c[0][2] + v * c[1][2] + w * c[2][2] - cz) + 0.5);
      }
    });
    var bondCache = [];
    function bondsOf(k) { if (!bondCache[k]) bondCache[k] = bonds(list[k]); return bondCache[k]; }
    var rot = [1, 0, 0, 0, 1, 0, 0, 0, 1], zoom = 1, current = 0, timer = null;
    var style = globalStyle, labels = globalLabels, layers = !!info.layers, showCell = true;
    // The style of an atom: with the ONIOM layers shown, the high layer in the chosen style,
    // the middle layer as sticks, and the rest as thin, faded lines.
    function atomStyle(k) {
      if (!layers || high[k]) return { st: STYLES[style], dim: false };
      if (middle[k]) return { st: style === 'wire' ? STYLES.wire : STYLES.stick, dim: false };
      return { st: STYLES.wire, dim: true };
    }
    function colorOf(el) { return el === 'C' ? palette.carbon : (COLOR[el] || '#ff1493'); }
    function draw() {
      var w = el.clientWidth, h = el.clientHeight, dpr = window.devicePixelRatio || 1;
      if (!w || !h) return;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.bg; ctx.fillRect(0, 0, w, h);
      var ext = extent + (style === 'spacefill' ? 1.2 : 0);
      var scale = zoom * 0.45 * Math.min(w, h) / ext;
      function P(x, y, z) {
        x -= cx; y -= cy; z -= cz;
        return { x: w / 2 + scale * (rot[0] * x + rot[1] * y + rot[2] * z),
                 y: h / 2 - scale * (rot[3] * x + rot[4] * y + rot[5] * z),
                 z: rot[6] * x + rot[7] * y + rot[8] * z };
      }
      var frame = list[current], atoms = frame.atoms;
      var p = atoms.map(function (a) { return P(a.x, a.y, a.z); });
      if (frame.cell && showCell) {
        var c = frame.cell, corner = function (u, v, s) {
          return P(u * c[0][0] + v * c[1][0] + s * c[2][0], u * c[0][1] + v * c[1][1] + s * c[2][1], u * c[0][2] + v * c[1][2] + s * c[2][2]);
        };
        ctx.strokeStyle = palette.cell; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
        [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0],
         [0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]
          .forEach(function (e) { var a = corner(e[0], e[1], e[2]), b = corner(e[3], e[4], e[5]); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); });
        ctx.setLineDash([]);
      }
      var items = [];
      bondsOf(current).forEach(function (b) {
        var ai = atoms[b.i], aj = atoms[b.j];
        // Each half of a bond belongs to its atom; a bond to an image in the next cell is drawn as
        // the two halves that start at the atoms of this cell.
        var mi = P((ai.x + aj.x + b.t[0]) / 2, (ai.y + aj.y + b.t[1]) / 2, (ai.z + aj.z + b.t[2]) / 2);
        var mj = b.image ? P((ai.x + aj.x - b.t[0]) / 2, (ai.y + aj.y - b.t[1]) / 2, (ai.z + aj.z - b.t[2]) / 2) : mi;
        items.push({ z: (p[b.i].z + mi.z) / 2 - 0.01, from: p[b.i], to: mi, atom: b.i, half: true });
        items.push({ z: (p[b.j].z + mj.z) / 2 - 0.01, from: p[b.j], to: mj, atom: b.j, half: true });
      });
      atoms.forEach(function (a, i) { items.push({ z: p[i].z, atom: i }); });
      items.sort(function (u, v) { return u.z - v.z; });
      items.forEach(function (it) {
        var k = it.atom, a = atoms[k], s = atomStyle(k), st = s.st;
        ctx.globalAlpha = s.dim ? 0.55 : 1;
        if (it.half) {
          if (st.bond === 0) return;
          ctx.lineWidth = st.bond < 0 ? (s.dim ? 1.2 : 1.8) : Math.max(2, st.bond * scale);
          ctx.lineCap = 'round';
          ctx.strokeStyle = s.dim ? palette.dim : colorOf(a.el);
          ctx.beginPath(); ctx.moveTo(it.from.x, it.from.y); ctx.lineTo(it.to.x, it.to.y); ctx.stroke();
          return;
        }
        if (st.atom) {
          var r = Math.max(st.shade ? 3 : 1.5, scale * st.atom(a.el));
          if (st.shade) {
            var g = ctx.createRadialGradient(p[k].x - r / 3, p[k].y - r / 3, r / 6, p[k].x, p[k].y, r);
            g.addColorStop(0, '#ffffff'); g.addColorStop(1, colorOf(a.el));
            ctx.fillStyle = g;
          } else {
            ctx.fillStyle = colorOf(a.el);
          }
          ctx.beginPath(); ctx.arc(p[k].x, p[k].y, r, 0, 2 * Math.PI); ctx.fill();
          if (st.shade) { ctx.strokeStyle = 'rgba(0,0,0,0.35)'; ctx.lineWidth = 1; ctx.stroke(); }
        } else {
          ctx.fillStyle = s.dim ? palette.dim : colorOf(a.el);
          ctx.beginPath(); ctx.arc(p[k].x, p[k].y, s.dim ? 1.5 : 2.5, 0, 2 * Math.PI); ctx.fill();
        }
        if (labels !== 'none' && !s.dim) {
          var name = labelOf(labels, a, k);
          ctx.font = '600 11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.lineWidth = 3; ctx.strokeStyle = palette.halo; ctx.strokeText(name, p[k].x, p[k].y);
          ctx.fillStyle = palette.ink; ctx.fillText(name, p[k].x, p[k].y);
        }
      });
      ctx.globalAlpha = 1;
    }
    // Turning (drag) and zooming (wheel).
    var drag = null;
    el.addEventListener('pointerdown', function (e) { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); el.style.cursor = 'grabbing'; });
    el.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag = { x: e.clientX, y: e.clientY };
      rot = multiply(rotation(dy * 0.01, dx * 0.01), rot);
      draw();
    });
    el.addEventListener('pointerup', function () { drag = null; el.style.cursor = 'grab'; });
    el.addEventListener('wheel', function (e) { e.preventDefault(); zoom *= e.deltaY < 0 ? 1.1 : 1 / 1.1; draw(); }, { passive: false });
    // Frames, display style, atom numbers, layers, the cell, downloads.
    var bar = document.createElement('div');
    bar.className = 'controls';
    function add(node) { bar.appendChild(node); return node; }
    function checkbox(text, on, change) {
      var box = document.createElement('label'), c = document.createElement('input');
      c.type = 'checkbox'; c.checked = on; c.onchange = function () { change(c.checked); draw(); };
      box.appendChild(c); box.appendChild(document.createTextNode(' ' + text));
      return add(box);
    }
    function button(text, click) { var b = document.createElement('button'); b.type = 'button'; b.textContent = text; b.onclick = click; return add(b); }
    var play = null, slider = null, frameInfo = null;
    if (list.length > 1) {
      play = button('Play', function () { if (timer) stop(); else start(); });
      slider = add(document.createElement('input'));
      slider.type = 'range'; slider.min = 0; slider.max = list.length - 1; slider.value = 0;
      slider.setAttribute('aria-label', 'Frame');
      slider.oninput = function () { stop(); show(parseInt(slider.value, 10)); };
      frameInfo = add(document.createElement('span'));
    }
    var select = add(document.createElement('select'));
    select.setAttribute('aria-label', 'Display style');
    Object.keys(STYLES).forEach(function (key) { select.appendChild(new Option(STYLES[key].name, key)); });
    select.value = style;
    select.onchange = function () { style = select.value; draw(); };
    var labelSelect = add(document.createElement('select'));
    labelSelect.setAttribute('aria-label', 'Labels of the atoms');
    Object.keys(LABELS).forEach(function (key) { labelSelect.appendChild(new Option(LABELS[key], key)); });
    labelSelect.value = labels;
    labelSelect.onchange = function () { labels = labelSelect.value; draw(); };
    if (info.layers) checkbox('ONIOM layers', true, function (on) { layers = on; });
    if (periodic) checkbox('cell', true, function (on) { showCell = on; });
    button(list.length > 1 ? 'Download frame (.xyz)' : 'Download .xyz', function () {
      download(m.name + (list.length > 1 ? '_frame' + (current + 1) : '') + '.xyz', list[current].text);
    });
    if (list.length > 1) button('Download displayed frames (.xyz)', function () { download(m.name + '.xyz', m.xyz); });
    el.parentNode.insertBefore(bar, el.nextSibling);
    function show(k) {
      current = k;
      // With m.captions the comment of the frame (what it is, its energy) follows its number.
      if (slider) { slider.value = k; frameInfo.textContent = 'frame ' + (k + 1) + ' / ' + list.length + (m.captions && list[k].comment ? ' · ' + list[k].comment.slice(0, 90) : ''); }
      draw();
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; } if (play) play.textContent = 'Play'; }
    function start() { stop(); play.textContent = 'Pause'; timer = setInterval(function () { show((current + 1) % list.length); }, 120); }
    var observer = null;
    var handle = {
      draw: draw,
      setStyle: function (s) { style = s; select.value = s; draw(); },
      setLabels: function (l) { labels = l; labelSelect.value = l; draw(); },
      remove: function () {
        stop();
        if (observer) observer.disconnect(); else window.removeEventListener('resize', draw);
        var k = viewers.indexOf(handle);
        if (k >= 0) viewers.splice(k, 1);
        bar.remove(); el.textContent = ''; el.moptViewer = null;
      }
    };
    viewers.push(handle);
    el.moptViewer = handle;
    if (window.ResizeObserver) { observer = new ResizeObserver(draw); observer.observe(el); } else window.addEventListener('resize', draw);
    show(0);
    if (m.animate && list.length > 1 && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) start();
    else stop();
    return handle;
  }
  document.querySelectorAll('.viewer').forEach(function (el) {
    var m = models[el.dataset.model];
    if (m) setup(el, m);
  });
  // For other scripts of the page (the report of run_mapper shows the structure of the node
  // chosen).
  window.moptViewer = { setup: setup };

  // One display style for every viewer.
  function setStyleAll(s) {
    if (!STYLES[s]) return;
    globalStyle = s; store('mopt-report-style', s);
    document.querySelectorAll('[data-style-choice]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.styleChoice === s)); });
    viewers.forEach(function (v) { v.setStyle(s); });
  }
  document.querySelectorAll('[data-style-choice]').forEach(function (b) {
    b.addEventListener('click', function () { setStyleAll(b.dataset.styleChoice); });
    b.setAttribute('aria-pressed', String(b.dataset.styleChoice === globalStyle));
  });
  // One kind of atom labels for every viewer.
  function setLabelsAll(l) {
    if (!LABELS[l]) return;
    globalLabels = l; store('mopt-report-labels', l);
    document.querySelectorAll('[data-label-choice]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.labelChoice === l)); });
    viewers.forEach(function (v) { v.setLabels(l); });
  }
  document.querySelectorAll('[data-label-choice]').forEach(function (b) {
    b.addEventListener('click', function () { setLabelsAll(b.dataset.labelChoice); });
    b.setAttribute('aria-pressed', String(b.dataset.labelChoice === globalLabels));
  });
})();
</script>
"##;

/// The style of the page: light and dark themes (by the system, or chosen on the page),
/// translucent cards over a soft background, and the colors of the charts drawn by
/// [`crate::core_infra::viz`] mapped to the theme (shared with the page of `relaxedscan`).
pub const STYLE: &str = r##":root {
  color-scheme: light;
  --bg: #eef1f7; --bg-a: rgba(120, 160, 255, .28); --bg-b: rgba(255, 170, 200, .22); --bg-c: rgba(120, 220, 200, .20);
  --glass: rgba(255, 255, 255, .64); --glass-strong: rgba(255, 255, 255, .82); --glass-edge: rgba(255, 255, 255, .75);
  --shadow: 0 1px 1px rgba(15, 23, 42, .04), 0 10px 30px rgba(15, 23, 42, .08);
  --fg: #111827; --muted: #5d6676; --line: rgba(17, 24, 39, .10); --line-strong: rgba(17, 24, 39, .55);
  --accent: #0a7cff; --accent-soft: rgba(10, 124, 255, .12);
  --ok: #138a4b; --ok-soft: rgba(19, 138, 75, .12); --bad: #d12d22; --bad-soft: rgba(209, 45, 34, .10);
  --s-blue: #2563eb; --s-red: #dc2626; --s-green: #059669; --s-amber: #d97706; --s-magenta: #c026d3;
  --viewer-bg: #ffffff; --viewer-ink: #111111; --viewer-halo: rgba(255, 255, 255, .85); --viewer-carbon: #505050;
  --viewer-cell: rgba(17, 24, 39, .45); --viewer-dim: rgba(110, 118, 130, .55);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0b0e14; --bg-a: rgba(70, 110, 255, .22); --bg-b: rgba(200, 80, 150, .16); --bg-c: rgba(40, 170, 150, .14);
    --glass: rgba(28, 33, 44, .62); --glass-strong: rgba(36, 42, 56, .86); --glass-edge: rgba(255, 255, 255, .08);
    --shadow: 0 1px 1px rgba(0, 0, 0, .3), 0 12px 32px rgba(0, 0, 0, .35);
    --fg: #e9edf4; --muted: #9aa4b4; --line: rgba(255, 255, 255, .10); --line-strong: rgba(233, 237, 244, .55);
    --accent: #4ea3ff; --accent-soft: rgba(78, 163, 255, .16);
    --ok: #3ddc84; --ok-soft: rgba(61, 220, 132, .14); --bad: #ff6b60; --bad-soft: rgba(255, 107, 96, .14);
    --s-blue: #6ea8ff; --s-red: #ff7b72; --s-green: #34d399; --s-amber: #fbbf24; --s-magenta: #f472b6;
    --viewer-bg: #10141c; --viewer-ink: #f2f4f8; --viewer-halo: rgba(16, 20, 28, .85); --viewer-carbon: #a3a3a3;
    --viewer-cell: rgba(233, 237, 244, .45); --viewer-dim: rgba(160, 168, 180, .5);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0b0e14; --bg-a: rgba(70, 110, 255, .22); --bg-b: rgba(200, 80, 150, .16); --bg-c: rgba(40, 170, 150, .14);
  --glass: rgba(28, 33, 44, .62); --glass-strong: rgba(36, 42, 56, .86); --glass-edge: rgba(255, 255, 255, .08);
  --shadow: 0 1px 1px rgba(0, 0, 0, .3), 0 12px 32px rgba(0, 0, 0, .35);
  --fg: #e9edf4; --muted: #9aa4b4; --line: rgba(255, 255, 255, .10); --line-strong: rgba(233, 237, 244, .55);
  --accent: #4ea3ff; --accent-soft: rgba(78, 163, 255, .16);
  --ok: #3ddc84; --ok-soft: rgba(61, 220, 132, .14); --bad: #ff6b60; --bad-soft: rgba(255, 107, 96, .14);
  --s-blue: #6ea8ff; --s-red: #ff7b72; --s-green: #34d399; --s-amber: #fbbf24; --s-magenta: #f472b6;
  --viewer-bg: #10141c; --viewer-ink: #f2f4f8; --viewer-halo: rgba(16, 20, 28, .85); --viewer-carbon: #a3a3a3;
  --viewer-cell: rgba(233, 237, 244, .45); --viewer-dim: rgba(160, 168, 180, .5);
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; color: var(--fg); background-color: var(--bg);
  background-image: radial-gradient(60vmax 50vmax at 0% 0%, var(--bg-a), transparent 60%),
    radial-gradient(50vmax 40vmax at 100% 10%, var(--bg-b), transparent 60%),
    radial-gradient(60vmax 50vmax at 50% 100%, var(--bg-c), transparent 60%);
  background-attachment: fixed;
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif;
}
.glass {
  background: var(--glass); border: 1px solid var(--glass-edge); box-shadow: var(--shadow);
  -webkit-backdrop-filter: blur(22px) saturate(170%); backdrop-filter: blur(22px) saturate(170%);
}
@media (prefers-reduced-transparency: reduce) { .glass { background: var(--glass-strong); -webkit-backdrop-filter: none; backdrop-filter: none; } }
.topbar { position: sticky; top: 0; z-index: 10; border-width: 0 0 1px; border-radius: 0; }
.topbar-inner { max-width: 1020px; margin: 0 auto; padding: 10px 16px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; }
.brand { font-weight: 650; letter-spacing: -.01em; margin-right: auto; }
nav { display: flex; flex-wrap: wrap; gap: 4px; }
nav a { color: var(--fg); text-decoration: none; font-size: .85rem; padding: 4px 10px; border-radius: 999px; }
nav a:hover, nav a:focus-visible { background: var(--accent-soft); color: var(--accent); outline: none; }
main { max-width: 1020px; margin: 0 auto; padding: 22px 16px 72px; }
.hero { margin: 8px 4px 18px; }
.hero h1 { margin: 0 0 4px; font-size: clamp(1.6rem, 4vw, 2.2rem); letter-spacing: -.02em; line-height: 1.15; }
.hero p { margin: 0 0 12px; color: var(--muted); }
section.card { border-radius: 22px; padding: 18px 20px; margin: 16px 0; scroll-margin-top: 76px; }
@media (max-width: 520px) { section.card { padding: 14px 14px; border-radius: 18px; } }
h2 { margin: 0 0 12px; font-size: 1.2rem; letter-spacing: -.01em; }
h3 { margin: 20px 0 8px; font-size: 1rem; }
.seg { display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 3px; border-radius: 12px; background: var(--line); }
.seg button { font: inherit; font-size: .82rem; border: 0; background: transparent; color: var(--fg); padding: 4px 12px; border-radius: 9px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--glass-strong); box-shadow: 0 1px 3px rgba(0, 0, 0, .14); font-weight: 600; }
.seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.seg-label { font-size: .8rem; color: var(--muted); margin-right: 6px; }
.toolrow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; }
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 14px; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; font-size: .8rem; }
tbody tr:last-child td { border-bottom: 0; }
td.num { text-align: right; }
.pill { display: inline-block; padding: 1px 9px; border-radius: 999px; background: var(--line); font-size: .78rem; font-weight: 500; }
.pill.ok { color: var(--ok); background: var(--ok-soft); }
.pill.bad { color: var(--bad); background: var(--bad-soft); }
h3 .pill { margin-left: 6px; vertical-align: 2px; }
.warn { background: var(--bad-soft); color: var(--fg); padding: 10px 14px; border-radius: 14px; font-size: .9rem; }
.warn::before { content: "⚠︎ "; color: var(--bad); font-weight: 700; }
.chart svg, svg.diagram { display: block; width: 100%; height: auto; }
.chart svg > rect[fill="#ffffff"] { fill: transparent; }
.chart svg[fill="#181c24"], .chart [fill="#181c24"] { fill: var(--fg); }
.chart [stroke="#181c24"] { stroke: var(--line-strong); }
.chart [stroke="#e2e8f0"] { stroke: var(--line); }
main svg [fill="#2563eb"] { fill: var(--s-blue); } main svg [stroke="#2563eb"] { stroke: var(--s-blue); }
main svg [fill="#dc2626"] { fill: var(--s-red); } main svg [stroke="#dc2626"] { stroke: var(--s-red); }
main svg [fill="#059669"] { fill: var(--s-green); } main svg [stroke="#059669"] { stroke: var(--s-green); }
main svg [fill="#d97706"] { fill: var(--s-amber); } main svg [stroke="#d97706"] { stroke: var(--s-amber); }
main svg [fill="#c026d3"] { fill: var(--s-magenta); } main svg [stroke="#c026d3"] { stroke: var(--s-magenta); }
svg.diagram .axis { fill: var(--muted); }
figure { margin: 12px 0; }
figcaption, .note { color: var(--muted); font-size: .85rem; }
.viewer { position: relative; width: 100%; height: min(420px, 70vw); min-height: 260px; border-radius: 16px; overflow: hidden; background: var(--viewer-bg); border: 1px solid var(--line); touch-action: none; cursor: grab; }
.viewer canvas { width: 100%; height: 100%; display: block; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin: 8px 2px; font-size: .84rem; color: var(--muted); }
.viewer svg.static-view { width: 100%; height: 100%; display: block; }
a.dl { text-decoration: none; display: inline-block; }
.controls button, button.dl, a.dl, .controls select {
  font: inherit; font-size: .82rem; padding: 4px 12px; border: 0; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent); cursor: pointer;
}
.controls select { padding-right: 8px; }
.controls button:focus-visible, button.dl:focus-visible, a.dl:focus-visible, .controls select:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.controls input[type=range] { flex: 1 1 160px; min-width: 120px; accent-color: var(--accent); }
.controls input[type=checkbox] { accent-color: var(--accent); }
.legend { display: inline-flex; gap: 10px; font-size: .8rem; color: var(--muted); }
details summary { cursor: pointer; color: var(--accent); }
pre { background: var(--line); padding: 12px 14px; border-radius: 14px; overflow-x: auto; font-size: .8rem; }
code { font-size: .88em; }
"##;
/// Frames of a path kept in the page at most (evenly spaced, the ends kept).
const MAX_FRAMES: usize = 200;

pub(super) fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |f| f.to_string_lossy().into_owned())
}

/// The buttons that choose the labels of the atoms of every viewer of a page (their numbers,
/// none, their elements, both); the script of the page ([`VIEWER`]) reads them, and every
/// viewer has the same choice for itself.
pub const LABEL_CONTROLS: &str = "<span class=\"seg-label\">Labels</span><div class=\"seg\" \
    role=\"group\" aria-label=\"Labels of the atoms of every molecule\"><button type=\"button\" \
    data-label-choice=\"number\">Numbers</button><button type=\"button\" \
    data-label-choice=\"none\">None</button><button type=\"button\" \
    data-label-choice=\"element\">Elements</button><button type=\"button\" \
    data-label-choice=\"both\">Elements and numbers</button></div>";

/// The frames of an xyz file as text, at most [`MAX_FRAMES`] of them.
fn frames_text(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    let mut frames: Vec<String> = Vec::new();
    let mut k = 0;
    while k < lines.len() {
        if lines[k].trim().is_empty() {
            k += 1;
            continue;
        }
        let n: usize = lines[k].trim().parse().ok()?;
        let end = (k + 2 + n).min(lines.len());
        frames.push(lines[k..end].join("\n"));
        k = end;
    }
    if frames.is_empty() {
        return None;
    }
    if frames.len() > MAX_FRAMES {
        let last = frames.len() - 1;
        let picked: Vec<String> = (0..MAX_FRAMES)
            .map(|i| frames[i * last / (MAX_FRAMES - 1)].clone())
            .collect();
        frames = picked;
    }
    Some(frames.join("\n") + "\n")
}

fn read_table(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.split(',').map(|c| c.trim().to_string()).collect())
                .collect()
        })
        .unwrap_or_default()
}

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            dirs.push(p);
        } else if p.file_name().is_some_and(|f| f == name) {
            return Some(p);
        }
    }
    dirs.sort();
    dirs.iter().find_map(|d| find_file(d, name))
}

pub(super) fn chart(title: &str, x: &str, y: &str, series: Vec<PlotSeries>) -> String {
    let mut fig = PlotFigure::new(title, x, y);
    fig.width = 760;
    fig.height = 420;
    for s in series {
        fig.add_series(s);
    }
    format!("<div class=\"chart\">{}</div>", fig.to_svg_string())
}

pub(super) fn series(
    label: &str,
    xs: Vec<f64>,
    ys: Vec<f64>,
    color: Rgb,
    style: PlotStyle,
) -> PlotSeries {
    PlotSeries {
        label: label.to_string(),
        xs,
        ys,
        color,
        style,
    }
}

/// The lowest wavenumber (cm^-1, negative for an imaginary one) of `normal_modes.txt`.
fn lowest_wavenumber(dir: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(dir.join("normal_modes.txt")).ok()?;
    text.lines()
        .filter(|l| l.starts_with("Freq [cm^-1]"))
        .flat_map(|l| {
            l.split(']')
                .nth(1)
                .unwrap_or("")
                .split_whitespace()
                .filter_map(|v| v.parse::<f64>().ok())
                .collect::<Vec<_>>()
        })
        .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.min(v))))
}

/// The animation of the lowest mode in `vibration_animation/`.
fn lowest_mode_animation(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir.join("vibration_animation")).ok()?;
    entries
        .flatten()
        .map(|e| e.path())
        .filter_map(|p| {
            // `mode_<k>_<wavenumber>_wave_number.xyz`; an imaginary mode is written `65i`.
            let name = file_name(&p);
            let token = name.split('_').nth(2)?;
            let freq: f64 = match token.strip_suffix('i') {
                Some(value) => -value.parse::<f64>().ok()?,
                None => token.parse().ok()?,
            };
            Some((freq, p))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, p)| p)
}

struct Model {
    id: String,
    /// The file name without its extension, for downloads.
    name: String,
    text: String,
    animate: bool,
    /// Whether the comment of every frame stands beside its number.
    captions: bool,
}

/// The structures of a page: each a viewer in its place, all read by the script of the page
/// ([`VIEWER`]) from one block of data ([`Viewers::models_json`]). Shared with the page of
/// `run_automecp`.
pub(super) struct Viewers {
    models: Vec<Model>,
}

impl Viewers {
    /// No structures yet.
    pub(super) fn new() -> Self {
        Self { models: Vec::new() }
    }

    fn push(&mut self, path: Option<&Path>, text: String, animate: bool) -> String {
        let id = format!("m{}", self.models.len());
        let name = path
            .and_then(Path::file_stem)
            .map_or_else(|| id.clone(), |s| s.to_string_lossy().into_owned());
        self.models.push(Model {
            id: id.clone(),
            name,
            text,
            animate,
            captions: false,
        });
        id
    }

    /// A viewer of the xyz frames `text`, saved under `name`: animated when `animate`, the
    /// comment of every frame beside its number when `captions`.
    pub(super) fn add_text(
        &mut self,
        name: &str,
        text: String,
        animate: bool,
        captions: bool,
        caption: &str,
    ) -> String {
        if text.trim().is_empty() {
            return format!("<p class=\"note\">{} (no structure)</p>", escape(caption));
        }
        let drawing = super::autots_static::drawing(&text);
        let frames = text
            .lines()
            .filter(|l| l.trim().parse::<usize>().is_ok())
            .count();
        let label = if frames > 1 {
            "Download displayed frames (.xyz)"
        } else {
            "Download .xyz"
        };
        let link = super::autots_static::download_link(&format!("{name}.xyz"), &text, label, "dl");
        let id = format!("m{}", self.models.len());
        self.models.push(Model {
            id: id.clone(),
            name: name.to_string(),
            text,
            animate,
            captions,
        });
        format!(
            "<figure class=\"viewer-box\"><div class=\"viewer\" data-model=\"{id}\">{drawing}</div>\
             <div class=\"controls static-controls\">{link}</div>\
             <figcaption>{}</figcaption></figure>",
            escape(caption)
        )
    }

    /// The structures as the data the script of the page reads (the element `models`).
    pub(super) fn models_json(&self) -> String {
        JsonValue::Object(
            self.models
                .iter()
                .map(|model| {
                    let mut m = std::collections::BTreeMap::new();
                    m.insert("xyz".to_string(), JsonValue::String(model.text.clone()));
                    m.insert("animate".to_string(), JsonValue::Bool(model.animate));
                    m.insert("captions".to_string(), JsonValue::Bool(model.captions));
                    m.insert("name".to_string(), JsonValue::String(model.name.clone()));
                    (model.id.clone(), JsonValue::Object(m))
                })
                .collect(),
        )
        .to_string_compact()
        .replace("</", "<\\/")
    }

    /// A link that saves the structures of `path` as an xyz file (the file is in the link,
    /// so it works without the script of the page), or nothing.
    fn download(&mut self, path: &Path, label: &str) -> String {
        let Some(text) = frames_text(path) else {
            return String::new();
        };
        let name = path.file_stem().map_or_else(
            || "structure".to_string(),
            |s| s.to_string_lossy().into_owned(),
        );
        super::autots_static::download_link(&format!("{name}.xyz"), &text, &escape(label), "dl")
    }

    /// A viewer of the frames of `path` (animated when `animate`), or a note.
    fn add(&mut self, path: Option<&Path>, animate: bool, caption: &str) -> String {
        let Some(text) = path.and_then(frames_text) else {
            return format!(
                "<p class=\"note\">{} (no structure file)</p>",
                escape(caption)
            );
        };
        // A drawing of the first frame and a link to the file stand in for the viewer and
        // its controls where the script of the page does not run; the script replaces them.
        let drawing = super::autots_static::drawing(&text);
        let name = path.and_then(Path::file_stem).map_or_else(
            || "structure".to_string(),
            |s| s.to_string_lossy().into_owned(),
        );
        let frames = text
            .lines()
            .filter(|l| l.trim().parse::<usize>().is_ok())
            .count();
        let label = if frames > 1 {
            "Download displayed frames (.xyz)"
        } else {
            "Download .xyz"
        };
        let link = super::autots_static::download_link(&format!("{name}.xyz"), &text, label, "dl");
        let id = self.push(path, text, animate);
        format!(
            "<figure class=\"viewer-box\"><div class=\"viewer\" data-model=\"{id}\">{drawing}</div>\
             <div class=\"controls static-controls\">{link}</div>\
             <figcaption>{}</figcaption></figure>",
            escape(caption)
        )
    }
}

pub(super) fn kcal(v: Option<f64>) -> String {
    v.map_or_else(|| "—".to_string(), |x| format!("{x:.2}"))
}

/// Two IRC ends are the same minimum when their energies differ by less than this (Hartree)
/// and no interatomic distance by more than [`SAME_MINIMUM_BOHR`] (judgement values: ends
/// optimized from two IRCs agree to about 1e-7 Hartree and 1e-3 Bohr in the runs so far).
const SAME_MINIMUM_HARTREE: f64 = 1.0e-5;
/// See [`SAME_MINIMUM_HARTREE`].
const SAME_MINIMUM_BOHR: f64 = 5.0e-2;

/// The distinct minima among the IRC ends of the transition states.
struct Minima {
    /// The first file of each minimum and its energy (Hartree).
    list: Vec<(PathBuf, f64)>,
    /// For each transition state (as given), the minima of its ends.
    labels: Vec<Vec<usize>>,
}

impl Minima {
    fn of(states: &[(usize, &TransitionState)]) -> Self {
        let mut list: Vec<(PathBuf, f64)> = Vec::new();
        let mut labels = Vec::new();
        for (_, t) in states {
            let mut mine = Vec::new();
            for (file, e) in &t.ends {
                let found = list.iter().position(|(f, known)| {
                    (known - e).abs() < SAME_MINIMUM_HARTREE
                        && super::autots::largest_distance_change(f, file)
                            .ok()
                            .flatten()
                            .is_some_and(|d| d < SAME_MINIMUM_BOHR)
                });
                let k = found.unwrap_or_else(|| {
                    list.push((file.clone(), *e));
                    list.len() - 1
                });
                mine.push(k);
            }
            labels.push(mine);
        }
        Self { list, labels }
    }

    /// The labels of the ends of the `k`-th state ("M1 → M2"), or a dash.
    fn ends_text(&self, k: usize) -> String {
        match self.labels.get(k).map(Vec::as_slice) {
            Some([a, b]) => format!("M{} → M{}", a + 1, b + 1),
            _ => "—".to_string(),
        }
    }
}

/// An energy diagram of the reactions on one energy scale: each transition state between
/// the two minima its IRC ends at, side by side; minima with the same label are the same
/// structure, so the diagram shows which transition states share a minimum.
fn energy_diagram(states: &[(usize, &TransitionState)], minima: &Minima) -> String {
    let groups: Vec<(usize, usize, &TransitionState)> = states
        .iter()
        .enumerate()
        .filter(|(_, (_, t))| t.ends.len() == 2)
        .map(|(i, (k, t))| (i, *k, *t))
        .collect();
    if groups.is_empty() {
        return String::new();
    }
    // The lowest end of all, the zero of the diagram, and its minimum.
    let (zero, lowest) = groups
        .iter()
        .flat_map(|(i, _, t)| {
            t.ends.iter().enumerate().map(move |(j, e)| {
                let label = minima.labels.get(*i).and_then(|l| l.get(j)).copied();
                (e.1, label)
            })
        })
        .fold((f64::INFINITY, None), |best, (e, label)| {
            if e < best.0 {
                (e, label)
            } else {
                best
            }
        });
    let rel = |e: f64| Units::HARTREE_TO_KCAL_MOL * (e - zero);
    let levels: Vec<([f64; 3], [String; 3])> = groups
        .iter()
        .map(|(i, k, t)| {
            let names = match minima.labels.get(*i).map(Vec::as_slice) {
                Some([a, b]) => [
                    format!("M{}", a + 1),
                    format!("TS {k}"),
                    format!("M{}", b + 1),
                ],
                _ => ["end 1".to_string(), format!("TS {k}"), "end 2".to_string()],
            };
            ([rel(t.ends[0].1), rel(t.energy), rel(t.ends[1].1)], names)
        })
        .collect();
    let svg = level_diagram(
        &levels,
        "Energy diagram of the transition states and the minima they connect",
    );
    let lowest = lowest.map_or_else(String::new, |m| format!(" (M{})", m + 1));
    format!(
        "<figure><div class=\"table-wrap\">{svg}</div><figcaption>Energies in kcal/mol above \
         the lowest minimum{lowest}. Each transition state is drawn between the two minima its \
         IRC ends at; the same label (M1, M2, ...) is the same structure, so transition states \
         that share a minimum connect.</figcaption></figure>"
    )
}

/// A diagram of energy levels on one scale: every group is three levels side by side (a
/// minimum, the point between, a minimum) with their energies (kcal/mol) and names, joined by
/// dashed lines. Shared with the page of `run_automecp`.
pub(super) fn level_diagram(groups: &[([f64; 3], [String; 3])], aria: &str) -> String {
    levels(groups, aria, 1.0, 1)
}

/// Draw the three levels in the units and precision of the reported quantity.
pub(super) fn levels(
    groups: &[([f64; 3], [String; 3])],
    aria: &str,
    minimum_top: f64,
    digits: usize,
) -> String {
    let hi = groups
        .iter()
        .flat_map(|(e, _)| e.iter().copied())
        .fold(minimum_top, f64::max);
    let lo = groups
        .iter()
        .flat_map(|(e, _)| e.iter().copied())
        .fold(0.0_f64, f64::min);
    let group_width = 240.0;
    let w = (40.0 + group_width * groups.len() as f64).max(760.0);
    let (h, top, bottom) = (320.0, 34.0, 46.0);
    let y = |e: f64| top + (h - top - bottom) * (1.0 - (e - lo) / (hi - lo));
    // Wide diagrams scroll instead of shrinking their labels below reading size.
    let min_width = (160.0 * groups.len() as f64).min(w);
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" class=\"diagram\" \
         style=\"min-width: {min_width}px\" role=\"img\" aria-label=\"{aria}\">"
    );
    let colors = [Rgb::BLUE, Rgb::RED, Rgb::EMERALD, Rgb::AMBER, Rgb::MAGENTA];
    let offset = (w - group_width * groups.len() as f64) / 2.0;
    for (n, (es, names)) in groups.iter().enumerate() {
        let c = colors[n % colors.len()].to_hex();
        let x0 = offset + group_width * n as f64;
        let xs = [x0 + 40.0, x0 + 120.0, x0 + 200.0];
        for j in 0..3 {
            let (xa, xb) = (xs[j] - 30.0, xs[j] + 30.0);
            let yj = y(es[j]);
            let _ = write!(
                svg,
                "<line x1=\"{xa}\" x2=\"{xb}\" y1=\"{yj:.1}\" y2=\"{yj:.1}\" stroke=\"{c}\" \
                 stroke-width=\"3\"/><text x=\"{}\" y=\"{:.1}\" text-anchor=\"middle\" \
                 font-size=\"12\" fill=\"{c}\">{:.digits$}</text><text x=\"{}\" y=\"{:.1}\" \
                 text-anchor=\"middle\" font-size=\"11\" class=\"axis\">{}</text>",
                xs[j],
                yj - 6.0,
                es[j],
                xs[j],
                yj + 15.0,
                names[j]
            );
            if j < 2 {
                let _ = write!(
                    svg,
                    "<line x1=\"{xb}\" x2=\"{}\" y1=\"{yj:.1}\" y2=\"{:.1}\" stroke=\"{c}\" \
                     stroke-dasharray=\"4 3\"/>",
                    xs[j + 1] - 30.0,
                    y(es[j + 1])
                );
            }
        }
    }
    svg.push_str("</svg>");
    svg
}

/// The table of the distinct minima: energy, the transition states they connect, file.
fn minima_table(
    states: &[(usize, &TransitionState)],
    minima: &Minima,
    viewers: &mut Viewers,
) -> String {
    if minima.list.is_empty() {
        return String::new();
    }
    let zero = minima
        .list
        .iter()
        .map(|m| m.1)
        .fold(f64::INFINITY, f64::min);
    let mut out = String::from(
        "<h3>Minima at the ends of the IRCs</h3><div class=\"table-wrap\"><table><thead><tr>\
         <th>minimum</th><th>energy (Hartree)</th><th>kcal/mol</th><th>transition states</th>\
         <th>file</th><th></th></tr></thead><tbody>",
    );
    for (m, (file, e)) in minima.list.iter().enumerate() {
        let linked: Vec<String> = states
            .iter()
            .zip(&minima.labels)
            .filter(|(_, l)| l.contains(&m))
            .map(|((k, _), _)| format!("TS {k}"))
            .collect();
        let _ = write!(
            out,
            "<tr><td>M{}</td><td class=\"num\">{e:.8}</td><td class=\"num\">{:.2}</td>\
             <td>{}</td><td>{}</td><td>{}</td></tr>",
            m + 1,
            Units::HARTREE_TO_KCAL_MOL * (e - zero),
            linked.join(", "),
            escape(&file_name(file)),
            viewers.download(file, "Download .xyz")
        );
    }
    out.push_str("</tbody></table></div>");
    out
}

fn afir_section(result: &AutotsResult, viewers: &mut Viewers) -> String {
    let method = match result.first_path_method.as_str() {
        "" => "not run".to_string(),
        m => m.replace("manual_AFIR", "AFIR"),
    };
    let mut out = format!(
        "<section class=\"card glass\" id=\"step1\"><h2>Step 1 · First path ({})</h2>",
        escape(&method)
    );
    let Some(dir) = &result.afir_directory else {
        if result.afir_path.is_some() {
            out.push_str("<p class=\"note\">Step 1 was skipped: the input is the path.</p>");
            out.push_str(&viewers.add(
                result.afir_path.as_deref(),
                true,
                "The given path (animated).",
            ));
        } else {
            out.push_str(if result.error.is_some() {
                "<p class=\"warn\">Step 1 did not finish: the run stopped with the error \
                 shown at the top.</p>"
            } else if result.progress.is_some() {
                "<p class=\"note\">Step 1 has not ended yet.</p>"
            } else {
                "<p class=\"note\">Step 1 did not run.</p>"
            });
        }
        out.push_str("</section>");
        return out;
    };
    let rows = read_table(&dir.join("energy_profile.csv"));
    let mut it = Vec::new();
    let (mut raw, mut biased) = (Vec::new(), Vec::new());
    for r in rows.iter().skip(1) {
        let (Some(k), Some(b), Some(e)) = (
            r.first().and_then(|v| v.parse::<f64>().ok()),
            r.get(1).and_then(|v| v.parse::<f64>().ok()),
            r.get(3).and_then(|v| v.parse::<f64>().ok()),
        ) else {
            continue;
        };
        it.push(k);
        biased.push(b);
        raw.push(e);
    }
    if let (Some(&e0), Some(&b0)) = (raw.first(), biased.first()) {
        let ys_raw: Vec<f64> = raw
            .iter()
            .map(|e| Units::HARTREE_TO_KCAL_MOL * (e - e0))
            .collect();
        let ys_b: Vec<f64> = biased
            .iter()
            .map(|b| Units::HARTREE_TO_KCAL_MOL * (b - b0))
            .collect();
        // BITSS writes the sum of the energies of its two images and its own function (the
        // energies with the penalties on their difference and on their distance).
        let bitss = result.first_path_method == "BITSS";
        let (title, raw_label, biased_label, note) = if bitss {
            (
                "Energy during BITSS (step 1)",
                "both images (E1 + E2)",
                "the BITSS function (shifted)",
                "The BITSS function adds to E1 + E2 the penalties on the difference of the two \
                 energies and on the distance of the images; it is shifted to start where \
                 E1 + E2 starts.",
            )
        } else {
            (
                "Energy along the optimization of step 1",
                "without the bias",
                "with the bias (shifted)",
                "The energy with the bias is shifted to start where the energy without it \
                 starts (a bias such as the AFIR function can have a large constant part).",
            )
        };
        out.push_str(&chart(
            title,
            "step",
            "kcal/mol (from the first step)",
            vec![
                series(
                    raw_label,
                    it.clone(),
                    ys_raw,
                    Rgb::BLUE,
                    PlotStyle::LinePoints,
                ),
                series(biased_label, it, ys_b, Rgb::RED, PlotStyle::LinePoints),
            ],
        ));
        let _ = write!(out, "<p class=\"note\">{note}</p>");
    }
    out.push_str(&viewers.add(
        result.afir_path.as_deref(),
        true,
        "The first path (animated).",
    ));
    out.push_str("</section>");
    out
}

/// Show each evaluated band beside its run's initial band. Image counts may change.
pub(super) fn band_history(
    profiles: &[BandProfile],
    title: &str,
    ordinate: &str,
    scale: f64,
) -> String {
    let Some(last) = profiles.last() else {
        return String::new();
    };
    let mut out = String::from("<div class=\"band-history\"><h3>Path relaxation</h3><div class=\"controls\"><label>Saved band <select onchange=\"this.closest('.band-history').querySelectorAll('[data-band]').forEach((panel, i) => panel.hidden = i !== this.selectedIndex)\">");
    for (i, p) in profiles.iter().enumerate() {
        let selected = if i + 1 == profiles.len() {
            " selected"
        } else {
            ""
        };
        let _ = write!(
            out,
            "<option{selected}>Run {} · Iteration {}</option>",
            p.run, p.iteration
        );
    }
    out.push_str("</select></label></div>");
    let mut initial = &profiles[0];
    for (i, p) in profiles.iter().enumerate() {
        if p.run != initial.run {
            initial = p;
        }
        let hidden = if i + 1 == profiles.len() {
            ""
        } else {
            " hidden"
        };
        let _ = write!(out, "<div data-band{hidden}>");
        if let Some(&zero) = initial.energies.first() {
            let curve = |label: &str, band: &BandProfile, color| {
                let intervals = band.energies.len().saturating_sub(1).max(1) as f64;
                let x = (0..band.energies.len())
                    .map(|j| j as f64 / intervals)
                    .collect();
                let y = band
                    .energies
                    .iter()
                    .map(|energy| (energy - zero) * scale)
                    .collect();
                series(label, x, y, color, PlotStyle::LinePoints)
            };
            let mut curves = vec![curve("Initial path", initial, Rgb::SLATE)];
            if !std::ptr::eq(initial, p) {
                curves.push(curve(&format!("Iteration {}", p.iteration), p, Rgb::BLUE));
            }
            out.push_str(&chart(title, "Fractional image index", ordinate, curves));
        }
        out.push_str("</div>");
    }
    let _ = write!(
        out,
        "<p class=\"note\">Latest saved band: run {}, iteration {}.</p></div>",
        last.run, last.iteration
    );
    out
}

fn path_section(result: &AutotsResult, viewers: &mut Viewers) -> String {
    let Some(stage) = &result.path_stage else {
        let why = if result.error.is_some() {
            "<p class=\"warn\">Step 2 did not run: the run stopped with the error shown at the \
             top before it, or in it.</p>"
        } else if result.progress.is_some() {
            "<p class=\"note\">Step 2 has not ended yet.</p>"
        } else {
            "<p class=\"note\">Step 2 did not run.</p>"
        };
        return format!(
            "<section class=\"card glass\" id=\"step2\"><h2>Step 2 · Path</h2>{why}</section>"
        );
    };
    let PathStage {
        method,
        path,
        energies,
        candidates,
        estimates,
    } = stage;
    let mut out = format!(
        "<section class=\"card glass\" id=\"step2\"><h2>Step 2 · Path ({})</h2>",
        escape(method)
    );
    if let (Some(file), false) = (path, energies.is_empty()) {
        // An energy profile along the frames of the path.
        let frames = read_xyz_trajectory_file(file, SpinCharge::default(), None)
            .map(|t| t.frames)
            .unwrap_or_default();
        let points: Vec<Col<f64>> = frames.iter().map(|f| f.system.coords_flat_bohr()).collect();
        let (lengths, x_label) = if points.len() == energies.len() && points.len() > 1 {
            (
                path_lengths(&points, ChainKind::Atoms),
                "length along the path (Bohr)",
            )
        } else {
            (
                (0..energies.len()).map(|k| k as f64).collect(),
                "point along the path",
            )
        };
        let zero = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let ys: Vec<f64> = energies
            .iter()
            .map(|e| Units::HARTREE_TO_KCAL_MOL * (e - zero))
            .collect();
        let (cx, cy): (Vec<f64>, Vec<f64>) = candidates
            .iter()
            .filter(|&&k| k < ys.len())
            .map(|&k| (lengths[k], ys[k]))
            .unzip();
        out.push_str(&chart(
            "Energy profile of the path",
            x_label,
            "kcal/mol (from the lowest point)",
            vec![
                series(
                    "points of the path",
                    lengths,
                    ys,
                    Rgb::BLUE,
                    PlotStyle::LinePoints,
                ),
                series("candidates", cx, cy, Rgb::RED, PlotStyle::Scatter),
            ],
        ));
    }
    if let Some(file) = path {
        out.push_str(&viewers.add(
            Some(file),
            true,
            &format!("The path of {method} (animated)."),
        ));
    }
    if estimates.is_empty() {
        out.push_str("<p class=\"note\">No estimate of a transition state.</p>");
    } else {
        out.push_str(&format!(
            "<p class=\"note\">{} estimate(s) of transition states; the highest {} went to \
             step 3.</p>",
            estimates.len(),
            result.candidates.len()
        ));
        if path.is_none() || energies.is_empty() {
            // No profile to show them on: the estimates themselves.
            for (k, file) in estimates.iter().enumerate() {
                out.push_str(&viewers.add(
                    Some(file),
                    false,
                    &format!("Estimate {} ({}).", k + 1, file_name(file)),
                ));
            }
        }
    }
    out.push_str(&band_history(
        &result.band_profiles,
        "Path energy during relaxation",
        "Energy relative to the first image of the initial path (kcal/mol)",
        Units::HARTREE_TO_KCAL_MOL,
    ));
    if let Some(iteration) = path.as_deref().and_then(Path::parent) {
        if iteration
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("path_ITR_"))
        {
            if let Some(dir) = iteration.parent() {
                out.push_str(&saved_band_figures(dir));
            }
        }
    }
    out.push_str("</section>");
    out
}

/// Saved figures of the band driver, available offline in the report and as SVG downloads.
fn saved_band_figures(dir: &Path) -> String {
    let mut figures = String::new();
    for (file, caption) in [
        ("energy_profile.svg", "Band energy profiles"),
        ("image_forces.svg", "Forces on the images"),
        ("convergence.svg", "Band convergence"),
    ] {
        if let Ok(bytes) = std::fs::read(dir.join(file)) {
            let data = super::autots_static::base64_bytes(&bytes);
            let _ = write!(figures, "<figure><img style=\"max-width:100%;height:auto\" alt=\"{caption}\" src=\"data:image/svg+xml;base64,{data}\"><figcaption>{caption} · <a class=\"dl\" download=\"{file}\" href=\"data:image/svg+xml;base64,{data}\">{file}</a></figcaption></figure>");
        }
    }
    if figures.is_empty() {
        figures
    } else {
        format!("<div class=\"band-figures\"><h3>Saved path figures</h3>{figures}</div>")
    }
}

/// What the optimization of a candidate gave: a label for the tables and, when it is not a
/// transition state, a warning that says why.
fn verdict(t: &TransitionState, checked: bool) -> (String, Option<String>) {
    if let Some(notice) = &t.meta_irc {
        return (
            "<span class=\"pill bad\">meta-IRC start, not a transition state</span>".to_string(),
            Some(escape(notice)),
        );
    }
    match (t.converged, t.imaginary_modes, t.same_as) {
        (false, _, _) => (
            "<span class=\"pill bad\">not converged</span>".to_string(),
            Some(
                "The optimization of this candidate did not converge, so this is not a \
                 transition state: the structure, energy and frequencies shown are those of the \
                 last step of the optimization. It was not taken to step 4. More steps \
                 (<code>NSTEP</code> in <code>step3_settings</code>), another method, or an \
                 exact Hessian more often (<code>calc_exact_hess</code>) may reach the \
                 transition state."
                    .to_string(),
            ),
        ),
        (true, _, Some(j)) => (format!("<span class=\"pill\">same as TS {j}</span>"), None),
        (true, Some(1), None) => (
            "<span class=\"pill ok\">transition state</span>".to_string(),
            None,
        ),
        (true, None, None) if checked => (
            "<span class=\"pill bad\">no frequencies</span>".to_string(),
            Some(
                "The vibrational analysis of this stationary point did not give its \
                 frequencies, so it is not known to be a transition state. It was not taken \
                 to step 4."
                    .to_string(),
            ),
        ),
        (true, None, None) => (
            "<span class=\"pill\">converged, not checked</span>".to_string(),
            Some(
                "The vibrational analysis is off (<code>\"check_frequencies\": \
                 false</code>), so it is not known whether this stationary point has exactly \
                 one imaginary frequency; it was taken to step 4 as a transition state."
                    .to_string(),
            ),
        ),
        (true, Some(0), None) => (
            "<span class=\"pill bad\">no imaginary frequency</span>".to_string(),
            Some(
                "The optimization converged to a minimum, not a transition state (no imaginary \
                 frequency). It was not taken to step 4."
                    .to_string(),
            ),
        ),
        (true, Some(n), None) => (
            format!("<span class=\"pill bad\">{n} imaginary</span>"),
            Some(format!(
                "The optimization converged to a saddle point with {n} imaginary frequencies, \
                 not a transition state (one). It was not taken to step 4."
            )),
        ),
    }
}

/// A transition state that goes on to step 4: converged, the first of its kind, and with
/// one imaginary frequency (when the check is off, also one that was not analysed).
fn is_transition_state(t: &TransitionState, checked: bool) -> bool {
    let frequencies = if checked {
        t.imaginary_modes == Some(1)
    } else {
        t.imaginary_modes.is_none_or(|n| n == 1)
    };
    t.converged && t.same_as.is_none() && frequencies && t.meta_irc.is_none()
}

/// The atoms of an ONIOM layer in the settings of the steps (`oniom_flag`, `oniom_middle`:
/// `ATOMS LINK CALCULATOR`): the 0-based atoms and the calculator.
fn oniom_layer(config: &AutotsConfig, key: &str) -> Option<(Vec<usize>, String)> {
    config.steps.iter().find_map(|s| {
        let tokens = s.get(key).and_then(JsonValue::as_string_vec).or_else(|| {
            s.get(key)
                .and_then(JsonValue::as_str)
                .map(|t| t.split_whitespace().map(str::to_string).collect())
        })?;
        let atoms = parse_atom_indices_one_based(tokens.first()?).ok()?;
        let calculator = tokens.get(2).cloned().unwrap_or_default();
        Some((atoms.into_iter().map(|i| i - 1).collect(), calculator))
    })
}

/// The periodic cell of the first frame of an extended xyz file, as lengths (Angstrom) and
/// angles (degrees).
fn cell_text(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let comment = text.lines().nth(1)?;
    let start = comment.find("Lattice=\"")? + "Lattice=\"".len();
    let end = start + comment[start..].find('"')?;
    let v: Vec<f64> = comment[start..end]
        .split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect();
    let [a1, a2, a3, b1, b2, b3, c1, c2, c3] = v.as_slice() else {
        return None;
    };
    let (a, b, c) = ([*a1, *a2, *a3], [*b1, *b2, *b3], [*c1, *c2, *c3]);
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let len = |x: [f64; 3]| dot(x, x).sqrt();
    let angle = |x: [f64; 3], y: [f64; 3]| {
        (dot(x, y) / (len(x) * len(y)))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    Some(format!(
        "periodic cell a = {:.3}, b = {:.3}, c = {:.3} Å, α = {:.1}°, β = {:.1}°, γ = {:.1}°",
        len(a),
        len(b),
        len(c),
        angle(b, c),
        angle(a, c),
        angle(a, b)
    ))
}

/// The ONIOM layers for the viewer (`null` without ONIOM) and a line on the system: the layers
/// and the periodic cell of the input.
fn page_system(config: &AutotsConfig) -> (JsonValue, String) {
    let mut line = String::new();
    let high = oniom_layer(config, "oniom_flag");
    let middle = oniom_layer(config, "oniom_middle");
    let layers = match &high {
        Some((atoms, low)) => {
            let list = |a: &[usize]| {
                JsonValue::Array(a.iter().map(|&i| JsonValue::Number(i as f64)).collect())
            };
            let mut doc = std::collections::BTreeMap::new();
            doc.insert("high".to_string(), list(atoms));
            let _ = write!(line, " · ONIOM: high layer {} atoms", atoms.len());
            if let Some((m, calc)) = &middle {
                doc.insert("middle".to_string(), list(m));
                let _ = write!(line, ", middle layer {} atoms ({calc})", m.len());
            }
            let _ = write!(line, ", low layer {low}");
            JsonValue::Object(doc)
        }
        None => JsonValue::Null,
    };
    if let Some(cell) = cell_text(&config.input) {
        let _ = write!(line, " · {cell}");
    }
    (layers, line)
}

/// The `Reference:` lines of the logs (`mopt.log`) under `dir`: what the calculators and
/// methods of the steps cite.
pub(super) fn logged_references(dir: &Path, depth: usize, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            if depth > 0 {
                logged_references(&path, depth - 1, out);
            }
        } else if path.file_name().is_some_and(|n| n == "mopt.log") {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for line in text.lines() {
                if let Some(r) = line.trim().strip_prefix("Reference: ") {
                    if !out.iter().any(|known| known == r) {
                        out.push(r.to_string());
                    }
                }
            }
        }
    }
}

/// The references of the run: the methods of the workflow, then what the steps logged.
fn references(result: &AutotsResult) -> Vec<String> {
    let mut out = Vec::new();
    if result.first_path_method.contains("AFIR") {
        out.push("S. Maeda, K. Morokuma, J. Chem. Phys. 2010, 132, 241102 (AFIR)".to_string());
        out.push("S. Maeda, K. Morokuma, J. Chem. Theory Comput. 2011, 7, 2335 (AFIR)".to_string());
    }
    let path = result
        .path_stage
        .as_ref()
        .map(|s| s.method.as_str())
        .unwrap_or_default();
    if result.first_path_method == "BITSS" || path.contains("BITSS") {
        out.push(
            "S. Avis, J. R. Panter, H. Kusumaatmaja, J. Chem. Phys. 2022, 157, 124107 (BITSS)"
                .to_string(),
        );
    }
    if path.contains("MaxFlux") {
        out.push(
            "S.-i. Koda, S. Saito, J. Chem. Theory Comput. 2024, 20, 2798 (direct MaxFlux)"
                .to_string(),
        );
    }
    logged_references(&result.directory, 4, &mut out);
    out
}

/// The rights and credits of the page.
fn credits(result: &AutotsResult) -> String {
    let mut refs = String::new();
    for r in references(result) {
        let _ = write!(refs, "<li>{}</li>", escape(&r));
    }
    if refs.is_empty() {
        refs.push_str("<li>No reference was logged by the steps of this run.</li>");
    }
    format!(
        "<section class=\"card glass\" id=\"credits\"><h2>Rights and credits</h2>\
         <p>This report was generated by mopt-rs {version}, Copyright © 2023–2026 ss0832, free \
         software under the GNU General Public License, version 3 or later \
         (GPL-3.0-or-later), with no warranty. The structure viewer, the style and the code \
         embedded in this page are part of mopt-rs, under the same license.</p>\
         <p>The structures, energies and other results on this page are the output of the \
         calculation that was run; mopt-rs claims no rights to them. The four steps follow the \
         AutoTS workflow of MultiOptPy (ss0832, GPL-3.0).</p>\
         <p>Element colors follow the CPK convention (as in Jmol); bonds are drawn from the \
         single-bond covalent radii of P. Pyykkö and M. Atsumi, Chem. Eur. J. 2009, 15, 186; \
         space-filling spheres use van der Waals radii mostly from A. Bondi, J. Phys. Chem. \
         1964, 68, 441 (approximate values for some metals).</p>\
         <h3>Methods and calculators of this run</h3><ul>{refs}</ul></section>",
        version = crate::VERSION
    )
}

/// Write the report of `result` to `path`.
pub fn write(config: &AutotsConfig, result: &AutotsResult, path: &Path) -> MoptResult<()> {
    let mut viewers = Viewers { models: Vec::new() };
    let name = config
        .input
        .file_stem()
        .map_or_else(|| "input".to_string(), |s| s.to_string_lossy().into_owned());
    let calculator = config
        .steps
        .iter()
        .find_map(|s| {
            s.get("othersoft")
                .or_else(|| s.get("usextb"))
                .and_then(JsonValue::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "the default calculator".to_string());
    let usable: Vec<(usize, &TransitionState)> = result
        .transition_states
        .iter()
        .enumerate()
        .filter(|(_, t)| is_transition_state(t, config.check_frequencies))
        .map(|(k, t)| (k + 1, t))
        .collect();
    let minima = Minima::of(&usable);
    // (The start of a meta-IRC is said as such, not as a failed optimization of step 3.)
    let meta = result
        .transition_states
        .iter()
        .filter(|t| t.meta_irc.is_some())
        .count();
    let not_ts = result.transition_states.len() - usable.len() - meta;

    // Summary.
    let mut summary = String::from(
        "<table><thead><tr><th>#</th><th>energy (Hartree)</th><th>lowest frequency (cm⁻¹)</th>\
         <th>status</th><th>IRC ends</th><th>barrier from end 1</th><th>barrier from end 2</th>\
         <th>reaction energy</th></tr></thead><tbody>",
    );
    for (k, t) in result.transition_states.iter().enumerate() {
        let (status, _) = verdict(t, config.check_frequencies);
        let mut ends = usable
            .iter()
            .position(|(n, _)| *n == k + 1)
            .map_or_else(|| "—".to_string(), |i| minima.ends_text(i));
        // An end that is not a verified minimum is said in its row (and in the warnings of
        // step 4).
        let unverified: Vec<String> = t
            .end_checks
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c, super::autots::EndCheck::NotVerified { .. }))
            .map(|(j, _)| (j + 1).to_string())
            .collect();
        if !unverified.is_empty() {
            let _ = write!(
                ends,
                " <span class=\"warn\">(end {} not verified)</span>",
                unverified.join(", ")
            );
        }
        let end = |j: usize| t.ends.get(j).map(|e| e.1);
        // The start of a meta-IRC is no transition state: no barrier or reaction energy.
        let ts_like = t.meta_irc.is_none();
        let barrier = |j: usize| {
            end(j)
                .filter(|_| ts_like)
                .map(|e| Units::HARTREE_TO_KCAL_MOL * (t.energy - e))
        };
        let reaction = match (end(0), end(1)) {
            (Some(a), Some(b)) if ts_like => Some(Units::HARTREE_TO_KCAL_MOL * (b - a)),
            _ => None,
        };
        let _ = write!(
            summary,
            "<tr><td>{}</td><td class=\"num\">{:.8}</td><td class=\"num\">{}</td><td>{status}</td>\
             <td>{ends}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            k + 1,
            t.energy,
            lowest_wavenumber(&t.directory).map_or_else(|| "—".to_string(), |w| format!("{w:.1}")),
            kcal(barrier(0)),
            kcal(barrier(1)),
            kcal(reaction)
        );
    }
    summary.push_str("</tbody></table><p class=\"note\">Barriers and reaction energies in kcal/mol; the reaction energy is end 2 minus end 1. IRC ends name the minima of the table below the diagram.</p>");
    if not_ts > 0 {
        let _ = write!(
            summary,
            "<p class=\"warn\">{not_ts} of the {} optimizations of step 3 did not give a \
             transition state (not converged, the wrong number of imaginary frequencies, or the \
             same as another); their rows are marked, and their energies are those of where the \
             optimization stopped.</p>",
            result.transition_states.len()
        );
    }

    // Step 3 and 4.
    let mut ts_section = String::from(
        "<section class=\"card glass\" id=\"step3\"><h2>Step 3 · Transition states</h2>",
    );
    let mut irc_section =
        String::from("<section class=\"card glass\" id=\"step4\"><h2>Step 4 · IRC</h2>");
    for note in &result.notes {
        let _ = write!(irc_section, "<p class=\"warn\">{}</p>", escape(note));
    }
    for line in &result.surface {
        let _ = write!(ts_section, "<p class=\"note\">{}</p>", escape(line));
    }
    let mut any_irc = false;
    for (k, t) in result.transition_states.iter().enumerate() {
        let (status, warning) = verdict(t, config.check_frequencies);
        let ok = is_transition_state(t, config.check_frequencies);
        let title = match (ok, t.meta_irc.is_some()) {
            (_, true) => "meta-IRC start",
            (true, false) => "TS",
            (false, false) => "Candidate",
        };
        let _ = write!(
            ts_section,
            "<h3>{title} {} · {} {status}</h3>",
            k + 1,
            escape(&file_name(&t.file))
        );
        if let Some(w) = &warning {
            let _ = write!(ts_section, "<p class=\"warn\">{w}</p>");
        }
        // Both energies on a surface with bias potentials: that of the calculator alone and
        // that of the surface the search ran on.
        let biased = result
            .surface
            .iter()
            .any(|s| s.starts_with("Bias potentials"));
        if biased && t.biased_energy.is_finite() {
            let _ = write!(
                ts_section,
                "<p class=\"note\">Energy of the calculator {:.10} Hartree; with the bias \
                 potentials of the surface {:.10} Hartree.</p>",
                t.energy, t.biased_energy
            );
        }
        let anim = lowest_mode_animation(&t.directory);
        let caption = match (lowest_wavenumber(&t.directory), t.converged) {
            (Some(w), _) if w < 0.0 => format!(
                "{title} {}: the imaginary mode, {:.1}i cm⁻¹ (animated).",
                k + 1,
                -w
            ),
            (Some(w), _) => format!(
                "{title} {}: the lowest mode, {w:.1} cm⁻¹ (animated; no imaginary mode).",
                k + 1
            ),
            (None, false) => format!("Candidate {}: the last step of its optimization.", k + 1),
            (None, true) => format!("{title} {}.", k + 1),
        };
        ts_section.push_str(&viewers.add(
            Some(anim.as_deref().unwrap_or(&t.file)),
            anim.is_some(),
            &caption,
        ));
        if let Some(dir) = &t.irc_directory {
            any_irc = true;
            match &t.meta_irc {
                Some(notice) => {
                    let _ = write!(
                        irc_section,
                        "<h3>meta-IRC from point {} (not a transition state)</h3>\
                         <p class=\"warn\">{}</p>",
                        k + 1,
                        escape(notice)
                    );
                }
                None => {
                    let _ = write!(irc_section, "<h3>IRC of TS {}</h3>", k + 1);
                }
            }
            if let Some(csv) = find_file(dir, "irc_profile.csv") {
                let rows = read_table(&csv);
                let (mut xs, mut es) = (Vec::new(), Vec::new());
                for r in rows.iter().skip(1) {
                    if let (Some(s), Some(e)) = (
                        r.get(1).and_then(|v| v.parse::<f64>().ok()),
                        r.get(2).and_then(|v| v.parse::<f64>().ok()),
                    ) {
                        xs.push(s);
                        es.push(e);
                    }
                }
                let ys: Vec<f64> = es
                    .iter()
                    .map(|e| Units::HARTREE_TO_KCAL_MOL * (e - t.energy))
                    .collect();
                let (title, axis, name) = if t.meta_irc.is_some() {
                    (
                        format!("Energy along the meta-IRC from point {}", k + 1),
                        "kcal/mol (from its start)",
                        "meta-IRC",
                    )
                } else {
                    (
                        format!("Energy along the IRC of TS {}", k + 1),
                        "kcal/mol (from the transition state)",
                        "IRC",
                    )
                };
                irc_section.push_str(&chart(
                    &title,
                    "arc length (mass-weighted, Bohr amu^1/2)",
                    axis,
                    vec![series(name, xs, ys, Rgb::BLUE, PlotStyle::LinePoints)],
                ));
            }
            let path = find_file(dir, "irc_combined_path.xyz");
            irc_section.push_str(&viewers.add(
                path.as_deref(),
                true,
                &if t.meta_irc.is_some() {
                    format!("The meta-IRC from point {} (animated).", k + 1)
                } else {
                    format!("The IRC of TS {} (animated).", k + 1)
                },
            ));
            let labels = usable
                .iter()
                .position(|(n, _)| *n == k + 1)
                .and_then(|i| minima.labels.get(i));
            let mut ends = String::from("<ul>");
            for (j, (file, e)) in t.ends.iter().enumerate() {
                let minimum = labels
                    .and_then(|l| l.get(j))
                    .map_or_else(String::new, |m| format!(" = M{}", m + 1));
                let button = viewers.download(file, "Download .xyz");
                let _ = write!(
                    ends,
                    "<li>End {}{minimum}: {e:.8} Hartree ({:+.2} kcal/mol from {}), {} {button}</li>",
                    j + 1,
                    Units::HARTREE_TO_KCAL_MOL * (e - t.energy),
                    if t.meta_irc.is_some() {
                        "its start"
                    } else {
                        "the transition state"
                    },
                    escape(&file_name(file))
                );
            }
            ends.push_str("</ul>");
            irc_section.push_str(&ends);
            // An optimized end above its transition state is not the minimum the IRC went to:
            // the end was optimized as another system (another cell or charge, for example).
            if t.meta_irc.is_none() && t.ends.iter().any(|e| e.1 > t.energy) {
                irc_section.push_str(
                    "<p class=\"warn\">An optimized end of this IRC lies above its transition \
                     state, so it is not the minimum the IRC went down to; its barriers are not \
                     meaningful. Compare the end structures with the last points of the \
                     IRC.</p>",
                );
            }
        }
    }
    ts_section.push_str("</section>");
    if !any_irc {
        irc_section.push_str(if result.error.is_some() {
            "<p class=\"warn\">Step 4 did not run: the run stopped with the error shown at the \
             top.</p>"
        } else if result.progress.is_some() {
            "<p class=\"note\">Step 4 has not run yet, or its IRCs have not ended.</p>"
        } else {
            "<p class=\"note\">Step 4 did not run: the IRC and the optimized ends come with \
             <code>--run_step4</code>; the transition states above are not yet linked to \
             minima.</p>"
        });
    }
    irc_section.push_str("</section>");

    let settings = super::autots_settings::text(&result.effective_settings);
    let step1 = afir_section(result, &mut viewers);
    let step2 = path_section(result, &mut viewers);
    let evaluations = evaluations_table(result);
    let diagram = format!(
        "{}{}",
        energy_diagram(&usable, &minima),
        minima_table(&usable, &minima, &mut viewers)
    );

    let (layers, system) = page_system(config);
    let page_info = {
        let mut doc = std::collections::BTreeMap::new();
        doc.insert("layers".to_string(), layers);
        JsonValue::Object(doc)
            .to_string_compact()
            .replace("</", "<\\/")
    };
    let models = viewers.models_json();

    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoTS {name}</title>
<style>{style}</style>
</head>
<body>
<header class="topbar glass"><div class="topbar-inner">
<div class="brand">AutoTS · {name}</div>
<nav aria-label="Sections"><a href="#summary">Summary</a><a href="#step1">First path</a><a href="#step2">Path</a><a href="#step3">Transition states</a><a href="#step4">IRC</a><a href="#settings">Settings</a><a href="#credits">Credits</a></nav>
<div class="seg" role="group" aria-label="Theme"><button type="button" data-theme-choice="auto">Auto</button><button type="button" data-theme-choice="light">Light</button><button type="button" data-theme-choice="dark">Dark</button></div>
</div></header>
<main>
<div class="hero">
<h1>AutoTS · {name}</h1>
<p>mopt-rs {version} · calculator: {calculator} · {count} transition state(s) from {candidates} candidate(s){not_ts}{system}</p>
{stopped}
<div class="toolrow"><span class="seg-label">Molecules</span><div class="seg" role="group" aria-label="Display style of every molecule"><button type="button" data-style-choice="ball">Ball and stick</button><button type="button" data-style-choice="stick">Sticks</button><button type="button" data-style-choice="spacefill">Space-filling</button><button type="button" data-style-choice="wire">Wireframe</button></div>{labels}</div>
</div>
<section class="card glass" id="summary"><h2>Summary</h2><div class="table-wrap">{summary}</div>{diagram}{evaluations}</section>
{step1}
{step2}
{ts_section}
{irc_section}
<section class="card glass" id="settings"><h2>Settings</h2><details><summary>The settings of the steps</summary><pre>{settings}</pre></details>
<p class="note">Working directory: {work}. The run's own files (logs, trajectories, CSV tables) are in it; the structures of this page are embedded in it.</p></section>
{credits}
</main>
<script type="application/json" id="models">{models}</script>
<script type="application/json" id="page-info">{page_info}</script>
{viewer}
</body>
</html>
"##,
        version = crate::VERSION,
        viewer = VIEWER,
        labels = LABEL_CONTROLS,
        style = STYLE,
        credits = credits(result),
        system = escape(&system),
        calculator = escape(&calculator),
        count = usable.len(),
        not_ts = if not_ts > 0 {
            format!(" · {not_ts} optimization(s) did not give one")
        } else {
            String::new()
        },
        candidates = result.candidates.len(),
        stopped = match (&result.error, &result.progress) {
            (Some(e), _) => format!(
                "<p class=\"warn\">The run stopped with an error; this page shows what was \
                 done before it. {}</p>",
                escape(e)
            ),
            (None, Some(stage)) => format!(
                "<p class=\"warn\">The run had not ended when this page was written ({}). Reload \
                 this page for the latest saved Step 2 band or completed workflow step.</p>",
                escape(stage)
            ),
            (None, None) => String::new(),
        },
        settings = escape(&settings),
        work = escape(&file_name(&result.directory)),
    );
    std::fs::write(path, html)?;
    Ok(())
}

/// The evaluations of every step and of the whole run: gradients, exact Hessians, energies
/// alone, and the microiterations of an ONIOM environment.
pub(super) fn evaluations_table(result: &AutotsResult) -> String {
    let all = result
        .evaluations
        .iter()
        .fold(crate::pes::counting::EvaluationCounts::default(), |a, b| {
            a.plus(b)
        });
    let row = |name: &str, e: &crate::pes::counting::EvaluationCounts| {
        format!(
            "<tr><td>{name}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            e.gradients,
            e.hessians,
            e.energies,
            if e.micro_relaxations > 0 {
                format!(
                    "{} in {} relaxation(s)",
                    e.micro_iterations, e.micro_relaxations
                )
            } else {
                "0".to_string()
            }
        )
    };
    let mut rows = String::new();
    for (k, e) in result.evaluations.iter().enumerate() {
        rows.push_str(&row(&format!("Step {}", k + 1), e));
    }
    rows.push_str(&row("The whole run", &all));
    format!(
        "<h3>Evaluations</h3><div class=\"table-wrap\"><table><thead><tr><th>step</th> \
        <th>gradient calculations</th><th>exact Hessian calculations</th><th>energies \
         alone</th><th>microiterations</th></tr></thead><tbody>{rows}</tbody></table></div> \
        <p class=\"note\">Counted as the calculator was asked: a Hessian from differences of \
         gradients counts as one Hessian; microiterations are the inner iterations of the \
         relaxations of an ONIOM environment, whose gradients of the lowest level are among the \
         gradient calculations. Evaluations a restarted run took from the journal of the run \
         before are not counted again.</p>"
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use super::*;
    use crate::cli::autots::{FirstPathMethod, PathMethod};

    /// H2 with the bond length 0.74 + `stretch` Angstrom, in a cell when `cell`.
    fn h2(dir: &Path, name: &str, stretch: f64, cell: bool) -> PathBuf {
        let path = dir.join(name);
        let comment = if cell {
            "Lattice=\"6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 7.0\" pbc=\"T T T\""
        } else {
            "0 1"
        };
        std::fs::write(
            &path,
            format!(
                "2\n{comment}\nH 0.0 0.0 0.0\nH 0.0 0.0 {:.4}\n",
                0.74 + stretch
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn candidates_that_are_not_transition_states_and_several_irc() {
        let d = std::env::temp_dir().join(format!("mopt_autots_report_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let m1 = h2(&d, "m1.xyz", 0.0, false);
        let m2 = h2(&d, "m2.xyz", 0.5, false);
        // The same minimum as m2, reached from another IRC.
        let m2_again = h2(&d, "m2_again.xyz", 0.5005, false);
        let m3 = h2(&d, "m3.xyz", 1.0, false);
        let state = |name: &str,
                     energy: f64,
                     converged: bool,
                     imaginary: Option<usize>,
                     ends: Vec<(PathBuf, f64)>| {
            TransitionState {
                file: h2(&d, name, 0.2, false),
                energy,
                biased_energy: energy,
                converged,
                imaginary_modes: imaginary,
                meta_irc: None,
                same_as: None,
                irc_directory: (!ends.is_empty()).then(|| d.clone()),
                end_checks: vec![crate::cli::autots::EndCheck::Minimum; ends.len()],
                ends,
                directory: d.clone(),
            }
        };
        let states = vec![
            state("ts_final_1.xyz", -1.00, false, None, vec![]),
            state(
                "ts_final_2.xyz",
                -1.02,
                true,
                Some(1),
                vec![(m1.clone(), -1.10), (m2, -1.05)],
            ),
            state(
                "ts_final_3.xyz",
                -1.01,
                true,
                Some(1),
                vec![(m2_again, -1.05 + 1e-7), (m3, -1.20)],
            ),
            state("ts_final_4.xyz", -0.99, true, Some(2), vec![]),
        ];
        let mut steps: [BTreeMap<String, JsonValue>; 4] = Default::default();
        steps[0].insert(
            "oniom_flag".to_string(),
            JsonValue::Array(
                ["1", "none", "gfnff"]
                    .iter()
                    .map(|s| JsonValue::String((*s).to_string()))
                    .collect(),
            ),
        );
        let config = AutotsConfig {
            input: h2(&d, "input.xyz", 0.0, true),
            work_dir: d.clone(),
            top_n: 4,
            steps,
            workflow: None,
            first_method: FirstPathMethod::Bias,
            path_method: PathMethod::Chain,
            skip_step1: false,
            run_step4: true,
            skip_to_step4: false,
            check_frequencies: true,
            html_report: true,
            settings: JsonValue::Null,
            document: JsonValue::Null,
            restart_dir: None,
            notes: Vec::new(),
        };
        let result = AutotsResult {
            directory: d.clone(),
            afir_path: None,
            afir_directory: None,
            first_path_method: String::new(),
            path_stage: None,
            candidates: states.iter().map(|s| s.file.clone()).collect(),
            transition_states: states,
            notes: vec!["Steps 3 and 4 do not run on the same surface <test>.".to_string()],
            error: None,
            progress: None,
            ..AutotsResult::default()
        };
        let page = d.join("report.html");
        write(&config, &result, &page).unwrap();
        let html = std::fs::read_to_string(&page).unwrap();
        // What the run warns about as a whole is in the report, escaped.
        assert!(html.contains("do not run on the same surface &lt;test&gt;."));
        assert!(!html.contains("stopped with an error"));
        // A run that stopped says so, with the error, and the steps after it say why they
        // did not run.
        let stopped = AutotsResult {
            error: Some("Calculator error: the charges did not converge <here>".to_string()),
            path_stage: None,
            afir_path: None,
            afir_directory: None,
            transition_states: Vec::new(),
            candidates: Vec::new(),
            ..result.clone()
        };
        write(&config, &stopped, &page).unwrap();
        let failed = std::fs::read_to_string(&page).unwrap();
        assert!(failed.contains("The run stopped with an error"));
        assert!(failed.contains("the charges did not converge &lt;here&gt;"));
        assert!(failed.contains("Step 1 did not finish"));
        assert!(!failed.contains("Step 1 did not run."));
        // A run that goes on says where it is.
        let going = AutotsResult {
            error: None,
            progress: Some("step 2 (the path) is running".to_string()),
            ..stopped.clone()
        };
        write(&config, &going, &page).unwrap();
        let partial = std::fs::read_to_string(&page).unwrap();
        assert!(partial.contains("had not ended when this page was written"));
        assert!(partial.contains("step 2 (the path) is running"));
        assert!(partial.contains("Step 2 has not ended yet."));
        write(&config, &result, &page).unwrap();
        let html = std::fs::read_to_string(&page).unwrap();
        // Not converged and the wrong number of imaginary frequencies: warned, not counted.
        assert!(html.contains("Candidate 1"), "candidate");
        assert!(
            html.contains("did not converge, so this is not a"),
            "warning"
        );
        assert!(html.contains("2 imaginary"), "order 2");
        assert!(
            html.contains("2 transition state(s) from 4 candidate(s)"),
            "count"
        );
        assert!(
            html.contains("2 optimization(s) did not give one"),
            "count of the others"
        );
        // Two transition states on one diagram, sharing the minimum M2.
        assert!(
            html.contains("M1 → M2") && html.contains("M2 → M3"),
            "minima of the IRCs"
        );
        assert!(!html.contains(">M4<"), "the shared minimum is one");
        assert!(
            html.contains(">TS 2<") && html.contains(">TS 3<"),
            "diagram"
        );
        // Downloads, the ONIOM layer and the cell; nothing from elsewhere.
        assert!(html.contains("class=\"dl\""), "downloads");
        // Without the script: a drawing in every viewer and downloads as links.
        assert!(
            !html.contains("<button class=\"dl\""),
            "downloads are links"
        );
        assert!(
            html.contains("href=\"data:chemical/x-xyz;base64,"),
            "file in the link"
        );
        assert_eq!(
            html.matches("<div class=\"viewer\"").count(),
            html.matches("<svg class=\"static-view\"").count(),
            "a drawing in every viewer"
        );
        assert!(html.contains("\"high\":[0]"), "ONIOM layer");
        assert!(html.contains("periodic cell a = 6.000"), "cell");
        assert!(
            !html.contains("<script src=") && !html.contains("<link "),
            "one file"
        );
        assert!(
            html.contains("Rights and credits") && html.contains("GPL-3.0-or-later"),
            "credits"
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[cfg(test)]
mod mode_tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn the_animated_mode_of_a_transition_state_is_its_imaginary_one() {
        let d = std::env::temp_dir().join(format!("mopt_report_modes_{}", std::process::id()));
        let movies = d.join("vibration_animation");
        std::fs::create_dir_all(&movies).unwrap();
        for name in [
            "mode_1_65i_wave_number.xyz",
            "mode_2_68_wave_number.xyz",
            "mode_3_120_wave_number.xyz",
        ] {
            std::fs::write(movies.join(name), "1\n\nH 0 0 0\n").unwrap();
        }
        let picked = lowest_mode_animation(&d).unwrap();
        assert_eq!(file_name(&picked), "mode_1_65i_wave_number.xyz");
        let _ = std::fs::remove_dir_all(&d);
    }
}
