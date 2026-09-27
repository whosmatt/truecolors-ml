"use strict";

const $ = (id) => document.getElementById(id);
const LOOKAHEAD_S = 0.1;
const TICK_MS = 25;
const START_DELAY_S = 0.03;
const ENV_BIN = 256; // samples per precomputed envelope bin; finer zooms read raw samples

const S = {
  songs: [], curId: null, row: null, auto: null, label: null,
  buf: null, env: null, dur: 0,
  bpm: 120, t0: 0, include: [], sections: [],
  evalRes: null, evalSeq: 0, evalTimer: null,
  loadSeq: 0, dirty: false, curSection: -1,
};

// ---------- audio ----------
let actx = null, trackGain = null, clickGain = null, clickBuf = null;
const P = { playing: false, src: null, ctxStart: 0, offset: 0, scheduledUntil: 0, clicks: [] };

function ensureCtx() {
  if (actx) return;
  actx = new (window.AudioContext || window.webkitAudioContext)();
  trackGain = actx.createGain();
  clickGain = actx.createGain();
  trackGain.connect(actx.destination);
  clickGain.connect(actx.destination);
  trackGain.gain.value = +$("trackVol").value;
  clickGain.gain.value = +$("clickVol").value;
  clickBuf = makeClick(actx);
}

function makeClick(c) {
  // Onset at sample 0 so the audible click sits exactly on the scheduled time.
  const sr = c.sampleRate, n = Math.round(sr * 0.025);
  const b = c.createBuffer(1, n, sr), d = b.getChannelData(0);
  let seed = 12345;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const noise = seed / 0x3fffffff - 1;
    d[i] = Math.exp(-t / 0.004) * (0.8 * Math.sin(2 * Math.PI * 2200 * t) + 0.25 * noise);
  }
  return b;
}

function pos() {
  if (!P.playing) return P.offset;
  return Math.min(Math.max(P.offset, P.offset + actx.currentTime - P.ctxStart), S.dur);
}

function play() {
  if (!S.buf || P.playing) return;
  ensureCtx();
  if (actx.state !== "running") {
    const want = S.curId;
    actx.resume().then(() => { if (S.curId === want && !P.playing) play(); }, () => {});
    return;
  }
  if (P.offset >= S.dur - 0.01) P.offset = 0;
  const src = actx.createBufferSource();
  src.buffer = S.buf;
  src.connect(trackGain);
  const when = actx.currentTime + START_DELAY_S;
  src.start(when, P.offset);
  P.src = src; P.ctxStart = when; P.playing = true; P.scheduledUntil = P.offset;
  src.onended = () => {
    if (P.src !== src) return;
    P.src = null; P.playing = false; P.offset = S.dur; cancelClicks(); updatePlayBtn();
  };
  schedule();
  updatePlayBtn();
}

function pause() {
  if (!P.playing) return;
  P.offset = pos();
  const s = P.src; P.src = null; P.playing = false;
  try { s.stop(); } catch (e) { /* already stopped */ }
  cancelClicks();
  updatePlayBtn();
}

function togglePlay() {
  if (!S.buf) { S.pendingPlay = !S.pendingPlay; return; }
  P.playing ? pause() : play();
}

function seek(t) {
  t = Math.min(Math.max(0, t), S.dur || 0);
  if (P.playing) { pause(); P.offset = t; play(); } else P.offset = t;
}

function cancelClicks(fromCtxTime = -Infinity) {
  const keep = [];
  for (const c of P.clicks) {
    if (c.when > fromCtxTime) { try { c.node.stop(); } catch (e) { /* not started */ } }
    else keep.push(c);
  }
  P.clicks = keep;
}

// Grid moved while playing: drop queued clicks and reschedule from now.
function regrid() {
  if (!P.playing) return;
  const now = actx.currentTime;
  cancelClicks(now);
  P.scheduledUntil = P.offset + (now - P.ctxStart);
  schedule();
}

function schedule() {
  if (!P.playing || !actx) return;
  const now = actx.currentTime;
  P.clicks = P.clicks.filter((c) => c.when > now - 0.2);
  if (!$("clickOn").checked) { P.scheduledUntil = P.offset + (now + LOOKAHEAD_S - P.ctxStart); return; }
  const from = Math.max(P.scheduledUntil, P.offset + (now - P.ctxStart));
  const to = P.offset + (now + LOOKAHEAD_S - P.ctxStart);
  if (to <= from) return;
  const T = 60 / S.bpm;
  for (let k = Math.ceil((from - S.t0) / T - 1e-9); ; k++) {
    const bt = S.t0 + k * T;
    if (bt >= to) break;
    if (bt < from || bt < 0 || bt >= S.dur) continue;
    const when = P.ctxStart + (bt - P.offset);
    if (when < now) continue;
    const n = actx.createBufferSource();
    n.buffer = clickBuf;
    n.connect(clickGain);
    n.start(when);
    P.clicks.push({ node: n, when });
  }
  P.scheduledUntil = to;
}
setInterval(schedule, TICK_MS);

function updatePlayBtn() { $("play").innerHTML = P.playing ? "&#10074;&#10074; pause" : "&#9654; play"; }

// ---------- data ----------
async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  let j = null;
  try { j = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok) throw new Error((j && j.error) || `${r.status} ${r.statusText}`);
  return j;
}

async function refreshList() {
  let j;
  try { j = await api("/api/songs"); } catch (e) { $("listCount").textContent = "list failed: " + e.message; return; }
  S.songs = j.songs;
  renderList();
  const cur = S.songs.find((s) => s.id === S.curId);
  if (cur && cur.analysed && !S.auto) loadLateAuto(S.curId);
}

function matches(s, f) {
  switch (f) {
    case "all": return true;
    case "unlabelled": return !s.label_status;
    case "labelled": return !!s.label_status;
    case "needs_review": return !s.label_status && s.needs_review !== false;
  }
  if (f.startsWith("h:")) return s.label_status === f.slice(2);
  if (f === "a:none") return !s.analysed;
  if (f.startsWith("a:")) return s.auto_verdict === f.slice(2);
  return true;
}

// Rows still downloading have no audio yet; only "all" lists them.
function filtered() {
  const f = $("filter").value;
  return S.songs.filter((s) => (f === "all" || s.has_audio) && matches(s, f));
}

function tag(text, cls) { const t = document.createElement("span"); t.className = "tag " + (cls || ""); t.textContent = text; return t; }

function renderList() {
  const ul = $("songList"), list = filtered();
  const scroll = ul.scrollTop;
  ul.textContent = "";
  for (const s of list) {
    const li = document.createElement("li");
    li.dataset.id = s.id;
    if (s.id === S.curId) li.className = "cur";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = `${s.artist || "?"} – ${s.title || s.id}`;
    name.title = name.textContent;
    const tags = document.createElement("div");
    tags.className = "tags";
    tags.append(s.analysed ? tag(s.auto_verdict || "?", s.auto_verdict) : tag("not analysed"));
    if (s.label_status) tags.append(tag("✓ " + s.label_status, s.label_status));
    if (!s.has_audio) tags.append(tag("no audio", "variable_tempo"));
    li.append(name, tags);
    li.onclick = () => loadSong(s.id);
    ul.append(li);
  }
  ul.scrollTop = scroll;
  const nl = S.songs.filter((s) => s.label_status).length;
  $("listCount").textContent = `${list.length} shown · ${S.songs.length} songs · ${nl} labelled`;
}

function neighbour(dir) {
  const list = filtered();
  const i = list.findIndex((s) => s.id === S.curId);
  if (i >= 0) return list[i + dir] || null;
  const full = S.songs.findIndex((s) => s.id === S.curId);
  if (full < 0) return list[0] || null;
  const order = (s) => S.songs.indexOf(s);
  return dir > 0 ? list.find((s) => order(s) > full) || null
                 : [...list].reverse().find((s) => order(s) < full) || null;
}

function go(dir) { const n = neighbour(dir); if (n) loadSong(n.id); }

async function loadSong(id) {
  const seq = ++S.loadSeq;
  // Play state carries over: a song change while playing (or while a previous load was
  // still pending with play wanted) starts the new song once it is decoded.
  S.pendingPlay = P.playing || !!S.pendingPlay;
  pause();
  P.offset = 0;
  S.curId = id; S.buf = null; S.env = null; S.dur = 0; S.evalRes = null;
  history.replaceState(null, "", "#" + id);
  renderList();
  $("loadState").textContent = "loading…";
  $("saved").textContent = ""; $("saved").className = "small";
  let d;
  try { d = await api("/api/song/" + encodeURIComponent(id)); }
  catch (e) { if (seq === S.loadSeq) $("loadState").textContent = "failed: " + e.message; return; }
  if (seq !== S.loadSeq) return;
  S.row = d.row; S.auto = d.auto; S.label = d.label; S.dur = d.auto?.duration_s || d.row.duration_s || 0;
  initGrid();
  renderMeta();
  drawAll();
  try {
    ensureCtx();
    const r = await fetch("/audio/" + encodeURIComponent(id));
    if (!r.ok) throw new Error(`audio ${r.status}`);
    const ab = await r.arrayBuffer();
    if (seq !== S.loadSeq) return;
    const buf = await actx.decodeAudioData(ab);
    if (seq !== S.loadSeq) return;
    S.buf = buf; S.dur = buf.duration;
    S.env = envelope(buf);
    $("loadState").textContent = `${buf.numberOfChannels} ch · ${buf.sampleRate} Hz · ${fmtT(buf.duration)}`
      + (S.auto ? "" : " · not analysed");
    if (S.pendingPlay) { S.pendingPlay = false; drawAll(); play(); }
  } catch (e) {
    if (seq === S.loadSeq) $("loadState").textContent = "audio failed: " + e.message;
  }
  drawAll();
}

async function loadLateAuto(id) {
  let a;
  try { a = await api("/api/auto/" + encodeURIComponent(id)); } catch (e) { return; }
  if (id !== S.curId || S.auto) return;
  S.auto = a;
  if (!S.dirty && !S.label) initGrid(); else initSections();
  renderMeta();
  drawAll();
  $("loadState").textContent += " · analysis arrived";
}

function initSections() {
  S.sections = S.auto?.sections || S.label?.sections || [];
  const ls = S.label?.sections || [];
  S.include = S.sections.map((s) => {
    const m = ls.find((l) => Math.abs(l.start_s - s.start_s) < 0.05);
    return m ? m.include !== false : true;
  });
}

function autoGrid() {
  const f = S.auto?.fit;
  if (f && isFinite(f.bpm) && isFinite(f.t0_s)) return [f.bpm, f.t0_s];
  const sp = S.row?.spotify_tempo;
  return [sp && sp > 0 ? sp : 120, 0];
}

function initGrid() {
  const L = S.label;
  let [bpm, t0] = autoGrid();
  if (L && L.bpm != null && L.t0_s != null && isFinite(L.bpm) && L.bpm > 0) { bpm = L.bpm; t0 = L.t0_s; }
  S.bpm = bpm; S.t0 = t0; S.dirty = false;
  $("note").value = L?.note || "";
  initSections();
  gridChanged(false);
}

// ---------- grid ----------
function normT0(t0) { const T = 60 / S.bpm; return ((t0 % T) + T) % T; }

function gridChanged(dirty = true) {
  S.t0 = normT0(S.t0);
  if (dirty) S.dirty = true;
  $("bpm").value = S.bpm.toFixed(4);
  $("t0").value = S.t0.toFixed(4);
  regrid();
  queueEval();
  renderHits();
  drawAll();
}

// Keep the first beat of the first included section fixed, so a tempo change
// pivots the grid about where the labelled span starts.
function setBpm(nb) {
  if (!isFinite(nb) || nb < 20 || nb > 400) { $("bpm").value = S.bpm.toFixed(4); return; }
  const T = 60 / S.bpm;
  const i = S.include.indexOf(true);
  const start = i >= 0 && S.sections[i] ? S.sections[i].start_s : 0;
  const anchor = S.t0 + Math.ceil((start - S.t0) / T - 1e-9) * T;
  S.bpm = nb; S.t0 = anchor;
  gridChanged();
}

function shiftT0(ds) { S.t0 += ds; gridChanged(); }

function queueEval() {
  clearTimeout(S.evalTimer);
  S.evalTimer = setTimeout(runEval, 150);
}

async function runEval() {
  const seq = ++S.evalSeq;
  if (!S.auto) { S.evalRes = null; $("evalMsg").textContent = "not analysed: no detections to compare"; renderEval(); return; }
  $("evalMsg").textContent = "evaluating…";
  try {
    const r = await api("/api/eval", { id: S.curId, bpm: S.bpm, t0_s: S.t0 });
    if (seq !== S.evalSeq) return;
    S.evalRes = r; $("evalMsg").textContent = "";
  } catch (e) {
    if (seq !== S.evalSeq) return;
    S.evalRes = null; $("evalMsg").textContent = "eval error: " + e.message;
  }
  renderEval();
}

// Hits within HIT_WIN_S of a beat vs of a half-beat, summed activation per grid beat;
// comparable to the analyser's fit.kick_on / kick_off.
const HIT_WIN_S = 0.03;
function hitStats(hits) {
  if (!Array.isArray(hits) || !hits.length || !S.dur) return null;
  const T = 60 / S.bpm, win = Math.min(HIT_WIN_S, T / 4);
  let on = 0, off = 0, nOn = 0, nOff = 0;
  for (const h of hits) {
    const t = h[0], a = h[1];
    if (!isFinite(t) || !isFinite(a)) continue;
    const ph = (((t - S.t0) / T) % 1 + 1) % 1;
    const dBeat = Math.min(ph, 1 - ph) * T, dHalf = Math.abs(ph - 0.5) * T;
    if (dBeat <= win) { on += a; nOn++; } else if (dHalf <= win) { off += a; nOff++; }
  }
  const beats = Math.max(1, Math.floor(S.dur / T));
  return { on: on / beats, off: off / beats, nOn, nOff };
}

function renderHits() {
  const el = $("hitsNow");
  const hits = S.auto?.hits;
  el.textContent = "";
  if (!hits) return;
  let offbeat = false;
  for (const k of ["kick", "snare"]) {
    const st = hitStats(hits[k]);
    if (!st) continue;
    const d = document.createElement("div");
    d.textContent = `${k.padEnd(5)} on/off beat ${st.on.toFixed(2)} / ${st.off.toFixed(2)}  (${st.nOn} / ${st.nOff} hits)`;
    if (st.off > st.on) { d.className = "warnline"; offbeat = true; }
    el.append(d);
  }
  if (offbeat) {
    const w = document.createElement("div");
    w.className = "warnline";
    w.textContent = "more hit activation on half-beats than beats: possible half-beat error, check by ear";
    el.append(w);
  }
}

function hintBpm() {
  const h = S.auto?.fit?.tempo_hint;
  const v = h == null ? NaN : parseFloat(String(h));
  return isFinite(v) && v > 0 ? v : null;
}

// ---------- rendering: text ----------
const fmtMs = (v, d = 1) => (v == null || !isFinite(v) ? "–" : v.toFixed(d));
const fmtPct = (v) => (v == null || !isFinite(v) ? "–" : Math.round(v * 100) + "%");
function fmtT(t, ms = false) {
  if (!isFinite(t)) return "–";
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${(ms ? s.toFixed(3) : Math.floor(s).toString()).padStart(ms ? 6 : 2, "0")}`;
}
function signed(v, d = 1) {
  if (v == null || !isFinite(v)) return "–";
  const s = v.toFixed(d);
  return +s === 0 ? (0).toFixed(d) : (v > 0 ? "+" : "") + s;
}

function kvRows(tbl, rows) {
  tbl.textContent = "";
  for (const [k, v, rowCls] of rows) {
    const tr = document.createElement("tr");
    if (rowCls) tr.className = rowCls;
    const a = document.createElement("td"), b = document.createElement("td");
    a.textContent = k; b.textContent = v; b.className = "mono";
    tr.append(a, b); tbl.append(tr);
  }
}

function renderMeta() {
  const r = S.row || {}, a = S.auto, v = a?.verdict, f = a?.fit;
  $("songTitle").textContent = `${r.artist || "?"} – ${r.title || r.id || ""}`;
  $("songSub").textContent = [r.album, r.genres, r.id].filter(Boolean).join(" · ")
    + (S.label ? ` · labelled ${S.label.status} ${S.label.labelled_at || ""}` : "");
  const av = $("autoVerdict");
  av.textContent = "";
  if (!a) av.append(tag("not analysed"));
  else {
    av.append(tag(v?.auto || "?", v?.auto));
    if (v?.needs_review) av.append(" ", tag("needs review", "insufficient_lock"));
  }
  const ul = $("autoReasons"); ul.textContent = "";
  for (const x of v?.reasons || []) { const li = document.createElement("li"); li.textContent = x; ul.append(li); }
  const hint = f?.tempo_hint, hb = hintBpm();
  const showHint = !!hint || v?.auto === "check_tempo";
  $("tempoHint").hidden = !showHint;
  $("tempoHintText").textContent = hint ? `tempo hint: ${hint}`
    : "check tempo: fit is a triplet-type ratio away from Spotify's";
  $("applyHint").hidden = hb == null;
  const num = (x, d) => (x == null || !isFinite(x) ? "–" : (+x).toFixed(d));
  // Hints only: off > on suggests a half-beat grid error, to be checked by ear.
  const onoff = (name, a, b) => (a == null && b == null ? [] : [[`${name} on/off beat`,
    `${num(a, 2)} / ${num(b, 2)}` + (a != null && b != null && b > a ? "  half-beat?" : ""),
    a != null && b != null && b > a ? "warnrow" : ""]]);
  kvRows($("autoFit"), f ? [
    ["bpm", num(f.bpm, 4) + (f.rounded ? " (rounded)" : "")],
    ...(f.bpm_free != null ? [["bpm free", num(f.bpm_free, 4)]] : []),
    ["t0_s", num(f.t0_s, 4)], ["err", fmtMs(f.err_ms) + " ms"],
    ...(f.err_ms_free != null ? [["err free", fmtMs(f.err_ms_free) + " ms"]] : []),
    ...onoff("kick", f.kick_on, f.kick_off),
    ...onoff("snare", f.snare_on, f.snare_off),
    ...("spotify_ratio" in f ? [["spotify ratio", num(f.spotify_ratio, 4)]] : []),
    ["uncertainty", fmtMs(f.uncertainty_ms, 2) + " ms"], ["drift", f.drift == null ? "–" : signed(f.drift * 100, 3) + " %"],
    ["span", fmtMs(f.span_s, 0) + " s"], ["locks used", String(f.locks_used)],
  ] : [["fit", a ? "none" : "–"]]);
  kvRows($("refTable"), [
    ["spotify tempo", r.spotify_tempo != null ? r.spotify_tempo.toFixed(3) : "–"],
    ["time signature", r.time_signature ?? "–"],
    ["duration", fmtT(r.duration_s || 0)],
    ["youtube", r.youtube_title ? `${r.youtube_title} (${r.youtube_channel || ""})` : "–"],
    ["model", a?.model || "–"], ["fe spec", a?.fe_spec_version ?? "–"], ["analysed", a?.analysed || "–"],
  ]);
  renderSections();
  renderEval();
  renderHits();
}

function cls(el, v, good, warn) {
  el.classList.remove("good", "warn", "badv");
  if (v == null || !isFinite(v)) return;
  el.classList.add(Math.abs(v) <= good ? "good" : Math.abs(v) <= warn ? "warn" : "badv");
}

function renderEval() {
  const t = S.evalRes?.total;
  $("curErr").textContent = t ? fmtMs(t.err_ms) + " ms" : "–";
  $("curOff").textContent = t ? signed(t.offset_ms) + " ms" : "–";
  $("curCov").textContent = t ? fmtPct(t.coverage) : "–";
  $("curBeats").textContent = t ? String(t.beats) : "–";
  cls($("curErr"), t?.err_ms, 10, 20);
  cls($("curOff"), t?.offset_ms, 3, 10);
  cls($("curCov"), t ? (1 - t.coverage) * 100 : null, 25, 50);
  $("applyOffset").disabled = !(t && t.offset_ms != null);
  const rows = $("sections").tBodies[0].rows, ev = S.evalRes?.sections || [];
  for (let i = 0; i < rows.length; i++) {
    const e = ev[i], c = rows[i].cells;
    c[8].textContent = e ? fmtMs(e.err_ms) : "–";
    c[9].textContent = e ? signed(e.offset_ms) : "–";
    c[10].textContent = e ? fmtPct(e.coverage) : "–";
  }
}

function secColour(s, inc) {
  if (!inc) return "#555";
  if (!s.solid) return "#6b7280";
  if (s.agrees === false) return "#e5534b";
  if (s.agrees === true) return "#3fb96a";
  return "#7fae8f"; // solid, agreement not tested
}

function renderSections() {
  const tb = $("sections").tBodies[0];
  tb.textContent = "";
  S.sections.forEach((s, i) => {
    const tr = document.createElement("tr");
    const cells = [
      i + 1, `${fmtT(s.start_s)}–${fmtT(s.end_s)}`, s.music != null ? s.music.toFixed(2) : "–",
      s.local_bpm != null ? s.local_bpm.toFixed(2) : "–", fmtMs(s.local_err_ms), fmtPct(s.local_coverage),
      s.solid == null ? "–" : s.solid ? "yes" : "no", s.agrees == null ? "–" : s.agrees ? "yes" : "no",
      "–", "–", "–",
    ];
    cells.forEach((v, j) => {
      const td = document.createElement("td");
      if (j === 0) {
        const dot = document.createElement("span");
        dot.className = "dot"; dot.style.background = secColour(s, true);
        td.append(dot);
      }
      td.append(String(v));
      if (j >= 8) td.className = "mono";
      tr.append(td);
    });
    const td = document.createElement("td");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = S.include[i];
    cb.onclick = (e) => e.stopPropagation();
    cb.onchange = () => setInclude(i, cb.checked);
    td.append(cb); tr.append(td);
    if (!S.include[i]) tr.classList.add("excl");
    if (i === S.curSection) tr.classList.add("cur");
    tr.onclick = () => seek(s.start_s);
    tb.append(tr);
  });
}

function setInclude(i, v) {
  S.include[i] = v; S.dirty = true;
  const tr = $("sections").tBodies[0].rows[i];
  if (tr) { tr.classList.toggle("excl", !v); tr.cells[11].firstChild.checked = v; }
  drawAll();
}

// ---------- rendering: canvases ----------
function envelope(buf) {
  const n = Math.ceil(buf.length / ENV_BIN);
  const mn = new Float32Array(n).fill(1), mx = new Float32Array(n).fill(-1);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let b = 0; b < n; b++) {
      let lo = mn[b], hi = mx[b];
      const e = Math.min(d.length, (b + 1) * ENV_BIN);
      for (let i = b * ENV_BIN; i < e; i++) { const v = d[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      mn[b] = lo; mx[b] = hi;
    }
  }
  return { mn, mx };
}

// min/max of the audio over [t0, t1) seconds.
function range(t0, t1) {
  const buf = S.buf, sr = buf.sampleRate;
  let i0 = Math.max(0, Math.floor(t0 * sr)), i1 = Math.min(buf.length, Math.ceil(t1 * sr));
  if (i1 <= i0) return null;
  let lo = 1, hi = -1;
  if (i1 - i0 > ENV_BIN * 2) {
    const e = S.env, b1 = Math.min(e.mn.length, Math.ceil(i1 / ENV_BIN));
    for (let b = Math.floor(i0 / ENV_BIN); b < b1; b++) { if (e.mn[b] < lo) lo = e.mn[b]; if (e.mx[b] > hi) hi = e.mx[b]; }
  } else {
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = i0; i < i1; i++) { const v = d[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }
  return [lo, hi];
}

function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.round(cv.clientHeight * dpr);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, cv.clientWidth, cv.clientHeight];
}

let hatch = null;
function hatchPattern(g) {
  if (hatch) return hatch;
  const c = document.createElement("canvas"); c.width = c.height = 8;
  const h = c.getContext("2d");
  h.strokeStyle = "rgba(200,200,200,0.35)"; h.lineWidth = 1.5;
  h.beginPath(); h.moveTo(0, 8); h.lineTo(8, 0); h.moveTo(-2, 2); h.lineTo(2, -2); h.moveTo(6, 10); h.lineTo(10, 6); h.stroke();
  hatch = g.createPattern(c, "repeat");
  return hatch;
}

const BAND_H = 14;
let ovLayer = null, ovKey = "";

function drawOverviewLayer(w, h) {
  const dpr = window.devicePixelRatio || 1;
  const key = [w, h, dpr, S.dur, S.bpm, S.t0, S.include.join(), S.curId, !!S.buf, S.sections.length].join("|");
  if (ovLayer && key === ovKey) return;
  ovKey = key;
  ovLayer = ovLayer || document.createElement("canvas");
  ovLayer.width = Math.round(w * dpr); ovLayer.height = Math.round(h * dpr);
  const g = ovLayer.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const dur = S.dur || 1, x = (t) => (t / dur) * w;
  S.sections.forEach((s, i) => {
    const x0 = x(s.start_s), x1 = x(s.end_s);
    g.fillStyle = secColour(s, S.include[i]);
    g.fillRect(x0 + 0.5, 1, Math.max(1, x1 - x0 - 1), BAND_H - 2);
    g.globalAlpha = 0.12; g.fillRect(x0, BAND_H, x1 - x0, h - BAND_H); g.globalAlpha = 1;
  });
  const top = BAND_H + 2, mid = top + (h - top) / 2, amp = (h - top) / 2 - 2;
  if (S.buf) {
    g.fillStyle = "#9aa6b8";
    for (let px = 0; px < w; px++) {
      const r = range((px / w) * dur, ((px + 1) / w) * dur);
      if (!r) continue;
      g.fillRect(px, mid - r[1] * amp, 1, Math.max(1, (r[1] - r[0]) * amp));
    }
  }
  const T = 60 / S.bpm;
  if ((T / dur) * w >= 4) {
    g.strokeStyle = "rgba(79,156,255,0.35)"; g.lineWidth = 1; g.beginPath();
    for (let t = S.t0; t < dur; t += T) { const xx = Math.round(x(t)) + 0.5; g.moveTo(xx, top); g.lineTo(xx, h); }
    g.stroke();
  }
  S.sections.forEach((s, i) => {
    if (S.include[i]) return;
    const x0 = x(s.start_s), x1 = x(s.end_s);
    g.fillStyle = "rgba(14,16,19,0.55)"; g.fillRect(x0, 0, x1 - x0, h);
    g.fillStyle = hatchPattern(g); g.fillRect(x0, 0, x1 - x0, h);
  });
}

function drawOverview() {
  const [g, w, h] = fitCanvas($("overview"));
  drawOverviewLayer(w, h);
  g.clearRect(0, 0, w, h);
  g.drawImage(ovLayer, 0, 0, w, h);
  const dur = S.dur || 1, p = pos(), zw = +$("zoom").value;
  g.fillStyle = "rgba(255,255,255,0.08)";
  g.fillRect(((zoomCentre(p) - zw / 2) / dur) * w, BAND_H, (zw / dur) * w, h - BAND_H);
  const px = Math.round((p / dur) * w) + 0.5;
  g.strokeStyle = "#fff"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
}

// Stationary window centred on the grid beat nearest the playhead, so beat alignment can
// be judged while playing; frozen while the zoom view is being dragged.
let zoomFrozen = null;
function zoomCentre(p) {
  if (zoomFrozen != null) return zoomFrozen;
  // Always the grid beat nearest the playhead, however narrow the window: the view
  // exists to show one beat, so it changes exactly once per beat and the playhead
  // may leave it in between.
  const T = 60 / S.bpm;
  return S.t0 + Math.round((p - S.t0) / T) * T;
}

function drawZoom() {
  const [g, w, h] = fitCanvas($("zoomView"));
  g.clearRect(0, 0, w, h);
  const zw = +$("zoom").value, p = pos(), c = zoomCentre(p), a = c - zw / 2, b = c + zw / 2;
  const x = (t) => ((t - a) / zw) * w;
  const axisH = 14, top = axisH, bot = h, mid = (top + bot) / 2, amp = (bot - top) / 2 - 2;

  S.sections.forEach((s, i) => {
    if (s.end_s < a || s.start_s > b) return;
    const x0 = Math.max(0, x(s.start_s)), x1 = Math.min(w, x(s.end_s));
    g.fillStyle = secColour(s, S.include[i]);
    g.globalAlpha = 0.5; g.fillRect(x0, 0, x1 - x0, 4); g.globalAlpha = 1;
    if (!S.include[i]) { g.fillStyle = hatchPattern(g); g.fillRect(x0, top, x1 - x0, bot - top); }
    if (s.start_s >= a) {
      g.strokeStyle = "rgba(255,255,255,0.25)"; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(Math.round(x(s.start_s)) + 0.5, top); g.lineTo(Math.round(x(s.start_s)) + 0.5, bot); g.stroke();
      g.setLineDash([]);
    }
  });

  if (S.buf) {
    g.fillStyle = "#6f7b8e";
    const spp = zw / w;
    for (let px = 0; px < w; px++) {
      const t = a + px * spp;
      if (t + spp < 0 || t > S.dur) continue;
      const r = range(t, t + spp);
      if (!r) continue;
      g.fillRect(px, mid - r[1] * amp, 1, Math.max(1, (r[1] - r[0]) * amp));
    }
  }

  g.fillStyle = "#8a93a1"; g.font = "10px ui-monospace, Menlo, monospace"; g.textBaseline = "top";
  const step = zw > 10 ? 2 : zw > 4 ? 1 : zw > 1.5 ? 0.5 : 0.1;
  for (let t = Math.ceil(a / step) * step; t < b; t += step) {
    const xx = Math.round(x(t)) + 0.5;
    g.fillRect(xx, axisH - 4, 1, 4);
    g.fillText(t.toFixed(step < 1 ? 1 : 0) + "s", xx + 2, 2);
  }

  const T = 60 / S.bpm;
  g.strokeStyle = "#4f9cff"; g.lineWidth = 1.5; g.beginPath();
  for (let k = Math.ceil((a - S.t0) / T); ; k++) {
    const t = S.t0 + k * T;
    if (t > b) break;
    if (t < 0) continue;
    const xx = Math.round(x(t)) + 0.5;
    g.moveTo(xx, top); g.lineTo(xx, bot);
  }
  g.stroke();

  const pk = S.auto?.peaks;
  if (pk && pk.length) {
    let lo = 0, hi = pk.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (pk[m][0] < a) lo = m + 1; else hi = m; }
    g.strokeStyle = "#f0a33a"; g.lineWidth = 2; g.beginPath();
    for (let i = lo; i < pk.length && pk[i][0] <= b; i++) {
      const xx = Math.round(x(pk[i][0])) + 0.5, act = Math.min(1, Math.max(0, pk[i][1]));
      g.moveTo(xx, bot); g.lineTo(xx, bot - act * (bot - top) * 0.6);
    }
    g.stroke();
  }

  // Kick/snare hit peaks hang from the top, below the axis.
  const hits = S.auto?.hits || {};
  for (const [k, col] of [["kick", "#e5534b"], ["snare", "#b78cf2"]]) {
    const hs = hits[k];
    if (!Array.isArray(hs) || !hs.length) continue;
    let lo = 0, hi = hs.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (hs[m][0] < a) lo = m + 1; else hi = m; }
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    for (let i = lo; i < hs.length && hs[i][0] <= b; i++) {
      const xx = Math.round(x(hs[i][0])) + 0.5, act = Math.min(1, Math.max(0, hs[i][1]));
      g.moveTo(xx, top); g.lineTo(xx, top + act * (bot - top) * 0.3);
    }
    g.stroke();
  }

  const px = Math.round(x(p)) + 0.5;
  g.strokeStyle = "#fff"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
}

function drawAll() { drawOverview(); drawZoom(); }

function frame() {
  const p = pos();
  $("time").textContent = `${fmtT(p, true)} / ${fmtT(S.dur)}`;
  let cur = -1;
  for (let i = 0; i < S.sections.length; i++) if (p >= S.sections[i].start_s && p < S.sections[i].end_s) { cur = i; break; }
  if (cur !== S.curSection) {
    const rows = $("sections").tBodies[0].rows;
    if (rows[S.curSection]) rows[S.curSection].classList.remove("cur");
    if (rows[cur]) rows[cur].classList.add("cur");
    S.curSection = cur;
  }
  drawAll();
  requestAnimationFrame(frame);
}

// ---------- pointer ----------
function overviewPointer() {
  const cv = $("overview");
  let drag = false;
  const tAt = (e) => (Math.min(Math.max(0, e.offsetX), cv.clientWidth) / cv.clientWidth) * (S.dur || 0);
  cv.addEventListener("pointerdown", (e) => {
    const t = tAt(e);
    if (e.offsetY < BAND_H) {
      const i = S.sections.findIndex((s) => t >= s.start_s && t < s.end_s);
      if (i >= 0) { setInclude(i, !S.include[i]); return; }
    }
    drag = true; cv.setPointerCapture(e.pointerId); seek(t);
  });
  cv.addEventListener("pointermove", (e) => { if (drag) seek(tAt(e)); });
  cv.addEventListener("pointerup", () => { if (drag && !P.playing) play(); drag = false; });
}

function zoomPointer() {
  const cv = $("zoomView");
  // Click or drag puts the playhead under the pointer; never starts playback (scrubbing
  // while paused is how grids get fine-tuned by eye).
  let drag = null;
  const tAt = (e) => zoomFrozen - (+$("zoom").value) / 2 + (e.offsetX / cv.clientWidth) * +$("zoom").value;
  cv.addEventListener("pointerdown", (e) => {
    zoomFrozen = zoomCentre(pos());
    drag = { x: e.clientX, moved: false }; cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (Math.abs(e.clientX - drag.x) > 2) drag.moved = true;
    if (drag.moved) seek(tAt(e));
  });
  const end = (e) => {
    if (!drag) return;
    if (!drag.moved && e.type === "pointerup") seek(tAt(e));
    drag = null; zoomFrozen = null;
  };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", end);
}

// ---------- saving ----------
async function save(status) {
  if (!S.curId) return;
  const next = neighbour(1);
  const body = {
    id: S.curId, status, bpm: S.bpm, t0_s: S.t0, note: $("note").value,
    sections: S.sections.map((s, i) => ({ start_s: s.start_s, end_s: s.end_s, include: S.include[i] })),
  };
  const el = $("saved");
  el.className = "small"; el.textContent = "saving…";
  try {
    const lab = await api("/api/label", body);
    const s = S.songs.find((x) => x.id === S.curId);
    if (s) s.label_status = lab.status;
    el.className = "small ok"; el.textContent = `saved ${lab.status} ✓`;
    const savedId = S.curId;
    renderList();
    if (next && next.id !== savedId) {
      await loadSong(next.id);
      $("saved").className = "small ok";
      $("saved").textContent = `saved previous (${lab.status})`;
    } else {
      S.label = lab; S.dirty = false; renderMeta();
    }
  } catch (e) {
    el.className = "small err"; el.textContent = "save failed: " + e.message;
  }
}

function skip() { if ($("recordSkip").checked) save("skipped"); else go(1); }

// ---------- wiring ----------
function isTyping(e) {
  const t = e.target;
  if (!t || !t.tagName) return false;
  if (t.isContentEditable || t.tagName === "TEXTAREA" || t.tagName === "SELECT") return true;
  return t.tagName === "INPUT" && !["checkbox", "range", "button"].includes(t.type);
}

document.addEventListener("keydown", (e) => {
  if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
  let handled = true;
  switch (e.key) {
    case " ": togglePlay(); break;
    case "m": case "M": $("clickOn").checked = !$("clickOn").checked; onClickToggle(); break;
    case "ArrowLeft": seek(pos() - 5); break;
    case "ArrowRight": seek(pos() + 5); break;
    case "[": case "{": shiftT0(-(e.shiftKey ? 1 : 5) / 1000); break;
    case "]": case "}": shiftT0((e.shiftKey ? 1 : 5) / 1000); break;
    case "n": case "N": go(1); break;
    case "p": case "P": go(-1); break;
    case "Enter": save("confirmed"); break;
    default: handled = false;
  }
  if (handled) e.preventDefault();
});

// Focused buttons would otherwise also react to space/Enter.
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest("button, input[type=checkbox], input[type=range]");
  if (b) b.blur();
});

function onClickToggle() {
  if (!P.playing) return;
  if ($("clickOn").checked) regrid(); else cancelClicks(actx.currentTime);
}

function wire() {
  $("play").onclick = togglePlay;
  $("clickOn").onchange = onClickToggle;
  $("clickVol").oninput = () => { if (clickGain) clickGain.gain.value = +$("clickVol").value; };
  $("trackVol").oninput = () => { if (trackGain) trackGain.gain.value = +$("trackVol").value; };
  $("zoom").oninput = () => { $("zoomVal").textContent = (+$("zoom").value).toFixed(1) + " s"; };
  $("filter").onchange = () => { renderList(); try { localStorage.setItem("labeler.filter", $("filter").value); } catch (e) { /* storage off */ } };
  $("prevSong").onclick = () => go(-1);
  $("nextSong").onclick = () => go(1);
  document.querySelectorAll("[data-bpm]").forEach((b) => { b.onclick = () => setBpm(S.bpm + +b.dataset.bpm); });
  document.querySelectorAll("[data-bpmmul]").forEach((b) => {
    const [n, d = "1"] = b.dataset.bpmmul.split("/");
    b.onclick = () => setBpm((S.bpm * +n) / +d);
  });
  $("applyHint").onclick = () => { const h = hintBpm(); if (h != null) setBpm(h); };
  document.querySelectorAll("[data-ms]").forEach((b) => { b.onclick = () => shiftT0(+b.dataset.ms / 1000); });
  $("bpm").onchange = () => setBpm(parseFloat($("bpm").value));
  $("t0").onchange = () => {
    const v = parseFloat($("t0").value);
    if (isFinite(v)) { S.t0 = v; gridChanged(); } else $("t0").value = S.t0.toFixed(4);
  };
  $("halfBeat").onclick = () => shiftT0(30 / S.bpm);
  $("applyOffset").onclick = () => {
    const o = S.evalRes?.total?.offset_ms;
    if (o != null && isFinite(o)) shiftT0(o / 1000);
  };
  $("resetGrid").onclick = () => { [S.bpm, S.t0] = autoGrid(); gridChanged(); };
  $("confirm").onclick = () => save("confirmed");
  $("rejVar").onclick = () => save("variable_tempo");
  $("rejNon").onclick = () => save("non_rhythmic");
  $("skip").onclick = skip;
  window.addEventListener("resize", drawAll);
  overviewPointer();
  zoomPointer();
}

async function boot() {
  try { const f = localStorage.getItem("labeler.filter"); if (f) $("filter").value = f; } catch (e) { /* storage off */ }
  wire();
  requestAnimationFrame(frame);
  await refreshList();
  const want = decodeURIComponent(location.hash.slice(1));
  const first = S.songs.find((s) => s.id === want) || filtered()[0] || S.songs[0];
  if (first) loadSong(first.id);
  setInterval(refreshList, 15000);
}

boot();
