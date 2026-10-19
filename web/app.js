// In-browser port of recsys/serve.py: the same user tower, retrieval, ranker features and ranker,
// run with ONNX Runtime Web. Results match the Python server (checked in tests/web parity).
"use strict";

const M = "model/";
const K_CANDIDATES = 100;
const PRESETS = {
  "90s sci-fi fan": ["The Matrix", "Terminator 2", "Jurassic Park", "Twelve Monkeys", "The Fifth Element"],
  "Animation lover": ["Toy Story", "Spirited Away", "Up", "WALL·E", "Inside Out"],
  "Crime & thrillers": ["Pulp Fiction", "Seven", "The Usual Suspects", "The Departed", "Prisoners"],
  "Romantic comedies": ["When Harry Met Sally", "Notting Hill", "Love Actually", "Crazy, Stupid, Love", "About Time"],
};

const S = { items: [], byIdx: new Map(), byMovie: new Map(), picked: [] };
const $ = (id) => document.getElementById(id);
const norm = (s) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9 ]+/g, " ").trim();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function bin(name, Type) {
  const r = await fetch(M + name);
  return new Type(await r.arrayBuffer());
}

async function load() {
  ort.env.wasm.numThreads = 1;
  const [meta, items, vectors, genres, decades, logPop, user, ranker] = await Promise.all([
    fetch(M + "meta.json").then((r) => r.json()),
    fetch(M + "items.json").then((r) => r.json()),
    bin("vectors.bin", Float32Array), bin("genres.bin", Float32Array), bin("decades.bin", Int32Array),
    bin("log_pop.bin", Float32Array),
    ort.InferenceSession.create(M + "user_tower.onnx"), ort.InferenceSession.create(M + "ranker.onnx"),
  ]);
  Object.assign(S, { meta, vectors, genres, decades, logPop, user, ranker, D: meta.dim, G: meta.n_genres, N: meta.n_items });
  for (const [idx, movieId, title, year, gs, tmdbId, likes] of items) {
    const m = { idx, movieId, title, year, genres: gs, tmdbId, likes, key: norm(`${title} ${year}`) };
    S.items.push(m);
    S.byIdx.set(idx, m);
    S.byMovie.set(movieId, m);
  }
}

// ── maths ───────────────────────────────────────────────────────────────
const vec = (i) => S.vectors.subarray(i * S.D, (i + 1) * S.D);
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

async function userVector(history) {
  const L = S.meta.max_history;
  const h = new BigInt64Array(L);
  history.slice(-L).reverse().forEach((v, i) => { h[i] = BigInt(v); });
  const out = await S.user.run({ history: new ort.Tensor("int64", h, [1, L]) });
  return out.user_vec.data;
}

async function recommend(history, k = 10, rerank = true) {
  const u = await userVector(history);
  const seen = new Set(history);
  const scores = new Float32Array(S.N);
  for (let i = 1; i < S.N; i++) scores[i] = seen.has(i) ? -Infinity : dot(vec(i), u);
  scores[0] = -Infinity;
  const order = Array.from(scores.keys()).filter((i) => isFinite(scores[i])).sort((a, b) => scores[b] - scores[a]);
  const cand = order.slice(0, K_CANDIDATES);
  let ranked = cand;
  if (rerank) {
    const r = await rankerScores(history, cand, cand.map((i) => scores[i]), u);
    ranked = cand.map((c, j) => [c, r[j]]).sort((a, b) => b[1] - a[1]).map((x) => x[0]);
  }
  return ranked.slice(0, k).map((i) => {
    let best = history[0], bs = -Infinity;
    for (const h of history) { const s = dot(vec(i), vec(h)); if (s > bs) { bs = s; best = h; } }
    return { ...S.byIdx.get(i), because: S.byIdx.get(best).title };
  });
}

async function rankerScores(history, cand, candScores, u) {
  const k = cand.length, D = S.D, G = S.G;
  const histGenre = new Float32Array(G);
  for (const h of history) for (let g = 0; g < G; g++) histGenre[g] += S.genres[h * G + g] / history.length;
  const hv = history.slice(-20), recent = hv.slice(-5);
  const idx = new BigInt64Array(k), dec = new BigInt64Array(k);
  const gen = new Float32Array(k * G), uv = new Float32Array(k * D), iv = new Float32Array(k * D), dense = new Float32Array(k * 7);
  cand.forEach((c, j) => {
    const cv = vec(c);
    idx[j] = BigInt(c);
    dec[j] = BigInt(S.decades[c]);
    gen.set(S.genres.subarray(c * G, (c + 1) * G), j * G);
    uv.set(u, j * D);
    iv.set(cv, j * D);
    let aff = 0; for (let g = 0; g < G; g++) aff += S.genres[c * G + g] * histGenre[g];
    let mx = -Infinity; for (const h of hv) mx = Math.max(mx, dot(cv, vec(h)));
    let mean = 0; for (const h of recent) mean += dot(cv, vec(h)) / recent.length;
    dense.set([candScores[j], S.logPop[c], (S.decades[c] - 9) / 3, aff, mx, mean, Math.log1p(history.length) / 5], j * 7);
  });
  const T = (type, data, dims) => new ort.Tensor(type, data, dims);
  const out = await S.ranker.run({
    item_idx: T("int64", idx, [k]), item_genres: T("float32", gen, [k, G]), item_decade: T("int64", dec, [k]),
    user_vec: T("float32", uv, [k, D]), item_vec: T("float32", iv, [k, D]), dense: T("float32", dense, [k, 7]),
  });
  return out.score.data;
}

// ── UI ──────────────────────────────────────────────────────────────────
function search(q, limit = 8) {
  q = norm(q);
  if (q.length < 2) return [];
  return S.items.filter((m) => m.key.includes(q) && !S.picked.includes(m.idx))
    .sort((a, b) => (b.key.startsWith(q) - a.key.startsWith(q)) || b.likes - a.likes).slice(0, limit);
}

function label(m) { return m.year ? `${m.title} (${m.year})` : m.title; }

function renderPicked() {
  $("picked").innerHTML = S.picked.map((i) =>
    `<button class="chip" data-idx="${i}" title="Remove">${esc(label(S.byIdx.get(i)))} <span aria-hidden="true">×</span></button>`).join("");
}

function card(m) {
  const title = m.tmdbId ? `<a href="https://www.themoviedb.org/movie/${m.tmdbId}" target="_blank" rel="noopener">${esc(m.title)} ↗</a>` : esc(m.title);
  return `<div class="card"><p class="yr">${m.year || ""}</p><h4>${title}</h4><p class="g">${esc(m.genres.join(" · "))}</p>` +
    `<p class="why">because you liked <b>${esc(m.because)}</b></p></div>`;
}

async function run() {
  renderPicked();
  if (!S.picked.length) { $("out").innerHTML = "<p class='muted'>Pick at least one movie you like (or a preset).</p>"; return; }
  const rerank = $("rerank").checked;
  const t = performance.now();
  const recs = await recommend(S.picked, 10, rerank);
  const ms = performance.now() - t;
  $("out").innerHTML = recs.map(card).join("");
  $("stats").textContent = `${rerank ? "two-tower retrieval → DLRM re-ranking of the top 100" : "two-tower retrieval only"} · ${ms.toFixed(0)} ms in your browser`;
}

function setPreset(name) {
  S.picked = PRESETS[name].map((t) => search(t, 1)[0]).filter(Boolean).map((m) => m.idx);
  run();
}

async function main() {
  try {
    await load();
  } catch (e) {
    $("stats").textContent = "Could not load the models: " + e.message;
    return;
  }
  $("presets").innerHTML = Object.keys(PRESETS).map((n) => `<button class="preset">${esc(n)}</button>`).join("");
  $("presets").addEventListener("click", (e) => { if (e.target.matches(".preset")) setPreset(e.target.textContent); });
  $("picked").addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (b) { S.picked = S.picked.filter((i) => i !== Number(b.dataset.idx)); run(); }
  });
  const q = $("q"), sug = $("suggest");
  q.disabled = false;
  q.addEventListener("input", () => {
    const hits = search(q.value);
    sug.innerHTML = hits.map((m) => `<li data-idx="${m.idx}">${esc(label(m))}</li>`).join("");
    sug.hidden = !hits.length;
  });
  sug.addEventListener("click", (e) => {
    const li = e.target.closest("li");
    if (!li) return;
    S.picked.push(Number(li.dataset.idx));
    q.value = "";
    sug.hidden = true;
    run();
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) sug.hidden = true; });
  $("rerank").addEventListener("change", run);
  window.recommendIds = async (movieIds, rerank = true) =>  // used by the parity check
    (await recommend(movieIds.map((m) => S.byMovie.get(m).idx), 10, rerank)).map((r) => r.movieId);
  setPreset("90s sci-fi fan");
}

main();
