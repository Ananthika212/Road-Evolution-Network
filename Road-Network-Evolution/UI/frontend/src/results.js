/* Result pages (Change Analysis, Road Network, Priority Review, Final Report).
   Imperative DOM code from the ROAD-EVO prototype; React mounts the empty <section id="pg-*"> hosts and calls renderAll(). */
const $ = id => document.getElementById(id);

const COL = {High: "#ef4444", Medium: "#f5b324", Low: "#4c8dff"};
let data = null, imgs = null, selected = null, sortBy = "priority";
let nav = () => {};
const allMaps = [];

/* tiny DOM helper: h("div", {class:"x", onclick:fn}, child, "text") — text is always set as text, never HTML */
function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "style") e.style.cssText = v; else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const svg = (inner, cls) => { const w = document.createElement("span"); w.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`; const s = w.firstChild; if (cls) s.setAttribute("class", cls); return s; };
const ICON = {
  check: '<path d="M20 6L9 17l-5-5"/>', chart: '<path d="M4 20V10m6 10V4m6 16v-7m4 7H2"/>', layers: '<path d="M12 2l9 5-9 5-9-5zM3 12l9 5 9-5M3 17l9 5 9-5"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/>', road: '<path d="M8 3L5 21M16 3l3 18M12 4v3m0 4v3m0 4v2"/>',
  alert: '<circle cx="12" cy="12" r="9"/><path d="M12 7v6m0 3v.5"/>', circle: '<circle cx="12" cy="12" r="8"/>', shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
  flag: '<path d="M5 21V4m0 0h12l-2 4 2 4H5"/>', down: '<path d="M12 3v12m0 0l-4-4m4 4l4-4M4 21h16"/>'
};
const fmt = n => Number(n).toLocaleString();
const pad2 = n => String(n).padStart(2, "0");
const loadImg = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
const dl = (name, text, type) => { const a = h("a", {href: URL.createObjectURL(new Blob([text], {type})), download: name}); a.click(); };


const go = page => nav(page);
export function relayout() { allMaps.forEach(m => m.relayout()); }
export async function loadImages(d) {
  const im = {};
  await Promise.all(["t1", "t2", "roads_t1", "roads_t2", "priority", "heat", "mask", "overlay"].map(async k => im[k] = await loadImg(d.images[k])));
  im.outputs = await Promise.all(d.images.outputs.map(loadImg));
  return im;
}
export function renderAll(d, im, navFn) { data = d; imgs = im; nav = navFn; selected = null; outputIdx = 0; buildAll(); }
/* ---------------- shared bits ---------------- */
const pageHead = (crumb, title, sub, pill) => h("div", {class: "headrow"},
  h("div", {}, h("div", {class: "crumb"}, crumb), h("h1", {}, title), h("div", {class: "sub"}, sub)), pill);
const okPill = txt => h("div", {class: "pills"}, h("span", {class: "pill ok"}, h("i"), txt));
const dateOf = k => (data.meta[k] || "").trim();

function buildAll() { allMaps.length = 0; buildChange(); buildRoad(); buildPrio(); buildReport(); }

/* ---------------- CHANGE ANALYSIS ---------------- */
let outputIdx = 0;
function buildChange() {
  const m = data.meta, k = data.metrics, host = $("pg-change"); host.replaceChildren();
  host.append(pageHead("ROAD-EVO / Change Analysis / LGMM", "Change Detection (LGMM)",
    "Detect road changes between multi-temporal satellite images using LGMM network", okPill("LGMM Analysis Completed")));

  const stat = (a, b) => h("div", {class: "stat"}, h("small", {}, a), h("b", {}, b));
  host.append(h("div", {class: "card banner"},
    h("div", {class: "check"}, svg(ICON.check)),
    h("div", {}, h("h2", {}, "Model Run Successful"), h("div", {class: "hint"}, "LGMM has processed the input images and generated change detection results.")),
    stat("Model", "LGMM"), stat("Inference Time", m.seconds_change + " s"), stat("Input Size", `${m.model_input} × ${m.model_input}`),
    stat("Output", `${m.n_outputs} map${m.n_outputs === 1 ? "" : "s"}`)));

  const box = (cap, sub, src) => h("div", {class: "imgbox"}, h("div", {class: "cap"}, cap, sub ? h("small", {}, sub) : null), h("img", {src, alt: cap}));
  const inputs = h("div", {class: "card"}, h("h3", {}, svg(ICON.image), "Input Images"),
    h("div", {class: "imgpair"}, box("T1 (Previous)", dateOf("t1_date"), data.images.t1), box("T2 (Current)", dateOf("t2_date"), data.images.t2)));

  const heatSrc = outputIdx === 0 ? data.images.heat : data.images.outputs[outputIdx];
  const probBox = h("div", {class: "imgbox sel"}, h("div", {class: "cap"}, outputIdx === 0 ? "Change Probability Map" : `Change Probability (Output ${outputIdx})`),
    h("div", {class: "cbar"}, h("img", {src: heatSrc, alt: "Change probability map"}),
      h("div", {class: "bar"}), h("div", {class: "ticks"}, h("span", {}, "1.0"), h("span", {}, "0.5"), h("span", {}, "0.0"))));
  const overlayBox = h("div", {class: "imgbox"}, h("div", {class: "cap"}, "Overlay (on T2)"),
    h("img", {src: data.images.overlay, alt: "Overlay on T2"}),
    h("div", {class: "legend row"}, h("span", {}, h("i", {style: "background:#ef4444"}), "Road Change"), h("span", {}, h("i", {style: "background:#3a4a43"}), "No Change")));
  const out = h("div", {class: "card"}, h("h3", {}, svg(ICON.chart), "LGMM Output"),
    h("div", {class: "outs3"}, probBox, box("Binary Change Mask", "", data.images.mask), overlayBox));

  host.append(h("div", {class: "grid2 wide"}, inputs, out));

  const met = (a, b) => h("div", {class: "metric"}, h("small", {}, a), h("b", {}, b));
  const metrics = h("div", {class: "card"}, h("h3", {}, svg(ICON.chart), "Analysis Metrics"),
    h("div", {class: "metrics3"}, met("Changed Pixels", fmt(k.changed_pixels)), met("Total Pixels", fmt(k.total_pixels)), met("Change Percentage", k.change_pct + "%")),
    h("div", {class: "okbox"}, h("div", {class: "check", style: "width:34px;height:34px"}, svg(ICON.check)),
      h("div", {}, h("b", {}, k.changed_pixels ? "LGMM detected road changes successfully!" : "No road changes detected"),
        h("span", {}, `The model has identified ${fmt(k.changed_pixels)} changed pixels (${k.change_pct}% of the image).`))));

  const thumbs = h("div", {class: "thumbs"}, data.images.outputs.map((src, i) =>
    h("button", {class: "thumb" + (i === outputIdx ? " sel" : ""), onclick: () => { outputIdx = i; buildChange(); }}, h("img", {src, alt: "Output " + i}), "Output " + i)));
  const all = h("div", {class: "card"}, h("h3", {}, svg(ICON.layers), `All Output Maps (${m.n_outputs})`), thumbs,
    h("div", {class: "rowend"}, h("button", {class: "btn", onclick: downloadCsv}, "Download Results")));
  host.append(h("div", {class: "grid2 wide"}, metrics, all));
}

function downloadCsv() {
  const cols = ["rank", "priority", "change_type", "area_m2", "area_px", "affected_road_length_m", "confidence_C", "road_association_RAS", "length_norm_L", "connectivity_K", "priority_score", "centroid_x", "centroid_y"];
  dl("road_changes.csv", [cols.join(",")].concat(data.regions.map(r => cols.map(c => r[c]).join(","))).join("\n"), "text/csv");
}

/* ---------------- MAP COMPONENT (canvas layers + pins + zoom + scale bar) ---------------- */
function createMap(opts) {
  const W = data.meta.width, H = data.meta.height, gsd = data.meta.gsd;
  const canvas = h("canvas", {width: W, height: H}), inner = h("div", {class: "mapinner"}, canvas),
        wrap = h("div", {class: "mapwrap"}, inner), box = h("div", {class: "mapbox"}, wrap);
  let zoom = 1, layers = [], pinEls = new Map(), scale = null;
  const ctx = canvas.getContext("2d");

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const L of layers) { if (L.fill) { ctx.fillStyle = L.fill; ctx.fillRect(0, 0, W, H); } else ctx.drawImage(L.img, 0, 0, W, H); }
  }
  function updateScale() {
    if (!scale || !inner.clientWidth) return;
    const mPerPx = gsd * W / inner.clientWidth;
    const steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    const meters = steps.find(s => s / mPerPx >= 70) || steps[steps.length - 1];
    scale.firstChild.textContent = meters >= 1000 ? meters / 1000 + " km" : meters + " m";
    scale.lastChild.style.width = Math.round(meters / mPerPx) + "px";
  }
  function setZoom(z) { zoom = Math.max(1, Math.min(6, z)); inner.style.width = zoom * 100 + "%"; updateScale(); }
  function mkPin(r) {
    const p = h("div", {class: "pin", title: `Region ${pad2(r.rank)} — ${r.priority}`, style: `left:${100 * r.centroid_x / W}%;top:${100 * r.centroid_y / H}%`, onclick: e => { e.stopPropagation(); opts.onPin && opts.onPin(r); }});
    p.innerHTML = `<svg viewBox="0 0 26 32"><path d="M13 31C13 31 2 19 2 12a11 11 0 0 1 22 0c0 7-11 19-11 19z" fill="${COL[r.priority]}" stroke="#fff" stroke-width="1.6"/></svg>`;
    p.append(h("span", {}, pad2(r.rank)));
    return p;
  }
  if (opts.legend) box.append(opts.legend);
  box.append(
    h("div", {class: "zoom"}, h("button", {onclick: () => setZoom(zoom * 1.5), "aria-label": "Zoom in"}, "+"), h("button", {onclick: () => setZoom(zoom / 1.5), "aria-label": "Zoom out"}, "−"),
      h("button", {onclick: () => { setZoom(1); wrap.scrollTo(0, 0); }, "aria-label": "Reset view", title: "Reset"}, "⌖")));
  scale = h("div", {class: "scalebar"}, h("span", {}, ""), h("div", {})); box.append(scale);
  if (opts.toggle) box.append(opts.toggle);

  const api = {
    el: box,
    setLayers(l) { layers = l; draw(); },
    setPins(list) { pinEls.forEach(e => e.remove()); pinEls.clear(); list.forEach(r => { const p = mkPin(r); pinEls.set(r.id, p); inner.append(p); }); },
    markSelected(id) { pinEls.forEach((e, k) => e.classList.toggle("sel", k === id)); },
    focus(r) {
      setZoom(Math.max(zoom, 2.5));
      requestAnimationFrame(() => {
        wrap.scrollTo({left: r.centroid_x / W * inner.clientWidth - wrap.clientWidth / 2, top: r.centroid_y / H * inner.clientHeight - wrap.clientHeight / 2, behavior: "smooth"});
      });
    },
    relayout: updateScale, toCanvas: () => canvas
  };
  allMaps.push(api);
  return api;
}

/* ---------------- ROAD NETWORK ---------------- */
function buildRoad() {
  const m = data.metrics, host = $("pg-road"); host.replaceChildren();
  host.append(pageHead("ROAD-EVO / Road Network", "Road Network Extraction (CRNet)",
    "Roads extracted from each image by CRNet. White lines show the road centrelines.", okPill("CRNet Extraction Completed")));
  const net = m.road_km_t2 - m.road_km_t1;
  const kpi = (cls, ico, a, b, c) => h("div", {class: "kpi"}, h("div", {class: "ico " + cls}, svg(ICON[ico])), h("div", {}, h("small", {}, a), h("b", {}, b), h("em", {}, c)));
  host.append(h("div", {class: "kpis", style: "grid-template-columns:repeat(4,1fr)"},
    kpi("g", "road", "Road Length — T1", m.road_km_t1 + " km", `${m.road_cov_t1}% of pixels are road`),
    kpi("g", "road", "Road Length — T2", m.road_km_t2 + " km", `${m.road_cov_t2}% of pixels are road`),
    kpi(net >= 0 ? "g" : "r", "chart", "Net Change", (net >= 0 ? "+" : "") + net.toFixed(2) + " km", "T2 minus T1"),
    kpi("b", "layers", "Model", "CRNet", `${data.meta.crnet_tile}×${data.meta.crnet_tile} tiles · ${data.meta.seconds_roads} s`)));
  const one = (tag, title, date, base, roads) => {
    const mp = createMap({}); mp.setLayers([{img: base}, {img: roads}]);
    return h("div", {class: "card"}, h("h3", {}, h("span", {class: "tag"}, tag), title, h("span", {class: "hint", style: "margin-left:auto"}, date)), h("div", {style: "margin-top:12px"}, mp.el));
  };
  host.append(h("div", {class: "grid2"}, one("T1", "Previous Road Network", dateOf("t1_date"), imgs.t1, imgs.roads_t1), one("T2", "Current Road Network", dateOf("t2_date"), imgs.t2, imgs.roads_t2)));
  requestAnimationFrame(() => host.querySelectorAll(".mapbox").forEach(() => {}));
}

/* ---------------- PRIORITY REVIEW ---------------- */
let map = null, mapTab = "overlay", roadsOnly = false;
function sortedRegions() {
  const a = data.regions.slice();
  const key = {priority: r => r.rank, confidence: r => -r.confidence_C, area: r => -r.area_px, length: r => -r.affected_road_length_m}[sortBy];
  return a.sort((x, y) => key(x) - key(y));
}
function applyLayers() {
  const bg = roadsOnly ? {fill: "#04100c"} : {img: mapTab === "t1" ? imgs.t1 : imgs.t2};
  const L = [bg];
  if (mapTab === "t1") L.push({img: imgs.roads_t1}); else L.push({img: imgs.roads_t2});
  if (mapTab === "overlay") L.push({img: imgs.priority});
  map.setLayers(L);
  map.setPins(mapTab === "overlay" ? data.regions : []); map.markSelected(selected);
}
function buildPrio() {
  const k = data.metrics, host = $("pg-prio"); host.replaceChildren();
  const range = [dateOf("t1_date"), dateOf("t2_date")].every(Boolean) ? h("span", {class: "pill"}, `T1: ${dateOf("t1_date")}   T2: ${dateOf("t2_date")}`) : null;
  host.append(h("div", {class: "headrow"},
    h("div", {}, h("div", {class: "crumb"}, "ROAD-EVO / Priority Review"), h("h1", {}, "Priority Review"),
      h("div", {class: "sub"}, "View and prioritize detected road changes based on impact, confidence and importance.")),
    h("div", {class: "pills"}, h("span", {class: "pill ok"}, h("i"), "All Analysis Completed"), range, h("button", {class: "btn ghost", onclick: () => go("report")}, "Export Report"))));
  const kpi = (cls, ico, a, b, c) => h("div", {class: "kpi"}, h("div", {class: "ico " + cls}, svg(ICON[ico])), h("div", {}, h("small", {}, a), h("b", {}, b), h("em", {}, c)));
  host.append(h("div", {class: "kpis"},
    kpi("g", "road", "Total Change Regions", k.total_regions, "Detected road changes"), kpi("r", "alert", "High Priority", k.counts.High, "Critical attention required"),
    kpi("y", "alert", "Medium Priority", k.counts.Medium, "Needs monitoring"), kpi("b", "circle", "Low Priority", k.counts.Low, "Minor changes"),
    kpi("g", "shield", "Avg. Confidence", k.avg_confidence.toFixed(2), "Model confidence score")));

  const legend = h("div", {class: "mlegend"}, ["High", "Medium", "Low"].map(p => h("div", {}, h("i", {style: "background:" + COL[p]}), p + " Priority")),
    h("div", {}, h("i", {style: "background:#fff;height:2px;vertical-align:3px"}), "Road Network"));
  const toggle = h("div", {class: "sat"}, ["Satellite", "Roads only"].map((t, i) =>
    h("button", {"aria-pressed": String((i === 1) === roadsOnly), onclick: e => { roadsOnly = i === 1; e.target.parentNode.childNodes.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i))); applyLayers(); }}, t)));
  map = createMap({legend, toggle, onPin: r => select(r.id, true)});
  const tabs = h("div", {class: "tabs"}, [["overlay", "Change Overlay"], ["t1", "T1 Road Network"], ["t2", "T2 Road Network"]].map(([id, label]) =>
    h("button", {class: "tab", "aria-selected": String(id === mapTab), onclick: e => { mapTab = id; e.target.parentNode.childNodes.forEach(b => b.setAttribute("aria-selected", "false")); e.target.setAttribute("aria-selected", "true"); applyLayers(); }}, label)));
  host.append(h("div", {class: "grid2", style: "grid-template-columns:1.5fr 1fr"},
    h("div", {class: "card"}, tabs, map.el),
    h("div", {class: "card listcard"},
      h("div", {class: "listhead"}, h("h3", {}, svg(ICON.flag), "Detected Road Changes"),
        h("label", {class: "hint"}, "Sort by: ", h("select", {id: "sort", onchange: e => { sortBy = e.target.value; renderList(); }},
          [["priority", "Priority"], ["confidence", "Confidence"], ["area", "Change area"], ["length", "Road length"]].map(([v, t]) => h("option", {value: v}, t))))),
      h("div", {class: "regions", id: "regions"}),
      h("div", {style: "margin-top:12px"}, h("button", {class: "btn dark", onclick: () => go("report")}, "Proceed to Final Report  ›")))));
  applyLayers(); renderList();
}
function renderList() {
  const host = $("regions"); host.replaceChildren();
  if (!data.regions.length) { host.append(h("div", {class: "empty"}, "No road changes were found in this image pair.")); return; }
  for (const r of sortedRegions()) {
    const more = h("div", {class: "more"},
      [["Type", r.change_type], ["Score", r.priority_score.toFixed(3)], ["Area", fmt(r.area_m2) + " m²"], ["Road assoc.", r.road_association_RAS.toFixed(2)],
       ["Connectivity", `${r.connectivity_K.toFixed(2)} (${r.affected_nodes}/${r.total_nodes})`], ["Length term", r.length_norm_L.toFixed(2)]]
        .map(([a, b]) => h("div", {}, a + ": ", h("b", {style: "text-transform:capitalize"}, b))));
    const card = h("div", {class: "rc" + (r.id === selected ? " sel open" : ""), "data-id": r.id, onclick: () => select(r.id, true)},
      h("img", {src: r.thumb, alt: "Region " + pad2(r.rank)}),
      h("div", {},
        h("div", {class: "t"}, h("b", {}, "Region " + pad2(r.rank)), h("span", {class: "badge " + r.priority}, r.priority.toUpperCase()), h("span", {class: "conf"}, "Confidence: " + r.confidence_C.toFixed(2))),
        h("div", {class: "d"}, r.description),
        h("div", {class: "m"}, "Change Area: " + fmt(r.area_px) + " px", h("span", {}, "|"), "Length: " + r.length_km.toFixed(2) + " km"),
        h("div", {class: "act"}, h("button", {class: "btn sm", onclick: e => { e.stopPropagation(); select(r.id, true); }}, "View Details  ›"))),
      more);
    host.append(card);
  }
}
function select(id, focus) {
  selected = id;
  document.querySelectorAll(".rc").forEach(c => { const on = Number(c.dataset.id) === id; c.classList.toggle("sel", on); c.classList.toggle("open", on); });
  if (map) { map.markSelected(id); const r = data.regions.find(x => x.id === id); if (focus && r) map.focus(r); }
  const el = document.querySelector(`.rc[data-id="${id}"]`); if (el) el.scrollIntoView({block: "nearest", behavior: "smooth"});
}

/* ---------------- FINAL REPORT ---------------- */
function buildReport() {
  const m = data.meta, k = data.metrics, host = $("pg-report"); host.replaceChildren();
  host.append(h("div", {class: "headrow"}, h("div", {}, h("div", {class: "crumb"}, "ROAD-EVO / Final Report"), h("h1", {}, "Road Change Report"),
    h("div", {class: "sub"}, "Detected road changes ranked for map review")),
    h("div", {class: "pills noprint"}, h("button", {class: "btn ghost", onclick: () => go("prio")}, "‹ Back"), h("button", {class: "btn", onclick: () => window.print()}, "Print / Save as PDF"))));
  const kv = (a, b) => h("div", {}, h("small", {}, a), h("b", {}, b || "—"));
  host.append(h("div", {class: "card"}, h("div", {class: "kv"},
    kv("Study location", m.location), kv("T1 date", dateOf("t1_date")), kv("T2 date", dateOf("t2_date")), kv("Image size", `${m.width} × ${m.height} px at ${m.gsd} m/px`),
    kv("Change regions", k.total_regions), kv("High / Medium / Low", `${k.counts.High} / ${k.counts.Medium} / ${k.counts.Low}`),
    kv("Average confidence", k.avg_confidence.toFixed(2)), kv("Road length T1 → T2", `${k.road_km_t1} → ${k.road_km_t2} km`))));
  const c = document.createElement("canvas"); c.width = m.width; c.height = m.height;
  const x = c.getContext("2d"); x.drawImage(imgs.t2, 0, 0); x.drawImage(imgs.roads_t2, 0, 0); x.drawImage(imgs.priority, 0, 0);
  x.font = `bold ${Math.max(14, Math.round(m.width / 55))}px sans-serif`; x.textAlign = "center"; x.textBaseline = "middle";
  const rad = Math.max(10, Math.round(m.width / 70));
  data.regions.forEach(r => { x.beginPath(); x.arc(r.centroid_x, r.centroid_y, rad, 0, 7); x.fillStyle = COL[r.priority]; x.fill(); x.lineWidth = 2; x.strokeStyle = "#fff"; x.stroke(); x.fillStyle = "#fff"; x.fillText(pad2(r.rank), r.centroid_x, r.centroid_y + 1); });
  const pic = h("img", {class: "rmap", alt: "Priority map", src: c.toDataURL("image/jpeg", .9)});
  host.append(h("div", {class: "card"}, h("h3", {}, "Priority map"), h("div", {style: "margin-top:12px"}, pic)));
  const head = ["#", "Priority", "Type", "Area (m²)", "Road length (m)", "Confidence", "Score"];
  const rows = data.regions.map(r => h("tr", {}, h("td", {}, pad2(r.rank)), h("td", {}, h("span", {class: "badge " + r.priority}, r.priority.toUpperCase())),
    h("td", {style: "text-transform:capitalize"}, r.change_type), h("td", {}, fmt(r.area_m2)), h("td", {}, fmt(r.affected_road_length_m)), h("td", {}, r.confidence_C.toFixed(2)), h("td", {}, r.priority_score.toFixed(3))));
  host.append(h("div", {class: "card"}, h("h3", {}, "Ranked road changes"), h("div", {style: "overflow-x:auto;margin-top:10px"},
    rows.length ? h("table", {class: "rtable"}, h("thead", {}, h("tr", {}, head.map(t => h("th", {}, t)))), h("tbody", {}, rows)) : h("div", {class: "empty"}, "No road changes were found."))));
}


window.addEventListener("resize", relayout);
