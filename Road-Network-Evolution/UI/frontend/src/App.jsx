import { useEffect, useRef, useState } from "react";
import "./App.css";
import "./results.css";
import { renderAll, loadImages, relayout } from "./results.js";

// Same-origin by default (the FastAPI server serves this app). For `npm run dev`, vite proxies /api to port 8000.
const API = import.meta.env.VITE_API || "";

function App() {
  const [location, setLocation] = useState("");
  const [date1, setDate1] = useState("");
  const [date2, setDate2] = useState("");

  // Satellite image states
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);

  // Loading and error states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Waiting for imagery");

  // Analysis states
  const [page, setPage] = useState("dash");
  const [gsd, setGsd] = useState("0.5");
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [preview1, setPreview1] = useState("");
  const [preview2, setPreview2] = useState("");
  const [health, setHealth] = useState({ status: "loading", error: "" });
  const [running, setRunning] = useState(false);
  const [runMsg, setRunMsg] = useState("");
  const [runError, setRunError] = useState("");
  const [done, setDone] = useState(false);
  const resultRef = useRef(null);
  const [coords, setCoords] = useState(null);
  const statusRef = useRef(null);

  // Poll the server until the models are loaded
  useEffect(() => {
    let stop = false;
    let timer;
    const poll = async () => {
      try {
        const r = await (await fetch(`${API}/api/health`)).json();
        if (stop) return;
        setHealth({ status: r.status, error: r.error || "" });
        if (r.status === "ready" || r.status === "error") return;
      } catch {
        if (stop) return;
        setHealth({ status: "offline", error: "" });
      }
      timer = setTimeout(poll, 2000);
    };
    poll();
    return () => { stop = true; clearTimeout(timer); };
  }, []);

  // Re-measure the maps whenever the page changes (hidden pages have no width)
  useEffect(() => {
    requestAnimationFrame(() => { relayout(); window.scrollTo(0, 0); });
  }, [page]);

  const pickFile = (n, f) => {
    if (!f) return;
    if (!/\.(png|jpe?g|webp|bmp)$/i.test(f.name)) {
      setRunError("Please add a PNG or JPG image (export GeoTIFFs as PNG first).");
      return;
    }
    setRunError("");
    const url = URL.createObjectURL(f);
    if (n === 1) { setFile1(f); setPreview1(url); } else { setFile2(f); setPreview2(url); }
  };

  const runAnalysis = async () => {
    setRunError("");
    setRunning(true);
    const t0 = Date.now();
    setRunMsg("Running CRNet and LGMM-Net…");
    setTimeout(() => statusRef.current && statusRef.current.scrollIntoView({ block: "center", behavior: "smooth" }), 50);
    const timer = setInterval(() => setRunMsg(
      `Running CRNet and LGMM-Net… ${Math.round((Date.now() - t0) / 1000)} s (large images can take a few minutes)`), 500);
    try {
      const fd = new FormData();
      if (uploadOk) {
        fd.append("t1", file1);
        fd.append("t2", file2);
      } else {
        fd.append("latitude", String(coords.latitude));
        fd.append("longitude", String(coords.longitude));
      }
      fd.append("gsd", gsd || "0.5");
      fd.append("location", location);
      fd.append("t1_date", date1);
      fd.append("t2_date", date2);
      const r = await fetch(`${API}/api/analyse`, { method: "POST", body: fd });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Server error " + r.status);
      const imgs = await loadImages(j);
      resultRef.current = { data: j, imgs };
      setDone(true);
      renderAll(j, imgs, setPage);
      setPage("change");
    } catch (e) {
      setRunError(e.message || String(e));
    } finally {
      clearInterval(timer);
      setRunMsg("");
      setRunning(false);
    }
  };

  const pill = {
    ready: ["AI Pipeline Ready", ""],
    loading: ["Loading models…", " warn"],
    error: ["Models not loaded", " err"],
    offline: ["Server not reachable", " err"],
  }[health.status] || ["Loading models…", " warn"];
  const uploadOk = Boolean(file1 && file2);
  const fetchedOk = Boolean(image1?.analysable && image2?.analysable && coords && date1 && date2);
  const canRun = health.status === "ready" && (uploadOk || fetchedOk) && !running;

  // Fetch one satellite image from FastAPI
  const fetchSatelliteImage = async (latitude, longitude, date) => {
    const response = await fetch(
      `${API}/api/satellite/preview`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          latitude: latitude,
          longitude: longitude,
          date: date,
        }),
      }
    );

    if (!response.ok) {
      throw new Error("Backend request failed.");
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.message || "No satellite image found.");
    }

    return data.data;
  };

  // Fetch both T1 and T2 images
  const handleFetchImagery = async () => {
    setError("");
    setImage1(null);
    setImage2(null);

    // Check location
    if (!location.trim()) {
      setError("Please enter latitude and longitude.");
      return;
    }

    // Check dates
    if (!date1 || !date2) {
      setError("Please select both T1 and T2 dates.");
      return;
    }

    // Convert "13.0827, 80.2707" into numbers
    const coordinates = location.split(",");

    if (coordinates.length !== 2) {
      setError(
        "Enter location as latitude, longitude. Example: 13.0827, 80.2707"
      );
      return;
    }

    const latitude = Number(coordinates[0].trim());
    const longitude = Number(coordinates[1].trim());

    if (
      Number.isNaN(latitude) ||
      Number.isNaN(longitude) ||
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      setError(
        "Invalid coordinates. Example: 13.0827, 80.2707"
      );
      return;
    }

    // Start loading
    setLoading(true);
    setStatus("Fetching satellite imagery...");

    try {
      // Fetch T1
      const t1 = await fetchSatelliteImage(
        latitude,
        longitude,
        date1
      );

      setImage1(t1);

      // Fetch T2
      const t2 = await fetchSatelliteImage(
        latitude,
        longitude,
        date2
      );

      setImage2(t2);
      setCoords({ latitude, longitude });
      if (t1.analysable && t2.analysable) setGsd(String(t1.gsd));

      setStatus(
        t1.analysable && t2.analysable
          ? "High-resolution imagery loaded: ready to analyse"
          : "Preview loaded (too coarse to analyse): upload T1 and T2 images"
      );
    } catch (err) {
      console.error(err);

      setError(
        err.message || "Unable to fetch satellite imagery."
      );

      setStatus("Unable to load imagery");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      {/* Sidebar */}
      <aside className="sidebar">

        <div className="logo">
          <div className="logo-icon">R</div>

          <div>
            <h1>ROAD-EVO</h1>
            <span>Road Network Intelligence</span>
          </div>
        </div>


        <nav className="navigation">
          {[["dash", "⌂", "Dashboard"], ["change", "◉", "Change Analysis"], ["road", "⌁", "Road Network"], ["prio", "⚑", "Priority Review"]].map(([id, icon, label]) => (
            <div
              key={id}
              className={"nav-item" + (page === id || (id === "prio" && page === "report") ? " active" : "") + (id !== "dash" && !done ? " disabled" : "")}
              onClick={() => { if (id === "dash" || done) setPage(id); }}
            >
              <span>{icon}</span>
              {label}
            </div>
          ))}
        </nav>


        <div className="sidebar-bottom">

          <div className="system-status">

            <div className="status-dot" style={health.status === "ready" ? {} : { background: health.status === "loading" ? "#f5b324" : "#ef4444" }}></div>

            <div>
              <strong>{health.status === "ready" ? "System Online" : health.status === "loading" ? "Starting up" : "System Offline"}</strong>
              <small>{health.status === "ready" ? "AI analysis ready" : health.status === "loading" ? "Loading models…" : "Models not available"}</small>
            </div>

          </div>

          <div className="sidebar-footer">
            CRNet + LGMM-Net
          </div>

        </div>

      </aside>


      {/* Main content */}
      <main className="main">

        {/* Header */}
        <header className="topbar" style={page === "dash" ? undefined : { display: "none" }}>

          <div>

            <div className="breadcrumb">
              ROAD-EVO / Dashboard
            </div>

            <h2>Road Network Evolution</h2>

            <p>
              Analyze road changes using multi-temporal satellite imagery
            </p>

          </div>


          <div className="header-badge" title={health.error || ""}>
            <span style={health.status === "ready" ? {} : { background: pill[1] === " warn" ? "#f5b324" : "#ef4444" }}></span>
            {pill[0]}
          </div>

        </header>


        <div style={page === "dash" ? undefined : { display: "none" }}>
        {/* Study Area */}
        <section className="control-card">

          <div className="section-heading">

            <div>

              <span className="section-label">
                01 — STUDY AREA
              </span>

              <h3>Select Analysis Parameters</h3>

              <p>
                Define the location and two observation dates for
                bi-temporal satellite analysis.
              </p>

            </div>

          </div>


          <div className="form-grid">

            {/* Location */}
            <div className="input-group location-input">

              <label>Study Location</label>

              <div className="input-wrapper">

                <span className="input-icon">
                  ⌖
                </span>

                <input
                  type="text"
                  placeholder="13.0827, 80.2707"
                  value={location}
                  onChange={(e) =>
                    setLocation(e.target.value)
                  }
                />

              </div>

              <small>
                Enter latitude, longitude
              </small>

            </div>


            {/* T1 */}
            <div className="input-group">

              <label>
                Previous Observation — T1
              </label>

              <div className="input-wrapper">

                <span className="input-icon">
                  ◷
                </span>

                <input
                  type="date"
                  value={date1}
                  onChange={(e) =>
                    setDate1(e.target.value)
                  }
                />

              </div>

              <small>
                Earlier satellite observation
              </small>

            </div>


            {/* T2 */}
            <div className="input-group">

              <label>
                Current Observation — T2
              </label>

              <div className="input-wrapper">

                <span className="input-icon">
                  ◷
                </span>

                <input
                  type="date"
                  value={date2}
                  onChange={(e) =>
                    setDate2(e.target.value)
                  }
                />

              </div>

              <small>
                Recent satellite observation
              </small>

            </div>

          </div>


          <div className="form-grid" style={{ marginTop: 16, gridTemplateColumns: "1fr 2fr" }}>
            <div className="input-group">
              <label>Pixel size (m)</label>
              <div className="input-wrapper">
                <span className="input-icon">⌗</span>
                <input type="number" step="0.01" min="0.05" value={gsd} onChange={(e) => setGsd(e.target.value)} />
              </div>
              <small>Models were trained at ~0.5 m per pixel</small>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}


          <div className="action-row">

            <div className="source-info">

              <span className="source-dot"></span>

              Satellite source: Sentinel-2 / Google Earth Engine

            </div>


            <button
              className="fetch-button"
              onClick={handleFetchImagery}
              disabled={loading}
            >

              <span>
                {loading ? "◌" : "↗"}
              </span>

              {loading
                ? "Fetching Imagery..."
                : "Fetch Satellite Imagery"}

            </button>

          </div>

        </section>


        {/* Imagery */}
        <section className="imagery-section">

          <div className="section-title-row">

            <div>

              <span className="section-label">
                02 — SATELLITE IMAGERY
              </span>

              <h3>
                Bi-temporal Observation
              </h3>

            </div>


            <div className="imagery-status">
              {status}
            </div>

          </div>


          <div className="image-grid">

            {/* T1 */}
            <div className="image-card">

              <div className="image-header">

                <div>

                  <span className="time-badge">
                    T1
                  </span>

                  <strong>
                    Previous Image
                  </strong>

                </div>


                <span className="image-date">

                  {(image1?.actual_date || date1 || "Not selected") + (image1?.source ? " · " + image1.source : "")}

                </span>

              </div>


              <div className="satellite-placeholder">

                {preview1 ? (
                  <img src={preview1} alt="T1 image for analysis" className="satellite-image" />
                ) : image1 ? (
                  <img
                    src={image1.thumbnail_url}
                    alt="T1 Sentinel-2 satellite imagery"
                    className="satellite-image"
                  />

                ) : (

                  <>
                    <div className="placeholder-grid"></div>

                    <div className="placeholder-content">

                      <div className="satellite-icon">
                        ◉
                      </div>

                      <strong>
                        Satellite image will appear here
                      </strong>

                      <span>
                        Select location and date
                      </span>

                    </div>
                  </>

                )}

              </div>

              <label className="upload-row">
                <input type="file" accept="image/png,image/jpeg,image/webp,image/bmp" onChange={(e) => pickFile(1, e.target.files[0])} />
                <span className="upload-btn">⇪ Upload T1 image for analysis</span>
                <span className="upload-name">{file1 ? file1.name : image1?.analysable ? "Fetched " + image1.source + " will be analysed" : image1 ? "Preview only: upload a high-resolution image" : "PNG or JPG, same size as the other image"}</span>
              </label>

            </div>


            {/* T2 */}
            <div className="image-card">

              <div className="image-header">

                <div>

                  <span className="time-badge current">
                    T2
                  </span>

                  <strong>
                    Current Image
                  </strong>

                </div>


                <span className="image-date">

                  {(image2?.actual_date || date2 || "Not selected") + (image2?.source ? " · " + image2.source : "")}

                </span>

              </div>


              <div className="satellite-placeholder">

                {preview2 ? (
                  <img src={preview2} alt="T2 image for analysis" className="satellite-image" />
                ) : image2 ? (
                  <img
                    src={image2.thumbnail_url}
                    alt="T2 Sentinel-2 satellite imagery"
                    className="satellite-image"
                  />

                ) : (

                  <>
                    <div className="placeholder-grid"></div>

                    <div className="placeholder-content">

                      <div className="satellite-icon">
                        ◉
                      </div>

                      <strong>
                        Satellite image will appear here
                      </strong>

                      <span>
                        Select location and date
                      </span>

                    </div>
                  </>

                )}

              </div>

              <label className="upload-row">
                <input type="file" accept="image/png,image/jpeg,image/webp,image/bmp" onChange={(e) => pickFile(2, e.target.files[0])} />
                <span className="upload-btn">⇪ Upload T2 image for analysis</span>
                <span className="upload-name">{file2 ? file2.name : image2?.analysable ? "Fetched " + image2.source + " will be analysed" : image2 ? "Preview only: upload a high-resolution image" : "PNG or JPG, same size as the other image"}</span>
              </label>

            </div>

          </div>

        </section>


        {/* Analysis Pipeline */}
        <section className="pipeline-section">

          <div className="section-title-row">

            <div>

              <span className="section-label">
                03 — ANALYSIS PIPELINE
              </span>

              <h3>
                AI Road Change Detection
              </h3>

            </div>

          </div>


          <div className="pipeline">

            <div className="pipeline-step">

              <div className="pipeline-number">
                01
              </div>

              <div>
                <strong>CRNet</strong>
                <span>Road Extraction</span>
              </div>

            </div>


            <div className="pipeline-arrow">
              →
            </div>


            <div className="pipeline-step">

              <div className="pipeline-number">
                02
              </div>

              <div>
                <strong>Road Masks</strong>
                <span>T1 / T2 Networks</span>
              </div>

            </div>


            <div className="pipeline-arrow">
              →
            </div>


            <div className="pipeline-step">

              <div className="pipeline-number">
                03
              </div>

              <div>
                <strong>LGMM-Net</strong>
                <span>Change Detection</span>
              </div>

            </div>


            <div className="pipeline-arrow">
              →
            </div>


            <div className="pipeline-step">

              <div className="pipeline-number">
                04
              </div>

              <div>
                <strong>Review</strong>
                <span>Priority & Recommendation</span>
              </div>

            </div>

          </div>

        </section>


        <section className="run-section">
          <div className="action-row" style={{ marginTop: 0, borderTop: "none", paddingTop: 0 }}>
            <div className="source-info">
              <span className="source-dot"></span>
              Analysis needs high-resolution imagery (about 0.5 m per pixel). Upload the T1 and T2 images, then run.
            </div>
            <button className="fetch-button" onClick={runAnalysis} disabled={!canRun}>
              <span>{running ? "◌" : "▶"}</span>
              {running ? "Analysing…" : "Run Analysis"}
            </button>
          </div>
          {health.status === "error" && (
            <div className="error-message">Models could not be loaded: {health.error}</div>
          )}
          {health.status === "offline" && (
            <div className="error-message">The server is not reachable. Start it with: python -m uvicorn app:app --port 8000</div>
          )}
          {!running && !canRun && (
            <div className="run-hint">
              {health.status !== "ready"
                ? "Run Analysis is disabled until the models finish loading (see the badge at the top)."
                : "Run Analysis is disabled until you have two analysable images: either upload T1 and T2 with the buttons above, or fetch a location that has high-resolution (NAIP, 0.6 m) imagery. Sentinel-2 previews are too coarse to analyse."}
            </div>
          )}
          {runMsg && <div className="run-status" ref={statusRef}><span className="spin"></span>{runMsg}</div>}
          {runError && <div className="error-message">{runError}</div>}
        </section>

        {/* Bottom stats */}
        <section className="stats-grid">

          <div className="stat-card">

            <span>
              ROAD EXTRACTION
            </span>

            <strong>
              CRNet
            </strong>

            <small>
              Semantic-spatial segmentation
            </small>

          </div>


          <div className="stat-card">

            <span>
              CHANGE DETECTION
            </span>

            <strong>
              LGMM-Net
            </strong>

            <small>
              Local-global change modeling
            </small>

          </div>


          <div className="stat-card">

            <span>
              DATA SOURCE
            </span>

            <strong>
              Sentinel-2
            </strong>

            <small>
              Google Earth Engine
            </small>

          </div>


          <div className="stat-card">

            <span>
              ANALYSIS TYPE
            </span>

            <strong>
              Bi-temporal
            </strong>

            <small>
              T1 → T2 comparison
            </small>

          </div>

        </section>
        </div>

        {/* Result pages (rendered by results.js) */}
        <div className="res">
          {["change", "road", "prio", "report"].map((id) => (
            <section key={id} className={"page" + (page === id ? " show" : "")} id={"pg-" + id}></section>
          ))}
        </div>
      </main>

    </div>
  );
}

export default App;