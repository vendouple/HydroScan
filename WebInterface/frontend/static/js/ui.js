const form = document.getElementById("analyze-form");
const filesInput = document.getElementById("files");
const descriptionInput = document.getElementById("description");
const debugInput = document.getElementById("debug");
const latInput = document.getElementById("lat");
const lonInput = document.getElementById("lon");
const locateBtn = document.getElementById("locate-btn");
const locationStatus = document.getElementById("location-status");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");

const timelineEl = document.getElementById("timeline");
const analysisIdEl = document.getElementById("analysis-id");

const resultsCard = document.getElementById("results-card");
const potabilityScoreEl = document.getElementById("potability-score");
const bandLabelEl = document.getElementById("band-label");
const confidenceScoreEl = document.getElementById("confidence-score");
const confidenceBandEl = document.getElementById("confidence-band");
const resultTimestampEl = document.getElementById("result-timestamp");
const componentContainer = document.getElementById("component-breakdown");
const secondaryMetricsEl = document.getElementById("secondary-metrics");
const externalDataEl = document.getElementById("external-data");

const debugCard = document.getElementById("debug-card");
const debugLogEl = document.getElementById("debug-log");
const debugGalleryGrid = document.getElementById("debug-gallery-grid");
const debugGalleryEmpty = document.getElementById("debug-gallery-empty");
const downloadDebugLink = document.getElementById("download-debug");
const openResultLink = document.getElementById("open-result-json");
const userAnalysisCard = document.getElementById("user-analysis");
const userAnalysisConclusionEl = document.getElementById(
  "user-analysis-conclusion"
);
const userAnalysisScoreEl = document.getElementById("user-analysis-score");
const userAnalysisConfidenceEl = document.getElementById(
  "user-analysis-confidence"
);
const userAnalysisModelEl = document.getElementById("user-analysis-model");
const userAnalysisRationaleEl = document.getElementById(
  "user-analysis-rationale"
);

const body = document.body;

const TIMELINE_LABELS = {
  preparing_media: "Preparing media",
  scene_classifier: "Running scene classifier",
  applying_filters: "Applying filters",
  running_detector: "Running detector",
  aggregating_results: "Aggregating results",
  external_data: "Fetching external data",
  scoring: "Scoring",
  debug_detections: "Detection snapshots",
  finalizing: "Finalizing",
  error: "Analysis error",
  user_input_analysis: "User input analysis",
};

const STATUS_LABELS = {
  pending: "Pending",
  "in-progress": "In progress",
  done: "Completed",
  warning: "Warning",
  error: "Error",
};

const COMPONENT_LABELS = {
  external: "External water data",
  visual: "Visual clarity & turbidity",
  model_confidence: "Model detection confidence",
  color: "Color & appearance",
  user_text: "User description",
  temporal: "Temporal stability",
  corroboration: "Media corroboration",
  detector_confidence: "Detector confidence",
  image_quality: "Image/video quality",
  media_corroboration: "Media corroboration",
};

const CONFIDENCE_BANDS = {
  high: { threshold: 80, label: "High" },
  moderate: { threshold: 50, label: "Moderate" },
  low: { threshold: 0, label: "Low" },
};

const TIMELINE_FLOW = [
  {
    step: "preparing_media",
    hint: "Validating uploads",
  },
  {
    step: "scene_classifier",
    hint: "Checking indoor/outdoor context",
  },
  {
    step: "applying_filters",
    hint: "Generating enhanced variants",
  },
  {
    step: "running_detector",
    hint: "Locating water features",
  },
  {
    step: "aggregating_results",
    hint: "Merging per-frame findings",
  },
  {
    step: "external_data",
    hint: "Looking up nearby stations",
  },
  {
    step: "scoring",
    hint: "Computing potability score",
  },
  {
    step: "finalizing",
    hint: "Saving results",
  },
];

let liveTimelineTimer = null;
let liveTimelineIndex = 0;
let liveTimelineActive = false;

const createTimelineItemMarkup = (step, status = "pending", detail = "") => {
  const title = TIMELINE_LABELS[step] || titleCase(step || "step");
  const statusLabel = STATUS_LABELS[status] || titleCase(status || "pending");
  const detailText = detail || statusLabel;

  return `
    <li class="timeline-item ${status}" data-step="${step}">
      <div class="marker"></div>
      <div class="content">
        <p class="title">${title}</p>
        <p class="detail">${detailText}</p>
      </div>
    </li>
  `;
};

const ensureTimelineItem = (step, status = "pending", detail = "") => {
  if (!step) return null;
  let item = timelineEl.querySelector(`[data-step="${step}"]`);
  if (!item) {
    timelineEl.insertAdjacentHTML(
      "beforeend",
      createTimelineItemMarkup(step, status, detail)
    );
    item = timelineEl.querySelector(`[data-step="${step}"]`);
  }
  return item;
};

const setTimelineStatus = (step, status = "pending", detail = "") => {
  const item = ensureTimelineItem(step, status, detail);
  if (!item) return;

  item.className = `timeline-item ${status}`;

  const titleEl = item.querySelector(".title");
  if (titleEl) {
    titleEl.textContent = TIMELINE_LABELS[step] || titleCase(step || "step");
  }

  const detailEl = item.querySelector(".detail");
  if (detailEl) {
    detailEl.textContent =
      detail || STATUS_LABELS[status] || titleCase(status || "pending");
  }
};

const renderTimelineSkeleton = () => {
  if (!TIMELINE_FLOW.length) {
    clearTimeline();
    return;
  }

  timelineEl.innerHTML = TIMELINE_FLOW.map((entry, index) => {
    const status = index === 0 ? "in-progress" : "pending";
    const baseDetail = entry.hint || STATUS_LABELS[status];
    return createTimelineItemMarkup(entry.step, status, baseDetail);
  }).join("");
};

const formatNumber = (value, digits = 1) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "--";

const titleCase = (value) =>
  value
    .split(/[_\-\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const toggleAnalysisMode = (active) => {
  body.classList.toggle("analysis-mode", Boolean(active));
};

const setFormEnabled = (enabled) => {
  analyzeBtn.disabled = !enabled;
  resetBtn.disabled = !enabled;
  filesInput.disabled = !enabled;
  descriptionInput.disabled = !enabled;
  latInput.disabled = !enabled;
  lonInput.disabled = !enabled;
  debugInput.disabled = !enabled;
  if (enabled) {
    locateBtn.disabled = false;
  } else {
    locateBtn.disabled = true;
  }
};

const clearTimeline = () => {
  timelineEl.innerHTML = `
    <li class="timeline-item idle">
      <div class="marker"></div>
      <div class="content">
        <p class="title">Waiting for upload</p>
        <p class="detail">Start an analysis to view processing stages.</p>
      </div>
    </li>
  `;
};

const renderTimeline = (entries = []) => {
  const isErrorOnly = entries.length === 1 && entries[0]?.status === "error";

  if (isErrorOnly) {
    timelineEl.innerHTML = createTimelineItemMarkup(
      entries[0].step || "error",
      "error",
      entries[0].detail || STATUS_LABELS.error
    );
    return;
  }

  if (!entries.length) {
    clearTimeline();
    return;
  }

  const flowSteps = new Set(TIMELINE_FLOW.map((entry) => entry.step));
  const entriesByStep = new Map(
    entries.map((entry) => [entry.step || "", entry])
  );

  const flowMarkup = TIMELINE_FLOW.map((entry) => {
    const matching = entriesByStep.get(entry.step) || {};
    const status = matching.status || "done";
    const detail =
      matching.detail ||
      matching.hint ||
      entry.hint ||
      STATUS_LABELS[status] ||
      titleCase(status);
    return createTimelineItemMarkup(entry.step, status, detail);
  });

  const extraEntries = entries.filter(
    (entry) => entry.step && !flowSteps.has(entry.step)
  );
  const extraMarkup = extraEntries.map((entry) =>
    createTimelineItemMarkup(
      entry.step,
      entry.status || "done",
      entry.detail ||
        STATUS_LABELS[entry.status || "done"] ||
        titleCase(entry.status || "done")
    )
  );

  timelineEl.innerHTML = [...flowMarkup, ...extraMarkup].join("");
};

const advanceLiveTimeline = () => {
  if (!liveTimelineActive || !TIMELINE_FLOW.length) return;

  const currentEntry = TIMELINE_FLOW[liveTimelineIndex];
  const hasNext = liveTimelineIndex < TIMELINE_FLOW.length - 1;

  if (currentEntry && hasNext) {
    setTimelineStatus(
      currentEntry.step,
      "done",
      STATUS_LABELS.done || "Completed"
    );
  }

  if (hasNext) {
    liveTimelineIndex += 1;
    const nextEntry = TIMELINE_FLOW[liveTimelineIndex];
    if (nextEntry) {
      setTimelineStatus(nextEntry.step, "in-progress", nextEntry.hint);
    }
  } else if (currentEntry) {
    setTimelineStatus(
      currentEntry.step,
      "in-progress",
      currentEntry.hint || STATUS_LABELS["in-progress"]
    );
    if (liveTimelineTimer) {
      window.clearInterval(liveTimelineTimer);
      liveTimelineTimer = null;
    }
  }
};

const startLiveTimeline = () => {
  stopLiveTimeline(false);

  if (!TIMELINE_FLOW.length) {
    clearTimeline();
    return;
  }

  liveTimelineActive = true;
  liveTimelineIndex = 0;
  renderTimelineSkeleton();

  const firstEntry = TIMELINE_FLOW[0];
  if (firstEntry) {
    setTimelineStatus(firstEntry.step, "in-progress", firstEntry.hint);
  }

  liveTimelineTimer = window.setInterval(advanceLiveTimeline, 2800);
};

const stopLiveTimeline = (completed = true) => {
  if (liveTimelineTimer) {
    window.clearInterval(liveTimelineTimer);
    liveTimelineTimer = null;
  }

  if (!liveTimelineActive) {
    return;
  }

  if (completed) {
    TIMELINE_FLOW.forEach((entry) =>
      setTimelineStatus(entry.step, "done", STATUS_LABELS.done)
    );
  } else {
    const currentEntry =
      TIMELINE_FLOW[Math.min(liveTimelineIndex, TIMELINE_FLOW.length - 1)];
    if (currentEntry) {
      setTimelineStatus(
        currentEntry.step,
        "warning",
        "Stopped before completion"
      );
    }
  }

  liveTimelineActive = false;
};

const renderComponents = (scores) => {
  componentContainer.innerHTML = "";
  if (!scores) return;

  const { potability, confidence } = scores.components || {};
  const createSection = (title, components) => {
    if (!components) return "";

    const items = Object.entries(components)
      .map(([key, info]) => {
        const friendly = COMPONENT_LABELS[key] || titleCase(key);
        const value = info?.value ?? 0;
        const weight = info?.weight ?? 0;
        const width = Math.min(100, Math.max(0, Number(value)));

        return `
          <div class="component">
            <h4>
              <span>${friendly}</span>
              <span class="muted">${formatNumber(
                value
              )} / weight ${formatNumber(weight, 0)}%</span>
            </h4>
            <div class="bar">
              <div class="bar-fill" style="width: ${width}%"></div>
            </div>
          </div>
        `;
      })
      .join("");

    return `
      <p class="component-title">${title}</p>
      ${items}
    `;
  };

  componentContainer.innerHTML = `
    ${createSection("Potability contributors", potability)}
    ${createSection("Confidence contributors", confidence)}
  `;
};

const renderSecondary = (result) => {
  const secondaryCards = [];

  if (result.scene) {
    const { majority, confidence } = result.scene;
    secondaryCards.push(`
      <div class="secondary-card">
        <h4>Scene classification</h4>
        <p><strong>${titleCase(majority || "unknown")}</strong> (${formatNumber(
      (confidence || 0) * 100
    )}% mean confidence)</p>
      </div>
    `);
  }

  if (result.aggregation) {
    const top = result.aggregation.top_detection;
    const classCount = result.aggregation.class_counts || {};
    const total = Object.values(classCount).reduce(
      (sum, val) => sum + Number(val || 0),
      0
    );
    const summary = Object.entries(classCount)
      .map(([cls, count]) => `${titleCase(cls)} (${count})`)
      .join(", ");

    secondaryCards.push(`
      <div class="secondary-card">
        <h4>Detections overview</h4>
        <p>Total detections: <strong>${total}</strong></p>
        ${
          top
            ? `<p>Top detection: <strong>${titleCase(
                top.class_name
              )}</strong> (${formatNumber(top.score * 100)}%)</p>`
            : ""
        }
        ${summary ? `<p class="muted">${summary}</p>` : ""}
      </div>
    `);
  }

  if (result.media) {
    const { frame_count, variant_count, saved_files = [] } = result.media;
    secondaryCards.push(`
      <div class="secondary-card">
        <h4>Media summary</h4>
        <p>Frames analysed: <strong>${frame_count || 0}</strong></p>
        <p>Variants generated: <strong>${variant_count || 0}</strong></p>
        <p class="muted">Files uploaded: ${saved_files.length}</p>
      </div>
    `);
  }

  secondaryMetricsEl.innerHTML = secondaryCards.join("") || "";
};

const renderExternalData = (external) => {
  if (!external || !Object.keys(external).length) {
    externalDataEl.innerHTML = "";
    externalDataEl.classList.add("hidden");
    return;
  }

  const stationId = external.station_id || "Unknown station";
  const distance = external.distance_km
    ? `${formatNumber(external.distance_km, 2)} km away`
    : "Distance unknown";
  const quality = external.overall_quality
    ? `${formatNumber(external.overall_quality)} quality`
    : null;
  const sampledAt = external.sample_date || external.last_updated;

  const parameters = external.parameters
    ? Object.entries(external.parameters)
        .map(([name, info]) => {
          const value = info?.value;
          const status = info?.status ? ` (${info.status})` : "";
          return `<li><strong>${titleCase(name)}</strong>: ${formatNumber(
            value
          )}${status}</li>`;
        })
        .join("")
    : null;

  externalDataEl.innerHTML = `
    <p><strong>${stationId}</strong></p>
    <p>${distance}${quality ? ` · ${quality}` : ""}</p>
    ${sampledAt ? `<p>Sampled: ${sampledAt}</p>` : ""}
    ${parameters ? `<ul>${parameters}</ul>` : ""}
  `;
  externalDataEl.classList.remove("hidden");
};

const renderUserAnalysis = (analysis) => {
  if (!userAnalysisCard) return;

  if (!analysis) {
    userAnalysisCard.classList.add("hidden");
    userAnalysisCard.classList.remove("unavailable");
    if (userAnalysisConfidenceEl) {
      userAnalysisConfidenceEl.classList.remove("muted");
    }
    if (userAnalysisModelEl) {
      userAnalysisModelEl.classList.add("hidden");
      userAnalysisModelEl.textContent = "";
    }
    if (userAnalysisRationaleEl) {
      userAnalysisRationaleEl.classList.add("hidden");
      userAnalysisRationaleEl.textContent = "";
    }
    return;
  }

  const available = Boolean(analysis.available);
  const conclusion = analysis.conclusion?.trim();
  const rationale = analysis.rationale?.trim();
  const reason = analysis.reason?.trim();
  const modelName = analysis.model_name?.trim();

  userAnalysisConclusionEl.textContent =
    conclusion ||
    (available ? "No conclusion generated." : "User input model unavailable.");

  if (available) {
    userAnalysisScoreEl.textContent = `${formatNumber(
      Number(analysis.score) || 0,
      0
    )} / 100`;
    userAnalysisConfidenceEl.textContent = `${formatNumber(
      Number(analysis.confidence) || 0,
      0
    )}% confidence`;
    userAnalysisConfidenceEl.classList.remove("muted");
  } else {
    userAnalysisScoreEl.textContent = "Not applied";
    userAnalysisConfidenceEl.textContent = reason || "Model unavailable";
    userAnalysisConfidenceEl.classList.add("muted");
  }

  if (modelName) {
    userAnalysisModelEl.textContent = `Model: ${modelName}`;
    userAnalysisModelEl.classList.remove("hidden");
  } else {
    userAnalysisModelEl.textContent = "";
    userAnalysisModelEl.classList.add("hidden");
  }

  if (available && rationale) {
    userAnalysisRationaleEl.textContent = rationale;
    userAnalysisRationaleEl.classList.remove("hidden");
  } else if (!available && reason) {
    userAnalysisRationaleEl.textContent = reason;
    userAnalysisRationaleEl.classList.remove("hidden");
  } else {
    userAnalysisRationaleEl.textContent = "";
    userAnalysisRationaleEl.classList.add("hidden");
  }

  userAnalysisCard.classList.toggle("unavailable", !available);
  userAnalysisCard.classList.remove("hidden");
};

const determineConfidenceBand = (score) => {
  if (!Number.isFinite(score)) return { label: "--", key: "" };
  if (score >= CONFIDENCE_BANDS.high.threshold)
    return { label: CONFIDENCE_BANDS.high.label, key: "high" };
  if (score >= CONFIDENCE_BANDS.moderate.threshold)
    return { label: CONFIDENCE_BANDS.moderate.label, key: "moderate" };
  return { label: CONFIDENCE_BANDS.low.label, key: "low" };
};

const renderResults = (result) => {
  if (!result?.scores) {
    resultsCard.classList.add("hidden");
    renderUserAnalysis(null);
    return;
  }

  const scores = result.scores;
  const potScore = Number(scores.potability_score ?? scores.potability);
  const confScore = Number(scores.confidence_score ?? scores.confidence);

  potabilityScoreEl.textContent = formatNumber(potScore);
  bandLabelEl.textContent = `Band: ${scores.band_label || "--"}`;

  confidenceScoreEl.textContent = formatNumber(confScore);
  const confBand = determineConfidenceBand(confScore);
  confidenceBandEl.textContent = confBand.label;
  confidenceBandEl.className = `confidence-pill ${confBand.key}`.trim();

  if (result.timestamp) {
    resultTimestampEl.textContent = new Date(result.timestamp).toLocaleString();
  } else {
    resultTimestampEl.textContent = "";
  }

  renderComponents(scores);
  renderSecondary(result);
  renderExternalData(result.external_data);
  renderUserAnalysis(result.user_analysis);

  resultsCard.classList.remove("hidden");
};

const buildDebugLog = (result, timelineEntries) => {
  const lines = [];
  lines.push(`Analysis ID: ${result.analysis_id || "unknown"}`);
  if (result.timestamp) {
    lines.push(`Completed: ${result.timestamp}`);
  }
  lines.push("--- Timeline ---");
  timelineEntries.forEach((entry) => {
    const status = entry.status || "in-progress";
    const label =
      TIMELINE_LABELS[entry.step] || titleCase(entry.step || "step");
    lines.push(`[${status}] ${label}: ${entry.detail || ""}`);
  });

  if (result.aggregation?.top_detection) {
    const top = result.aggregation.top_detection;
    lines.push("--- Top detection ---");
    lines.push(
      `${titleCase(top.class_name)} · ${formatNumber(
        top.score * 100
      )}% confidence`
    );
  }

  if (result.external_data) {
    lines.push("--- External data ---");
    lines.push(
      `Station ${
        result.external_data.station_id || "n/a"
      }, overall quality ${formatNumber(result.external_data.overall_quality)}`
    );
  }

  if (result.media) {
    lines.push("--- Media ---");
    lines.push(
      `Frames: ${result.media.frame_count || 0}, Variants: ${
        result.media.variant_count || 0
      }`
    );
  }

  return lines.join("\n");
};

const renderDebugGallery = (images = [], analysisId) => {
  if (!images.length) {
    debugGalleryGrid.innerHTML = "";
    debugGalleryEmpty.classList.remove("hidden");
    debugGalleryEmpty.textContent = "No detection snapshots generated yet.";
    return;
  }

  debugGalleryEmpty.classList.add("hidden");
  debugGalleryGrid.innerHTML = images
    .map((img) => {
      const label = `${titleCase(img.variant || "variant")} · frame ${
        Number(img.frame_index) + 1 || 1
      }`;
      const detections = img.detections || 0;
      const url =
        img.url || `/api/results/${analysisId}/artifacts/${img.relative_path}`;

      return `
        <figure class="gallery-card">
          <a href="${url}" target="_blank" rel="noopener">
            <img src="${url}" alt="Detection snapshot" loading="lazy" />
          </a>
          <figcaption class="caption">
            <strong>${label}</strong>
            <span>${detections} detections</span>
            <a href="${url}" target="_blank" rel="noopener">Open full size</a>
          </figcaption>
        </figure>
      `;
    })
    .join("");
};

const renderDebug = (result, analysisId, debugImages = []) => {
  const timelineEntries = result.timeline || [];
  debugLogEl.textContent = buildDebugLog(result, timelineEntries);

  renderDebugGallery(debugImages, analysisId);

  if (downloadDebugLink) {
    downloadDebugLink.href = `/api/results/${analysisId}/artifacts/debug.json`;
  }
  if (openResultLink) {
    openResultLink.href = `/api/results/${analysisId}?include_debug=true`;
  }

  debugCard.classList.remove("hidden");
};

const resetDebug = () => {
  debugLogEl.textContent = "Debug not enabled.";
  debugGalleryGrid.innerHTML = "";
  debugGalleryEmpty.classList.remove("hidden");
  debugGalleryEmpty.textContent = "No detection snapshots generated yet.";
  if (downloadDebugLink) {
    downloadDebugLink.href = "#";
  }
  if (openResultLink) {
    openResultLink.href = "#";
  }
  debugCard.classList.add("hidden");
};

const fetchResult = async (analysisId, includeDebug) => {
  const response = await fetch(
    `/api/results/${analysisId}?include_debug=${
      includeDebug ? "true" : "false"
    }`
  );
  if (!response.ok) {
    throw new Error(`Unable to retrieve results (${response.status})`);
  }
  return response.json();
};

const handleSubmit = async (event) => {
  event.preventDefault();
  if (!filesInput.files?.length) {
    filesInput.focus();
    return;
  }

  const debugEnabled = debugInput.checked;
  setFormEnabled(false);
  toggleAnalysisMode(true);
  startLiveTimeline();
  resetDebug();
  analysisIdEl.textContent = "Processing...";

  const fd = new FormData();
  Array.from(filesInput.files).forEach((file) => fd.append("files", file));
  if (descriptionInput.value) fd.append("description", descriptionInput.value);
  if (latInput.value) fd.append("lat", latInput.value);
  if (lonInput.value) fd.append("lon", lonInput.value);
  fd.append("debug", debugEnabled ? "true" : "false");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: fd,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.detail || "Analysis failed");
    }

    const analysisId = payload.analysis_id;
    analysisIdEl.textContent = analysisId;

    let resultData;
    try {
      resultData = await fetchResult(analysisId, debugEnabled);
    } catch (err) {
      console.warn(
        "Unable to fetch stored result; falling back to response",
        err
      );
      resultData = {
        analysis_id: analysisId,
        scores: payload.scores,
        scene: payload.scene,
        external_data: payload.external_data,
        timeline: payload.timeline || [],
      };
    }

    stopLiveTimeline(true);
    renderTimeline(resultData.timeline || payload.timeline || []);
    renderResults(resultData);

    if (debugEnabled) {
      const debugImagesSource =
        resultData?.debug?.detection_images ||
        payload.debug_images ||
        (resultData.debug_artifacts?.detection_images ?? []);
      renderDebug(resultData, analysisId, debugImagesSource || []);
    } else {
      resetDebug();
    }
  } catch (error) {
    console.error(error);
    analysisIdEl.textContent = "Analysis failed";
    stopLiveTimeline(false);
    renderTimeline([
      {
        step: "error",
        status: "error",
        detail: error?.message || "An unexpected error occurred",
      },
    ]);
    resultsCard.classList.add("hidden");
    resetDebug();
  } finally {
    setFormEnabled(true);
    locateBtn.disabled = false;
  }
};

const handleReset = () => {
  stopLiveTimeline(false);
  analysisIdEl.textContent = "No analysis yet";
  resultsCard.classList.add("hidden");
  toggleAnalysisMode(false);
  clearTimeline();
  resetDebug();
  renderUserAnalysis(null);
  locationStatus.textContent = "Location not set";
};

const handleLocate = () => {
  if (!navigator.geolocation) {
    locationStatus.textContent = "Geolocation not supported";
    return;
  }

  locationStatus.textContent = "Locating...";
  locateBtn.disabled = true;

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const { latitude, longitude } = pos.coords;
      latInput.value = latitude.toFixed(6);
      lonInput.value = longitude.toFixed(6);
      locationStatus.textContent = `Lat ${latitude.toFixed(
        4
      )}, Lon ${longitude.toFixed(4)}`;
      locateBtn.disabled = false;
    },
    (err) => {
      locationStatus.textContent = err.message || "Location unavailable";
      locateBtn.disabled = false;
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 600000 }
  );
};

form?.addEventListener("submit", handleSubmit);
form?.addEventListener("reset", handleReset);
locateBtn?.addEventListener("click", handleLocate);

// Initialize default state
clearTimeline();
resetDebug();
