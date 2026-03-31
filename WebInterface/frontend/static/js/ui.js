const form = document.getElementById("analyze-form");
const filesInput = document.getElementById("files");
const descriptionInput = document.getElementById("description");
const debugInput = document.getElementById("debug");
const dontSaveInput = document.getElementById("dont-save");
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
const debugSnapshotsEl = document.getElementById("debug-snapshots");
const debugSnapshotsEmptyEl = document.getElementById("debug-snapshots-empty");
const snapshotGroupsEl = document.getElementById("snapshot-groups");
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

const historyCard = document.getElementById("history-card");
const historyEmptyEl = document.getElementById("history-empty");
const historyListEl = document.getElementById("history-list");
const refreshHistoryBtn = document.getElementById("refresh-history");

const body = document.body;

const TIMELINE_LABELS = {
  preparing_media: "Media ingestion",
  scene_detection: "Scene detection (Places365)",
  scene_branch: "Scene routing",
  filter_strategy: "Adaptive filter stack",
  detector_inference: "Detection engines",
  aggregation: "Aggregating results",
  outdoor_external: "Outdoor external data",
  outdoor_detector: "Outdoor detector selection",
  packaging_scan: "Packaging detection",
  ocr_scan: "Brand & OCR analysis",
  water_confirmation: "Water confirmation",
  external_data: "External data",
  user_input_analysis: "User input analysis",
  scoring: "Scoring",
  debug_detections: "Detection snapshots",
  finalizing: "Finalizing",
  error: "Analysis error",
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
    hint: "Validating uploads & deduplicating frames",
  },
  { step: "scene_detection", hint: "Identifying indoor/outdoor context" },
  { step: "scene_branch", hint: "Selecting outdoor or indoor workflow" },
  { step: "filter_strategy", hint: "Selecting adaptive filters" },
  { step: "detector_inference", hint: "Running detectors" },
  { step: "aggregation", hint: "Merging detections & metrics" },
  {
    step: "outdoor_external",
    hint: "Fetching outdoor external data",
    optional: true,
  },
  {
    step: "outdoor_detector",
    hint: "Selecting outdoor detector",
    optional: true,
  },
  {
    step: "packaging_scan",
    hint: "Scanning for packaging",
    optional: true,
  },
  { step: "ocr_scan", hint: "Analyzing branding", optional: true },
  { step: "water_confirmation", hint: "Confirming water presence" },
  {
    step: "user_input_analysis",
    hint: "Interpreting user notes",
    optional: true,
  },
  { step: "scoring", hint: "Computing potability score" },
  { step: "finalizing", hint: "Saving results" },
];

const LIVE_TIMELINE_FLOW = TIMELINE_FLOW.filter((entry) => !entry.optional);

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
  if (!LIVE_TIMELINE_FLOW.length) {
    clearTimeline();
    return;
  }

  timelineEl.innerHTML = LIVE_TIMELINE_FLOW.map((entry, index) => {
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

  const flowMarkup = TIMELINE_FLOW.filter(
    (entry) => !entry.optional || entriesByStep.has(entry.step)
  ).map((entry) => {
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
  if (!liveTimelineActive || !LIVE_TIMELINE_FLOW.length) return;

  const currentEntry = LIVE_TIMELINE_FLOW[liveTimelineIndex];
  const hasNext = liveTimelineIndex < LIVE_TIMELINE_FLOW.length - 1;

  if (currentEntry && hasNext) {
    setTimelineStatus(
      currentEntry.step,
      "done",
      STATUS_LABELS.done || "Completed"
    );
  }

  if (hasNext) {
    liveTimelineIndex += 1;
    const nextEntry = LIVE_TIMELINE_FLOW[liveTimelineIndex];
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

  if (!LIVE_TIMELINE_FLOW.length) {
    clearTimeline();
    return;
  }

  liveTimelineActive = true;
  liveTimelineIndex = 0;
  renderTimelineSkeleton();

  const firstEntry = LIVE_TIMELINE_FLOW[0];
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
    LIVE_TIMELINE_FLOW.forEach((entry) =>
      setTimelineStatus(entry.step, "done", STATUS_LABELS.done)
    );
  } else {
    const currentEntry =
      LIVE_TIMELINE_FLOW[
        Math.min(liveTimelineIndex, LIVE_TIMELINE_FLOW.length - 1)
      ];
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

const renderSnapshotGroups = (snapshots = {}, analysisId) => {
  if (!snapshotGroupsEl || !debugSnapshotsEmptyEl) return;

  const categories = Object.entries(snapshots).filter(
    ([, entries]) => Array.isArray(entries) && entries.length
  );

  if (!categories.length) {
    snapshotGroupsEl.innerHTML = "";
    debugSnapshotsEmptyEl.classList.remove("hidden");
    debugSnapshotsEmptyEl.textContent = "No detection snapshots generated yet.";
    return;
  }

  debugSnapshotsEmptyEl.classList.add("hidden");
  snapshotGroupsEl.innerHTML = categories
    .map(([category, entries]) => {
      const friendly = titleCase(category || "snapshots");
      const cards = entries
        .map((entry, index) => {
          const frameLabel =
            entry.frame_index !== undefined && entry.frame_index !== null
              ? `Frame ${Number(entry.frame_index) + 1}`
              : null;
          const variantLabel = entry.variant
            ? titleCase(entry.variant)
            : entry.label
            ? titleCase(entry.label)
            : null;
          const detectionCount =
            entry.detections !== undefined && entry.detections !== null
              ? `${entry.detections} detections`
              : null;
          const detailParts = [frameLabel, variantLabel, detectionCount].filter(
            Boolean
          );
          const labelText =
            entry.label || variantLabel || `Snapshot ${index + 1}`;
          const url =
            entry.url ||
            (analysisId && entry.relative_path
              ? `/api/results/${analysisId}/artifacts/${entry.relative_path}`
              : null);

          return `
            <figure class="snapshot-card">
              ${
                url
                  ? `<a href="${url}" target="_blank" rel="noopener">
                      <img src="${url}" alt="${friendly} snapshot" loading="lazy" />
                    </a>`
                  : ""
              }
              <div class="snapshot-meta">
                <strong>${titleCase(labelText)}</strong>
                ${
                  detailParts.length
                    ? `<span>${detailParts.join(" · ")}</span>`
                    : ""
                }
                ${
                  url
                    ? `<a href="${url}" target="_blank" rel="noopener">Open full size</a>`
                    : ""
                }
              </div>
            </figure>
          `;
        })
        .join("");

      const openAttr = category === "detector" ? "open" : "";

      return `
        <details class="snapshot-group" ${openAttr}>
          <summary>
            <span>${friendly}</span>
            <span class="badge">${entries.length}</span>
          </summary>
          <div class="snapshot-body">
            ${cards}
          </div>
        </details>
      `;
    })
    .join("");
};

const renderDebug = (result, analysisId, debugImages = []) => {
  const timelineEntries = result.timeline || [];
  debugLogEl.textContent = buildDebugLog(result, timelineEntries);

  const debugSection = result.debug || {};
  const snapshotSource = debugSection.snapshots || {};
  const mergedSnapshots = { ...snapshotSource };

  const legacyImages =
    debugImages.length > 0 ? debugImages : debugSection.detection_images || [];
  if (legacyImages.length) {
    mergedSnapshots.detector = [
      ...(mergedSnapshots.detector || []),
      ...legacyImages,
    ];
  }

  renderSnapshotGroups(mergedSnapshots, analysisId);

  const historySaved = result.history_saved !== false;
  if (downloadDebugLink) {
    if (historySaved) {
      downloadDebugLink.href = `/api/results/${analysisId}/artifacts/debug.json`;
      downloadDebugLink.removeAttribute("aria-disabled");
    } else {
      downloadDebugLink.href = "#";
      downloadDebugLink.setAttribute("aria-disabled", "true");
    }
  }
  if (openResultLink) {
    if (historySaved) {
      openResultLink.href = `/api/results/${analysisId}?include_debug=true`;
      openResultLink.removeAttribute("aria-disabled");
    } else {
      openResultLink.href = "#";
      openResultLink.setAttribute("aria-disabled", "true");
    }
  }

  debugCard.classList.remove("hidden");
};

const resetDebug = () => {
  debugLogEl.textContent = "Debug not enabled.";
  if (snapshotGroupsEl) {
    snapshotGroupsEl.innerHTML = "";
  }
  if (debugSnapshotsEmptyEl) {
    debugSnapshotsEmptyEl.classList.remove("hidden");
    debugSnapshotsEmptyEl.textContent = "No detection snapshots generated yet.";
  }
  if (downloadDebugLink) {
    downloadDebugLink.href = "#";
    downloadDebugLink.setAttribute("aria-disabled", "true");
  }
  if (openResultLink) {
    openResultLink.href = "#";
    openResultLink.setAttribute("aria-disabled", "true");
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
  const akinatorEnabled = document.getElementById("enable-akinator")?.checked;
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

    // Start Akinator session with visual context
    if (akinatorController && akinatorEnabled) {
      if (!akinatorController.isActive) {
        akinatorController.show();
      }
      const visualContext = {
        scores: resultData.scores || payload.scores,
        scene: resultData.scene || payload.scene,
        components: resultData.components || payload.components,
        potability: resultData.potability || payload.potability,
      };
      akinatorController.startSession(visualContext, analysisId);
    } else if (akinatorController) {
      akinatorController.hide();
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
  locationStatus.textContent = "Location features coming soon";
  
  // Reset Akinator controller
  if (akinatorController) {
    akinatorController.reset();
  }
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

// History management
const loadHistory = async () => {
  if (!historyListEl) {
    return;
  }

  try {
    const response = await fetch("/api/history");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    renderHistory(data.analyses || []);
  } catch (error) {
    console.error("Failed to load history:", error);
    if (historyEmptyEl) {
      historyEmptyEl.textContent = "Failed to load history. Please try again.";
      historyEmptyEl.classList.remove("hidden");
    }
    historyListEl.classList.add("hidden");
  }
};

const renderHistory = (analyses) => {
  if (!historyListEl) {
    return;
  }

  if (!analyses || analyses.length === 0) {
    if (historyEmptyEl) {
      historyEmptyEl.classList.remove("hidden");
    }
    historyListEl.classList.add("hidden");
    return;
  }

  if (historyEmptyEl) {
    historyEmptyEl.classList.add("hidden");
  }
  historyListEl.classList.remove("hidden");

  historyListEl.innerHTML = "";

  analyses.forEach((analysis) => {
    const li = document.createElement("li");
    li.className = "history-item";

    const date = new Date(analysis.timestamp_parsed);
    const dateStr = date.toLocaleDateString();
    const timeStr = date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    // Get band color class
    const bandClass = getBandColorClass(analysis.potability_score);

    li.innerHTML = `
      <div class="history-item-header">
        <div class="history-item-title">
          <span class="history-item-date">${dateStr} ${timeStr}</span>
          <span class="history-item-scene">${analysis.scene}</span>
        </div>
        <div class="history-item-actions">
          <button type="button" class="btn-icon" onclick="loadAnalysis('${
            analysis.analysis_id
          }')" title="Load analysis">
            📄
          </button>
          <button type="button" class="btn-icon" onclick="deleteAnalysis('${
            analysis.analysis_id
          }')" title="Delete analysis">
            🗑️
          </button>
        </div>
      </div>
      <div class="history-item-details">
        <div class="history-item-score ${bandClass}">
          ${analysis.potability_score}% ${analysis.band_label}
        </div>
        <div class="history-item-meta">
          <span>Confidence: ${analysis.confidence_score}%</span>
          <span>${analysis.media_count} media file${
      analysis.media_count !== 1 ? "s" : ""
    }</span>
          ${
            analysis.debug_available
              ? '<span class="debug-badge">Debug</span>'
              : ""
          }
        </div>
        ${
          analysis.description
            ? `<div class="history-item-description">${analysis.description.substring(
                0,
                100
              )}${analysis.description.length > 100 ? "..." : ""}</div>`
            : ""
        }
      </div>
    `;

    historyListEl.appendChild(li);
  });
};

const getBandColorClass = (score) => {
  if (score >= 100) return "score-drinkable";
  if (score >= 51) return "score-very-clean";
  if (score >= 50) return "score-clean";
  if (score >= 26) return "score-less-clean";
  return "score-unclean";
};

const loadAnalysis = async (analysisId) => {
  try {
    const response = await fetch(
      `/api/results/${analysisId}?include_debug=true`
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();

    // Reset the form and clear current analysis
    handleReset();

    // Render the loaded results
    renderResults(result);

    // Show results card
    resultsCard.classList.remove("hidden");

    // If debug data is available, show it
    if (result.debug) {
      renderDebug(result, analysisId, result.debug.detection_images || []);
    }

    // Update analysis ID display
    analysisIdEl.textContent = `Loaded: ${analysisId}`;
  } catch (error) {
    console.error("Failed to load analysis:", error);
    alert("Failed to load analysis. Please try again.");
  }
};

const deleteAnalysis = async (analysisId) => {
  if (
    !confirm(
      "Are you sure you want to delete this analysis? This action cannot be undone."
    )
  ) {
    return;
  }

  try {
    const response = await fetch(`/api/history/${analysisId}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    // Refresh the history list
    await loadHistory();
  } catch (error) {
    console.error("Failed to delete analysis:", error);
    alert("Failed to delete analysis. Please try again.");
  }
};

const refreshHistory = () => {
  loadHistory();
};

form?.addEventListener("submit", handleSubmit);
form?.addEventListener("reset", handleReset);
locateBtn?.addEventListener("click", handleLocate);
refreshHistoryBtn?.addEventListener("click", refreshHistory);

// ============================================
// AKINATOR CONTROLLER CLASS
// ============================================

class AkinatorController {
  constructor() {
    this.sessionId = null;
    this.currentQuestion = null;
    this.roundNumber = 0;
    this.maxRounds = 10;
    this.isActive = false;
    this.analysisId = null;
    this.visualContext = null;
    this.messageHistory = [];
    
    // DOM elements
    this.chatEl = document.getElementById("akinator-chat");
    this.bodyEl = document.getElementById("akinator-body");
    this.headerEl = document.getElementById("akinator-header");
    this.toggleBtn = document.getElementById("akinator-toggle-btn");
    this.floatBtn = document.getElementById("akinator-float-btn");
    this.statusEl = document.getElementById("akinator-status");
    
    this.init();
  }
  
  init() {
    // Header click to expand/collapse
    this.headerEl?.addEventListener("click", (e) => {
      if (e.target !== this.toggleBtn) {
        this.toggleCollapse();
      }
    });
    
    // Toggle button
    this.toggleBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleCollapse();
    });
    
    // Float button to show chat
    this.floatBtn?.addEventListener("click", () => {
      this.show();
    });
    
    // Show the chat by default (Akinator mode ON by default)
    this.show();
  }
  
  show() {
    this.chatEl?.classList.remove("hidden");
    this.floatBtn?.classList.add("hidden");
    this.isActive = true;
  }
  
  hide() {
    this.chatEl?.classList.add("hidden");
    this.floatBtn?.classList.remove("hidden");
    this.isActive = false;
  }
  
  toggleCollapse() {
    this.chatEl?.classList.toggle("collapsed");
    const icon = this.toggleBtn?.textContent;
    if (this.toggleBtn) {
      this.toggleBtn.textContent = icon === "v" ? "^" : "v";
    }
  }
  
  async startSession(visualContext = null, analysisId = null) {
    try {
      this.visualContext = visualContext;
      this.analysisId = analysisId;
      this.messageHistory = [];
      this.roundNumber = 0;
      
      // Clear the body
      this.bodyEl.innerHTML = "";
      
      // Add loading status
      this.addStatus("Starting AI analysis session...");
      
      // Call API to start session
      const response = await fetch("/api/akinator/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_context: visualContext,
          analysis_id: analysisId,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      this.sessionId = data.session_id;
      this.roundNumber = data.round_number;
      
      // Remove loading status
      this.clearStatus();
      
      if (data.question) {
        this.currentQuestion = data.question;
        this.renderQuestion(data.question);
      } else if (data.inference) {
        this.renderInference(data.inference);
      }
      
    } catch (error) {
      console.error("Failed to start Akinator session:", error);
      this.addStatus("Failed to start AI session. Using fallback mode.");
      // Start with fallback question
      this.startFallbackSession();
    }
  }
  
  startFallbackSession() {
    this.sessionId = `fallback-${Date.now()}`;
    this.roundNumber = 1;
    this.currentQuestion = {
      id: "q-smell",
      text: "Does the water have any unusual smell? (chlorine, sulfur/rotten egg, metallic, earthy/musty, or no smell)",
      type: "choice",
      options: ["Yes, strong smell", "Yes, slight smell", "No smell", "Not sure"],
    };
    this.renderQuestion(this.currentQuestion);
  }
  
  renderQuestion(question) {
    const questionEl = document.createElement("div");
    questionEl.className = "akinator-question";
    questionEl.id = `question-${question.id}`;
    
    const roundInfo = document.createElement("div");
    roundInfo.className = "akinator-round";
    roundInfo.textContent = `Round ${this.roundNumber} of ${this.maxRounds}`;
    
    const textEl = document.createElement("p");
    textEl.textContent = question.text;
    
    const optionsEl = document.createElement("div");
    optionsEl.className = "akinator-options";
    
    if (question.type === "yesno" || question.type === "choice") {
      const options = question.options || ["Yes", "No", "Not sure"];
      options.forEach((option) => {
        const btn = document.createElement("button");
        btn.textContent = option;
        btn.addEventListener("click", () => this.submitAnswer(question.id, option));
        optionsEl.appendChild(btn);
      });
    } else if (question.type === "text") {
      // Text input for open-ended questions
      const input = document.createElement("input");
      input.type = "text";
      input.placeholder = "Type your answer...";
      input.className = "akinator-text-input";
      input.style.cssText = "width: 100%; padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.05); color: white; margin-bottom: 8px;";
      
      const submitBtn = document.createElement("button");
      submitBtn.textContent = "Submit";
      submitBtn.addEventListener("click", () => {
        if (input.value.trim()) {
          this.submitAnswer(question.id, input.value.trim());
        }
      });
      
      optionsEl.appendChild(input);
      optionsEl.appendChild(submitBtn);
    }
    
    questionEl.appendChild(roundInfo);
    questionEl.appendChild(textEl);
    questionEl.appendChild(optionsEl);
    
    this.bodyEl.appendChild(questionEl);
    this.scrollToBottom();
  }
  
  async submitAnswer(questionId, answer) {
    try {
      // Add the answer to the chat
      this.addAnswer(answer);
      
      // Disable the current question buttons
      const questionEl = document.getElementById(`question-${questionId}`);
      if (questionEl) {
        const buttons = questionEl.querySelectorAll("button");
        buttons.forEach((btn) => {
          btn.disabled = true;
          if (btn.textContent === answer) {
            btn.classList.add("selected");
          }
        });
      }
      
      // Add thinking status
      this.addStatus("Analyzing your answer...");
      
      // Submit to API
      const response = await fetch("/api/akinator/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: this.sessionId,
          question_id: questionId,
          answer: answer,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      this.roundNumber = data.round_number;
      this.clearStatus();
      
      // Store in history
      this.messageHistory.push({
        question: this.currentQuestion,
        answer: answer,
      });
      
      if (data.inference) {
        // We have a result!
        this.renderInference(data.inference);
      } else if (data.question) {
        // More questions needed
        this.currentQuestion = data.question;
        this.renderQuestion(data.question);
      } else {
        // Something went wrong
        this.addStatus("Unable to continue analysis. Please try again.");
      }
      
    } catch (error) {
      console.error("Failed to submit answer:", error);
      this.clearStatus();
      
      // Fallback: continue with next question locally
      this.continueFallback(questionId, answer);
    }
  }
  
  continueFallback(questionId, answer) {
    // Simple fallback question flow
    const fallbackQuestions = [
      { id: "q-color", text: "What color is the water?", type: "choice", options: ["Clear/transparent", "Slightly yellow/tinted", "Brown/muddy", "Green (algae)", "Other"] },
      { id: "q-clarity", text: "How clear is the water?", type: "choice", options: ["Crystal clear", "Slightly cloudy", "Very cloudy/turbid", "Opaque"] },
      { id: "q-particles", text: "Can you see any particles or sediment in the water?", type: "choice", options: ["No particles visible", "Small particles suspended", "Settled sediment at bottom", "Large debris visible"] },
      { id: "q-source", text: "What is the source of this water?", type: "choice", options: ["Tap/municipal supply", "Well/borehole", "River/stream", "Lake/pond", "Rainwater", "Unknown"] },
      { id: "q-container", text: "Is the water in its original container or has it been transferred?", type: "choice", options: ["Original sealed bottle", "Transferred to another container", "Open container", "Natural water body"] },
    ];
    
    const currentIndex = this.roundNumber - 1;
    
    if (currentIndex < fallbackQuestions.length && this.roundNumber < this.maxRounds) {
      this.currentQuestion = fallbackQuestions[currentIndex];
      this.roundNumber++;
      this.renderQuestion(this.currentQuestion);
    } else {
      // Generate a fallback inference based on answers
      this.generateFallbackInference();
    }
  }
  
  generateFallbackInference() {
    // Simple inference based on collected answers
    let score = 70; // Start with moderate score
    let confidence = 0.5;
    let reasoning = "Based on your observations: ";
    let contaminants = [];
    
    // Analyze answers
    this.messageHistory.forEach(({ question, answer }) => {
      if (question.id === "q-smell") {
        if (answer.includes("strong") || answer.includes("slight")) {
          score -= 15;
          reasoning += "Unusual smell detected. ";
          contaminants.push("Potential chemical contamination");
        }
      }
      if (question.id === "q-color") {
        if (answer.includes("Brown") || answer.includes("muddy")) {
          score -= 20;
          reasoning += "Discoloration suggests contamination. ";
          contaminants.push("Sediment/particulate matter");
        } else if (answer.includes("Green")) {
          score -= 25;
          reasoning += "Green tint suggests algae growth. ";
          contaminants.push("Algal contamination");
        }
      }
      if (question.id === "q-clarity") {
        if (answer.includes("cloudy") || answer.includes("turbid") || answer.includes("Opaque")) {
          score -= 20;
          reasoning += "Low clarity indicates impurities. ";
        }
      }
    });
    
    // Clamp score
    score = Math.max(0, Math.min(100, score));
    confidence = Math.min(0.85, confidence + (this.messageHistory.length * 0.05));
    
    const waterQuality = score >= 70 ? "Likely Safe" : score >= 40 ? "Questionable" : "Potentially Unsafe";
    
    this.renderInference({
      water_quality: waterQuality,
      confidence: confidence,
      reasoning: reasoning,
      recommendations: this.generateRecommendations(score),
      contaminants: contaminants,
    });
  }
  
  generateRecommendations(score) {
    if (score >= 80) {
      return ["Water appears safe based on visual inspection", "Regular testing recommended for confirmation"];
    } else if (score >= 60) {
      return ["Consider boiling before drinking", "Test for specific contaminants if concerned", "Use filtration if available"];
    } else if (score >= 40) {
      return ["Do not drink without proper treatment", "Boil and filter before any use", "Consider laboratory testing"];
    } else {
      return ["Do not consume this water", "Seek alternative water source", "Contact local water authorities if from municipal supply"];
    }
  }
  
  renderInference(inference) {
    const resultEl = document.createElement("div");
    resultEl.className = "akinator-result";
    
    const titleEl = document.createElement("h4");
    titleEl.textContent = "Analysis Complete!";
    
    const qualityEl = document.createElement("p");
    qualityEl.innerHTML = `<strong>Water Quality:</strong> ${inference.water_quality}`;
    
    const reasoningEl = document.createElement("p");
    reasoningEl.textContent = inference.reasoning || "Based on visual and contextual analysis.";
    
    const confidenceBar = document.createElement("div");
    confidenceBar.className = "confidence-bar";
    
    const confidenceFill = document.createElement("div");
    confidenceFill.className = "confidence-fill";
    confidenceFill.style.width = `${(inference.confidence || 0.5) * 100}%`;
    
    confidenceBar.appendChild(confidenceFill);
    
    const confidenceLabel = document.createElement("p");
    confidenceLabel.style.fontSize = "0.8rem";
    confidenceLabel.style.color = "rgba(255,255,255,0.6)";
    confidenceLabel.textContent = `Confidence: ${Math.round((inference.confidence || 0.5) * 100)}%`;
    
    resultEl.appendChild(titleEl);
    resultEl.appendChild(qualityEl);
    resultEl.appendChild(reasoningEl);
    resultEl.appendChild(confidenceLabel);
    resultEl.appendChild(confidenceBar);
    
    // Add recommendations if available
    if (inference.recommendations && inference.recommendations.length > 0) {
      const recsEl = document.createElement("div");
      recsEl.style.marginTop = "12px";
      
      const recsTitle = document.createElement("p");
      recsTitle.style.fontSize = "0.85rem";
      recsTitle.style.color = "rgba(255,255,255,0.8)";
      recsTitle.innerHTML = "<strong>Recommendations:</strong>";
      recsEl.appendChild(recsTitle);
      
      const recsList = document.createElement("ul");
      recsList.style.margin = "8px 0";
      recsList.style.paddingLeft = "20px";
      recsList.style.fontSize = "0.8rem";
      inference.recommendations.forEach((rec) => {
        const li = document.createElement("li");
        li.textContent = rec;
        li.style.color = "rgba(255,255,255,0.7)";
        recsList.appendChild(li);
      });
      recsEl.appendChild(recsList);
      resultEl.appendChild(recsEl);
    }
    
    // Add done button
    const doneBtn = document.createElement("button");
    doneBtn.textContent = "Done";
    doneBtn.style.cssText = "margin-top: 16px; width: 100%; padding: 10px; border-radius: 8px; border: none; background: linear-gradient(135deg, var(--primary, #00d4ff), var(--accent-green, #00ff88)); color: #0a0a0f; font-weight: 600; cursor: pointer;";
    doneBtn.addEventListener("click", () => {
      this.hide();
      // Update the main results display with the inference
      this.updateMainResults(inference);
    });
    
    resultEl.appendChild(doneBtn);
    
    this.bodyEl.appendChild(resultEl);
    this.scrollToBottom();
    
    // Also update main results
    this.updateMainResults(inference);
  }
  
  updateMainResults(inference) {
    // Map water quality to score
    let score = 50;
    if (inference.water_quality === "Likely Safe" || inference.water_quality === "Clean") {
      score = 80;
    } else if (inference.water_quality === "Questionable" || inference.water_quality === "Moderate") {
      score = 50;
    } else if (inference.water_quality === "Potentially Unsafe" || inference.water_quality === "Dirty") {
      score = 25;
    }
    
    // Update potability score if available
    if (potabilityScoreEl) {
      potabilityScoreEl.textContent = score;
    }
    if (confidenceScoreEl) {
      confidenceScoreEl.textContent = `${Math.round(inference.confidence * 100)}%`;
    }
    if (confidenceBandEl) {
      const band = inference.confidence >= 0.8 ? "High" : inference.confidence >= 0.5 ? "Moderate" : "Low";
      confidenceBandEl.textContent = band;
    }
    
    // Show results card
    if (resultsCard) {
      resultsCard.classList.remove("hidden");
    }
  }
  
  addAnswer(answer) {
    const answerEl = document.createElement("div");
    answerEl.className = "akinator-answer";
    
    const textEl = document.createElement("p");
    textEl.textContent = answer;
    
    answerEl.appendChild(textEl);
    this.bodyEl.appendChild(answerEl);
    this.scrollToBottom();
  }
  
  addStatus(message) {
    const statusEl = document.createElement("div");
    statusEl.className = "akinator-status";
    statusEl.textContent = message;
    this.bodyEl.appendChild(statusEl);
    this.scrollToBottom();
  }
  
  clearStatus() {
    const statusEl = this.bodyEl.querySelector(".akinator-status:last-child");
    if (statusEl) {
      statusEl.remove();
    }
  }
  
  scrollToBottom() {
    if (this.bodyEl) {
      this.bodyEl.scrollTop = this.bodyEl.scrollHeight;
    }
  }
  
  reset() {
    this.sessionId = null;
    this.currentQuestion = null;
    this.roundNumber = 0;
    this.visualContext = null;
    this.analysisId = null;
    this.messageHistory = [];
    
    if (this.bodyEl) {
      this.bodyEl.innerHTML = "";
      this.addStatus("Ready to analyze. Upload media to begin.");
    }
  }
}

// Initialize Akinator controller
const akinatorController = new AkinatorController();

// ============================================
// END AKINATOR CONTROLLER
// ============================================

// Initialize default state
clearTimeline();
resetDebug();
loadHistory(); // Load history on page load
