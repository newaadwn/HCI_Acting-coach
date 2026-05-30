const app = document.querySelector("#app");

const MEDIAPIPE_VISION_BUNDLE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
const MEDIAPIPE_WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const LIVE_MODEL_PATH = "/backend/face_landmarker.task";

const EMOTION_META = [
  { id: "happiness", label: "Happiness", color: "#f59e0b" },
  { id: "sadness", label: "Sadness", color: "#60a5fa" },
  { id: "anger", label: "Anger", color: "#f87171" },
  { id: "surprise", label: "Surprise", color: "#22d3ee" },
  { id: "neutral", label: "Neutral", color: "#94a3b8" },
];

const LIVE_EMOTION_META = EMOTION_META.filter((emotion) => emotion.id !== "neutral");
const LIVE_EMOTION_COLOR = Object.fromEntries(EMOTION_META.map((emotion) => [emotion.id, emotion.color]));
const LIVE_DISTRIBUTION_PATHS = {
  anger: "/backend/acting_coach/anger_distribution.csv",
  neutral: "/backend/acting_coach/neutral_distribution.csv",
};
const LIVE_USER_NEUTRAL_COMPARE_KEYS = [
  "browDownLeft",
  "browDownRight",
  "eyeSquintLeft",
  "eyeSquintRight",
  "mouthPressLeft",
  "mouthPressRight",
  "noseSneerLeft",
  "noseSneerRight",
  "mouthSmileLeft",
  "mouthSmileRight",
  "mouthFrownLeft",
  "mouthFrownRight",
  "browInnerUp",
  "browOuterUpLeft",
  "browOuterUpRight",
  "eyeWideLeft",
  "eyeWideRight",
  "jawOpen",
];
const LIVE_AIHUB_COMPARE_KEYS = [
  "browDownLeft",
  "browDownRight",
  "eyeSquintLeft",
  "eyeSquintRight",
  "mouthPressLeft",
  "mouthPressRight",
  "noseSneerLeft",
  "noseSneerRight",
  "mouthSmileLeft",
  "mouthSmileRight",
  "mouthFrownLeft",
  "mouthFrownRight",
  "browInnerUp",
  "eyeWideLeft",
  "eyeWideRight",
  "jawOpen",
];
const LIVE_MEDIAPIPE_NEUTRAL_THRESHOLD = 15;
const LIVE_AIHUB_STD_FLOOR = 0.02;
const LIVE_USER_STD_FLOOR = 0.015;
const LIVE_USER_NEUTRAL_DISTANCE_THRESHOLD = 1.8;
const LIVE_USER_NEUTRAL_DELTA_ENERGY_THRESHOLD = 0.025;
const LIVE_LOW_INTENSITY_NEUTRAL_THRESHOLD = 40;
const LIVE_LOW_INTENSITY_NEUTRAL_AVERAGE = 28;
const LIVE_LOW_INTENSITY_NEUTRAL_ANGER_CORE = 0.22;
const LIVE_LOW_INTENSITY_WEAK_ANGER_SCORE = 62;
const LIVE_BLENDSHAPE_SMOOTHING_ALPHA = 0.65;

const ANALYSIS_STEPS = [
  { label: "Inspecting upload integrity", detail: "Checking codec, dimensions, and frame timing." },
  { label: "Tracking facial landmarks", detail: "Locating eyebrow, jaw, lip, and eye movement signals." },
  { label: "Projecting emotion vectors", detail: "Scoring micro-expression intensity across every frame." },
  { label: "Preparing rehearsal guidance", detail: "Summarizing the actor's emotional baseline for comparison." },
];

const WORKFLOW_STEPS = [
  { id: "upload", label: "Upload", detail: "Choose the actor clip" },
  { id: "analysis", label: "Analyze", detail: "Build the reference profile" },
  { id: "recording", label: "Record", detail: "Calibrate, rehearse, and capture" },
  { id: "results", label: "Results", detail: "Compare the take and coaching notes" },
];

const SUPPORT_ISSUE_URL = "https://github.com/tpgk4456-ui/HCI-acting-coach/issues/new";
const APP_SESSION_STORAGE_KEY = "actingCoachSessionId";
const APP_SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{8,128}$/;
const LOCAL_HOSTNAMES = new Set(["localhost", "127.0.0.1", "::1"]);

const TAKE_ANALYSIS_STEPS = [
  { label: "Securing the recording", detail: "Stopping capture and packaging the last frames from your take." },
  { label: "Uploading your take", detail: "Sending the rehearsal clip to the analysis server for processing." },
  { label: "Running emotion inference", detail: "Tracking facial landmarks and scoring expression intensity frame by frame." },
  { label: "Preparing the comparison board", detail: "Calculating the match score and drafting coaching notes for the result screen." },
];

function createAppSessionId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }

  return `session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

function getOrCreateAppSessionId() {
  try {
    const stored = window.localStorage.getItem(APP_SESSION_STORAGE_KEY);
    if (typeof stored === "string" && APP_SESSION_ID_PATTERN.test(stored)) {
      return stored;
    }

    const created = createAppSessionId();
    window.localStorage.setItem(APP_SESSION_STORAGE_KEY, created);
    return created;
  } catch {
    return createAppSessionId();
  }
}

function isLocalDesktopWebcamAvailable() {
  return LOCAL_HOSTNAMES.has(window.location.hostname.toLowerCase());
}

function buildApiHeaders(extraHeaders = {}) {
  return {
    Accept: "application/json",
    "X-Session-Id": appSessionId,
    ...extraHeaders,
  };
}

const appSessionId = getOrCreateAppSessionId();

const appState = {
  sessionId: appSessionId,
  routeCleanup: null,
  isDragging: false,
  referenceFile: null,
  referenceUrl: "",
  referenceMeta: null,
  isDemoReference: false,
  actorProfile: null,
  actorTimeline: [],
  analysisProgress: 0,
  analysisFrameCount: 0,
  analysisError: "",
  analysisMode: "",
  sourceLabel: "",
  sourceDetail: "",
  recordingStream: null,
  mediaRecorder: null,
  recordingChunks: [],
  recordingStartedAt: 0,
  recordingDuration: 0,
  recordedBlob: null,
  recordedUrl: "",
  userProfile: null,
  similarityScore: null,
  feedback: [],
  lastTakeSource: "camera",
  cameraError: "",
  takeAnalysisInFlight: false,
  takeAnalysisStartedAt: 0,
  takeAnalysisMessage: "",
  takeAnalysisError: "",
  recordingCountdownActive: false,
  recordingCountdownValue: 0,
  synchronizedReferencePlayback: false,
  neutralCalibrationActive: false,
  neutralCalibrationReady: false,
  neutralCalibrationProgressMs: 0,
  neutralCalibrationTargetMs: 3000,
  neutralCalibrationLastTickAt: 0,
  neutralCalibrationSamples: [],
  neutralBlendshapeBaseline: null,
  neutralBlendshapeStd: null,
  liveTrackingAvailable: false,
  liveTrackingMessage: "",
};

let liveFaceLandmarker = null;
let liveFaceLandmarkerPromise = null;
let liveAihubDistributionPromise = null;
let livePreviousRelativeBlendshapeMap = null;
let liveAnalysisFrameId = 0;
let liveAnalysisSession = 0;
let referenceAnalysisFrameId = 0;
let referenceAnalysisSession = 0;
let recordingCountdownTimer = 0;

function cleanupRoute() {
  if (typeof appState.routeCleanup === "function") {
    appState.routeCleanup();
  }
  appState.routeCleanup = null;
}

function navigate(path) {
  if (window.location.pathname !== path) {
    history.pushState({}, "", path);
  }
  renderRoute();
}

window.addEventListener("popstate", () => renderRoute());

function renderRoute() {
  cleanupRoute();
  const route = window.location.pathname;
  const routes = {
    "/": renderUploadPage,
    "/analysis": renderAnalysisPage,
    "/analysis-error": renderAnalysisErrorPage,
    "/recording": renderRecordingPage,
    "/results": renderResultsPage,
  };

  const renderer = routes[route] || renderUploadPage;
  renderer();
}

function setReferenceFile(file) {
  if (!file || !file.type.startsWith("video/")) {
    appState.analysisError = "Please choose a valid video file.";
    navigate("/analysis-error");
    return;
  }

  if (appState.referenceUrl) {
    URL.revokeObjectURL(appState.referenceUrl);
  }
  if (appState.recordedUrl) {
    URL.revokeObjectURL(appState.recordedUrl);
  }

  stopCamera();
  appState.referenceFile = file;
  appState.referenceUrl = URL.createObjectURL(file);
  appState.referenceMeta = null;
  appState.isDemoReference = false;
  appState.actorProfile = null;
  appState.actorTimeline = [];
  appState.userProfile = null;
  appState.feedback = [];
  appState.similarityScore = null;
  appState.recordingDuration = 0;
  appState.recordedUrl = "";
  appState.recordedBlob = null;
  appState.mediaRecorder = null;
  appState.cameraError = "";
  appState.analysisError = "";
  appState.analysisMode = "";
  appState.sourceLabel = "";
  appState.sourceDetail = "";
  appState.takeAnalysisInFlight = false;
  appState.takeAnalysisMessage = "";
  appState.takeAnalysisError = "";
  resetNeutralCalibration();
  appState.liveTrackingMessage = "";

  readVideoMetadata(appState.referenceUrl)
    .then((meta) => {
      appState.referenceMeta = meta;
      renderRoute();
    })
    .catch(() => {
      appState.referenceMeta = {
        duration: 0,
        width: 0,
        height: 0,
      };
      renderRoute();
    });

  renderRoute();
}

function loadDemoReference() {
  if (appState.referenceUrl) {
    URL.revokeObjectURL(appState.referenceUrl);
  }
  if (appState.recordedUrl) {
    URL.revokeObjectURL(appState.recordedUrl);
  }

  stopCamera();
  appState.referenceFile = {
    name: "hci-acting-coach-reference.mp4",
    size: 24_800_000,
    type: "video/mp4",
  };
  appState.referenceUrl = "";
  appState.referenceMeta = {
    duration: 58,
    width: 1920,
    height: 1080,
  };
  appState.isDemoReference = true;
  appState.actorProfile = null;
  appState.actorTimeline = [];
  appState.userProfile = null;
  appState.feedback = [];
  appState.similarityScore = null;
  appState.recordingDuration = 0;
  appState.recordedUrl = "";
  appState.recordedBlob = null;
  appState.mediaRecorder = null;
  appState.cameraError = "";
  appState.analysisError = "";
  appState.analysisMode = "";
  appState.sourceLabel = "";
  appState.sourceDetail = "";
  appState.takeAnalysisInFlight = false;
  appState.takeAnalysisMessage = "";
  appState.takeAnalysisError = "";
  resetNeutralCalibration();
  appState.liveTrackingMessage = "";
  renderRoute();
}

function readVideoMetadata(url) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.src = url;
    video.onloadedmetadata = () => {
      resolve({
        duration: Number.isFinite(video.duration) ? video.duration : 0,
        width: video.videoWidth || 0,
        height: video.videoHeight || 0,
      });
    };
    video.onerror = () => reject(new Error("metadata-failed"));
  });
}

function formatDuration(seconds) {
  const safeSeconds = Math.max(0, Math.round(seconds || 0));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatFileSize(size) {
  if (!size) return "Unknown size";
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

async function readJsonOrThrow(response, fallbackPrefix) {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.message || `${fallbackPrefix}-${response.status}`);
  }
  return payload;
}

async function fetchReferenceProfile({ demo = false } = {}) {
  const suffix = demo ? "?demo=1" : "";
  const response = await fetch(`/api/analysis/reference${suffix}`, {
    headers: buildApiHeaders(),
  });

  return readJsonOrThrow(response, "reference-profile");
}

async function fetchSampleComparison() {
  const response = await fetch("/api/analysis/sample-comparison", {
    headers: buildApiHeaders(),
  });

  return readJsonOrThrow(response, "sample-comparison");
}

async function uploadVideoForAnalysis(url, blob, filename, extraHeaders = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: buildApiHeaders({
      "X-Filename": encodeURIComponent(filename),
      ...extraHeaders,
    }),
    body: blob,
  });

  return readJsonOrThrow(response, "video-analysis");
}

async function analyzeReferenceVideo(file) {
  return uploadVideoForAnalysis("/api/analysis/reference-upload", file, file.name || "reference-video.mp4");
}

async function analyzeRecordedTake(blob) {
  return uploadVideoForAnalysis("/api/analysis/user-upload", blob, "recorded-take.webm");
}

async function runDesktopWebcamAnalysis() {
  const response = await fetch("/api/analysis/run-webcam", {
    method: "POST",
    headers: buildApiHeaders(),
  });

  return readJsonOrThrow(response, "webcam-analysis");
}

function applyReferenceProfile(profile) {
  appState.actorProfile = profile;
  appState.actorTimeline = Array.isArray(profile.timeline) ? profile.timeline : [];
  appState.analysisFrameCount = profile.frameCount || 0;
  appState.analysisMode = profile.analysisMode || "";
  appState.sourceLabel = profile.sourceLabel || "";
  appState.sourceDetail = profile.sourceDetail || "";
}

function getTakeSourceLabel(source) {
  if (source === "sample-model") {
    return "HCI sample CSV";
  }

  if (source === "recorded-video") {
    return "Recorded video";
  }

  if (source === "desktop-webcam") {
    return "Desktop webcam";
  }

  return source === "camera" ? "Live camera" : "Generated fallback";
}

function getFinalSourceLabel(source) {
  if (source === "csv_label_aux") {
    return "CSV label assist";
  }

  if (source === "mediapipe+csv_score") {
    return "MediaPipe + CSV score";
  }

  return source === "mediapipe" ? "MediaPipe" : "Backend mixed";
}

function getProfileDominantEmotion(profile, valueKey) {
  if (profile?.dominantSummary) {
    return {
      id: profile.dominantSummary.id,
      label: profile.dominantSummary.label,
      color: profile.dominantSummary.color,
      [valueKey]: Math.round(profile.dominantSummary.averagePercent || 0),
      averagePercent: Math.round(profile.dominantSummary.averagePercent || 0),
      share: profile.dominantSummary.share || 0,
      count: profile.dominantSummary.count || 0,
    };
  }

  const expressiveEmotions = [...(profile?.emotions || [])]
    .filter((emotion) => emotion.id !== "neutral")
    .sort((a, b) => (b[valueKey] || 0) - (a[valueKey] || 0));

  return expressiveEmotions[0] || profile?.topEmotion || null;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function deriveComparison(actorProfile, userProfile) {
  const bars = actorProfile.emotions.map((emotion) => {
    const userMatch = userProfile.emotions.find((item) => item.id === emotion.id);
    return {
      ...emotion,
      user: userMatch ? userMatch.user : 0,
    };
  });

  // Neutral is a derived fallback, so keep the main score focused on expressive channels.
  const expressiveBars = bars.filter((emotion) => emotion.id !== "neutral");
  const leadEmotion = expressiveBars.reduce(
    (best, emotion) => (emotion.actor > best.actor ? emotion : best),
    expressiveBars[0] || bars[0]
  );
  const weightedGap = expressiveBars.reduce((sum, emotion) => {
    const weight = emotion.id === leadEmotion.id ? 2 : 1;
    return sum + Math.abs(emotion.actor - emotion.user) * weight;
  }, 0);
  const weightTotal = expressiveBars.reduce((sum, emotion) => sum + (emotion.id === leadEmotion.id ? 2 : 1), 0);
  const actorIntensity =
    expressiveBars.reduce((sum, emotion) => sum + emotion.actor, 0) / Math.max(1, expressiveBars.length);
  const userIntensity =
    expressiveBars.reduce((sum, emotion) => sum + emotion.user, 0) / Math.max(1, expressiveBars.length);
  const expressiveGap = weightTotal ? weightedGap / weightTotal : 0;
  const intensityGap = Math.abs(actorIntensity - userIntensity);
  const similarityScore = Math.round(clamp(100 - expressiveGap * 1.5 - intensityGap * 0.75, 0, 100));

  const feedback = bars
    .map((emotion) => {
      const gap = emotion.user - emotion.actor;
      if (Math.abs(gap) <= 8) {
        return {
          id: `${emotion.id}-good`,
          emotion: emotion.label,
          type: "success",
          title: "Strong match",
          message: `${emotion.label} is closely aligned with the reference. Keep this shape and timing in your next take.`,
        };
      }

      if (gap > 0) {
        return {
          id: `${emotion.id}-high`,
          emotion: emotion.label,
          type: gap > 16 ? "warning" : "info",
          title: "Dial it back slightly",
          message: `${emotion.label} is ${gap}% higher than the actor. Soften that channel to keep the scene grounded.`,
        };
      }

      return {
        id: `${emotion.id}-low`,
        emotion: emotion.label,
        type: Math.abs(gap) > 16 ? "danger" : "warning",
        title: "Push this emotion further",
        message: `${emotion.label} is ${Math.abs(gap)}% lower than the actor. Add more facial tension and clearer transitions here.`,
      };
    })
    .sort((a, b) => {
      const priority = { danger: 0, warning: 1, info: 2, success: 3 };
      return priority[a.type] - priority[b.type];
    });

  return {
    bars,
    similarityScore,
    feedback,
  };
}

function applyComparisonPayload(payload, takeSource) {
  if (payload.actorProfile) {
    applyReferenceProfile(payload.actorProfile);
  }

  appState.lastTakeSource = takeSource;
  appState.userProfile = payload.userProfile;
  appState.recordingDuration = payload.takeDurationSeconds || appState.recordingDuration || 0;
  appState.similarityScore = payload.comparison?.similarityScore ?? null;
  appState.feedback = payload.comparison?.feedback || [];
}

function setTakeAnalysisState(isRunning, message = "", error = "") {
  if (isRunning && !appState.takeAnalysisInFlight) {
    appState.takeAnalysisStartedAt = performance.now();
  }

  if (!isRunning) {
    appState.takeAnalysisStartedAt = 0;
  }

  appState.takeAnalysisInFlight = isRunning;
  appState.takeAnalysisMessage = message;
  appState.takeAnalysisError = error;
}

function clearRecordedTakeArtifacts() {
  if (appState.recordedUrl) {
    URL.revokeObjectURL(appState.recordedUrl);
  }

  appState.recordingChunks = [];
  appState.recordingDuration = 0;
  appState.recordedBlob = null;
  appState.recordedUrl = "";
  appState.mediaRecorder = null;
  appState.userProfile = null;
  appState.similarityScore = null;
  appState.feedback = [];
  appState.lastTakeSource = "camera";
  setTakeAnalysisState(false);
  clearRecordingCountdown();
  appState.synchronizedReferencePlayback = false;
  resetNeutralCalibration();
}

function resetNeutralCalibration() {
  appState.neutralCalibrationActive = false;
  appState.neutralCalibrationReady = false;
  appState.neutralCalibrationProgressMs = 0;
  appState.neutralCalibrationLastTickAt = 0;
  appState.neutralCalibrationSamples = [];
  appState.neutralBlendshapeBaseline = null;
  appState.neutralBlendshapeStd = null;
  livePreviousRelativeBlendshapeMap = null;
}

function formatCalibrationSeconds(progressMs) {
  return `${(progressMs / 1000).toFixed(1)}s`;
}

function startNeutralCalibration() {
  appState.neutralCalibrationActive = true;
  appState.neutralCalibrationReady = false;
  appState.neutralCalibrationProgressMs = 0;
  appState.neutralCalibrationLastTickAt = 0;
  appState.neutralCalibrationSamples = [];
  appState.neutralBlendshapeBaseline = null;
  appState.neutralBlendshapeStd = null;
  livePreviousRelativeBlendshapeMap = null;
  appState.liveTrackingMessage = "Hold a neutral face for 3 seconds so the live preview can tune to your resting expression.";
}

function getLiveCalibrationProgressLabel() {
  return `${formatCalibrationSeconds(appState.neutralCalibrationProgressMs)} / ${formatCalibrationSeconds(
    appState.neutralCalibrationTargetMs
  )}`;
}

function computeAverage(values) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeStandardDeviation(values) {
  if (!values.length) {
    return 0;
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function buildLiveNeutralProfile(samples) {
  const keys = new Set();
  samples.forEach((sample) => {
    Object.keys(sample).forEach((key) => keys.add(key));
  });

  const baseline = {};
  const std = {};

  keys.forEach((key) => {
    const values = samples.map((sample) => Number(sample[key] || 0));
    baseline[key] = computeAverage(values);
    std[key] = computeStandardDeviation(values);
  });

  return { baseline, std };
}

function smoothLiveBlendshapeMap(currentMap, previousMap) {
  if (!previousMap) {
    return { ...currentMap };
  }

  return Object.keys(currentMap).reduce((smoothed, key) => {
    const previousValue = previousMap[key] ?? currentMap[key];
    smoothed[key] = previousValue * LIVE_BLENDSHAPE_SMOOTHING_ALPHA + currentMap[key] * (1 - LIVE_BLENDSHAPE_SMOOTHING_ALPHA);
    return smoothed;
  }, {});
}

function applyLiveNeutralCalibration(rawBlendshapeMap) {
  if (!appState.neutralBlendshapeBaseline || !appState.neutralBlendshapeStd) {
    return rawBlendshapeMap;
  }

  const relativeMap = Object.keys(rawBlendshapeMap).reduce((result, key) => {
    const rawValue = Number(rawBlendshapeMap[key] || 0);
    const baselineValue = Number(appState.neutralBlendshapeBaseline[key] || 0);
    const delta = rawValue - baselineValue;

    result[key] = Math.max(delta, 0);
    return result;
  }, {});

  const smoothed = smoothLiveBlendshapeMap(relativeMap, livePreviousRelativeBlendshapeMap);
  livePreviousRelativeBlendshapeMap = smoothed;
  return smoothed;
}

function getEmotionMetaById(id) {
  return EMOTION_META.find((emotion) => emotion.id === id) || EMOTION_META[0];
}

function stopReferenceEmotionTracking() {
  referenceAnalysisSession += 1;
  if (referenceAnalysisFrameId) {
    cancelAnimationFrame(referenceAnalysisFrameId);
    referenceAnalysisFrameId = 0;
  }
}

function findTimelineFrameAtTimeMs(timeline, timeMs) {
  if (!Array.isArray(timeline) || !timeline.length) {
    return null;
  }

  if (timeMs <= timeline[0].timestamp_ms) {
    return timeline[0];
  }

  let low = 0;
  let high = timeline.length - 1;

  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const frame = timeline[middle];
    if (frame.timestamp_ms === timeMs) {
      return frame;
    }
    if (frame.timestamp_ms < timeMs) {
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }

  return timeline[Math.max(0, high)] || timeline[0];
}

function renderReferenceEmotionHud(target, frame, status = "") {
  if (!target) {
    return;
  }

  if (status) {
    target.innerHTML = `
      <div class="emotion-hud-card is-loading">
        <strong>${escapeHtml(status)}</strong>
        <span>Reference timeline is getting ready.</span>
      </div>
    `;
    return;
  }

  if (!frame?.scores) {
    const topEmotion = appState.actorProfile?.topEmotion;
    target.innerHTML = `
      <div class="emotion-hud-card">
        <div class="emotion-hud-header">
          <span>Reference read</span>
          <strong>${escapeHtml(topEmotion ? `${topEmotion.label} ${Math.round(topEmotion.actor)}%` : "Ready")}</strong>
        </div>
        <span>Play the reference clip to see its time-synced emotion analysis.</span>
      </div>
    `;
    return;
  }

  const dominantMeta = getEmotionMetaById(frame.dominantEmotion);
  const dominantValue = Math.round(frame.scores[frame.dominantEmotion] || 0);

  target.innerHTML = `
    <div class="emotion-hud-card">
      <div class="emotion-hud-header">
        <span>Reference read</span>
        <strong>${escapeHtml(`${dominantMeta.label} ${dominantValue}%`)}</strong>
      </div>
      <div class="emotion-hud-bars">
        ${EMOTION_META.map((emotion) => {
          const value = Math.round(frame.scores[emotion.id] || 0);
          return `
            <div class="emotion-hud-row">
              <div class="emotion-hud-label">
                <span>${emotion.label}</span>
                <span>${value}%</span>
              </div>
              <div class="emotion-hud-meter">
                <div class="emotion-hud-fill" style="--emotion-color:${emotion.color}; width:${value}%;"></div>
              </div>
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function startReferenceEmotionTracking(videoElement, hudElement) {
  stopReferenceEmotionTracking();

  if (!videoElement || !hudElement) {
    return;
  }

  if (!appState.actorTimeline.length) {
    renderReferenceEmotionHud(hudElement, null, "Reference analysis unavailable");
    return;
  }

  const sessionId = referenceAnalysisSession;
  let lastMs = -1;

  const renderFrame = () => {
    if (sessionId !== referenceAnalysisSession) {
      return;
    }

    const currentMs = Math.max(0, Math.round((videoElement.currentTime || 0) * 1000));
    if (currentMs !== lastMs) {
      renderReferenceEmotionHud(hudElement, findTimelineFrameAtTimeMs(appState.actorTimeline, currentMs));
      lastMs = currentMs;
    }

    referenceAnalysisFrameId = requestAnimationFrame(renderFrame);
  };

  renderFrame();
}

function seekVideoToStart(videoElement) {
  if (!(videoElement instanceof HTMLVideoElement)) {
    return;
  }

  try {
    videoElement.currentTime = 0;
  } catch {
    // Ignore failed seeks before metadata is ready.
  }
}

function resetReferenceVideoPlayback(videoElement) {
  if (!(videoElement instanceof HTMLVideoElement)) {
    return;
  }

  videoElement.pause();
  seekVideoToStart(videoElement);
}

function primeReferenceVideoPlayback(videoElement) {
  if (!(videoElement instanceof HTMLVideoElement) || videoElement.readyState < 2) {
    return;
  }

  videoElement.muted = true;
  videoElement.defaultMuted = true;
  videoElement.playsInline = true;
  seekVideoToStart(videoElement);

  const primePromise = videoElement.play();
  if (primePromise && typeof primePromise.then === "function") {
    primePromise
      .then(() => {
        window.requestAnimationFrame(() => {
          resetReferenceVideoPlayback(videoElement);
        });
      })
      .catch(() => {
        resetReferenceVideoPlayback(videoElement);
      });
    return;
  }

  window.requestAnimationFrame(() => {
    resetReferenceVideoPlayback(videoElement);
  });
}

function playReferenceVideoFromStart(videoElement) {
  if (!(videoElement instanceof HTMLVideoElement)) {
    return;
  }

  videoElement.muted = true;
  videoElement.defaultMuted = true;
  videoElement.playsInline = true;

  const attemptPlayback = () => {
    seekVideoToStart(videoElement);

    const playPromise = videoElement.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
        videoElement.addEventListener(
          "canplay",
          () => {
            seekVideoToStart(videoElement);
            videoElement.play().catch(() => {});
          },
          { once: true }
        );
      });
    }
  };

  if (videoElement.readyState >= 2) {
    attemptPlayback();
    return;
  }

  videoElement.addEventListener("loadeddata", attemptPlayback, { once: true });
}

function updateRecordingCountdownOverlay() {
  const overlays = document.querySelectorAll("[data-countdown-overlay]");
  overlays.forEach((overlay) => {
    if (!(overlay instanceof HTMLElement)) {
      return;
    }

    if (!appState.recordingCountdownActive) {
      overlay.classList.add("hidden");
      overlay.innerHTML = "";
      return;
    }

    overlay.classList.remove("hidden");
    overlay.innerHTML = `
      <div class="frame-countdown-content">
        <span class="frame-countdown-label">Starting together in</span>
        <strong class="frame-countdown-value">${appState.recordingCountdownValue}</strong>
      </div>
    `;
  });
}

function clearRecordingCountdown() {
  if (recordingCountdownTimer) {
    clearInterval(recordingCountdownTimer);
    recordingCountdownTimer = 0;
  }

  appState.recordingCountdownActive = false;
  appState.recordingCountdownValue = 0;
  updateRecordingCountdownOverlay();
}

function startRecordingCountdown() {
  if (recordingCountdownTimer || appState.takeAnalysisInFlight) {
    return;
  }

  appState.cameraError = "";
  appState.recordingCountdownActive = true;
  appState.recordingCountdownValue = 3;
  updateRecordingCountdownOverlay();
  syncRecordingControls();
  updateRecordingStatusMessage();

  const referenceVideo = document.querySelector("#reference-preview");
  if (referenceVideo instanceof HTMLVideoElement) {
    referenceVideo.preload = "auto";
    resetReferenceVideoPlayback(referenceVideo);
    primeReferenceVideoPlayback(referenceVideo);
    if (referenceVideo.readyState < 2) {
      referenceVideo.load();
    }
  }

  recordingCountdownTimer = window.setInterval(() => {
    appState.recordingCountdownValue -= 1;

    if (appState.recordingCountdownValue <= 0) {
      clearRecordingCountdown();
      appState.synchronizedReferencePlayback = true;
      beginCameraRecording();
      if (referenceVideo instanceof HTMLVideoElement) {
        playReferenceVideoFromStart(referenceVideo);
      }
      return;
    }

    updateRecordingCountdownOverlay();
    updateRecordingStatusMessage();
  }, 1000);
}

function syncRecordingBadge() {
  const badge = document.querySelector("[data-recording-badge]");
  if (!badge) {
    return;
  }

  const isRecording = appState.mediaRecorder && appState.mediaRecorder.state === "recording";
  badge.className = "record-badge";

  if (appState.takeAnalysisInFlight) {
    badge.classList.add("is-analyzing");
    badge.textContent = "Analyzing take";
    return;
  }

  if (isRecording) {
    badge.textContent = "Recording live";
    return;
  }

  badge.classList.add("is-ready");
  badge.textContent = "Ready to rehearse";
}

function syncRecordingControls() {
  const startButton = document.querySelector('[data-action="start-recording"]');
  const stopButton = document.querySelector('[data-action="stop-recording"]');
  const enableCameraButton = document.querySelector('[data-action="enable-camera"]');
  const desktopButton = document.querySelector('[data-action="desktop-webcam"]');
  const sampleButton = document.querySelector('[data-action="fallback-recording"]');
  const isRecording = appState.mediaRecorder && appState.mediaRecorder.state === "recording";
  const calibrationPending = Boolean(appState.recordingStream) && appState.neutralCalibrationActive && !appState.neutralCalibrationReady;
  const isBusy = appState.takeAnalysisInFlight || appState.recordingCountdownActive;

  if (startButton) {
    startButton.disabled =
      !appState.recordingStream ||
      appState.takeAnalysisInFlight ||
      appState.recordingCountdownActive ||
      calibrationPending ||
      isRecording;
    startButton.textContent = isRecording
      ? "Recording..."
      : appState.recordingCountdownActive
        ? `Starting in ${appState.recordingCountdownValue}...`
        : calibrationPending
          ? `Calibrating ${Math.ceil((appState.neutralCalibrationTargetMs - appState.neutralCalibrationProgressMs) / 1000)}s...`
        : "Start recording";
  }

  if (stopButton) {
    stopButton.disabled = !isRecording || appState.takeAnalysisInFlight;
    stopButton.classList.toggle("button-primary", Boolean(isRecording));
    stopButton.classList.toggle("button-secondary", !isRecording);
  }

  if (enableCameraButton) {
    enableCameraButton.disabled = isBusy;
  }

  if (desktopButton) {
    desktopButton.disabled = isBusy;
  }

  if (sampleButton) {
    sampleButton.disabled = isBusy;
  }

  syncRecordingBadge();
}

function renderNeutralCalibrationStatus() {
  const panel = document.querySelector("#neutral-calibration-panel");
  if (!panel) {
    return;
  }

  const statusLabel = appState.liveTrackingAvailable ? "CSV overlay ready" : "Camera only";
  const bodyCopy = appState.liveTrackingAvailable
    ? appState.liveTrackingMessage || "Face tracking uses MediaPipe plus the AI-Hub anger/neutral CSV distributions."
    : appState.liveTrackingMessage || "If the overlay is still loading, recording can still start and backend analysis will run after you stop.";
  const stateClass = appState.liveTrackingAvailable ? "is-ready" : "";

  panel.innerHTML = `
    <div class="neutral-calibration-card ${stateClass}">
      <div class="neutral-calibration-head">
        <strong>Live CSV status</strong>
        <span>${escapeHtml(statusLabel)}</span>
      </div>
      <p>${escapeHtml(bodyCopy)}</p>
    </div>
  `;
}

function updateRecordingStatusMessage() {
  const status = document.querySelector("#recording-status-message");
  if (!status) {
    return;
  }

  if (appState.takeAnalysisError) {
    status.innerHTML = `<div class="callout">${escapeHtml(appState.takeAnalysisError)}</div>`;
    return;
  }

  if (appState.mediaRecorder && appState.mediaRecorder.state === "recording") {
    status.innerHTML =
      '<div class="callout">Recording is already running. Perform your take, then press <strong>Stop and analyze</strong> to send it to the backend.</div>';
    return;
  }

  if (appState.recordingCountdownActive) {
    status.innerHTML =
      '<div class="callout">Countdown is running. When it reaches zero, the reference clip and your recording will start together.</div>';
    return;
  }

  if (appState.takeAnalysisInFlight) {
    status.innerHTML = `<div class="callout">${escapeHtml(appState.takeAnalysisMessage)}</div>`;
    return;
  }

  if (appState.cameraError) {
    status.innerHTML = `<div class="callout callout-danger">${escapeHtml(appState.cameraError)}</div>`;
    return;
  }

  status.innerHTML =
    '<div class="status-note">After you stop recording, the final comparison opens automatically.</div>';
}

function inferTakeAnalysisStepIndex(message) {
  const normalized = String(message || "").toLowerCase();

  if (normalized.includes("window should appear") || normalized.includes("perform the take there")) {
    return 0;
  }

  if (normalized.includes("finishing")) {
    return 0;
  }

  if (normalized.includes("uploading") || normalized.includes("bundled hci sample take")) {
    return 1;
  }

  if (normalized.includes("desktop webcam") || normalized.includes("opencv webcam") || normalized.includes("running")) {
    return 2;
  }

  return 0;
}

function getTakeAnalysisVisualState() {
  const elapsedMs = appState.takeAnalysisStartedAt ? Math.max(0, performance.now() - appState.takeAnalysisStartedAt) : 0;
  const timedStep = Math.min(TAKE_ANALYSIS_STEPS.length - 1, Math.floor(elapsedMs / 1800));
  const messageStep = inferTakeAnalysisStepIndex(appState.takeAnalysisMessage);
  const activeStep = Math.max(timedStep, messageStep);
  const baselineProgress = clamp(14 + elapsedMs / 120, 14, 92);
  const stagedProgress = 18 + activeStep * 21;

  return {
    activeStep,
    progress: Math.round(Math.max(baselineProgress, stagedProgress)),
  };
}

function buildTakeAnalysisTimelineMarkup(activeStep) {
  return TAKE_ANALYSIS_STEPS.map((step, index) => {
    const stateClass = index < activeStep ? "is-complete" : index === activeStep ? "is-active" : "";
    const stateLabel = index < activeStep ? "Done" : index === activeStep ? "Running" : "Queued";

    return `
      <div class="timeline-step ${stateClass}">
        <div class="timeline-dot"></div>
        <div>
          <strong>${step.label}</strong>
          <div class="small-label">${step.detail}</div>
        </div>
        <div class="timeline-status">${stateLabel}</div>
      </div>
    `;
  }).join("");
}

function renderTakeAnalysisOverlay() {
  const visualState = getTakeAnalysisVisualState();
  const activeStep = TAKE_ANALYSIS_STEPS[visualState.activeStep];
  const message = appState.takeAnalysisMessage || "Your take is being prepared for the comparison board.";

  return `
    <div class="take-analysis-overlay" id="take-analysis-overlay">
      <div class="take-analysis-card">
        <div class="take-analysis-head">
          <div>
            <span class="eyebrow">Stop And Analyze</span>
            <h2 class="take-analysis-title">Analyzing your take</h2>
            <p class="take-analysis-copy">The recording is saved. We are now scoring your expressions against the actor reference.</p>
          </div>
          <div class="take-analysis-spinner" aria-hidden="true"></div>
        </div>

        <div class="take-analysis-progress-copy">
          <strong id="take-analysis-stage">${activeStep.label}</strong>
          <span id="take-analysis-progress-value">${visualState.progress}%</span>
        </div>
        <div class="progress-track take-analysis-progress-track">
          <div id="take-analysis-progress-bar" class="progress-bar take-analysis-progress-bar" style="width:${visualState.progress}%;"></div>
        </div>
        <p id="take-analysis-detail" class="muted-copy take-analysis-detail">${activeStep.detail}</p>

        <div class="take-analysis-meta-grid">
          <div class="micro-card">
            <strong>${appState.actorProfile?.frameCount || appState.analysisFrameCount || 0}</strong>
            <span>Reference frames ready</span>
          </div>
          <div class="micro-card">
            <strong>${formatDuration(appState.recordingDuration || 0)}</strong>
            <span>Captured take length</span>
          </div>
          <div class="micro-card">
            <strong>${escapeHtml(getTakeSourceLabel(appState.lastTakeSource))}</strong>
            <span>Current source</span>
          </div>
        </div>

        <div class="take-analysis-status">
          <span class="take-analysis-status-label">Backend status</span>
          <strong id="take-analysis-message">${escapeHtml(message)}</strong>
        </div>

        <div id="take-analysis-timeline" class="timeline take-analysis-timeline">
          ${buildTakeAnalysisTimelineMarkup(visualState.activeStep)}
        </div>
      </div>
    </div>
  `;
}

function mountTakeAnalysisOverlay() {
  const overlay = document.querySelector("#take-analysis-overlay");
  if (!overlay) {
    return () => {};
  }

  const stageEl = overlay.querySelector("#take-analysis-stage");
  const detailEl = overlay.querySelector("#take-analysis-detail");
  const progressBarEl = overlay.querySelector("#take-analysis-progress-bar");
  const progressValueEl = overlay.querySelector("#take-analysis-progress-value");
  const messageEl = overlay.querySelector("#take-analysis-message");
  const timelineEl = overlay.querySelector("#take-analysis-timeline");

  const render = () => {
    const visualState = getTakeAnalysisVisualState();
    const activeStep = TAKE_ANALYSIS_STEPS[visualState.activeStep];

    if (stageEl) {
      stageEl.textContent = activeStep.label;
    }

    if (detailEl) {
      detailEl.textContent = activeStep.detail;
    }

    if (progressBarEl) {
      progressBarEl.style.width = `${visualState.progress}%`;
    }

    if (progressValueEl) {
      progressValueEl.textContent = `${visualState.progress}%`;
    }

    if (messageEl) {
      messageEl.textContent = appState.takeAnalysisMessage || "Your take is being prepared for the comparison board.";
    }

    if (timelineEl) {
      timelineEl.innerHTML = buildTakeAnalysisTimelineMarkup(visualState.activeStep);
    }
  };

  render();
  const interval = window.setInterval(render, 220);
  return () => window.clearInterval(interval);
}

function toPercent(rawValue, emotionId) {
  const sensitivity = {
    happiness: 4.0,
    joy: 4.0,
    sadness: 6.0,
    anger: 4.0,
    surprise: 4.5,
  }[emotionId] ?? 3.0;

  const adjusted = rawValue * sensitivity;
  if (adjusted <= 0) {
    return 0;
  }

  return clamp((adjusted / (adjusted + 1)) * 100, 0, 100);
}

function parseLiveDistributionCsv(csvText) {
  const [headerLine, ...lines] = csvText.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  const distribution = {};

  lines.filter(Boolean).forEach((line) => {
    const values = line.split(",");
    const row = headers.reduce((entry, header, index) => {
      entry[header] = values[index] ?? "";
      return entry;
    }, {});

    distribution[row.blendshape] = {
      mean: Number(row.mean) || 0,
      std: Number(row.std) || 0,
    };
  });

  return distribution;
}

async function ensureLiveAihubDistributions() {
  if (!liveAihubDistributionPromise) {
    liveAihubDistributionPromise = Promise.all(
      Object.entries(LIVE_DISTRIBUTION_PATHS).map(([key, path]) =>
        fetch(path, {
          headers: { Accept: "text/csv" },
        }).then((response) => {
          if (!response.ok) {
            throw new Error(`Could not load ${key} distribution CSV.`);
          }
          return response.text().then((csvText) => [key, parseLiveDistributionCsv(csvText)]);
        })
      )
    )
      .then((entries) => Object.fromEntries(entries))
      .catch((error) => {
        liveAihubDistributionPromise = null;
        throw error;
      });
  }

  return liveAihubDistributionPromise;
}

function getLiveEmotionMeta(id) {
  return EMOTION_META.find((emotion) => emotion.id === id) || EMOTION_META[0];
}

function calculateLiveDistributionDistance(blendshapeMap, distribution, keys, stdFloor) {
  const squaredZValues = [];

  keys.forEach((key) => {
    const stats = distribution[key];
    if (!stats) {
      return;
    }

    const value = Number(blendshapeMap[key] || 0);
    const stdValue = Math.max(Number(stats.std) || 0, stdFloor);
    const zValue = (value - Number(stats.mean || 0)) / stdValue;
    squaredZValues.push(zValue ** 2);
  });

  if (!squaredZValues.length) {
    return null;
  }

  return Math.sqrt(squaredZValues.reduce((sum, value) => sum + value, 0) / squaredZValues.length);
}

function calculateLiveUserNeutralScores(rawBlendshapeMap) {
  if (!appState.neutralBlendshapeBaseline || !appState.neutralBlendshapeStd) {
    return {
      userNeutralDistance: null,
      userNeutralDeltaEnergy: 0,
      userNeutralScore: 0,
    };
  }

  const neutralDistribution = LIVE_USER_NEUTRAL_COMPARE_KEYS.reduce((distribution, key) => {
    distribution[key] = {
      mean: Number(appState.neutralBlendshapeBaseline[key] || 0),
      std: Number(appState.neutralBlendshapeStd[key] || 0),
    };
    return distribution;
  }, {});
  const userNeutralDistance = calculateLiveDistributionDistance(
    rawBlendshapeMap,
    neutralDistribution,
    LIVE_USER_NEUTRAL_COMPARE_KEYS,
    LIVE_USER_STD_FLOOR
  );
  const deltaValues = LIVE_USER_NEUTRAL_COMPARE_KEYS.map((key) =>
    Math.abs(Number(rawBlendshapeMap[key] || 0) - Number(appState.neutralBlendshapeBaseline[key] || 0))
  );
  const userNeutralDeltaEnergy = deltaValues.length
    ? deltaValues.reduce((sum, value) => sum + value, 0) / deltaValues.length
    : 0;

  return {
    userNeutralDistance,
    userNeutralDeltaEnergy,
    userNeutralScore: userNeutralDistance == null ? 0 : 100 / (1 + userNeutralDistance),
  };
}

function calculateLiveAihubScores(rawBlendshapeMap, distributions) {
  if (!distributions?.anger || !distributions?.neutral) {
    return {
      referenceEmotion: "",
      neutralDistance: null,
      angerDistance: null,
      neutralScore: 0,
      angerScore: 0,
    };
  }

  const neutralDistance = calculateLiveDistributionDistance(
    rawBlendshapeMap,
    distributions.neutral,
    LIVE_AIHUB_COMPARE_KEYS,
    LIVE_AIHUB_STD_FLOOR
  );
  const angerDistance = calculateLiveDistributionDistance(
    rawBlendshapeMap,
    distributions.anger,
    LIVE_AIHUB_COMPARE_KEYS,
    LIVE_AIHUB_STD_FLOOR
  );

  if (neutralDistance == null || angerDistance == null) {
    return {
      referenceEmotion: "",
      neutralDistance,
      angerDistance,
      neutralScore: 0,
      angerScore: 0,
    };
  }

  const totalDistance = neutralDistance + angerDistance;
  const neutralScore = totalDistance === 0 ? 50 : (angerDistance / totalDistance) * 100;
  const angerScore = totalDistance === 0 ? 50 : (neutralDistance / totalDistance) * 100;

  return {
    referenceEmotion: angerDistance < neutralDistance ? "Anger" : "Neutral",
    neutralDistance,
    angerDistance,
    neutralScore,
    angerScore,
  };
}

function calculateLiveAngerCore(blendshapeMap) {
  return (
    (blendshapeMap.browDownLeft || 0) +
    (blendshapeMap.browDownRight || 0) +
    (blendshapeMap.eyeSquintLeft || 0) +
    (blendshapeMap.eyeSquintRight || 0) +
    (blendshapeMap.mouthPressLeft || 0) +
    (blendshapeMap.mouthPressRight || 0)
  ) / 6;
}

function getLiveDominantEmotion(scores, aihubDetails, rawBlendshapeMap) {
  const mediapipeDominant = LIVE_EMOTION_META.reduce((best, emotion) => {
    return (scores[emotion.id] || 0) > (scores[best] || 0) ? emotion.id : best;
  }, LIVE_EMOTION_META[0].id);
  const mediapipePercent = scores[mediapipeDominant] || 0;
  const neutralScore = aihubDetails?.neutralScore || 0;
  const angerCore = calculateLiveAngerCore(rawBlendshapeMap || {});
  const mediapipeLabel = getLiveEmotionMeta(mediapipeDominant).label;
  const expressiveAverage =
    LIVE_EMOTION_META.reduce((sum, emotion) => sum + (scores[emotion.id] || 0), 0) / LIVE_EMOTION_META.length;
  const neutralPercent = Math.max(100 - mediapipePercent, neutralScore);

  const lowIntensityNeutral =
    mediapipePercent < LIVE_LOW_INTENSITY_NEUTRAL_THRESHOLD &&
    expressiveAverage < LIVE_LOW_INTENSITY_NEUTRAL_AVERAGE &&
    angerCore < LIVE_LOW_INTENSITY_NEUTRAL_ANGER_CORE &&
    (aihubDetails?.angerScore || 0) < LIVE_LOW_INTENSITY_WEAK_ANGER_SCORE;

  if (lowIntensityNeutral) {
    return {
      id: "neutral",
      label: "Neutral",
      percent: neutralPercent,
      finalSource: "csv_label_aux",
      mediapipeLabel,
      mediapipePercent,
    };
  }

  if (mediapipePercent < LIVE_MEDIAPIPE_NEUTRAL_THRESHOLD) {
    return {
      id: "neutral",
      label: "Neutral",
      percent: neutralPercent,
      finalSource: "csv_label_aux",
      mediapipeLabel,
      mediapipePercent,
    };
  }

  const meta = getLiveEmotionMeta(mediapipeDominant);
  return {
    id: mediapipeDominant,
    label: meta.label,
    percent: mediapipePercent,
    finalSource: "mediapipe",
    mediapipeLabel: meta.label,
    mediapipePercent,
  };
}

function mapBlendshapesToObject(blendshapes = []) {
  const scores = {};
  for (const category of blendshapes) {
    scores[category.categoryName] = category.score;
  }
  return scores;
}

function calculateLiveEmotionScores(rawBlendshapeMap, distributions = null) {
  const emotionBlendshapeMap =
    appState.neutralBlendshapeBaseline && appState.neutralBlendshapeStd
      ? applyLiveNeutralCalibration(rawBlendshapeMap)
      : rawBlendshapeMap;

  const mouthSmileLeft = emotionBlendshapeMap.mouthSmileLeft || 0;
  const mouthSmileRight = emotionBlendshapeMap.mouthSmileRight || 0;
  const mouthFrownLeft = emotionBlendshapeMap.mouthFrownLeft || 0;
  const mouthFrownRight = emotionBlendshapeMap.mouthFrownRight || 0;
  const mouthPressLeft = emotionBlendshapeMap.mouthPressLeft || 0;
  const mouthPressRight = emotionBlendshapeMap.mouthPressRight || 0;
  const browInnerUp = emotionBlendshapeMap.browInnerUp || 0;
  const browDownLeft = emotionBlendshapeMap.browDownLeft || 0;
  const browDownRight = emotionBlendshapeMap.browDownRight || 0;
  const browOuterUpLeft = emotionBlendshapeMap.browOuterUpLeft || 0;
  const browOuterUpRight = emotionBlendshapeMap.browOuterUpRight || 0;
  const eyeSquintLeft = emotionBlendshapeMap.eyeSquintLeft || 0;
  const eyeSquintRight = emotionBlendshapeMap.eyeSquintRight || 0;
  const eyeWideLeft = emotionBlendshapeMap.eyeWideLeft || 0;
  const eyeWideRight = emotionBlendshapeMap.eyeWideRight || 0;
  const noseSneerLeft = emotionBlendshapeMap.noseSneerLeft || 0;
  const noseSneerRight = emotionBlendshapeMap.noseSneerRight || 0;
  const jawOpen = emotionBlendshapeMap.jawOpen || 0;

  const smileAvg = (mouthSmileLeft + mouthSmileRight) / 2;
  const eyeWideAvg = (eyeWideLeft + eyeWideRight) / 2;
  const mouthPressAvg = (mouthPressLeft + mouthPressRight) / 2;
  const browOuterUpAvg = (browOuterUpLeft + browOuterUpRight) / 2;
  const browUpAvg = (browInnerUp + browOuterUpAvg) / 2;

  const joyRaw =
    mouthSmileLeft * 0.45 +
    mouthSmileRight * 0.45 +
    eyeSquintLeft * 0.05 +
    eyeSquintRight * 0.05;

  const sadnessRaw =
    mouthFrownLeft * 0.3 +
    mouthFrownRight * 0.3 +
    browInnerUp * 0.35 +
    mouthPressLeft * 0.025 +
    mouthPressRight * 0.025;

  let angerRaw =
    browDownLeft * 0.23 +
    browDownRight * 0.23 +
    eyeSquintLeft * 0.13 +
    eyeSquintRight * 0.13 +
    noseSneerLeft * 0.1 +
    noseSneerRight * 0.1 +
    mouthPressLeft * 0.04 +
    mouthPressRight * 0.04;

  if (smileAvg < 0.15 && eyeWideAvg > 0.18) {
    angerRaw += eyeWideAvg * 0.35;
  }

  if (mouthPressAvg > 0.15) {
    angerRaw += mouthPressAvg * 0.25;
  }

  if (!(smileAvg < 0.15 && eyeWideAvg > 0.18)) {
    angerRaw *= 1 - jawOpen * 0.4;
  }

  const angerSignal =
    (browDownLeft +
      browDownRight +
      eyeSquintLeft +
      eyeSquintRight +
      noseSneerLeft +
      noseSneerRight +
      mouthPressLeft +
      mouthPressRight) /
    8;

  let surpriseRaw = jawOpen * 0.3 + browUpAvg * 0.3 + eyeWideLeft * 0.2 + eyeWideRight * 0.2;
  surpriseRaw *= 1 - angerSignal * 0.5;

  if (jawOpen > 0.4 && browInnerUp < 0.15 && eyeWideAvg < 0.15) {
    surpriseRaw *= 0.4;
  }

  if (smileAvg < 0.15 && eyeWideAvg > 0.18 && jawOpen < 0.25) {
    surpriseRaw *= 0.5;
  }

  const scores = {
    happiness: Math.round(toPercent(joyRaw, "happiness")),
    sadness: Math.round(toPercent(sadnessRaw, "sadness")),
    anger: Math.round(toPercent(angerRaw, "anger")),
    surprise: Math.round(toPercent(surpriseRaw, "surprise")),
  };
  const aihub = calculateLiveAihubScores(rawBlendshapeMap, distributions);
  const dominant = getLiveDominantEmotion(scores, aihub, emotionBlendshapeMap);

  return {
    rawBlendshapeMap,
    relativeBlendshapeMap: emotionBlendshapeMap,
    raw: {
      happiness: joyRaw,
      sadness: sadnessRaw,
      anger: angerRaw,
      surprise: surpriseRaw,
    },
    scores,
    dominant,
    finalSource: dominant.finalSource,
    mediapipeLabel: dominant.mediapipeLabel,
    mediapipePercent: dominant.mediapipePercent,
    userNeutral: {
      score: 0,
      distance: null,
      deltaEnergy: 0,
    },
    aihub: {
      angerScore: Math.round(aihub.angerScore || 0),
      neutralScore: Math.round(aihub.neutralScore || 0),
      referenceEmotion: aihub.referenceEmotion || "",
      angerDistance: aihub.angerDistance,
      neutralDistance: aihub.neutralDistance,
    },
  };
}

function buildLiveFaceBox(landmarks) {
  if (!landmarks?.length) {
    return null;
  }

  const xs = landmarks.map((landmark) => landmark.x);
  const ys = landmarks.map((landmark) => landmark.y);
  const paddingX = 0.04;
  const paddingY = 0.06;

  return {
    xMin: clamp(Math.min(...xs) - paddingX, 0, 1),
    yMin: clamp(Math.min(...ys) - paddingY, 0, 1),
    xMax: clamp(Math.max(...xs) + paddingX, 0, 1),
    yMax: clamp(Math.max(...ys) + paddingY, 0, 1),
  };
}

function resizeOverlayCanvas(canvas, videoElement) {
  const width = Math.round(videoElement.clientWidth || videoElement.videoWidth || 0);
  const height = Math.round(videoElement.clientHeight || videoElement.videoHeight || 0);

  if (!width || !height) {
    return;
  }

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function getFittedVideoRect(canvas, videoElement, fit = "contain") {
  const sourceWidth = videoElement?.videoWidth || 0;
  const sourceHeight = videoElement?.videoHeight || 0;
  const frameWidth = canvas?.width || 0;
  const frameHeight = canvas?.height || 0;

  if (!sourceWidth || !sourceHeight || !frameWidth || !frameHeight) {
    return null;
  }

  const scaleX = frameWidth / sourceWidth;
  const scaleY = frameHeight / sourceHeight;
  const scale = fit === "cover" ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;

  return {
    x: (frameWidth - width) / 2,
    y: (frameHeight - height) / 2,
    width,
    height,
  };
}

function drawLivePreviewCanvas(canvas, videoElement, analysisResult) {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);

  const renderRect =
    videoElement?.readyState >= 2 && canvas.width > 0 && canvas.height > 0
      ? getFittedVideoRect(canvas, videoElement, "contain")
      : null;

  if (renderRect) {
    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(
      videoElement,
      canvas.width - renderRect.x - renderRect.width,
      renderRect.y,
      renderRect.width,
      renderRect.height
    );
    context.restore();
  }

  if (!analysisResult?.box || !analysisResult?.dominant) {
    return;
  }

  const { box, dominant } = analysisResult;
  if (dominant.id === "neutral") {
    return;
  }
  const color = LIVE_EMOTION_COLOR[dominant.id] || LIVE_EMOTION_COLOR.neutral;
  const frameX = renderRect?.x ?? 0;
  const frameY = renderRect?.y ?? 0;
  const frameWidth = renderRect?.width ?? canvas.width;
  const frameHeight = renderRect?.height ?? canvas.height;
  const x = frameX + (1 - box.xMax) * frameWidth;
  const y = frameY + box.yMin * frameHeight;
  const width = (box.xMax - box.xMin) * frameWidth;
  const height = (box.yMax - box.yMin) * frameHeight;
  const label = dominant.id === "neutral" ? "Neutral" : `${dominant.label} ${Math.round(dominant.percent)}%`;

  context.strokeStyle = color;
  context.lineWidth = 4;
  context.strokeRect(x, y, width, height);

  context.font = "600 18px Avenir Next, Segoe UI, sans-serif";
  const textWidth = context.measureText(label).width;
  const labelHeight = 34;
  const labelX = clamp(x, 8, Math.max(8, canvas.width - textWidth - 26));
  const labelY = Math.max(8, y - labelHeight - 10);

  context.fillStyle = color;
  context.fillRect(labelX, labelY, textWidth + 18, labelHeight);

  context.fillStyle = "#07111f";
  context.fillText(label, labelX + 9, labelY + 22);
}

function renderLiveEmotionHud(target, analysisResult, status = "") {
  if (!target) {
    return;
  }

  if (status) {
    target.innerHTML = `
      <div class="emotion-hud-card is-loading">
        <strong>${escapeHtml(status)}</strong>
        <span>Preparing live emotion tracking...</span>
      </div>
    `;
    return;
  }

  if (!analysisResult) {
    target.innerHTML = `
      <div class="emotion-hud-card">
        <strong>Waiting for face</strong>
        <span>Center your face in the guide to start live emotion tracking.</span>
      </div>
    `;
    return;
  }

  const dominantLabel =
    analysisResult.dominant.id === "neutral"
      ? "Neutral"
      : `${analysisResult.dominant.label} ${Math.round(analysisResult.dominant.percent)}%`;
  const sourceLabel = getFinalSourceLabel(analysisResult.finalSource);

  target.innerHTML = `
    <div class="emotion-hud-card">
      <div class="emotion-hud-header">
        <span>Live read · ${escapeHtml(sourceLabel)}</span>
        <strong>${escapeHtml(dominantLabel)}</strong>
      </div>
      <div class="emotion-hud-bars">
        ${LIVE_EMOTION_META.map((emotion) => {
          const value = Math.round(analysisResult.scores[emotion.id] || 0);
          return `
            <div class="emotion-hud-row">
              <div class="emotion-hud-label">
                <span>${emotion.label}</span>
                <span>${value}%</span>
              </div>
              <div class="emotion-hud-meter">
                <div class="emotion-hud-fill" style="--emotion-color:${emotion.color}; width:${value}%;"></div>
              </div>
            </div>
          `;
        }).join("")}
      </div>
      <div class="emotion-hud-footnote">
        ${
          analysisResult.mediapipeLabel
            ? `MediaPipe: ${escapeHtml(analysisResult.mediapipeLabel)} ${Math.round(analysisResult.mediapipePercent || 0)}%`
            : "Live preview is following the strengthened backend emotion rules."
        }
      </div>
    </div>
  `;
}

async function ensureLiveFaceLandmarker() {
  if (liveFaceLandmarker) {
    return liveFaceLandmarker;
  }

  if (!liveFaceLandmarkerPromise) {
    liveFaceLandmarkerPromise = (async () => {
      const visionTasks = await import(MEDIAPIPE_VISION_BUNDLE_URL);
      const resolver = await visionTasks.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);

      try {
        liveFaceLandmarker = await visionTasks.FaceLandmarker.createFromOptions(resolver, {
          baseOptions: {
            modelAssetPath: LIVE_MODEL_PATH,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch {
        liveFaceLandmarker = await visionTasks.FaceLandmarker.createFromOptions(resolver, {
          baseOptions: {
            modelAssetPath: LIVE_MODEL_PATH,
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      }

      appState.liveTrackingAvailable = true;
      appState.liveTrackingMessage = "";
      return liveFaceLandmarker;
    })().catch((error) => {
      appState.liveTrackingAvailable = false;
      appState.liveTrackingMessage = "Live face tracking failed to load.";
      liveFaceLandmarkerPromise = null;
      throw error;
    });
  }

  return liveFaceLandmarkerPromise;
}

function stopLiveEmotionTracking() {
  liveAnalysisSession += 1;
  if (liveAnalysisFrameId) {
    cancelAnimationFrame(liveAnalysisFrameId);
    liveAnalysisFrameId = 0;
  }
}

function updateNeutralCalibrationFromAnalysis(analysisResult, timestampMs) {
  if (!appState.neutralCalibrationActive) {
    return;
  }

  const hasFace = Boolean(analysisResult?.box);
  const rawBlendshapeMap = analysisResult?.rawBlendshapeMap;

  if (!hasFace || !rawBlendshapeMap) {
    appState.neutralCalibrationProgressMs = 0;
    appState.neutralCalibrationLastTickAt = timestampMs;
    appState.neutralCalibrationSamples = [];
    renderNeutralCalibrationStatus();
    updateRecordingStatusMessage();
    syncRecordingControls();
    return;
  }

  appState.neutralCalibrationSamples.push(rawBlendshapeMap);

  if (!appState.neutralCalibrationLastTickAt) {
    appState.neutralCalibrationLastTickAt = timestampMs;
  } else {
    const deltaMs = Math.max(0, timestampMs - appState.neutralCalibrationLastTickAt);
    appState.neutralCalibrationProgressMs = clamp(
      appState.neutralCalibrationProgressMs + deltaMs,
      0,
      appState.neutralCalibrationTargetMs
    );
    appState.neutralCalibrationLastTickAt = timestampMs;
  }

  renderNeutralCalibrationStatus();
  updateRecordingStatusMessage();
  syncRecordingControls();

  if (appState.neutralCalibrationProgressMs >= appState.neutralCalibrationTargetMs) {
    if (appState.neutralCalibrationSamples.length >= 10) {
      const profile = buildLiveNeutralProfile(appState.neutralCalibrationSamples);
      appState.neutralBlendshapeBaseline = profile.baseline;
      appState.neutralBlendshapeStd = profile.std;
      appState.liveTrackingMessage = "Neutral calibration ready. Live preview is now tuned to your resting face.";
    } else {
      startNeutralCalibration();
      renderNeutralCalibrationStatus();
      updateRecordingStatusMessage();
      syncRecordingControls();
      return;
    }

    appState.neutralCalibrationActive = false;
    appState.neutralCalibrationReady = true;
    appState.neutralCalibrationLastTickAt = 0;
    appState.neutralCalibrationSamples = [];
    renderNeutralCalibrationStatus();
    updateRecordingStatusMessage();
    syncRecordingControls();
  }
}

async function startLiveEmotionTracking(videoElement, canvasElement, hudElement) {
  stopLiveEmotionTracking();
  const sessionId = liveAnalysisSession;

  renderLiveEmotionHud(hudElement, null, "Loading MediaPipe");

  try {
    const [faceLandmarker, distributions] = await Promise.all([
      ensureLiveFaceLandmarker(),
      ensureLiveAihubDistributions().catch((error) => {
        console.warn(error);
        return null;
      }),
    ]);
    if (sessionId !== liveAnalysisSession) {
      return;
    }

    appState.liveTrackingMessage = distributions
      ? "Overlay ready with the AI-Hub anger/neutral CSV distribution standard."
      : "Overlay ready in MediaPipe-only preview mode. Final backend scoring still uses the AI-Hub CSV pipeline.";

    renderNeutralCalibrationStatus();
    updateRecordingStatusMessage();
    syncRecordingControls();

    let lastVideoTime = -1;

    const renderFrame = () => {
      if (sessionId !== liveAnalysisSession) {
        return;
      }

      if (videoElement.readyState < 2) {
        liveAnalysisFrameId = requestAnimationFrame(renderFrame);
        return;
      }

      resizeOverlayCanvas(canvasElement, videoElement);

      if (videoElement.currentTime !== lastVideoTime) {
        const timestampMs = performance.now();
        const result = faceLandmarker.detectForVideo(videoElement, timestampMs);
        lastVideoTime = videoElement.currentTime;

        if (result?.faceLandmarks?.length && result?.faceBlendshapes?.length) {
          const rawBlendshapeMap = mapBlendshapesToObject(result.faceBlendshapes[0].categories);
          const liveAnalysis = calculateLiveEmotionScores(rawBlendshapeMap, distributions);
          const box = buildLiveFaceBox(result.faceLandmarks[0]);
          const payload = {
            ...liveAnalysis,
            box,
          };
          drawLivePreviewCanvas(canvasElement, videoElement, payload);
          updateNeutralCalibrationFromAnalysis(payload, timestampMs);
          if (appState.neutralCalibrationActive) {
            renderLiveEmotionHud(hudElement, null, `Calibrating neutral face ${getLiveCalibrationProgressLabel()}`);
          } else {
            renderLiveEmotionHud(hudElement, payload);
          }
        } else {
          drawLivePreviewCanvas(canvasElement, videoElement, null);
          updateNeutralCalibrationFromAnalysis(null, timestampMs);
          renderLiveEmotionHud(
            hudElement,
            null,
            appState.neutralCalibrationActive ? "Center your face to finish neutral calibration." : ""
          );
        }
      } else {
        drawLivePreviewCanvas(canvasElement, videoElement, null);
      }

      liveAnalysisFrameId = requestAnimationFrame(renderFrame);
    };

    renderFrame();
  } catch (error) {
    console.error(error);
    appState.liveTrackingAvailable = false;
    appState.liveTrackingMessage = "Live face tracking failed to load.";
    renderLiveEmotionHud(hudElement, null, "Live tracking unavailable");
    hudElement.insertAdjacentHTML(
      "beforeend",
      `<div class="emotion-hud-footnote">Could not load the browser-side face tracker. Backend recording still works after stop.</div>`
    );
    renderNeutralCalibrationStatus();
    updateRecordingStatusMessage();
    syncRecordingControls();
  }
}

function routeHead(kicker, title, subtitle, rightAction = "") {
  return `
    <div class="route-head">
      <div class="brand-mark">
        <div class="brand-badge">🎭</div>
        <div class="brand-text">
          <strong>${kicker}</strong>
          <span>${subtitle}</span>
        </div>
      </div>
      ${rightAction}
    </div>
    <h1 class="section-title">${title}</h1>
  `;
}

function renderWorkflowStepper(activeStepId, caption = "") {
  const activeIndex = WORKFLOW_STEPS.findIndex((step) => step.id === activeStepId);

  return `
    <div class="workflow-strip">
      <div class="workflow-grid">
        ${WORKFLOW_STEPS.map((step, index) => {
          const stateClass = index < activeIndex ? "is-complete" : index === activeIndex ? "is-active" : "";
          return `
            <div class="workflow-step ${stateClass}">
              <div class="workflow-step-index">${index + 1}</div>
              <strong>${step.label}</strong>
            </div>
          `;
        }).join("")}
      </div>
      ${caption ? `<p class="workflow-caption">${escapeHtml(caption)}</p>` : ""}
    </div>
  `;
}

function buildSupportSummary(contextLabel = "") {
  const referenceLabel = appState.isDemoReference
    ? "Demo reference"
    : appState.referenceFile?.name || "No reference file selected";
  const csvPreviewLabel = appState.liveTrackingAvailable ? "Ready" : appState.liveTrackingMessage || "Not loaded";

  return [
    "HCI Acting Coach support summary",
    `Context: ${contextLabel || "general"}`,
    `Session: ${appState.sessionId}`,
    `Route: ${window.location.pathname}`,
    `Reference: ${referenceLabel}`,
    `Analysis mode: ${appState.analysisMode || "Pending"}`,
    `Last take source: ${getTakeSourceLabel(appState.lastTakeSource)}`,
    `Camera active: ${appState.recordingStream ? "Yes" : "No"}`,
    `Live CSV preview: ${csvPreviewLabel}`,
    `Live overlay: ${appState.liveTrackingAvailable ? "Ready" : appState.liveTrackingMessage || "Unavailable"}`,
    `Analysis error: ${appState.analysisError || "None"}`,
    `Take error: ${appState.takeAnalysisError || "None"}`,
    "",
    "What I tried:",
    "",
    "What happened instead:",
    "",
  ].join("\n");
}

async function copySupportSummary(button, contextLabel = "") {
  const summary = buildSupportSummary(contextLabel);
  const status = button?.closest(".support-card, .error-card")?.querySelector("[data-support-status]");

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(summary);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = summary;
      textarea.setAttribute("readonly", "true");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }

    if (status) {
      status.textContent = "Copied. Paste this into your team chat or GitHub issue with a screenshot.";
    }
  } catch (error) {
    if (status) {
      status.textContent = "Clipboard copy failed. You can still open a GitHub issue and describe the problem there.";
    }
  }
}

function renderSupportCard({ context, title, intro, steps, detailTitle = "Need help or want to report an issue?" }) {
  return `
    <div class="panel-card support-card">
      <div class="small-label">Quick guide</div>
      <div class="divider"></div>
      <strong class="support-title">${escapeHtml(title)}</strong>
      <p class="support-copy">${escapeHtml(intro)}</p>
      <div class="guide-list">
        ${steps
          .map(
            (step, index) => `
              <div class="guide-step">
                <span class="guide-step-index">${index + 1}</span>
                <div>
                  <strong>${escapeHtml(step.title)}</strong>
                  <div class="small-label">${escapeHtml(step.copy)}</div>
                </div>
              </div>
            `
          )
          .join("")}
      </div>
      <details class="support-details">
        <summary>${escapeHtml(detailTitle)}</summary>
        <div class="support-details-body">
          <p class="support-copy">Open the project issue tracker or copy a ready-to-send help summary for your team.</p>
          <div class="support-note-list">
            <div class="support-note">Include a screenshot of the current screen.</div>
            <div class="support-note">Mention which step you were on when the issue happened.</div>
            <div class="support-note">If the camera or analysis failed, include the copied support summary.</div>
          </div>
          <div class="hero-actions" style="margin-top: 1rem;">
            <button class="button button-secondary" data-action="copy-support-summary" data-context="${escapeHtml(context)}">
              Copy help summary
            </button>
            <a class="button button-ghost" href="${SUPPORT_ISSUE_URL}" target="_blank" rel="noreferrer">
              Open GitHub issue
            </a>
          </div>
          <p class="support-status" data-support-status></p>
        </div>
      </details>
    </div>
  `;
}

function attachSupportActions() {
  document.querySelectorAll('[data-action="copy-support-summary"]').forEach((button) => {
    button.addEventListener("click", () => {
      copySupportSummary(button, button.dataset.context || "");
    });
  });
}

function renderRecordingControlGuide() {
  const alternateTitle = isLocalDesktopWebcamAvailable() ? "Use an alternate path if needed" : "Use the sample take if needed";
  const alternateCopy = isLocalDesktopWebcamAvailable()
    ? "Choose the Python webcam model or skip to the bundled sample comparison."
    : "If the browser camera path is blocked, skip to the bundled sample comparison.";

  return `
    <div class="control-guide-grid">
      <div class="control-guide-card">
        <div class="small-label">Enable camera</div>
        <strong>Open the browser rehearsal preview</strong>
        <p>Starts face tracking with the same AI-Hub anger/neutral CSV reference used by backend analysis.</p>
      </div>
      <div class="control-guide-card">
        <div class="small-label">Start recording</div>
        <strong>Begin the synchronized take</strong>
        <p>Runs the countdown, then starts the actor clip and your recording together.</p>
      </div>
      <div class="control-guide-card">
        <div class="small-label">Stop and analyze</div>
        <strong>Send the take to the backend</strong>
        <p>Opens the comparison results automatically when backend scoring finishes.</p>
      </div>
      <div class="control-guide-card">
        <div class="small-label">Desktop webcam / sample take</div>
        <strong>${alternateTitle}</strong>
        <p>${alternateCopy}</p>
      </div>
    </div>
  `;
}

function renderUploadPage() {
  const file = appState.referenceFile;
  const meta = appState.referenceMeta;

  app.innerHTML = `
    <section class="screen">
      <div class="hero-card upload-focus">
        ${renderWorkflowStepper("upload")}
        <div class="upload-layout">
          <div class="upload-zone ${appState.isDragging ? "is-dragging" : ""}" id="upload-zone">
            <input id="reference-input" type="file" accept="video/*" />
            <div>
              <div class="upload-icon">⬆</div>
              <h2>Upload actor video</h2>
              <p class="muted-copy">Drop a clear face video here, or click to choose one.</p>
              ${
                file
                  ? `<div class="upload-meta" style="justify-content:center;">
                      <span class="meta-chip">Ready for analysis</span>
                      <span class="meta-chip">${escapeHtml(file.type || "video")}</span>
                    </div>`
                  : ""
              }
            </div>
          </div>

          <div class="upload-side">
            <span class="eyebrow">Acting Coach</span>
            <h1 class="hero-title">Choose a reference. Then rehearse.</h1>
            <p class="hero-copy">
              Upload one actor clip, analyze it, and record your take against the same emotion profile.
            </p>
            <div class="hero-actions" style="margin-top: 1.3rem;">
              <button class="button button-primary" data-action="go-analysis" ${file ? "" : "disabled"}>
                Analyze reference
              </button>
              <button class="button button-secondary" data-action="load-demo">Use demo</button>
            </div>

            <div class="reference-summary">
              <div class="small-label">Current reference</div>
              ${
                file
                  ? `
                    <strong>${escapeHtml(file.name)}</strong>
                    <div class="upload-meta">
                      <span class="meta-chip">${formatFileSize(file.size)}</span>
                      <span class="meta-chip">${meta ? formatDuration(meta.duration) : "Reading duration..."}</span>
                      <span class="meta-chip">${meta && meta.width ? `${meta.width}x${meta.height}` : "Video metadata pending"}</span>
                    </div>
                  `
                  : `<p class="muted-copy">No file selected yet.</p>`
              }
            </div>

            <div class="video-shell video-shell-compact">
              ${
                file && appState.referenceUrl
                  ? `<video src="${appState.referenceUrl}" controls preload="metadata"></video>`
                  : `<div class="empty-video">
                      <div>
                        <strong style="display:block; font-size: 1.05rem; margin-bottom: 0.4rem;">${appState.isDemoReference ? "Demo loaded" : "Preview"}</strong>
                        <span>${appState.isDemoReference ? "Ready to analyze the demo reference." : "Your selected clip will appear here."}</span>
                      </div>
                    </div>`
              }
            </div>
          </div>
        </div>
      </div>
    </section>
  `;

  const zone = document.querySelector("#upload-zone");
  const input = document.querySelector("#reference-input");

  zone.addEventListener("dragenter", onDragState(true));
  zone.addEventListener("dragover", onDragState(true));
  zone.addEventListener("dragleave", onDragState(false));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    appState.isDragging = false;
    const [droppedFile] = event.dataTransfer?.files || [];
    if (droppedFile) {
      setReferenceFile(droppedFile);
    } else {
      renderRoute();
    }
  });

  input.addEventListener("change", (event) => {
    const [selectedFile] = event.target.files || [];
    if (selectedFile) setReferenceFile(selectedFile);
  });

  document.querySelector('[data-action="go-analysis"]')?.addEventListener("click", () => {
    navigate("/analysis");
  });

  document.querySelector('[data-action="load-demo"]')?.addEventListener("click", async () => {
    loadDemoReference();
  });
  attachSupportActions();
}

function onDragState(isDragging) {
  return (event) => {
    event.preventDefault();
    if (appState.isDragging !== isDragging) {
      appState.isDragging = isDragging;
      renderRoute();
    }
  };
}

function renderAnalysisPage() {
  if (!appState.referenceFile) {
    navigate("/");
    return;
  }

  app.innerHTML = `
    <section class="screen">
      <div class="panel-card">
        ${routeHead(
          "Reference Analysis",
          "Analyzing reference",
          appState.isDemoReference ? "Loading demo profile" : "Running video inference",
          '<button class="button button-ghost" data-action="back-home">Choose another clip</button>'
        )}
        ${renderWorkflowStepper("analysis", "Recording opens automatically when analysis finishes.")}
        <div class="analysis-grid" style="margin-top: 1.5rem;">
          <div class="panel-card" style="background: rgba(7, 17, 31, 0.52);">
            <div class="video-shell">
              ${
                appState.referenceUrl
                  ? `<video src="${appState.referenceUrl}" controls preload="metadata"></video>`
                  : `<div class="empty-video">
                      <div>
                        <strong style="display:block; font-size: 1.05rem; margin-bottom: 0.5rem;">Demo reference profile</strong>
                        <span>This run uses the merged HCI-acting-coach reference signature so you can test the full workflow without an upload.</span>
                      </div>
                    </div>`
              }
            </div>
            <p class="video-helper-note">Preview only. Playback does not change the analysis result.</p>
            <div class="progress-copy">
              <strong id="analysis-stage">${ANALYSIS_STEPS[0].label}</strong>
              <span id="analysis-progress-value">0%</span>
            </div>
            <div class="progress-track" style="margin-top: 0.6rem;">
              <div id="analysis-progress-bar" class="progress-bar" style="width: 0%;"></div>
            </div>
            <p id="analysis-stage-detail" class="muted-copy" style="margin-top: 0.9rem;">
              ${ANALYSIS_STEPS[0].detail}
            </p>
          </div>
          <div class="panel-card">
            <div class="small-label">Pipeline status</div>
            <div class="divider"></div>
            <div id="analysis-timeline" class="timeline"></div>
          </div>
        </div>
      </div>
    </section>
  `;

  document.querySelector('[data-action="back-home"]').addEventListener("click", () => navigate("/"));

  const stageEl = document.querySelector("#analysis-stage");
  const detailEl = document.querySelector("#analysis-stage-detail");
  const barEl = document.querySelector("#analysis-progress-bar");
  const percentEl = document.querySelector("#analysis-progress-value");
  const timelineEl = document.querySelector("#analysis-timeline");
  let progress = 0;
  let routeIsActive = true;
  let analysisComplete = false;
  let analysisResult = null;
  let analysisError = false;

  const renderTimeline = () => {
    const activeStep = Math.min(ANALYSIS_STEPS.length - 1, Math.floor(progress / 26));
    timelineEl.innerHTML = ANALYSIS_STEPS.map((step, index) => {
      const stateClass = index < activeStep ? "is-complete" : index === activeStep ? "is-active" : "";
      return `
        <div class="timeline-step ${stateClass}">
          <div class="timeline-dot"></div>
          <div>
            <strong>${step.label}</strong>
            <div class="small-label">${step.detail}</div>
          </div>
          <div class="timeline-status">${index < activeStep ? "Done" : index === activeStep ? "Running" : "Queued"}</div>
        </div>
      `;
    }).join("");
  };

  renderTimeline();

  const finalizeAnalysis = () => {
    if (!routeIsActive || analysisComplete || progress < 100) {
      return;
    }

    if (analysisError) {
      analysisComplete = true;
      appState.analysisError = typeof analysisError === "string" ? analysisError : "The HCI backend did not return a usable reference profile. Try another clip or use the demo flow.";
      navigate("/analysis-error");
      return;
    }

    if (!analysisResult) {
      return;
    }

    analysisComplete = true;
    applyReferenceProfile(analysisResult);
    navigate("/recording");
  };

  const profilePromise = appState.isDemoReference
    ? fetchReferenceProfile({ demo: true })
    : analyzeReferenceVideo(appState.referenceFile);

  profilePromise
    .then((profile) => {
      if (!routeIsActive) {
        return;
      }
      analysisResult = profile;
      finalizeAnalysis();
    })
    .catch((error) => {
      if (!routeIsActive) {
        return;
      }
      analysisError = error?.message || true;
      finalizeAnalysis();
    });

  const interval = window.setInterval(() => {
    if (progress < 100) {
      progress = Math.min(100, progress + Math.ceil(Math.random() * 13));
    }

    const activeStep = Math.min(ANALYSIS_STEPS.length - 1, Math.floor(progress / 26));

    if (progress >= 100 && !analysisResult && !analysisError) {
      stageEl.textContent = "Finalizing backend payload";
      detailEl.textContent =
        "The analysis service is packaging the analyzed reference profile for the rehearsal flow.";
    } else {
      stageEl.textContent = ANALYSIS_STEPS[activeStep].label;
      detailEl.textContent = ANALYSIS_STEPS[activeStep].detail;
    }

    barEl.style.width = `${progress}%`;
    percentEl.textContent = `${progress}%`;
    renderTimeline();
    finalizeAnalysis();

    if (analysisComplete) {
      window.clearInterval(interval);
    }
  }, 520);

  appState.routeCleanup = () => {
    routeIsActive = false;
    window.clearInterval(interval);
  };
}

function renderAnalysisErrorPage() {
  app.innerHTML = `
    <section class="screen">
      <div class="panel-card error-card">
        ${routeHead(
          "Analysis Error",
          "We couldn't complete the reference analysis.",
          "Try another clip or restart the workflow"
        )}
        ${renderWorkflowStepper("analysis", "You are still on the analysis step. Fix the issue, then continue to recording.")}
        <p class="hero-copy" style="margin-top: 0.8rem;">
          ${escapeHtml(appState.analysisError || "The emotion extraction pipeline was interrupted before a usable profile was produced.")}
        </p>
        <div class="summary-list" style="margin-top: 1.2rem;">
          <div class="summary-item">
            <strong>Try a simpler clip first.</strong>
            <p class="feedback-body">A short MP4, MOV, or browser-recorded clip with one clearly visible face tends to work best.</p>
          </div>
          <div class="summary-item">
            <strong>If the problem repeats, send a support summary.</strong>
            <p class="feedback-body">Use the support action below so your team can see which step failed.</p>
          </div>
        </div>
        <div class="hero-actions" style="margin-top: 1.3rem;">
          <button class="button button-primary" data-action="retry-home">Back to upload</button>
          <button class="button button-secondary" data-action="retry-analysis" ${appState.referenceFile ? "" : "disabled"}>
            Retry analysis
          </button>
          <button class="button button-ghost" data-action="copy-support-summary" data-context="analysis-error">Copy help summary</button>
          <a class="button button-ghost" href="${SUPPORT_ISSUE_URL}" target="_blank" rel="noreferrer">Open GitHub issue</a>
        </div>
        <p class="support-status" data-support-status style="margin-top: 0.8rem;"></p>
      </div>
    </section>
  `;

  document.querySelector('[data-action="retry-home"]').addEventListener("click", () => navigate("/"));
  document.querySelector('[data-action="retry-analysis"]').addEventListener("click", () => navigate("/analysis"));
  attachSupportActions();
}

async function prepareCamera(videoElement) {
  if (!navigator.mediaDevices?.getUserMedia) {
    appState.cameraError = isLocalDesktopWebcamAvailable()
      ? "This browser does not expose a camera API. You can still upload a recorded take or use the desktop webcam model."
      : "This browser does not expose a camera API. You can still upload a recorded take or use the bundled sample take.";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    appState.recordingStream = stream;
    videoElement.srcObject = stream;
    await videoElement.play();
    appState.cameraError = "";
    resetNeutralCalibration();
    appState.liveTrackingMessage = "Camera enabled. Live analysis uses the AI-Hub anger/neutral CSV distribution standard.";
  } catch (error) {
    appState.cameraError = isLocalDesktopWebcamAvailable()
      ? "Camera access was denied or unavailable. You can still run the desktop webcam model or use the bundled sample take."
      : "Camera access was denied or unavailable. You can still upload a recorded take or use the bundled sample take.";
  }
}

function stopCamera() {
  appState.recordingStream?.getTracks().forEach((track) => track.stop());
  appState.recordingStream = null;
}

function beginCameraRecording() {
  if (appState.takeAnalysisInFlight) {
    return;
  }

  if (!appState.recordingStream || !window.MediaRecorder) {
    return;
  }

  appState.recordingChunks = [];
  appState.recordingStartedAt = performance.now();
  appState.lastTakeSource = "camera";
  setTakeAnalysisState(false);
  syncRecordingControls();
  updateRecordingStatusMessage();
  renderNeutralCalibrationStatus();

  const recorder = new MediaRecorder(appState.recordingStream, {
    mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm",
  });

  recorder.addEventListener("dataavailable", (event) => {
    if (event.data?.size) appState.recordingChunks.push(event.data);
  });

  recorder.addEventListener("stop", () => {
    const blob = new Blob(appState.recordingChunks, { type: recorder.mimeType || "video/webm" });
    if (appState.recordedUrl) {
      URL.revokeObjectURL(appState.recordedUrl);
    }

    appState.recordedBlob = blob;
    appState.recordedUrl = URL.createObjectURL(blob);
    appState.recordingDuration = Math.max(1, (performance.now() - appState.recordingStartedAt) / 1000);
    resetNeutralCalibration();
    setTakeAnalysisState(true, "Uploading your recorded take and running HCI video analysis on the server.");
    renderRoute();

    analyzeRecordedTake(blob)
      .then((payload) => {
        setTakeAnalysisState(false);
        applyComparisonPayload(payload, "recorded-video");
        navigate("/results");
      })
      .catch((error) => {
        setTakeAnalysisState(false, "", error?.message || "The recorded take could not be analyzed.");
        renderRoute();
      });
  });

  recorder.start();
  appState.mediaRecorder = recorder;
  syncRecordingControls();
  updateRecordingStatusMessage();
  renderNeutralCalibrationStatus();
}

function startRecording() {
  if (appState.takeAnalysisInFlight) {
    return;
  }

  if (!appState.recordingStream || !window.MediaRecorder) {
    appState.cameraError = isLocalDesktopWebcamAvailable()
      ? "Recording is not supported in this browser. Try the desktop webcam model instead."
      : "Recording is not supported in this browser. Use the bundled sample take instead.";
    renderRoute();
    return;
  }

  if (appState.mediaRecorder && appState.mediaRecorder.state === "recording") {
    return;
  }

  startRecordingCountdown();
}

function stopRecordingAndAnalyze() {
  clearRecordingCountdown();
  appState.synchronizedReferencePlayback = false;
  const referenceVideo = document.querySelector("#reference-preview");
  if (referenceVideo instanceof HTMLVideoElement) {
    referenceVideo.pause();
  }
  if (appState.mediaRecorder && appState.mediaRecorder.state !== "inactive") {
    setTakeAnalysisState(true, "Finishing the recording before sending it to the backend.");
    renderRoute();
    appState.mediaRecorder.stop();
  }
}

async function runFallbackTake() {
  try {
    setTakeAnalysisState(true, "Loading the bundled HCI sample take for comparison.");
    renderRoute();
    const payload = await fetchSampleComparison();
    setTakeAnalysisState(false);
    applyComparisonPayload(payload, "sample-model");
    navigate("/results");
  } catch (error) {
    setTakeAnalysisState(false, "", error?.message || "The sample take could not be loaded.");
    renderRoute();
  }
}

async function runDesktopWebcamTake() {
  if (!isLocalDesktopWebcamAvailable()) {
    appState.takeAnalysisError = "The desktop webcam model is available only while running the app locally.";
    renderRoute();
    return;
  }

  stopCamera();
  appState.lastTakeSource = "desktop-webcam";
  setTakeAnalysisState(
    true,
    "A desktop OpenCV webcam window should appear. Perform the take there, then press q in that window to finish."
  );
  renderRoute();

  try {
    const payload = await runDesktopWebcamAnalysis();
    setTakeAnalysisState(false);
    applyComparisonPayload(payload, "desktop-webcam");
    navigate("/results");
  } catch (error) {
    setTakeAnalysisState(false, "", error?.message || "The desktop webcam analysis did not finish successfully.");
    renderRoute();
  }
}

function renderRecordingPage() {
  if (!appState.actorProfile) {
    navigate("/");
    return;
  }

  const desktopWebcamAvailable = isLocalDesktopWebcamAvailable();
  const activeRecorder = appState.mediaRecorder && appState.mediaRecorder.state === "recording";
  const expressiveEmotions = [...appState.actorProfile.emotions]
    .filter((emotion) => emotion.id !== "neutral")
    .sort((a, b) => b.actor - a.actor);
  const leadEmotion = getProfileDominantEmotion(appState.actorProfile, "actor") || expressiveEmotions[0] || appState.actorProfile.topEmotion;
  const supportEmotion = expressiveEmotions[1] || appState.actorProfile.secondaryEmotion || leadEmotion;
  const topTargets = expressiveEmotions.slice(0, 3);
  const leadEmotionMeta = appState.actorProfile.dominantSummary
    ? `${appState.actorProfile.dominantSummary.share}% of analyzed frames`
    : `${leadEmotion.actor}% reference weight`;

  app.innerHTML = `
    <section class="screen recording-screen">
      <div class="panel-card recording-shell">
        ${routeHead(
          "Record Your Take",
          "Record your take",
          `${appState.analysisFrameCount} reference frames are ready`,
          '<button class="button button-ghost" data-action="back-analysis">Re-run analysis</button>'
        )}
        ${renderWorkflowStepper("recording", "Enable camera, record, then analyze.")}

        <div class="recording-toolbar">
          <div class="toolbar-copy">
            <div class="small-label">Controls</div>
            <strong>Enable camera > Record > Analyze</strong>
          </div>
          <div class="inline-actions recording-actions recording-actions-primary">
            <button class="button button-secondary" data-action="enable-camera" ${
              appState.takeAnalysisInFlight ? "disabled" : ""
            }>
              ${appState.recordingStream ? "Refresh camera" : "Enable camera"}
            </button>
            <button class="button button-primary" data-action="start-recording" ${
              appState.recordingStream &&
              !appState.takeAnalysisInFlight &&
              !activeRecorder
                ? ""
                : "disabled"
            }>
              ${activeRecorder ? "Recording..." : "Start recording"}
            </button>
            <button class="button ${activeRecorder ? "button-primary" : "button-secondary"}" data-action="stop-recording" ${
              activeRecorder && !appState.takeAnalysisInFlight ? "" : "disabled"
            }>
              Stop and analyze
            </button>
          </div>
          <details class="secondary-actions">
            <summary>More options</summary>
            <div class="inline-actions">
            ${
              desktopWebcamAvailable
                ? `<button class="button button-secondary" data-action="desktop-webcam" ${
                    appState.takeAnalysisInFlight ? "disabled" : ""
                  }>
              Run desktop webcam model
            </button>`
                : ""
            }
            <button class="button button-ghost" data-action="fallback-recording" ${
              appState.takeAnalysisInFlight ? "disabled" : ""
            }>
              Use sample take
            </button>
            </div>
          </details>
        </div>

        <div class="rehearsal-stage-shell">
          <div class="rehearsal-stage-header">
            <div>
              <div class="small-label">Reference rehearsal mode</div>
              <strong>Actor guide on the left, your live response on the right.</strong>
            </div>
            <div class="stage-header-pills">
              <span class="meta-chip">${appState.analysisFrameCount} frames mapped</span>
              <span class="meta-chip">${leadEmotion.label} lead</span>
              <span class="meta-chip">${supportEmotion.label} support</span>
            </div>
          </div>

          <div class="rehearsal-stage">
            <article class="rehearsal-pane">
              <div class="pane-header">
                <div class="pane-heading">
                  <div class="pane-icon">🎬</div>
                  <div class="pane-title">
                    <div class="small-label">Reference</div>
                    <strong>Actor clip</strong>
                  </div>
                </div>
                <span class="meta-chip">${appState.isDemoReference ? "Demo source" : "Your upload"}</span>
              </div>
              <div class="reference-frame">
                ${
                  appState.referenceUrl
                    ? `<video id="reference-preview" src="${appState.referenceUrl}" controls playsinline preload="auto" muted></video>`
                    : `<div class="empty-video">
                        <div>
                          <strong style="display:block; font-size: 1.05rem; margin-bottom: 0.5rem;">Demo reference profile</strong>
                          <span>This path uses the bundled HCI demo reference because no uploaded clip URL is available.</span>
                        </div>
                      </div>`
                }
                <div class="reference-overlay">
                  <div class="frame-countdown-overlay hidden" data-countdown-overlay></div>
                </div>
              </div>
              <div class="pane-analysis-stack">
                <div id="reference-analysis-hud" class="emotion-hud emotion-hud-inline emotion-hud-reference"></div>
              </div>
            </article>

            <article class="rehearsal-pane rehearsal-pane-live">
              <div class="pane-header">
                <div class="pane-heading">
                  <div class="pane-icon pane-icon-live">📷</div>
                  <div class="pane-title">
                    <div class="small-label">Your take</div>
                    <strong>Live camera</strong>
                  </div>
                </div>
                <span class="meta-chip">Live preview</span>
              </div>
              <div class="camera-frame">
                <video id="camera-preview" playsinline muted class="source-camera-stream ${appState.recordedUrl ? "hidden" : ""}"></video>
                <canvas id="live-preview-canvas" class="live-preview-canvas ${appState.recordedUrl ? "hidden" : ""}"></canvas>
                ${
                  appState.recordedUrl
                    ? `<video src="${appState.recordedUrl}" controls preload="metadata"></video>`
                    : `<div class="empty-video ${appState.recordingStream ? "hidden" : ""}" id="camera-empty">
                        <div>
                          <strong style="display:block; font-size: 1.15rem; margin-bottom: 0.5rem;">Camera preview</strong>
                          <span>Enable your camera to begin.</span>
                        </div>
                      </div>`
                }
                <div class="camera-overlay">
                  <div class="record-badge ${
                    appState.takeAnalysisInFlight ? "is-analyzing" : activeRecorder ? "" : "is-ready"
                  }" data-recording-badge>${
                    appState.takeAnalysisInFlight ? "Analyzing take" : activeRecorder ? "Recording live" : "Ready to rehearse"
                  }</div>
                  <div class="frame-countdown-overlay hidden" data-countdown-overlay></div>
                </div>
              </div>
              <div class="pane-analysis-stack">
                <div id="live-analysis-hud" class="emotion-hud emotion-hud-inline"></div>
                <div class="camera-footer camera-footer-static">
                  <div>
                    <div class="small-label">Target emotions</div>
                    <div class="tag-grid">
                      ${topTargets
                        .map(
                          (emotion) =>
                            `<span class="emotion-chip" style="--chip-color:${emotion.color};">${emotion.label} ${emotion.actor}%</span>`
                        )
                        .join("")}
                    </div>
                  </div>
                  <div class="small-label">Mirror the actor's rise, hold, and release.</div>
                </div>
              </div>
            </article>
          </div>
        </div>

        <div class="recording-support-grid">
          <div id="neutral-calibration-panel"></div>
          <div id="recording-status-message"></div>
        </div>

        ${appState.takeAnalysisInFlight ? renderTakeAnalysisOverlay() : ""}
      </div>
    </section>
  `;

  document.querySelector('[data-action="back-analysis"]').addEventListener("click", () => navigate("/analysis"));
  document.querySelector('[data-action="enable-camera"]').addEventListener("click", async () => {
    clearRecordedTakeArtifacts();
    stopCamera();
    renderRoute();
    const preview = document.querySelector("#camera-preview");
    const empty = document.querySelector("#camera-empty");
    await prepareCamera(preview);
    if (appState.recordingStream) {
      empty?.classList.add("hidden");
    }
    renderRoute();
  });
  document.querySelector('[data-action="start-recording"]').addEventListener("click", startRecording);
  document.querySelector('[data-action="stop-recording"]').addEventListener("click", stopRecordingAndAnalyze);
  document.querySelector('[data-action="desktop-webcam"]')?.addEventListener("click", runDesktopWebcamTake);
  document.querySelector('[data-action="fallback-recording"]').addEventListener("click", runFallbackTake);

  const referenceVideo = document.querySelector("#reference-preview");
  const referenceHud = document.querySelector("#reference-analysis-hud");
  const preview = document.querySelector("#camera-preview");
  const overlayCanvas = document.querySelector("#live-preview-canvas");
  const overlayHud = document.querySelector("#live-analysis-hud");

  if (referenceVideo instanceof HTMLVideoElement && referenceHud) {
    if (appState.synchronizedReferencePlayback && appState.referenceUrl) {
      playReferenceVideoFromStart(referenceVideo);
    }

    startReferenceEmotionTracking(referenceVideo, referenceHud);
  } else if (referenceHud) {
    renderReferenceEmotionHud(referenceHud, null);
  }

  if (appState.recordingStream) {
    preview.srcObject = appState.recordingStream;
    preview.play().catch(() => {});
    document.querySelector("#camera-empty")?.classList.add("hidden");

    if (!appState.recordedUrl && !appState.takeAnalysisInFlight) {
      startLiveEmotionTracking(preview, overlayCanvas, overlayHud);
    }
  } else if (overlayHud) {
    renderLiveEmotionHud(overlayHud, null);
  }

  renderNeutralCalibrationStatus();
  updateRecordingStatusMessage();
  syncRecordingControls();
  updateRecordingCountdownOverlay();
  const cleanupTakeAnalysisOverlay = appState.takeAnalysisInFlight ? mountTakeAnalysisOverlay() : () => {};
  attachSupportActions();

  appState.routeCleanup = () => {
    cleanupTakeAnalysisOverlay();
    stopLiveEmotionTracking();
    stopReferenceEmotionTracking();
    clearRecordingCountdown();
    if (!window.location.pathname.startsWith("/recording")) {
      appState.synchronizedReferencePlayback = false;
      resetNeutralCalibration();
      stopCamera();
    }
  };
}

function renderResultsPage() {
  if (!appState.actorProfile || !appState.userProfile || appState.similarityScore == null) {
    navigate("/recording");
    return;
  }

  const comparison = deriveComparison(appState.actorProfile, appState.userProfile);
  const actorLeadEmotion = getProfileDominantEmotion(appState.actorProfile, "actor") || appState.actorProfile.topEmotion;
  const userLeadEmotion = getProfileDominantEmotion(appState.userProfile, "user") || appState.userProfile.topEmotion;
  const backendSource = appState.userProfile.sourceSummary?.primary || "";
  appState.similarityScore = comparison.similarityScore;
  appState.feedback = comparison.feedback;

  app.innerHTML = `
    <section class="screen">
      <div class="panel-card">
        ${routeHead(
          "Performance Results",
          "Emotion comparison analysis and rehearsal feedback",
          "Actor vs your latest take",
          '<button class="button button-ghost" data-action="retake">Record another take</button>'
        )}
        ${renderWorkflowStepper("results", "You have reached the final comparison screen. Use the notes below to plan the next take.")}

        <div class="results-grid" style="margin-top: 1.5rem;">
          <div class="panel-card">
            <div class="small-label">Emotion match breakdown</div>
            <div class="divider"></div>
            <div class="bars">
              ${comparison.bars
                .map(
                  (emotion) => `
                    <div class="bar-row">
                      <div class="bar-header">
                        <strong>${emotion.label}</strong>
                        <span class="small-label">${Math.abs(emotion.actor - emotion.user)}% gap</span>
                      </div>
                      <div class="bar-pair">
                        <div class="bar-label"><span>Actor reference</span><span>${emotion.actor}%</span></div>
                        <div class="bar-meter"><div class="bar-fill actor" style="width:${emotion.actor}%;"></div></div>
                      </div>
                      <div class="bar-pair">
                        <div class="bar-label"><span>Your take</span><span>${emotion.user}%</span></div>
                        <div class="bar-meter"><div class="bar-fill user" style="width:${emotion.user}%;"></div></div>
                      </div>
                    </div>
                  `
                )
                .join("")}
            </div>
          </div>

          <div class="panel-card">
            <div class="comparison-score" style="--score:${comparison.similarityScore};">
              <div style="text-align:center;">
                <strong>${comparison.similarityScore}</strong>
                <span>match score</span>
              </div>
            </div>

            <div class="micro-grid" style="margin-top: 1rem;">
              <div class="micro-card">
                <strong>${comparison.bars.filter((emotion) => Math.abs(emotion.actor - emotion.user) <= 8).length}</strong>
                <span>Strong matches</span>
              </div>
              <div class="micro-card">
                <strong>${getTakeSourceLabel(appState.lastTakeSource)}</strong>
                <span>Take source</span>
              </div>
              <div class="micro-card">
                <strong>${formatDuration(appState.recordingDuration)}</strong>
                <span>Take duration</span>
              </div>
              <div class="micro-card">
                <strong>${userLeadEmotion ? `${userLeadEmotion.label} ${userLeadEmotion.averagePercent || userLeadEmotion.user || 0}%` : "Unavailable"}</strong>
                <span>Your final dominant emotion</span>
              </div>
            </div>

            <div class="callout" style="margin-top: 1rem;">
              ${
                appState.lastTakeSource === "sample-model"
                  ? "This score compares the current actor reference with the bundled HCI sample user CSV."
                  : appState.lastTakeSource === "desktop-webcam"
                    ? "Both sides of this score came from local Python inference: uploaded actor reference on one side and the desktop webcam pipeline on the other."
                    : "This score compares the uploaded actor reference with a freshly analyzed recorded take sent to the analysis server."
              }${backendSource ? ` Final backend decision source: ${getFinalSourceLabel(backendSource)}.` : ""}
            </div>
            <div class="status-note" style="margin-top: 1rem;">
              If you want a cleaner retry, return to recording and focus on matching the top one or two emotions first before adjusting the smaller supporting signals.
            </div>
          </div>
        </div>

        <div class="hero-grid" style="margin-top: 1.5rem;">
          <div class="panel-card">
            <div class="small-label">Detailed coaching notes</div>
            <div class="divider"></div>
            <div class="feedback-list">
              ${appState.feedback
                .map(
                  (item) => `
                    <article class="feedback-card is-${item.type}">
                      <span class="feedback-kicker">${item.title} · ${item.emotion}</span>
                      <p class="feedback-body">${escapeHtml(item.message)}</p>
                    </article>
                  `
                )
                .join("")}
            </div>
          </div>

          <div class="panel-card">
            <div class="small-label">Next rehearsal moves</div>
            <div class="divider"></div>
            <div class="summary-list">
              <div class="summary-item">
                <strong>Shape the strongest emotion first.</strong>
                <p class="feedback-body">Match ${actorLeadEmotion.label.toLowerCase()} before refining the smaller supporting channels around it.</p>
              </div>
              <div class="summary-item">
                <strong>Watch transitions, not just peaks.</strong>
                <p class="feedback-body">The reference clip reads because tension rises and releases with rhythm. Practice those switches out loud with the beat of the scene.</p>
              </div>
              <div class="summary-item">
                <strong>Record one sharper alternate.</strong>
                <p class="feedback-body">Do another take pushing the top two emotions 10% harder, then compare whether the scene lands closer to the actor.</p>
              </div>
            </div>
            <div class="hero-actions" style="margin-top: 1rem;">
              <button class="button button-primary" data-action="retake-primary">Record another take</button>
              <button class="button button-secondary" data-action="restart">Start over</button>
            </div>
          </div>
        </div>
      </div>
    </section>
  `;

  document.querySelector('[data-action="retake"]').addEventListener("click", () => {
    clearRecordedTakeArtifacts();
    navigate("/recording");
  });
  document.querySelector('[data-action="retake-primary"]').addEventListener("click", () => {
    clearRecordedTakeArtifacts();
    navigate("/recording");
  });
  document.querySelector('[data-action="restart"]').addEventListener("click", () => navigate("/"));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

renderRoute();
