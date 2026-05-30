import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const host = process.env.HOST || "0.0.0.0";
const port = Number(process.env.PORT || 3001);

const sampleActorCsvPath = join(root, "backend", "actor_expression.csv");
const sampleUserCsvPath = join(root, "backend", "user_expression.csv");
const generatedDir = join(root, "backend", "generated");
const uploadsDir = join(root, "backend", "uploads");
const modelAssetPath = join(root, "backend", "face_landmarker.task");
const pythonWorkdir = join(root, "backend", "acting_coach");
const pythonBin = process.env.PYTHON_BIN || "python3";
const csvFps = 30;
const DEFAULT_SESSION_ID = "shared-demo";
const SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{8,128}$/;

const baseEmotionMeta = [
  { id: "happiness", label: "Happiness", color: "#f59e0b" },
  { id: "sadness", label: "Sadness", color: "#60a5fa" },
  { id: "anger", label: "Anger", color: "#f87171" },
  { id: "surprise", label: "Surprise", color: "#22d3ee" },
];

const allEmotionMeta = [...baseEmotionMeta, { id: "neutral", label: "Neutral", color: "#94a3b8" }];

const hciSensitivity = {
  happiness: 4.0,
  sadness: 6.0,
  anger: 4.0,
  surprise: 4.5,
};

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml; charset=utf-8",
  ".task": "application/octet-stream",
  ".webm": "video/webm",
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function roundToTenths(value) {
  return Math.round(value * 10) / 10;
}

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  response.end(JSON.stringify(payload));
}

function sendFile(pathname, response) {
  const safePath = normalize(pathname).replace(/^(\.\.[/\\])+/, "");
  const absolutePath = join(root, safePath);

  if (!existsSync(absolutePath) || statSync(absolutePath).isDirectory()) {
    return false;
  }

  response.writeHead(200, {
    "Content-Type": mimeTypes[extname(absolutePath)] || "application/octet-stream",
    "Cache-Control": "no-store",
  });

  createReadStream(absolutePath).pipe(response);
  return true;
}

function parseCsvRows(csvPath) {
  if (!existsSync(csvPath)) {
    throw new Error(`CSV file not found: ${csvPath}`);
  }

  const csv = readFileSync(csvPath, "utf8").replace(/^\uFEFF/, "").trim();
  if (!csv) {
    throw new Error(`CSV file is empty: ${csvPath}`);
  }

  const [headerLine, ...lines] = csv.split(/\r?\n/);
  const headers = headerLine.split(",");

  return lines.filter(Boolean).map((line) => {
    const values = line.split(",");
    return headers.reduce((row, header, index) => {
      const rawValue = values[index] ?? "";
      const trimmed = rawValue.trim();
      const numericValue = trimmed === "" ? Number.NaN : Number(trimmed);
      row[header] = Number.isFinite(numericValue) ? numericValue : trimmed;
      return row;
    }, {});
  });
}

function rawToPercent(rawValue, emotionId) {
  const sensitivity = hciSensitivity[emotionId] ?? 3.0;
  const adjusted = rawValue * sensitivity;
  const percent = adjusted <= 0 ? 0 : (adjusted / (adjusted + 1)) * 100;
  return roundToTenths(clamp(percent, 0, 100));
}

function addNeutralScore(scores) {
  const strongestExpressive = Math.max(...baseEmotionMeta.map((emotion) => scores[emotion.id] || 0));
  return {
    ...scores,
    neutral: strongestExpressive < 15 ? roundToTenths(100 - strongestExpressive) : 0,
  };
}

function getDominantEmotion(scores) {
  return allEmotionMeta.reduce((current, emotion) => {
    return (scores[emotion.id] || 0) > (scores[current] || 0) ? emotion.id : current;
  }, allEmotionMeta[0].id);
}

function getEmotionMetaById(id) {
  return allEmotionMeta.find((emotion) => emotion.id === id) || allEmotionMeta[0];
}

function pickFirstFinite(...values) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function normalizePercentValue(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }

  return roundToTenths(clamp(value > 1.01 ? value : value * 100, 0, 100));
}

function resolveDominantEmotionLabel(value, fallbackScores) {
  if (typeof value !== "string") {
    return getDominantEmotion(fallbackScores);
  }

  const normalized = value.trim().toLowerCase();
  const map = {
    joy: "happiness",
    happiness: "happiness",
    sadness: "sadness",
    anger: "anger",
    surprise: "surprise",
    neutral: "neutral",
  };

  return map[normalized] || getDominantEmotion(fallbackScores);
}

function normalizeEmotion(row, emotionId, rawKeys, valueKeys) {
  const rawScore = pickFirstFinite(...rawKeys.map((key) => row[key]));
  const directValue = pickFirstFinite(...valueKeys.map((key) => row[key]));

  let percentScore = null;
  let resolvedRawScore = rawScore;

  if (directValue !== null) {
    if (directValue > 1.01) {
      percentScore = directValue;
    } else if (resolvedRawScore === null) {
      resolvedRawScore = directValue;
    }
  }

  if (percentScore === null) {
    percentScore = resolvedRawScore !== null ? rawToPercent(resolvedRawScore, emotionId) : 0;
  }

  return {
    rawScore: resolvedRawScore ?? 0,
    percentScore: roundToTenths(clamp(percentScore, 0, 100)),
  };
}

function buildTimeline(rows) {
  return rows.map((row) => {
    const happiness = normalizeEmotion(row, "happiness", ["happiness_raw", "joy_raw"], ["happiness", "joy"]);
    const sadness = normalizeEmotion(row, "sadness", ["sadness_raw"], ["sadness"]);
    const anger = normalizeEmotion(row, "anger", ["anger_raw"], ["anger"]);
    const surprise = normalizeEmotion(row, "surprise", ["surprise_raw"], ["surprise"]);
    const explicitNeutral = pickFirstFinite(row.neutral);

    const expressiveScores = {
      happiness: happiness.percentScore,
      sadness: sadness.percentScore,
      anger: anger.percentScore,
      surprise: surprise.percentScore,
    };

    const scores =
      explicitNeutral !== null
        ? {
            ...expressiveScores,
            neutral: normalizePercentValue(explicitNeutral) ?? 0,
          }
        : addNeutralScore(expressiveScores);

    const dominantEmotion = getDominantEmotion(scores);
    const dominantScore = scores[dominantEmotion] ?? 0;
    const mediapipeScore = normalizePercentValue(pickFirstFinite(row.mediapipe_percent));

    return {
      frame: Math.round(row.frame || 0),
      timestamp_ms: Math.round(((row.frame || 0) / csvFps) * 1000),
      dominantEmotion,
      dominantScore,
      finalSource: typeof row.final_source === "string" ? row.final_source : "",
      mediapipeLabel: resolveDominantEmotionLabel(row.mediapipe_label, scores),
      mediapipeScore,
      userNeutral: {
        score: normalizePercentValue(pickFirstFinite(row.user_neutral_score)) ?? 0,
        distance: pickFirstFinite(row.user_neutral_distance),
        deltaEnergy: pickFirstFinite(row.user_neutral_delta_energy),
      },
      angerConfirmation:
        typeof row.anger_csv_confirms === "boolean"
          ? row.anger_csv_confirms
          : String(row.anger_csv_confirms || "").toLowerCase() === "true",
      aihubReferenceEmotion: typeof row.aihub_reference_emotion === "string" ? row.aihub_reference_emotion : "",
      aihubScores: {
        anger: normalizePercentValue(pickFirstFinite(row.aihub_anger_score)) ?? 0,
        neutral: normalizePercentValue(pickFirstFinite(row.aihub_neutral_score)) ?? 0,
      },
      aihubDistances: {
        anger: pickFirstFinite(row.aihub_anger_distance),
        neutral: pickFirstFinite(row.aihub_neutral_distance),
      },
      scores,
      rawScores: {
        happiness: happiness.rawScore,
        sadness: sadness.rawScore,
        anger: anger.rawScore,
        surprise: surprise.rawScore,
        neutral: pickFirstFinite(row.neutral) || 0,
      },
    };
  });
}

function summarizeDominantFrames(timeline) {
  const buckets = new Map();

  for (const frame of timeline) {
    const emotionId = frame.dominantEmotion || getDominantEmotion(frame.scores);
    const bucket = buckets.get(emotionId) || {
      id: emotionId,
      count: 0,
      totalPercent: 0,
      latestPercent: 0,
      latestTimestamp: -1,
    };

    bucket.count += 1;
    bucket.totalPercent += frame.dominantScore || 0;

    if ((frame.timestamp_ms || 0) >= bucket.latestTimestamp) {
      bucket.latestTimestamp = frame.timestamp_ms || 0;
      bucket.latestPercent = frame.dominantScore || 0;
    }

    buckets.set(emotionId, bucket);
  }

  const dominantBuckets = [...buckets.values()].sort((a, b) => {
    if (b.count !== a.count) {
      return b.count - a.count;
    }
    return b.totalPercent - a.totalPercent;
  });

  const winner = dominantBuckets[0];
  if (!winner) {
    return null;
  }

  const meta = getEmotionMetaById(winner.id);
  return {
    id: winner.id,
    label: meta.label,
    color: meta.color,
    count: winner.count,
    share: Math.round((winner.count / timeline.length) * 100),
    averagePercent: roundToTenths(winner.totalPercent / winner.count),
    latestPercent: roundToTenths(winner.latestPercent),
  };
}

function summarizeSourceUsage(timeline) {
  const counts = timeline.reduce((summary, frame) => {
    if (!frame.finalSource) {
      return summary;
    }

    summary[frame.finalSource] = (summary[frame.finalSource] || 0) + 1;
    return summary;
  }, {});

  const topEntry = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return {
    counts,
    primary: topEntry ? topEntry[0] : "",
  };
}

function buildProfileFromTimeline(timeline, valueKey, sourceLabel, sourceDetail, analysisMode) {
  if (!timeline.length) {
    throw new Error("Timeline is empty.");
  }

  const emotions = allEmotionMeta.map((emotion) => {
    const total = timeline.reduce((sum, frame) => sum + (frame.scores[emotion.id] || 0), 0);
    return {
      id: emotion.id,
      label: emotion.label,
      color: emotion.color,
      [valueKey]: Math.round(total / timeline.length),
    };
  });

  const sorted = [...emotions].sort((a, b) => b[valueKey] - a[valueKey]);
  const lastFrame = timeline.at(-1);
  const durationSeconds = (lastFrame?.timestamp_ms || 0) / 1000;
  const topEmotion = sorted[0];
  const dominantSummary = summarizeDominantFrames(timeline);
  const sourceSummary = summarizeSourceUsage(timeline);

  return {
    emotions,
    timeline,
    topEmotion,
    secondaryEmotion: sorted[1] || topEmotion,
    dominantSummary,
    sourceSummary,
    frameCount: lastFrame?.frame || timeline.length,
    sampleCount: timeline.length,
    pacing:
      durationSeconds >= 45
        ? "Measured tempo"
        : durationSeconds >= 20
          ? "Scene-building tempo"
          : "Compact tempo",
    intensity:
      topEmotion[valueKey] >= 45
        ? "High-contrast delivery"
        : topEmotion[valueKey] >= 28
          ? "Expressive balance"
          : "Low-amplitude subtlety",
    referenceDurationSeconds: durationSeconds,
    analysisMode,
    sourceLabel,
    sourceDetail,
  };
}

function normalizeSessionId(value) {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  if (!SESSION_ID_PATTERN.test(trimmed)) {
    return null;
  }

  return trimmed;
}

function getSessionPaths(sessionId) {
  const safeSessionId = normalizeSessionId(sessionId) || DEFAULT_SESSION_ID;
  const sessionGeneratedDir = join(generatedDir, "sessions", safeSessionId);
  const sessionUploadsDir = join(uploadsDir, safeSessionId);

  return {
    sessionId: safeSessionId,
    sessionGeneratedDir,
    sessionUploadsDir,
    activeActorCsvPath: join(sessionGeneratedDir, "actor_expression.csv"),
    activeUserCsvPath: join(sessionGeneratedDir, "video_expression_mediapipe.csv"),
    activeWebcamCsvPath: join(sessionGeneratedDir, "webcam_expression_mediapipe_with_aihub_csv.csv"),
  };
}

function resolveActorCsvPath(mode = "active", sessionId = DEFAULT_SESSION_ID) {
  if (mode === "sample") {
    return sampleActorCsvPath;
  }
  const { activeActorCsvPath } = getSessionPaths(sessionId);
  return existsSync(activeActorCsvPath) ? activeActorCsvPath : sampleActorCsvPath;
}

function resolveUserCsvPath(mode = "active", sessionId = DEFAULT_SESSION_ID, source = "active") {
  if (mode === "sample") {
    return sampleUserCsvPath;
  }
  const { activeUserCsvPath, activeWebcamCsvPath } = getSessionPaths(sessionId);
  if (source === "webcam" && existsSync(activeWebcamCsvPath)) {
    return activeWebcamCsvPath;
  }
  if (source === "recorded" && existsSync(activeUserCsvPath)) {
    return activeUserCsvPath;
  }
  if (existsSync(activeUserCsvPath)) {
    return activeUserCsvPath;
  }
  if (existsSync(activeWebcamCsvPath)) {
    return activeWebcamCsvPath;
  }
  return sampleUserCsvPath;
}

function buildActorProfile(mode = "active", sessionId = DEFAULT_SESSION_ID) {
  const { activeActorCsvPath } = getSessionPaths(sessionId);
  const csvPath = resolveActorCsvPath(mode, sessionId);
  const timeline = buildTimeline(parseCsvRows(csvPath));
  const isGenerated = csvPath === activeActorCsvPath;

  return buildProfileFromTimeline(
    timeline,
    "actor",
    isGenerated ? "Analyzed actor reference" : "HCI-acting-coach demo actor baseline",
    isGenerated
      ? "Profile derived from the uploaded reference video using the latest HCI-acting-coach MediaPipe + AI-Hub CSV reference pipeline."
      : "Profile derived from the bundled actor_expression.csv sample.",
    isGenerated ? "hci-acting-coach-actor-upload" : "hci-acting-coach-actor-demo"
  );
}

function buildUserProfile(mode = "active", sessionId = DEFAULT_SESSION_ID, source = "active") {
  const { activeUserCsvPath, activeWebcamCsvPath } = getSessionPaths(sessionId);
  const csvPath = resolveUserCsvPath(mode, sessionId, source);
  const timeline = buildTimeline(parseCsvRows(csvPath));
  const isRecorded = csvPath === activeUserCsvPath;
  const isWebcam = csvPath === activeWebcamCsvPath;
  const isGenerated = isRecorded || isWebcam;
  const isSample = mode === "sample" || !isGenerated;

  return buildProfileFromTimeline(
    timeline,
    "user",
    isSample && !isGenerated
      ? "HCI-acting-coach sample user take"
      : isWebcam
        ? "Analyzed webcam take with AI-Hub CSV reference"
        : "Analyzed user take with AI-Hub CSV reference",
    isSample && !isGenerated
      ? "Profile derived from the bundled user_expression.csv sample."
      : isWebcam
        ? "Profile derived from a freshly analyzed webcam take using MediaPipe plus the AI-Hub anger/neutral CSV distribution pipeline."
        : "Profile derived from a freshly analyzed recorded take using MediaPipe plus the AI-Hub anger/neutral CSV distribution pipeline.",
    isSample && !isGenerated
      ? "hci-acting-coach-user-demo"
      : isWebcam
        ? "hci-acting-coach-webcam-aihub-csv"
        : "hci-acting-coach-recorded-video-aihub-csv"
  );
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
          message: `${emotion.label} is closely aligned with the actor baseline. Keep that emotional contour.`,
        };
      }

      if (gap > 0) {
        return {
          id: `${emotion.id}-high`,
          emotion: emotion.label,
          type: gap > 16 ? "warning" : "info",
          title: "Dial it back slightly",
          message: `${emotion.label} is ${gap}% stronger than the actor baseline. Ease that channel down a little.`,
        };
      }

      return {
        id: `${emotion.id}-low`,
        emotion: emotion.label,
        type: Math.abs(gap) > 16 ? "danger" : "warning",
        title: "Push this emotion further",
        message: `${emotion.label} is ${Math.abs(gap)}% lower than the actor baseline. Add more facial energy there.`,
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

function buildComparisonPayload({
  actorMode = "active",
  userMode = "active",
  userSource = "active",
  sourceLabel,
  sourceDetail,
  sessionId = DEFAULT_SESSION_ID,
} = {}) {
  const actorProfile = buildActorProfile(actorMode, sessionId);
  const userProfile = buildUserProfile(userMode, sessionId, userSource);

  return {
    actorProfile,
    userProfile,
    comparison: deriveComparison(actorProfile, userProfile),
    takeDurationSeconds: userProfile.referenceDurationSeconds,
    sourceLabel: sourceLabel || `${actorProfile.sourceLabel} vs ${userProfile.sourceLabel}`,
    sourceDetail:
      sourceDetail ||
      `${actorProfile.sourceDetail} ${userProfile.sourceDetail}`.trim(),
  };
}

async function readRequestBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks);
}

function sanitizeFilename(filename) {
  return filename.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "") || "upload";
}

function inferVideoExtension(filename, contentType) {
  const extension = extname(filename || "");
  if (extension) {
    return extension.toLowerCase();
  }

  if (contentType.includes("webm")) return ".webm";
  if (contentType.includes("quicktime")) return ".mov";
  if (contentType.includes("mp4")) return ".mp4";
  if (contentType.includes("matroska")) return ".mkv";
  return ".mp4";
}

async function saveUploadedVideo(request, prefix, sessionId = DEFAULT_SESSION_ID) {
  const body = await readRequestBody(request);
  if (!body.length) {
    throw new Error("Upload body was empty.");
  }

  const { sessionUploadsDir } = getSessionPaths(sessionId);
  mkdirSync(sessionUploadsDir, { recursive: true });
  const headerValue = request.headers["x-filename"];
  const decodedName = Array.isArray(headerValue) ? headerValue[0] : headerValue || `${prefix}.mp4`;
  const safeName = sanitizeFilename(decodeURIComponent(decodedName));
  const extension = inferVideoExtension(safeName, String(request.headers["content-type"] || ""));
  const filePath = join(sessionUploadsDir, `${prefix}-${Date.now()}-${randomUUID()}${extension}`);

  writeFileSync(filePath, body);
  return filePath;
}

function readRequestHeader(request, headerName) {
  const value = request.headers[headerName.toLowerCase()];
  return Array.isArray(value) ? value[0] : value;
}

function getRequestSessionId(request) {
  return normalizeSessionId(readRequestHeader(request, "x-session-id")) || DEFAULT_SESSION_ID;
}

function isLocalHostRequest(request) {
  const hostValue = String(
    readRequestHeader(request, "x-forwarded-host") ||
      readRequestHeader(request, "host") ||
      ""
  )
    .split(",")[0]
    .trim()
    .toLowerCase();
  const hostname = hostValue.split(":")[0];
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

function runPythonScript(scriptName, args = []) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonBin, [scriptName, ...args], {
      cwd: pythonWorkdir,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }

      reject(new Error(stderr.trim() || stdout.trim() || `Python exited with code ${code}`));
    });
  });
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host}`);
  const sessionId = getRequestSessionId(request);

  try {
    if (request.method === "GET" && url.pathname === "/api/health") {
      const { activeActorCsvPath, activeUserCsvPath, activeWebcamCsvPath } = getSessionPaths(sessionId);
      sendJson(response, 200, {
        ok: true,
        backend: "node",
        model: "hci-acting-coach",
        sessionId,
        hasModelAsset: existsSync(modelAssetPath),
        hasSampleActorCsv: existsSync(sampleActorCsvPath),
        hasSampleUserCsv: existsSync(sampleUserCsvPath),
        hasGeneratedActorCsv: existsSync(activeActorCsvPath),
        hasGeneratedUserCsv: existsSync(activeUserCsvPath) || existsSync(activeWebcamCsvPath),
        hasGeneratedRecordedVideoCsv: existsSync(activeUserCsvPath),
        hasGeneratedWebcamCsv: existsSync(activeWebcamCsvPath),
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/actor-emotions") {
      const mode = url.searchParams.get("demo") === "1" ? "sample" : "active";
      sendJson(response, 200, buildActorProfile(mode, sessionId).timeline);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/user-emotions") {
      const mode = url.searchParams.get("demo") === "1" ? "sample" : "active";
      sendJson(response, 200, buildUserProfile(mode, sessionId).timeline);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/analysis/reference") {
      const mode = url.searchParams.get("demo") === "1" ? "sample" : "active";
      sendJson(response, 200, buildActorProfile(mode, sessionId));
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/analysis/sample-user") {
      sendJson(response, 200, buildUserProfile("sample", sessionId));
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/analysis/sample-comparison") {
      sendJson(
        response,
        200,
        buildComparisonPayload({
          actorMode: "active",
          userMode: "sample",
          sessionId,
          sourceLabel: "Actor reference vs HCI sample take",
          sourceDetail: "Comparison built from the current actor reference and the bundled HCI sample user_expression.csv.",
        })
      );
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/analysis/reference-upload") {
      const uploadPath = await saveUploadedVideo(request, "actor-reference", sessionId);
      const { sessionGeneratedDir, activeActorCsvPath } = getSessionPaths(sessionId);
      mkdirSync(sessionGeneratedDir, { recursive: true });
      await runPythonScript("analyze_actor_video.py", [
        uploadPath,
        "--model-path",
        modelAssetPath,
        "--output-csv",
        activeActorCsvPath,
        "--headless",
      ]);

      sendJson(response, 200, buildActorProfile("active", sessionId));
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/analysis/user-upload") {
      const uploadPath = await saveUploadedVideo(request, "user-take", sessionId);
      const { sessionGeneratedDir, activeUserCsvPath } = getSessionPaths(sessionId);
      mkdirSync(sessionGeneratedDir, { recursive: true });
      const scriptArgs = [
        uploadPath,
        "--model-path",
        modelAssetPath,
        "--output-csv",
        activeUserCsvPath,
        "--headless",
      ];

      await runPythonScript("analyze_user_video.py", scriptArgs);

      sendJson(
        response,
        200,
        buildComparisonPayload({
          actorMode: "active",
          userMode: "active",
          sessionId,
          sourceLabel: "Recorded take comparison",
          sourceDetail: "Comparison built from the current actor reference and a freshly analyzed recorded take.",
        })
      );
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/analysis/run-webcam") {
      if (!isLocalHostRequest(request)) {
        sendJson(response, 403, {
          error: "desktop-webcam-unavailable",
          message: "The desktop webcam route is available only in a local environment.",
        });
        return;
      }

      const { sessionGeneratedDir, activeWebcamCsvPath } = getSessionPaths(sessionId);
      mkdirSync(sessionGeneratedDir, { recursive: true });
      await runPythonScript("analyze_webcam.py", [
        "--model-path",
        modelAssetPath,
        "--output-csv",
        activeWebcamCsvPath,
      ]);

      sendJson(
        response,
        200,
        buildComparisonPayload({
          actorMode: "active",
          userMode: "active",
          userSource: "webcam",
          sessionId,
          sourceLabel: "Desktop webcam comparison",
          sourceDetail:
            "Comparison built from the current actor reference and a freshly analyzed desktop webcam take using the AI-Hub anger/neutral CSV distribution pipeline.",
        })
      );
      return;
    }
  } catch (error) {
    sendJson(response, 500, {
      error: "analysis-failed",
      message: error instanceof Error ? error.message : "Unexpected server error",
    });
    return;
  }

  if (url.pathname.startsWith("/api/")) {
    sendJson(response, 404, { error: "not-found" });
    return;
  }

  const pathname = url.pathname === "/" ? "/index.html" : url.pathname;

  if (sendFile(pathname, response)) {
    return;
  }

  if (sendFile("/index.html", response)) {
    return;
  }

  response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  response.end("Not found");
});

server.listen(port, host, () => {
  const previewHost = host === "0.0.0.0" ? "127.0.0.1" : host;
  console.log(`Acting Emotion Analysis Tool is running at http://${previewHost}:${port}`);
});
