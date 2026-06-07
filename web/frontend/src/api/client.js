// Resolve API base URL from environment or fall back to local dev server.
export const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function assertOk(res, fallbackMessage) {
  if (res.ok) {
    return;
  }

  let detail = fallbackMessage;
  try {
    const data = await res.json();
    detail = data.detail || fallbackMessage;
  } catch {
    detail = fallbackMessage;
  }

  throw new Error(detail);
}

// Upload an image and request detection results.
export async function detectImage(file, { labels, conf }) {
  const form = new FormData();
  form.append("file", file);

  const params = new URLSearchParams({ labels, conf });
  const res = await fetch(`${API_BASE}/api/image?${params.toString()}`, {
    method: "POST",
    body: form,
  });

  await assertOk(res, "Image detection failed");
  return res.json();
}

// Upload a video and start background processing.
export async function uploadVideo(file, { labels, conf }) {
  const form = new FormData();
  form.append("file", file);

  const params = new URLSearchParams({ labels, conf });
  const res = await fetch(`${API_BASE}/api/video/upload?${params.toString()}`, {
    method: "POST",
    body: form,
  });

  await assertOk(res, "Video upload failed");
  return res.json();
}

// Fetch status and metrics for a video job.
export async function getVideoStatus(jobId, { avgWindow = "minute" } = {}) {
  const params = new URLSearchParams({ avg_window: avgWindow });
  const res = await fetch(`${API_BASE}/api/video/${jobId}?${params.toString()}`);
  await assertOk(res, "Video status failed");
  return res.json();
}

// Download realtime metrics history as an Excel workbook.
export async function exportVideoMetrics(jobId, { avgWindow = "minute" } = {}) {
  const params = new URLSearchParams({ avg_window: avgWindow });
  const res = await fetch(
    `${API_BASE}/api/video/${jobId}/metrics/export?${params.toString()}`
  );
  await assertOk(res, "Metrics export failed");
  return res.blob();
}

// Start offline video processing for a previously uploaded video.
export async function startVideoProcessing(jobId, { labels, conf }) {
  const params = new URLSearchParams({ labels, conf });
  const res = await fetch(
    `${API_BASE}/api/video/${jobId}/process?${params.toString()}`,
    { method: "POST" }
  );
  await assertOk(res, "Video processing failed");
  return res.json();
}

// Save a manual normalized ROI polygon for a video job.
export async function setVideoRoi(jobId, roi) {
  const res = await fetch(`${API_BASE}/api/video/${jobId}/roi`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(roi),
  });
  await assertOk(res, "ROI update failed");
  return res.json();
}

// Clear manual ROI for a video job.
export async function resetVideoRoi(jobId) {
  const res = await fetch(`${API_BASE}/api/video/${jobId}/roi`, {
    method: "DELETE",
  });
  await assertOk(res, "ROI reset failed");
  return res.json();
}
