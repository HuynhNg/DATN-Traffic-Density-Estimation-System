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
export async function getVideoStatus(jobId) {
  const res = await fetch(`${API_BASE}/api/video/${jobId}`);
  await assertOk(res, "Video status failed");
  return res.json();
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
