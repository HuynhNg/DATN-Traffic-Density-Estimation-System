// Resolve API base URL from environment or fall back to local dev server.
export const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

// Upload an image and request detection results.
export async function detectImage(file, { labels, conf }) {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({ labels, conf });
  const res = await fetch(`${API_BASE}/api/image?${params.toString()}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error("Image detection failed");
  }
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
  if (!res.ok) {
    throw new Error("Video upload failed");
  }
  return res.json();
}

// Fetch status and metrics for a video job.
export async function getVideoStatus(jobId) {
  const res = await fetch(`${API_BASE}/api/video/${jobId}`);
  if (!res.ok) {
    throw new Error("Video status failed");
  }
  return res.json();
}
