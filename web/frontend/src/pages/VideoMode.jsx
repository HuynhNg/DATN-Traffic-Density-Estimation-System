import { useEffect, useRef, useState } from "react";
import { API_BASE, getVideoStatus, uploadVideo } from "../api/client.js";
import Toggle from "../components/Toggle.jsx";
import StatCard from "../components/StatCard.jsx";
import ChartPanel from "../components/ChartPanel.jsx";

// Video upload mode with MJPEG preview and analytics.
export default function VideoMode() {
  const allowedVideoTypes = [
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm"
  ];
  const allowedVideoExts = [".mp4", ".mov", ".avi", ".mkv", ".webm"];

  const previewRef = useRef(null);

  const [labels, setLabels] = useState(true);
  const [conf, setConf] = useState(true);
  const [metrics, setMetrics] = useState({
    fps: 0,
    avg_objects: 0,
    objects_in_frame: 0,
    occupancy_pct: 0,
    pce_count: 0,
    alert_label: "",
    alert_message: ""
  });
  const [series, setSeries] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [annotating, setAnnotating] = useState(false);
  const [targetFps, setTargetFps] = useState(12);
  const [streamUrl, setStreamUrl] = useState(null);
  const [error, setError] = useState(null);

  const alertTone = (() => {
    switch (metrics.alert_level) {
      case 1:
        return "bg-amber-50 text-amber-800 border-amber-200";
      case 2:
        return "bg-orange-50 text-orange-800 border-orange-200";
      case 3:
        return "bg-rose-50 text-rose-800 border-rose-200";
      default:
        return "bg-emerald-50 text-emerald-800 border-emerald-200";
    }
  })();

  // Poll job status for progress and live metrics.
  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      const status = await getVideoStatus(jobId);
      setJob(status);
      if (status.live_metrics) {
        setMetrics(status.live_metrics);
      }
      if (Array.isArray(status.live_series)) {
        setSeries(status.live_series);
      }
      if (status.status === "done" || status.status === "failed") {
        clearInterval(timer);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [jobId]);

  // Auto-play the uploaded video preview when available.
  useEffect(() => {
    if (!uploadedUrl) return;
    const url = uploadedUrl;
    const preview = previewRef.current;
    if (preview) {
      preview.currentTime = 0;
      preview.play().catch(() => undefined);
    }
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [uploadedUrl]);

  // Cleanup MJPEG stream on unmount.
  useEffect(() => {
    return () => {
      stopAnnotatedStream();
    };
  }, []);

  // Upload a video and initialize the job status view.
  async function handleVideoUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf("."));
    const isAllowed =
      allowedVideoTypes.includes(file.type) ||
      (ext && allowedVideoExts.includes(ext));
    if (!isAllowed) {
      setError("Unsupported video type. Use MP4, MOV, AVI, MKV, or WEBM.");
      return;
    }
    setError(null);
    stopAnnotatedStream();
    if (uploadedUrl) {
      URL.revokeObjectURL(uploadedUrl);
    }
    setUploadedUrl(URL.createObjectURL(file));
    const data = await uploadVideo(file, { labels, conf });
    setJobId(data.job_id);
    setJob((prev) => ({
      ...(prev || {}),
      fps: data.fps,
      total_frames: data.total_frames
    }));
  }

  // Start MJPEG stream for the current job.
  function startUploadStream() {
    if (!jobId) return;
    stopAnnotatedStream();
    const params = new URLSearchParams({
      labels: String(labels),
      conf: String(conf),
      target_fps: String(targetFps)
    });
    setStreamUrl(`${API_BASE}/api/video/${jobId}/stream?${params.toString()}`);
    setAnnotating(true);
  }

  // Stop MJPEG stream and clear preview URL.
  function stopAnnotatedStream() {
    setAnnotating(false);
    setStreamUrl(null);
  }

  return (
    <section className="grid gap-6">
      <div className="space-y-6">
        <div className="grid gap-6 lg:grid-cols-[1.6fr_0.7fr]">
          <div className="glass rounded-3xl p-6 shadow-soft">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <label className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-white">
                  Upload Video
                  <input
                    type="file"
                    accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
                    className="hidden"
                    onChange={handleVideoUpload}
                  />
                </label>
                {uploadedUrl && (
                  <div className="flex items-center gap-2">
                    <button
                      className={`rounded-full px-3 py-2 text-sm ${
                        annotating ? "bg-slate-200 text-slate-500" : "bg-accent text-white"
                      }`}
                      onClick={startUploadStream}
                      disabled={annotating}
                    >
                      Run AI
                    </button>
                    <button
                      className={`rounded-full px-3 py-2 text-sm ${
                        annotating ? "bg-rose-500 text-white" : "bg-slate-200 text-slate-500"
                      }`}
                      onClick={stopAnnotatedStream}
                      disabled={!annotating}
                    >
                      Stop AI
                    </button>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-4">
                <Toggle label="Labels" checked={labels} onChange={setLabels} />
                <Toggle label="Conf" checked={conf} onChange={setConf} />
                <div className="flex items-center gap-2 text-sm text-slate-500">
                  <span>Target FPS</span>
                  <input
                    type="number"
                    min="1"
                    max="60"
                    value={targetFps}
                    onChange={(e) => setTargetFps(Number(e.target.value) || 1)}
                    className="w-16 rounded-md border border-slate-200 bg-white px-2 py-1"
                  />
                </div>
              </div>
            </div>

            <div className="mt-6 flex h-[420px] items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-white/70">
              {error && <p className="text-rose-500">{error}</p>}
              {streamUrl ? (
                <img src={streamUrl} alt="Annotated" className="max-h-full rounded-2xl shadow-lg" />
              ) : uploadedUrl ? (
                <video ref={previewRef} src={uploadedUrl} className="max-h-full rounded-2xl shadow-lg" controls />
              ) : (
                <p className="text-slate-400">Upload a video to start</p>
              )}
            </div>
          </div>

          <div className="glass rounded-3xl p-6 shadow-soft">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase text-slate-400">Traffic Status</p>
                <h3 className="text-lg font-semibold text-ink">Live Density</h3>
              </div>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">
                Live
              </span>
            </div>

            <div className="mt-4 grid gap-3">
              <StatCard title="Avg Vehicles" value={metrics.avg_objects} />
              <StatCard title="Active Objects" value={metrics.objects_in_frame} />
              <StatCard title="PCE Count" value={metrics.pce_count?.toFixed?.(2) ?? metrics.pce_count} />
              <StatCard
                title="Occupancy"
                value={`${metrics.occupancy_pct?.toFixed?.(1) ?? metrics.occupancy_pct}%`}
              />
            </div>

            {metrics.alert_label && (
              <div
                className={`mt-4 rounded-2xl border px-4 py-3 text-sm shadow-soft ${alertTone}`}
              >
                <span className="font-semibold">{metrics.alert_label}</span>
                <span className="ml-2">{metrics.alert_message}</span>
              </div>
            )}
          </div>
        </div>

        <ChartPanel data={series} />
      </div>
    </section>
  );
}
