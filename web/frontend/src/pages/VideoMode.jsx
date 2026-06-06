import { useEffect, useRef, useState } from "react";
import {
  API_BASE,
  getVideoStatus,
  resetVideoRoi,
  setVideoRoi,
  uploadVideo,
} from "../api/client.js";
import Toggle from "../components/Toggle.jsx";
import StatCard from "../components/StatCard.jsx";
import ChartPanel from "../components/ChartPanel.jsx";
import { VIDEO_ACCEPT, isAllowedVideo } from "../utils/fileTypes.js";

const PREVIEW_PANEL_CLASS =
  "mt-6 flex h-[420px] items-center justify-center rounded-2xl border " +
  "border-dashed border-slate-200 bg-white/70";

const DEFAULT_METRICS = {
  fps: 0,
  avg_objects: 0,
  objects_in_frame: 0,
  occupancy_pct: 0,
  pce_count: 0,
  alert_label: "",
  alert_message: "",
};

function getAlertTone(level) {
  switch (level) {
    case 1:
      return "bg-amber-50 text-amber-800 border-amber-200";
    case 2:
      return "bg-orange-50 text-orange-800 border-orange-200";
    case 3:
      return "bg-rose-50 text-rose-800 border-rose-200";
    default:
      return "bg-emerald-50 text-emerald-800 border-emerald-200";
  }
}

function clampPoint(point) {
  return {
    x: Math.max(0, Math.min(Number(point.x) || 0, 1)),
    y: Math.max(0, Math.min(Number(point.y) || 0, 1)),
  };
}

function boxToPolygon(box) {
  const x = Math.max(0, Math.min(Number(box.x) || 0, 0.99));
  const y = Math.max(0, Math.min(Number(box.y) || 0, 0.99));
  const w = Math.max(0.02, Math.min(Number(box.w) || 1, 1 - x));
  const h = Math.max(0.02, Math.min(Number(box.h) || 1, 1 - y));
  return {
    type: "polygon",
    points: [
      { x, y },
      { x: x + w, y },
      { x: x + w, y: y + h },
      { x, y: y + h },
    ],
  };
}

function normalizeRoi(roi) {
  if (!roi) return null;
  if (roi.type === "polygon" && Array.isArray(roi.points)) {
    const points = roi.points.map(clampPoint);
    if (points.length >= 3) {
      return { type: "polygon", points };
    }
  }
  if ("x" in roi && "y" in roi && "w" in roi && "h" in roi) {
    return boxToPolygon(roi);
  }
  return null;
}

function movePolygon(points, dx, dy) {
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const safeDx = Math.max(-minX, Math.min(dx, 1 - maxX));
  const safeDy = Math.max(-minY, Math.min(dy, 1 - maxY));
  return points.map((point) => ({
    x: point.x + safeDx,
    y: point.y + safeDy,
  }));
}

function polygonPointsAttr(points) {
  return points.map((point) => `${point.x * 100},${point.y * 100}`).join(" ");
}

// Video upload mode with MJPEG preview and analytics.
export default function VideoMode() {
  const previewRef = useRef(null);
  const panelRef = useRef(null);
  const streamImageRef = useRef(null);
  const roiDragRef = useRef(null);

  const [labels, setLabels] = useState(true);
  const [conf, setConf] = useState(true);
  const [metrics, setMetrics] = useState(DEFAULT_METRICS);
  const [series, setSeries] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [annotating, setAnnotating] = useState(false);
  const [targetFps, setTargetFps] = useState(30);
  const [streamUrl, setStreamUrl] = useState(null);
  const [roi, setRoi] = useState(null);
  const [roiSource, setRoiSource] = useState(null);
  const [selectedRoiPoint, setSelectedRoiPoint] = useState(null);
  const [, setRoiLayoutVersion] = useState(0);
  const [error, setError] = useState(null);

  const alertTone = getAlertTone(metrics.alert_level);

  // Poll job status for progress and live metrics.
  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      try {
        const status = await getVideoStatus(jobId);
        setJob(status);
        if (status.live_metrics) {
          setMetrics(status.live_metrics);
        }
        if (Array.isArray(status.live_series)) {
          setSeries(status.live_series);
        }
        if (!roiDragRef.current) {
          setRoi(normalizeRoi(status.roi || status.roi_box));
          setRoiSource(status.roi_source || null);
        }
        if (status.status === "done" || status.status === "failed") {
          clearInterval(timer);
        }
      } catch (err) {
        setError(err.message || "Video status failed.");
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

  useEffect(() => {
    function refreshRoiLayout() {
      setRoiLayoutVersion((version) => version + 1);
    }

    window.addEventListener("resize", refreshRoiLayout);
    return () => window.removeEventListener("resize", refreshRoiLayout);
  }, []);

  // Upload a video and initialize the job status view.
  async function handleVideoUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isAllowedVideo(file)) {
      setError("Unsupported video type. Use MP4, MOV, AVI, MKV, or WEBM.");
      return;
    }
    setError(null);
    stopAnnotatedStream();
    setJobId(null);
    setJob(null);
    setMetrics(DEFAULT_METRICS);
    setSeries([]);
    setRoi(null);
    setRoiSource(null);
    setSelectedRoiPoint(null);
    if (uploadedUrl) {
      URL.revokeObjectURL(uploadedUrl);
    }
    const nextPreviewUrl = URL.createObjectURL(file);
    setUploadedUrl(nextPreviewUrl);

    try {
      const data = await uploadVideo(file, { labels, conf });
      setJobId(data.job_id);
      setJob((prev) => ({
        ...(prev || {}),
        fps: data.fps,
        total_frames: data.total_frames,
      }));
    } catch (err) {
      setError(err.message || "Video upload failed.");
      setUploadedUrl(null);
      URL.revokeObjectURL(nextPreviewUrl);
    }
  }

  // Start MJPEG stream for the current job.
  function startUploadStream() {
    if (!jobId) return;
    stopAnnotatedStream();
    const params = new URLSearchParams({
      labels: String(labels),
      conf: String(conf),
      target_fps: String(targetFps),
    });
    setStreamUrl(`${API_BASE}/api/video/${jobId}/stream?${params.toString()}`);
    setAnnotating(true);
  }

  // Stop MJPEG stream and clear preview URL.
  function stopAnnotatedStream() {
    setAnnotating(false);
    setStreamUrl(null);
  }

  function getImageBox() {
    const panel = panelRef.current;
    const image = streamImageRef.current;
    if (!panel || !image) return null;

    const panelRect = panel.getBoundingClientRect();
    const imageRect = image.getBoundingClientRect();
    return {
      left: imageRect.left - panelRect.left,
      top: imageRect.top - panelRect.top,
      width: imageRect.width,
      height: imageRect.height,
    };
  }

  function roiOverlayStyle() {
    if (!roi) return null;
    const box = getImageBox();
    if (!box) return null;
    return {
      left: box.left,
      top: box.top,
      width: box.width,
      height: box.height,
    };
  }

  function beginRoiEdit(e, mode, pointIndex = null) {
    if (!roi || !jobId) return;
    const imageBox = getImageBox();
    if (!imageBox) return;

    e.preventDefault();
    setSelectedRoiPoint(pointIndex);
    e.currentTarget.setPointerCapture(e.pointerId);
    roiDragRef.current = {
      mode,
      pointIndex,
      startX: e.clientX,
      startY: e.clientY,
      startPoints: roi.points.map((point) => ({ ...point })),
      latestRoi: roi,
      imageBox,
    };
  }

  function updateRoiEdit(e) {
    const drag = roiDragRef.current;
    if (!drag) return;

    const dx = (e.clientX - drag.startX) / Math.max(drag.imageBox.width, 1);
    const dy = (e.clientY - drag.startY) / Math.max(drag.imageBox.height, 1);
    const points =
      drag.mode === "move"
        ? movePolygon(drag.startPoints, dx, dy)
        : drag.startPoints.map((point, index) =>
            index === drag.pointIndex
              ? clampPoint({ x: point.x + dx, y: point.y + dy })
              : point
          );
    const next = { type: "polygon", points };

    drag.latestRoi = next;
    setRoi(next);
    setRoiSource("manual");
  }

  async function finishRoiEdit(e) {
    const drag = roiDragRef.current;
    if (!drag || !jobId) return;

    const nextRoi = drag.latestRoi || roi;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Pointer capture can already be released by the browser.
    }
    roiDragRef.current = null;
    if (!nextRoi) return;
    try {
      const data = await setVideoRoi(jobId, nextRoi);
      setRoi(normalizeRoi(data.roi || data.roi_box));
      setRoiSource(data.roi_source);
    } catch (err) {
      setError(err.message || "ROI update failed.");
    }
  }

  async function persistRoi(nextRoi) {
    if (!jobId) return;
    setRoi(nextRoi);
    setRoiSource("manual");
    try {
      const data = await setVideoRoi(jobId, nextRoi);
      setRoi(normalizeRoi(data.roi || data.roi_box));
      setRoiSource(data.roi_source);
    } catch (err) {
      setError(err.message || "ROI update failed.");
    }
  }

  function handleAddRoiPoint() {
    if (!roi || roi.points.length < 3) return;

    let insertAfter = 0;
    let longest = -1;
    roi.points.forEach((point, index) => {
      const next = roi.points[(index + 1) % roi.points.length];
      const dx = next.x - point.x;
      const dy = next.y - point.y;
      const length = dx * dx + dy * dy;
      if (length > longest) {
        longest = length;
        insertAfter = index;
      }
    });

    const a = roi.points[insertAfter];
    const b = roi.points[(insertAfter + 1) % roi.points.length];
    const midpoint = clampPoint({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
    const points = [
      ...roi.points.slice(0, insertAfter + 1),
      midpoint,
      ...roi.points.slice(insertAfter + 1),
    ];
    setSelectedRoiPoint(insertAfter + 1);
    persistRoi({ type: "polygon", points });
  }

  function handleRemoveRoiPoint() {
    if (!roi || roi.points.length <= 3) return;
    const removeIndex =
      selectedRoiPoint !== null && selectedRoiPoint < roi.points.length
        ? selectedRoiPoint
        : roi.points.length - 1;
    const points = roi.points.filter((_, index) => index !== removeIndex);
    setSelectedRoiPoint(Math.min(removeIndex, points.length - 1));
    persistRoi({ type: "polygon", points });
  }

  async function handleResetRoi() {
    if (!jobId) return;
    try {
      const data = await resetVideoRoi(jobId);
      setRoi(normalizeRoi(data.roi || data.roi_box));
      setRoiSource(data.roi_source || null);
      setSelectedRoiPoint(null);
    } catch (err) {
      setError(err.message || "ROI reset failed.");
    }
  }

  const currentRoiStyle = roiOverlayStyle();

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
                    accept={VIDEO_ACCEPT}
                    className="hidden"
                    onChange={handleVideoUpload}
                  />
                </label>
                {jobId && (
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

            <div ref={panelRef} className={`${PREVIEW_PANEL_CLASS} relative`}>
              {error && <p className="text-rose-500">{error}</p>}
              {streamUrl ? (
                <>
                  <img
                    ref={streamImageRef}
                    src={streamUrl}
                    alt="Annotated"
                    className="max-h-full rounded-2xl shadow-lg"
                    onLoad={() => setRoiLayoutVersion((version) => version + 1)}
                  />
                  {currentRoiStyle && roi && (
                    <svg
                      className="absolute overflow-visible"
                      style={currentRoiStyle}
                      viewBox="0 0 100 100"
                      preserveAspectRatio="none"
                    >
                      <polygon
                        points={polygonPointsAttr(roi.points)}
                        className="cursor-move fill-emerald-400/20 stroke-emerald-400"
                        strokeWidth="0.45"
                        vectorEffect="non-scaling-stroke"
                        onPointerDown={(e) => beginRoiEdit(e, "move")}
                        onPointerMove={updateRoiEdit}
                        onPointerUp={finishRoiEdit}
                      />
                      {roi.points.map((point, index) => (
                        <circle
                          key={`${point.x}-${point.y}-${index}`}
                          cx={point.x * 100}
                          cy={point.y * 100}
                          r="1.4"
                          className={`cursor-grab stroke-white ${
                            selectedRoiPoint === index
                              ? "fill-amber-400"
                              : "fill-emerald-500"
                          }`}
                          strokeWidth="0.35"
                          vectorEffect="non-scaling-stroke"
                          onPointerDown={(e) => {
                            e.stopPropagation();
                            beginRoiEdit(e, "point", index);
                          }}
                          onPointerMove={updateRoiEdit}
                          onPointerUp={finishRoiEdit}
                        />
                      ))}
                    </svg>
                  )}
                  {roi && (
                    <div className="absolute right-4 top-4 flex gap-2">
                      <button
                        className="rounded-full bg-white/90 px-3 py-1 text-xs font-medium text-slate-600 shadow-soft disabled:opacity-50"
                        onClick={handleAddRoiPoint}
                        type="button"
                      >
                        Add Point
                      </button>
                      <button
                        className="rounded-full bg-white/90 px-3 py-1 text-xs font-medium text-slate-600 shadow-soft disabled:opacity-50"
                        onClick={handleRemoveRoiPoint}
                        disabled={roi.points.length <= 3}
                        type="button"
                      >
                        Remove Point
                      </button>
                      {roiSource === "manual" && (
                        <button
                          className="rounded-full bg-white/90 px-3 py-1 text-xs font-medium text-slate-600 shadow-soft"
                          onClick={handleResetRoi}
                          type="button"
                        >
                          Reset ROI
                        </button>
                      )}
                    </div>
                  )}
                </>
              ) : uploadedUrl ? (
                <video
                  ref={previewRef}
                  src={uploadedUrl}
                  className="max-h-full rounded-2xl shadow-lg"
                  controls
                />
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
              <StatCard
                title="PCE Count"
                value={metrics.pce_count?.toFixed?.(2) ?? metrics.pce_count}
              />
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
