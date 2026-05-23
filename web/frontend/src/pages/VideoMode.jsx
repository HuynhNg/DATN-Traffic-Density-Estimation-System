import { useEffect, useMemo, useRef, useState } from "react";
import { getVideoStatus, getWsUrl, uploadVideo } from "../api/client.js";
import Toggle from "../components/Toggle.jsx";
import StatCard from "../components/StatCard.jsx";
import ChartPanel from "../components/ChartPanel.jsx";
import VideoControls from "../components/VideoControls.jsx";

export default function VideoMode() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const previewRef = useRef(null);
  const wsRef = useRef(null);
  const streamTimer = useRef(null);
  const streamSource = useRef(null);
  const frameLoopId = useRef(null);
  const lastSentTs = useRef(0);
  const lastChartTs = useRef(0);

  const [labels, setLabels] = useState(true);
  const [conf, setConf] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [frame, setFrame] = useState(null);
  const [metrics, setMetrics] = useState({ fps: 0, avg_objects: 0, objects_in_frame: 0 });
  const [series, setSeries] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [annotating, setAnnotating] = useState(false);
  const [targetFps, setTargetFps] = useState(12);
  const [streamUrl, setStreamUrl] = useState(null);

  const streamImage = useMemo(() => (frame ? `data:image/jpeg;base64,${frame}` : null), [frame]);

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      const status = await getVideoStatus(jobId);
      setJob(status);
      if (streamSource.current === "upload" && status.live_metrics) {
        setMetrics(status.live_metrics);
        if (Array.isArray(status.live_series)) {
          setSeries(status.live_series);
        }
      }
      if (status.status === "done" || status.status === "failed") {
        clearInterval(timer);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [jobId]);

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

  useEffect(() => {
    return () => {
      stopAnnotatedStream();
    };
  }, []);

  async function handleVideoUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (isRunning) {
      stopStream();
    }
    stopAnnotatedStream();
    setFrame(null);
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

  function openWebSocket() {
    wsRef.current = new WebSocket(getWsUrl());
    wsRef.current.onopen = () => {
      wsRef.current.send(JSON.stringify({ labels, conf }));
    };
    wsRef.current.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      setFrame(payload.image_b64);
      setMetrics(payload.metrics);
      const now = performance.now();
      if (now - lastChartTs.current > 500) {
        lastChartTs.current = now;
        setSeries((prev) => {
          const next = [...prev, { t: new Date().toLocaleTimeString(), count: payload.metrics.objects_in_frame }];
          return next.slice(-40);
        });
      }
    };
    wsRef.current.onclose = () => {
      if (streamSource.current === "webcam") {
        setIsRunning(false);
      }
      if (streamSource.current === "upload") {
        setAnnotating(false);
      }
      stopAnnotatedStream();
    };
    wsRef.current.onerror = () => {
      if (streamSource.current === "upload") {
        setAnnotating(false);
      }
    };
  }

  function startFrameLoop(sourceVideo, desiredFps) {
    const safeTarget = Math.max(1, Math.min(desiredFps || 12, 60));
    const minInterval = 1000 / safeTarget;
    lastSentTs.current = 0;

    const loop = (ts) => {
      frameLoopId.current = requestAnimationFrame(loop);
      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        return;
      }
      if (sourceVideo.readyState < 2 || sourceVideo.paused || sourceVideo.ended) {
        return;
      }
      if (wsRef.current.bufferedAmount > 2_000_000) {
        return;
      }
      if (ts - lastSentTs.current < minInterval) {
        return;
      }
      lastSentTs.current = ts;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = sourceVideo.videoWidth;
      canvas.height = sourceVideo.videoHeight;
      ctx.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then((buffer) => wsRef.current.send(buffer));
        }
      }, "image/jpeg", 0.8);
    };

    frameLoopId.current = requestAnimationFrame(loop);
  }

  async function startStream() {
    if (isRunning) return;
    setIsRunning(true);
    stopAnnotatedStream();
    const media = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = media;
    await videoRef.current.play();

    streamSource.current = "webcam";
    openWebSocket();
    startFrameLoop(videoRef.current, targetFps);
  }

  function stopStream() {
    setIsRunning(false);
    if (streamSource.current === "webcam") {
      stopAnnotatedStream();
    }
    const tracks = videoRef.current?.srcObject?.getTracks();
    tracks?.forEach((t) => t.stop());
  }

  function startUploadStream() {
    if (!jobId) return;
    stopAnnotatedStream();
    streamSource.current = "upload";
    const params = new URLSearchParams({
      labels: String(labels),
      conf: String(conf),
      target_fps: String(targetFps)
    });
    setStreamUrl(`http://localhost:8000/api/video/${jobId}/stream?${params.toString()}`);
    setAnnotating(true);
  }

  function stopAnnotatedStream() {
    if (streamTimer.current) clearInterval(streamTimer.current);
    if (frameLoopId.current) cancelAnimationFrame(frameLoopId.current);
    if (wsRef.current) wsRef.current.close();
    streamTimer.current = null;
    frameLoopId.current = null;
    wsRef.current = null;
    streamSource.current = null;
    setAnnotating(false);
    setStreamUrl(null);
  }

  return (
    <section className="grid gap-6">
      <div className="space-y-6">
        <div className="glass rounded-3xl p-6 shadow-soft">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <label className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-white">
                Upload Video
                <input type="file" accept="video/*" className="hidden" onChange={handleVideoUpload} />
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
              <VideoControls onStart={startStream} onStop={stopStream} isRunning={isRunning} />
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
            {streamImage ? (
              <img src={streamImage} alt="Realtime" className="max-h-full rounded-2xl shadow-lg" />
            ) : streamUrl ? (
              <img src={streamUrl} alt="Annotated" className="max-h-full rounded-2xl shadow-lg" />
            ) : uploadedUrl ? (
              <video ref={previewRef} src={uploadedUrl} className="max-h-full rounded-2xl shadow-lg" controls />
            ) : (
              <p className="text-slate-400">Start webcam stream or upload a video</p>
            )}
          </div>
          <video ref={videoRef} className="hidden" playsInline />
          <canvas ref={canvasRef} className="hidden" />
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <StatCard title="Avg Vehicles" value={metrics.avg_objects} />
          <StatCard title="Active Objects" value={metrics.objects_in_frame} />
        </div>

        <ChartPanel data={series} />
      </div>
    </section>
  );
}
