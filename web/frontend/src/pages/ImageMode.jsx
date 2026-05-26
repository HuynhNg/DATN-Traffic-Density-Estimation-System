import { useMemo, useState } from "react";
import { detectImage } from "../api/client.js";
import Toggle from "../components/Toggle.jsx";
import DetectionList from "../components/DetectionList.jsx";

// Image upload mode with detection preview and analytics.
export default function ImageMode() {
  const allowedImageTypes = ["image/jpeg", "image/png", "image/webp"];
  const allowedImageExts = [".jpg", ".jpeg", ".png", ".webp"];

  const [file, setFile] = useState(null);
  const [image, setImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [timing, setTiming] = useState({ inference: null, processing: null });
  const [labels, setLabels] = useState(true);
  const [conf, setConf] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Build a data URL for the annotated image.
  const preview = useMemo(() => (image ? `data:image/jpeg;base64,${image}` : null), [image]);

  // Upload image to backend and hydrate UI with results.
  async function handleUpload(e) {
    const selected = e.target.files?.[0];
    if (!selected) return;
    const ext = selected.name.toLowerCase().slice(selected.name.lastIndexOf("."));
    const isAllowed =
      allowedImageTypes.includes(selected.type) ||
      (ext && allowedImageExts.includes(ext));
    if (!isAllowed) {
      setError("Unsupported image type. Use JPG, PNG, or WEBP.");
      return;
    }
    setError(null);
    setFile(selected);
    setLoading(true);

    try {
      const data = await detectImage(selected, { labels, conf });
      setImage(data.image_b64);
      setDetections(data.detections || []);
      setTiming({ inference: data.inference_ms, processing: data.processing_ms });
    } finally {
      setLoading(false);
    }
  }

  // Download the annotated image as a file.
  function downloadResult() {
    if (!preview) return;
    const link = document.createElement("a");
    link.href = preview;
    link.download = "trafficai_result.jpg";
    link.click();
  }

  return (
    <section className="grid gap-6 lg:grid-cols-[1.4fr_0.6fr]">
      <div className="glass rounded-3xl p-6 shadow-soft">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <label className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-white">
              Upload Image
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={handleUpload}
              />
            </label>
            <button
              className="rounded-full border border-slate-200 px-4 py-2 text-sm"
              onClick={downloadResult}
              disabled={!image}
            >
              Export
            </button>
          </div>
          <div className="flex items-center gap-4">
            <Toggle label="Labels" checked={labels} onChange={setLabels} />
            <Toggle label="Conf" checked={conf} onChange={setConf} />
          </div>
        </div>

        <div className="mt-6 flex h-[520px] items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-white/70">
          {error && <p className="text-rose-500">{error}</p>}
          {loading && <p className="text-slate-400">Processing...</p>}
          {!loading && preview && (
            <img src={preview} alt="Detection" className="max-h-full rounded-2xl shadow-lg" />
          )}
          {!loading && !preview && <p className="text-slate-400">Upload an image to start</p>}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="glass rounded-2xl p-4 shadow-soft">
          <h3 className="text-base font-semibold">Image Analytics</h3>
          <p className="text-xs text-slate-500">File: {file?.name || "-"}</p>
          <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Objects</p>
              <p className="text-lg font-semibold">{detections.length}</p>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Classes</p>
              <p className="text-lg font-semibold">{new Set(detections.map((d) => d.class_name)).size}</p>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Inference</p>
              <p className="text-lg font-semibold">{timing.inference ?? "-"} ms</p>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Processing</p>
              <p className="text-lg font-semibold">{timing.processing ?? "-"} ms</p>
            </div>
          </div>
        </div>
        <DetectionList detections={detections} />
      </div>
    </section>
  );
}
