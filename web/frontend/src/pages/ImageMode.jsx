import { useMemo, useState } from "react";
import { detectImage } from "../api/client.js";
import Toggle from "../components/Toggle.jsx";
import DetectionList from "../components/DetectionList.jsx";
import { IMAGE_ACCEPT, isAllowedImage } from "../utils/fileTypes.js";

const PREVIEW_PANEL_CLASS =
  "mt-3 flex min-h-0 flex-1 items-center justify-center rounded-2xl border " +
  "border-dashed border-slate-200 bg-white/70";

// Image upload mode with detection preview and analytics.
export default function ImageMode() {
  const [file, setFile] = useState(null);
  const [image, setImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [timing, setTiming] = useState({ inference: null, processing: null });
  const [labels, setLabels] = useState(true);
  const [conf, setConf] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Build a data URL for the annotated image.
  const preview = useMemo(
    () => (image ? `data:image/jpeg;base64,${image}` : null),
    [image]
  );

  // Upload image to backend and hydrate UI with results.
  async function handleUpload(e) {
    const selected = e.target.files?.[0];
    if (!selected) return;

    if (!isAllowedImage(selected)) {
      setError("Định dạng ảnh chưa được hỗ trợ. Hãy dùng JPG, PNG hoặc WEBP.");
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
    } catch (err) {
      setError(err.message || "Nhận diện ảnh thất bại.");
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
    <section className="grid h-[calc(100dvh-5.75rem)] min-h-0 items-start gap-3 overflow-hidden lg:grid-cols-[1.75fr_0.65fr]">
      <div className="glass flex h-full min-h-0 flex-col rounded-3xl p-4 shadow-soft">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <label className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-white">
              Tải ảnh
              <input
                type="file"
                accept={IMAGE_ACCEPT}
                className="hidden"
                onChange={handleUpload}
              />
            </label>
            <button
              className="rounded-full border border-slate-200 px-4 py-2 text-sm"
              onClick={downloadResult}
              disabled={!image}
            >
              Xuất ảnh
            </button>
          </div>
          <div className="flex items-center gap-4">
            <Toggle label="Nhãn" checked={labels} onChange={setLabels} />
            <Toggle label="Độ tin cậy" checked={conf} onChange={setConf} />
          </div>
        </div>

        <div className={PREVIEW_PANEL_CLASS}>
          {error && <p className="text-rose-500">{error}</p>}
          {loading && <p className="text-slate-400">Đang xử lý...</p>}
          {!loading && preview && (
            <img
              src={preview}
              alt="Kết quả nhận diện"
              className="max-h-full rounded-2xl shadow-lg"
            />
          )}
          {!loading && !preview && (
            <p className="text-slate-400">Tải ảnh để bắt đầu</p>
          )}
        </div>
      </div>

      <div className="flex min-h-0 flex-col gap-3 self-start">
        <div className="glass rounded-2xl p-3 shadow-soft">
          <h3 className="text-base font-semibold">Phân tích ảnh</h3>
          <p className="text-xs text-slate-500">Tệp: {file?.name || "-"}</p>
          <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Đối tượng</p>
              <p className="text-lg font-semibold">{detections.length}</p>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Loại xe</p>
              <p className="text-lg font-semibold">
                {new Set(detections.map((d) => d.class_name)).size}
              </p>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Suy luận</p>
              <p className="text-lg font-semibold">{timing.inference ?? "-"} ms</p>
            </div>
            <div className="rounded-xl bg-white px-3 py-2">
              <p className="text-xs text-slate-400">Xử lý</p>
              <p className="text-lg font-semibold">{timing.processing ?? "-"} ms</p>
            </div>
          </div>
        </div>
        <DetectionList detections={detections} />
      </div>
    </section>
  );
}
