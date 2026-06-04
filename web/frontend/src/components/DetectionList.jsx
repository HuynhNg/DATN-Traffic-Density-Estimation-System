// Render a scrollable list of detections with confidence and bounds.
export default function DetectionList({ detections }) {
  return (
    <div className="glass h-full rounded-2xl p-4 shadow-soft">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-base font-semibold">Detections</h3>
        <span className="text-xs text-slate-500">{detections.length} objects</span>
      </div>
      <div className="space-y-2 overflow-y-auto pr-2" style={{ maxHeight: 320 }}>
        {detections.map((det) => (
          <div
            key={det.object_id}
            className="rounded-xl border border-slate-200 bg-white px-3 py-2"
          >
            <div className="flex items-center justify-between text-sm font-medium">
              <span>{det.class_name.toUpperCase()}_{det.object_id}</span>
              <span className="text-xs text-slate-500">{(det.confidence * 100).toFixed(1)}%</span>
            </div>
            <p className="text-xs text-slate-500">
              [{det.x1}, {det.y1}] - [{det.x2}, {det.y2}]
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
