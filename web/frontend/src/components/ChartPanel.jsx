import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from "recharts";

function chartMeta(windowMode) {
  if (windowMode === "hour") {
    return {
      subtitle: "Số xe duy nhất được gom theo giờ",
      badge: "Theo giờ",
    };
  }
  return {
    subtitle: "Số xe duy nhất được gom theo phút",
    badge: "Theo phút",
  };
}

// Line chart for vehicle count over time buckets.
export default function ChartPanel({ data, windowMode }) {
  const meta = chartMeta(windowMode);

  return (
    <div className="glass rounded-2xl px-4 py-1.5 shadow-soft">
      <div className="mb-1 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">Xu hướng lưu lượng xe</h3>
          <p className="text-xs text-slate-500">{meta.subtitle}</p>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs">
          {meta.badge}
        </span>
      </div>
      <div className="h-32 xl:h-36">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: 6, right: 10, top: 8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E6EAF7" />
            <XAxis dataKey="t" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#2563EB" strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
