export default function StatCard({ title, value, trend }) {
  return (
    <div className="glass rounded-2xl px-4 py-3 shadow-soft">
      <p className="text-xs uppercase text-slate-400">{title}</p>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-2xl font-semibold text-ink">{value}</span>
        {trend && <span className="text-xs text-emerald-500">{trend}</span>}
      </div>
    </div>
  );
}
