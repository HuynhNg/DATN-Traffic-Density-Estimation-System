// Small metric card for numeric highlights.
export default function StatCard({ title, value, trend, className = "" }) {
  return (
    <div className={`glass rounded-xl px-3 py-1.5 shadow-soft ${className}`}>
      <p className="text-[10px] uppercase leading-none text-slate-400">{title}</p>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-base font-semibold leading-none text-ink">{value}</span>
        {trend && <span className="text-xs text-emerald-500">{trend}</span>}
      </div>
    </div>
  );
}
