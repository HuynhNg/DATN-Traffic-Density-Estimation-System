export default function Header({ modes, mode, onModeChange }) {
  return (
    <header className="sticky top-0 z-20 w-full border-b border-slate-200 bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div>
          <p className="text-sm font-semibold text-slate-500">TrafficAI</p>
          <h1 className="text-xl font-semibold text-ink">Traffic Console</h1>
        </div>
        <div className="flex items-center gap-2 rounded-full bg-slate-100 p-1">
          {modes.map((m) => (
            <button
              key={m.id}
              className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                mode === m.id
                  ? "bg-white text-ink shadow"
                  : "text-slate-500 hover:text-ink"
              }`}
              onClick={() => onModeChange(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3">
          <button className="rounded-full bg-slate-100 px-4 py-2 text-sm font-medium">
            Export Logs
          </button>
          <div className="h-9 w-9 rounded-full bg-slate-200"></div>
        </div>
      </div>
    </header>
  );
}
