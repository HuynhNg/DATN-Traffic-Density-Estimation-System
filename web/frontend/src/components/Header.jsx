// Top navigation with mode switching controls.
export default function Header({ modes, mode, onModeChange }) {
  return (
    <header
      className="sticky top-0 z-20 w-full border-b border-slate-200 bg-white/80 backdrop-blur"
    >
      <div className="mx-auto grid max-w-7xl grid-cols-[1fr_auto_1fr] items-center px-6 py-3">
        <div>
          <p className="text-xs font-semibold text-slate-500">TrafficAI</p>
          <h1 className="text-lg font-semibold text-ink">Bảng điều khiển giao thông</h1>
        </div>
        <div className="flex items-center gap-2 rounded-full bg-slate-100 p-1">
          {modes.map((m) => (
            <button
              key={m.id}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition ${
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
        <div />
      </div>
    </header>
  );
}
