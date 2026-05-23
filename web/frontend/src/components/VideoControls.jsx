export default function VideoControls({ onStart, onStop, isRunning }) {
  return (
    <div className="flex items-center gap-3">
      <button
        className={`rounded-full px-4 py-2 text-sm font-medium ${
          isRunning ? "bg-slate-200 text-slate-500" : "bg-accent text-white"
        }`}
        onClick={onStart}
        disabled={isRunning}
      >
        Start Stream
      </button>
      <button
        className={`rounded-full px-4 py-2 text-sm font-medium ${
          isRunning ? "bg-rose-500 text-white" : "bg-slate-200 text-slate-500"
        }`}
        onClick={onStop}
        disabled={!isRunning}
      >
        Stop
      </button>
    </div>
  );
}
