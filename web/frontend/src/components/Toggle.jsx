export default function Toggle({ label, checked, onChange }) {
  return (
    <label className="flex items-center gap-2 text-sm text-slate-600">
      <span>{label}</span>
      <button
        className={`relative h-6 w-12 rounded-full transition ${
          checked ? "bg-accent" : "bg-slate-200"
        }`}
        onClick={() => onChange(!checked)}
        type="button"
      >
        <span
          className={`absolute top-1 h-4 w-4 rounded-full bg-white transition ${
            checked ? "left-7" : "left-1"
          }`}
        ></span>
      </button>
    </label>
  );
}
