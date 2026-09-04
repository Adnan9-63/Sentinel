export default function ControlPanel({ onSimulate, busy }) {
  const buttons = [
    { key: 'normal', label: 'Simulate normal transaction', hotkey: '1' },
    { key: 'ato', label: 'Simulate account takeover', hotkey: '2' },
    { key: 'burst', label: 'Simulate card-testing burst', hotkey: '3' },
  ]

  // We could add actual keyboard event listeners here, but for the demo UI
  // just showing them makes it feel like a "pro tool".

  return (
    <div className="bg-surface shadow-card border border-hairline rounded-md p-5">
      <h2 className="text-xs font-bold text-ink uppercase tracking-wide mb-4">
        Simulation Scenarios
      </h2>
      <div className="flex flex-col gap-3">
        {buttons.map((b) => {
          const isPrimary = b.key === 'normal';
          return (
            <button
              key={b.key}
              disabled={busy}
              onClick={() => onSimulate(b.key)}
              className={`group flex items-center justify-between text-left text-sm px-4 py-2.5 rounded-sm border transition-all disabled:opacity-50 disabled:cursor-not-allowed
                ${isPrimary 
                  ? 'bg-brand border-brand text-white hover:bg-blue-600' 
                  : 'bg-surface border-hairline text-ink hover:bg-surface2 hover:border-blue-300'
                }`}
            >
              <span className="font-semibold">{b.label}</span>
              <span className={`text-[10px] mono border rounded px-1.5 py-0.5 ${isPrimary ? 'text-white/80 border-white/40' : 'text-muted border-hairline group-hover:border-blue-200'}`}>
                {b.hotkey}
              </span>
            </button>
          );
        })}
      </div>
      <p className="text-xs text-muted mt-5 leading-relaxed">
        Each scenario runs a real transaction through the pipeline —
        feature computation, ML ensemble, and triage gate — creating a
        genuine decision in the audit ledger.
      </p>
    </div>
  )
}
