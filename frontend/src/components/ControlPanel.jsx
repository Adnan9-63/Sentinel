export default function ControlPanel({ onSimulate, busy }) {
  const buttons = [
    { key: 'normal', label: 'Simulate normal transaction', tone: 'border-hairline hover:border-teal' },
    { key: 'ato', label: 'Simulate account takeover', tone: 'border-hairline hover:border-flagged' },
    { key: 'burst', label: 'Simulate card-testing burst', tone: 'border-hairline hover:border-review' },
  ]

  return (
    <div className="bg-surface border border-hairline rounded-lg p-4">
      <h2 className="text-xs font-semibold tracking-wider text-muted uppercase mb-3">
        Run a scenario
      </h2>
      <div className="flex flex-col gap-2">
        {buttons.map((b) => (
          <button
            key={b.key}
            disabled={busy}
            onClick={() => onSimulate(b.key)}
            className={`text-left text-sm px-3 py-2.5 rounded-md border bg-surface2 ${b.tone}
                        transition-colors disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            {b.label}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted mt-3 leading-relaxed">
        Each scenario runs a real transaction through the actual pipeline —
        feature computation, the ML ensemble, the triage gate — and logs a
        genuine decision to the audit ledger below.
      </p>
    </div>
  )
}
