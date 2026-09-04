const PATH_LABELS = {
  auto_allow: 'Auto-allowed',
  llm_reasoned: 'AI-reasoned',
  auto_flag_obvious: 'Auto-flagged',
}

const PATH_COLORS = {
  auto_allow: 'bg-allow',
  llm_reasoned: 'bg-review',
  auto_flag_obvious: 'bg-flagged',
}

export default function StatsPanel({ stats, clusters }) {
  const total = stats?.total || 0
  const pathCounts = stats?.path_counts || {}

  return (
    <div className="bg-surface shadow-card border border-hairline rounded-md p-5">
      <h2 className="text-xs font-bold text-ink uppercase tracking-wide mb-5">
        Session Signal
      </h2>

      <div className="flex items-baseline gap-3 mb-6 pb-4 border-b border-hairline">
        <div className="mono text-4xl font-bold tracking-tight text-ink">{total}</div>
        <div className="text-xs text-muted uppercase tracking-wider font-semibold">
          Decisions<br/>Logged
        </div>
      </div>

      <div className="space-y-4 mb-6">
        {Object.keys(PATH_LABELS).map((key) => {
          const count = pathCounts[key] || 0
          const pct = total > 0 ? (count / total) * 100 : 0
          return (
            <div key={key}>
              <div className="flex justify-between text-xs font-semibold mb-1.5">
                <span className="text-muted tracking-wide">{PATH_LABELS[key]}</span>
                <span className="mono text-ink">{count}</span>
              </div>
              <div className="h-1.5 bg-surface2 rounded-sm overflow-hidden">
                <div
                  className={`h-full ${PATH_COLORS[key]} transition-all duration-500`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div className="border-t border-hairline pt-5">
        <div className="flex justify-between text-xs font-semibold mb-2">
          <span className="text-ink">Accounts in flagged clusters</span>
          <span className="mono text-flagged">{clusters?.n_flagged_accounts ?? '—'}</span>
        </div>
        {clusters?.by_type &&
          Object.entries(clusters.by_type).map(([type, count]) => (
            <div key={type} className="flex justify-between text-[11px] text-muted mt-1">
              <span className="pl-4 border-l-2 border-hairline">
                {type === 'coordinated_ring' ? 'Coordinated rings' : 'Card-testing bursts'}
              </span>
              <span className="mono">{count}</span>
            </div>
          ))}
      </div>
    </div>
  )
}
