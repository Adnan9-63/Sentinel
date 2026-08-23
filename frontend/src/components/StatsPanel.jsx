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
    <div className="bg-surface border border-hairline rounded-lg p-4">
      <h2 className="text-xs font-semibold tracking-wider text-muted uppercase mb-3">
        Session signal
      </h2>

      <div className="mono text-3xl font-semibold mb-1">{total}</div>
      <div className="text-xs text-muted mb-4">decisions logged this session</div>

      <div className="space-y-2 mb-4">
        {Object.keys(PATH_LABELS).map((key) => {
          const count = pathCounts[key] || 0
          const pct = total > 0 ? (count / total) * 100 : 0
          return (
            <div key={key}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-muted">{PATH_LABELS[key]}</span>
                <span className="mono">{count}</span>
              </div>
              <div className="h-1.5 bg-surface2 rounded-full overflow-hidden">
                <div
                  className={`h-full ${PATH_COLORS[key]} transition-all duration-500`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      <div className="border-t border-hairline pt-3">
        <div className="flex justify-between text-xs">
          <span className="text-muted">Accounts in flagged clusters</span>
          <span className="mono">{clusters?.n_flagged_accounts ?? '—'}</span>
        </div>
        {clusters?.by_type &&
          Object.entries(clusters.by_type).map(([type, count]) => (
            <div key={type} className="flex justify-between text-xs mt-1">
              <span className="text-muted pl-3">
                {type === 'coordinated_ring' ? 'Coordinated rings' : 'Card-testing bursts'}
              </span>
              <span className="mono">{count}</span>
            </div>
          ))}
      </div>
    </div>
  )
}
