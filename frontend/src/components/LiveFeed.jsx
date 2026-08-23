import { useState } from 'react'

const STATUS_STYLE = {
  allowed: { dot: 'bg-allow', text: 'text-allow', label: 'Allowed' },
  flagged_for_review: { dot: 'bg-flagged', text: 'text-flagged', label: 'Flagged for review' },
}

const PATH_LABEL = {
  auto_allow: 'auto-allowed · below threshold',
  llm_reasoned: 'AI-reasoned · ambiguous band',
  auto_flag_obvious: 'auto-flagged · above threshold',
}

function shortHash(h) {
  return h ? h.slice(0, 10) : '—'
}

function riskColor(score) {
  if (score < 0.4) return 'text-allow'
  if (score < 0.85) return 'text-review'
  return 'text-flagged'
}

function Entry({ entry, isLast, expanded, onToggle }) {
  const style = STATUS_STYLE[entry.final_status] || STATUS_STYLE.flagged_for_review
  return (
    <div className="relative pl-8">
      {/* chain link line down to next entry */}
      {!isLast && (
        <div className="absolute left-[13px] top-7 bottom-[-12px] w-px bg-hairline" />
      )}
      <div className={`absolute left-2 top-1.5 w-3 h-3 rounded-full ${style.dot}`} />

      <button
        onClick={onToggle}
        className="w-full text-left bg-surface hover:bg-surface2 border border-hairline
                   rounded-md px-3 py-2.5 mb-3 transition-colors"
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <span className="mono text-xs text-muted truncate">{entry.transaction_id}</span>
            {entry.simulated && (
              <span className="text-[10px] uppercase tracking-wider text-teal border border-teal/30 rounded px-1.5 py-0.5 shrink-0">
                live
              </span>
            )}
          </div>
          <div className="flex items-center gap-4 shrink-0">
            <span className={`mono text-sm font-semibold ${riskColor(entry.risk_score)}`}>
              {entry.risk_score?.toFixed(3)}
            </span>
            <span className={`text-xs font-medium ${style.text}`}>{style.label}</span>
          </div>
        </div>

        {expanded && (
          <div className="mt-3 pt-3 border-t border-hairline space-y-2">
            <div className="text-xs text-muted">{PATH_LABEL[entry.path]}</div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted mb-1">Evidence</div>
              <ul className="space-y-1">
                {(entry.evidence || []).map((e, i) => (
                  <li key={i} className="text-xs text-ink/90 flex gap-2">
                    <span className="text-teal shrink-0">·</span>
                    <span>{e}</span>
                  </li>
                ))}
              </ul>
            </div>
            {entry.ring_context && (
              <div className="text-xs text-review">
                Member of a flagged {entry.ring_context.cluster_type === 'coordinated_ring' ? 'coordinated ring' : 'card-testing cluster'}
                {' '}({entry.ring_context.cluster_size} accounts, ring score {entry.ring_context.ring_risk_score?.toFixed(2)})
              </div>
            )}
            <div className="flex gap-4 pt-1">
              <span className="mono text-[10px] text-muted">
                hash: {shortHash(entry.entry_hash)}
              </span>
              <span className="mono text-[10px] text-muted">
                prev: {shortHash(entry.prev_hash)}
              </span>
            </div>
          </div>
        )}
      </button>
    </div>
  )
}

export default function LiveFeed({ transactions }) {
  const [expandedId, setExpandedId] = useState(null)

  if (!transactions || transactions.length === 0) {
    return (
      <div className="bg-surface border border-hairline rounded-lg p-8 text-center">
        <p className="text-muted text-sm">
          No decisions logged yet. Run a scenario from the panel to see the pipeline work.
        </p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <span className="w-2 h-2 rounded-full bg-teal live-dot" />
        <h2 className="text-xs font-semibold tracking-wider text-muted uppercase">
          Live decision chain
        </h2>
        <span className="text-xs text-muted">— each entry is cryptographically linked to the one before it</span>
      </div>
      <div>
        {transactions.map((entry, i) => (
          <Entry
            key={entry.transaction_id + i}
            entry={entry}
            isLast={i === transactions.length - 1}
            expanded={expandedId === entry.transaction_id}
            onToggle={() =>
              setExpandedId(expandedId === entry.transaction_id ? null : entry.transaction_id)
            }
          />
        ))}
      </div>
    </div>
  )
}
