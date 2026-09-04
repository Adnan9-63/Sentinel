import { useState } from 'react'

const STATUS_STYLE = {
  allowed: { dot: 'bg-allow', text: 'text-allow', bg: 'bg-allow/10', label: 'Allowed' },
  flagged_for_review: { dot: 'bg-flagged', text: 'text-flagged', bg: 'bg-flagged/10', label: 'Flagged for review' },
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
  if (score < 0.4) return 'bg-allow text-allow'
  if (score < 0.85) return 'bg-review text-review'
  return 'bg-flagged text-flagged'
}

function Entry({ entry, isLast, expanded, onToggle }) {
  const style = STATUS_STYLE[entry.final_status] || STATUS_STYLE.flagged_for_review
  const rColorObj = riskColor(entry.risk_score)
  const rBg = rColorObj.split(' ')[0]
  const rText = rColorObj.split(' ')[1]
  const fillWidth = Math.min(Math.max((entry.risk_score || 0) * 100, 2), 100) + '%'

  return (
    <div className="relative pl-8">
      {/* chain link line down to next entry */}
      {!isLast && (
        <div className="absolute left-[13px] top-7 bottom-[-12px] w-px bg-hairline" />
      )}
      <div className={`absolute left-2 top-3.5 w-2.5 h-2.5 rounded-full ${style.dot}`} />

      <button
        onClick={onToggle}
        className="w-full text-left bg-surface shadow-card hover:bg-surface2 border border-hairline
                   rounded-md px-4 py-3 mb-3 transition-colors"
      >
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <span className="mono text-xs font-medium text-ink truncate">{entry.transaction_id}</span>
            {entry.simulated && (
              <span className="text-[10px] uppercase font-bold tracking-wider text-brand bg-brand/10 border border-brand/20 rounded px-1.5 py-0.5 shrink-0">
                LIVE
              </span>
            )}
          </div>
          <div className="flex items-center gap-4 shrink-0">
            <div className="flex items-center gap-2 w-28 justify-end">
              <span className={`mono text-xs font-bold ${rText}`}>
                {entry.risk_score?.toFixed(3)}
              </span>
              <div className="w-16 risk-bar shrink-0">
                <div className={`risk-bar-fill ${rBg}`} style={{ width: fillWidth }} />
              </div>
            </div>
            <span className={`text-[11px] font-bold w-32 text-center rounded px-2 py-1 uppercase tracking-wider ${style.text} ${style.bg}`}>
              {style.label}
            </span>
          </div>
        </div>

        {expanded && (
          <div className="mt-4 pt-3 border-t border-hairline space-y-3">
            <div className="text-xs text-muted font-medium">{PATH_LABEL[entry.path]}</div>
            <div>
              <div className="text-[10px] uppercase font-bold tracking-widest text-muted mb-2">Evidence</div>
              <ul className="space-y-1.5">
                {(entry.evidence || []).map((e, i) => (
                  <li key={i} className="text-sm text-ink flex gap-2">
                    <span className="text-brand shrink-0 font-bold">·</span>
                    <span className="leading-snug">{e}</span>
                  </li>
                ))}
              </ul>
            </div>
            {entry.grounding_warnings && entry.grounding_warnings.length > 0 && (
              <div className="bg-review/10 border border-review/30 rounded-md p-3">
                <div className="text-[10px] font-bold uppercase tracking-widest text-review mb-1.5">
                  Grounding check flagged this
                </div>
                <ul className="space-y-1">
                  {entry.grounding_warnings.map((w, i) => (
                    <li key={i} className="text-[13px] text-review flex gap-2 font-medium">
                      <span className="shrink-0">⚠</span>
                      <span>{w}</span>
                    </li>
                  ))}
                </ul>
                <div className="text-xs text-review/80 mt-2">
                  A claim in the AI's own reasoning didn't match the real input data —
                  forced to human review regardless of what the AI recommended.
                </div>
              </div>
            )}
            {entry.ring_context && (
              <div className="text-xs text-flagged font-medium mt-1 bg-flagged/10 px-2 py-1 rounded inline-block">
                Member of a flagged {entry.ring_context.cluster_type === 'coordinated_ring' ? 'coordinated ring' : 'card-testing cluster'}
                {' '}({entry.ring_context.cluster_size} accounts, ring score {entry.ring_context.ring_risk_score?.toFixed(2)})
              </div>
            )}
            <div className="flex gap-4 pt-2">
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
      <div className="bg-surface shadow-card border border-hairline rounded-md p-8 text-center h-48 flex items-center justify-center">
        <p className="text-muted text-sm max-w-sm">
          No decisions logged yet. Run a scenario from the panel to see the pipeline work.
        </p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-5">
        <span className="w-2 h-2 rounded-full bg-brand live-dot" />
        <h2 className="text-xs font-bold tracking-wide text-ink uppercase">
          Live Decision Chain
        </h2>
        <span className="text-xs text-muted font-medium ml-2">— each entry is cryptographically linked to the one before it</span>
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
