import { useEffect, useState, useCallback } from 'react'
import { api } from './api'
import ControlPanel from './components/ControlPanel'
import StatsPanel from './components/StatsPanel'
import LiveFeed from './components/LiveFeed'

export default function App() {
  const [transactions, setTransactions] = useState([])
  const [stats, setStats] = useState(null)
  const [clusters, setClusters] = useState(null)
  const [connected, setConnected] = useState(null) // null = checking, true/false after
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const [tx, st] = await Promise.all([api.recentTransactions(50), api.statsSummary()])
      setTransactions(tx.transactions || [])
      setStats(st)
      setConnected(true)
      setError(null)
    } catch (e) {
      setConnected(false)
      setError('Cannot reach the Sentinel API at localhost:8000. Is the backend running?')
    }
  }, [])

  useEffect(() => {
    api.clusters().then(setClusters).catch(() => {})
    refresh()
    const interval = setInterval(refresh, 4000)
    return () => clearInterval(interval)
  }, [refresh])

  async function handleSimulate(kind) {
    setBusy(true)
    try {
      if (kind === 'normal') await api.simulateNormal()
      else if (kind === 'ato') await api.simulateAto()
      else if (kind === 'burst') await api.simulateBurst(15)
      await refresh()
    } catch (e) {
      setError('Simulation request failed — check the backend is running and reachable.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen bg-base text-ink">
      <header className="border-b border-hairline">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold tracking-tight">SENTINEL</h1>
            <span className="text-xs text-muted hidden sm:inline">
              coordinated abuse-ring &amp; fraud-spike risk console
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            {connected === null ? (
              <span className="text-muted">connecting…</span>
            ) : connected ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-allow live-dot" />
                <span className="text-muted">
                  chain {stats?.ledger_intact === false ? (
                    <span className="text-flagged font-medium">TAMPERED</span>
                  ) : (
                    <span className="text-allow">intact</span>
                  )}
                </span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-flagged" />
                <span className="text-flagged">disconnected</span>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-6">
        {error && (
          <div className="mb-4 text-sm text-flagged bg-flagged/10 border border-flagged/30 rounded-md px-4 py-3">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">
          <aside className="space-y-4 order-2 lg:order-1">
            <ControlPanel onSimulate={handleSimulate} busy={busy} />
            <StatsPanel stats={stats} clusters={clusters} />
          </aside>

          <section className="order-1 lg:order-2">
            <LiveFeed transactions={transactions} />
          </section>
        </div>
      </main>
    </div>
  )
}
