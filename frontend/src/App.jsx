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
    setError(null)
    try {
      if (kind === 'normal') await api.simulateNormal()
      else if (kind === 'ato') await api.simulateAto()
      else if (kind === 'burst') await api.simulateBurst(15)
      await refresh()
    } catch (e) {
      const msg = e.message === 'RATE_LIMITED'
        ? 'Slow down — rate limit hit. Wait a moment and try again.'
        : 'Simulation request failed — check the backend is running and reachable.'
      setError(msg)
      setTimeout(() => setError(null), 4000)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen bg-base text-ink flex flex-col font-sans">
      <div className="header-accent w-full" />
      <header className="bg-deep text-surface shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold tracking-tight">Razorpay Sentinel</h1>
            <div className="h-4 w-px bg-surface/20 hidden sm:block"></div>
            <span className="text-xs text-surface/80 font-medium uppercase tracking-wider">
              Risk Console
            </span>
            <span className="text-[10px] text-surface/60 mono border border-surface/20 px-1.5 py-0.5 rounded ml-2">
              SEN-01
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs mono">
            {connected === null ? (
              <span className="text-surface/70">connecting…</span>
            ) : connected ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-allow live-dot" />
                <span className="text-surface/90">
                  chain {stats?.ledger_intact === false ? (
                    <span className="text-flagged font-medium bg-surface px-1 rounded">TAMPERED</span>
                  ) : (
                    <span className="text-allow">intact</span>
                  )}
                </span>
              </>
            ) : (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-flagged" />
                <span className="text-flagged bg-surface px-1 rounded">disconnected</span>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">
        {error && (
          <div className="mb-6 text-sm text-flagged bg-red-50 border border-red-200 rounded-md px-4 py-3 shadow-sm">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-8">
          <aside className="space-y-6 order-2 lg:order-1">
            <ControlPanel onSimulate={handleSimulate} busy={busy} />
            <StatsPanel stats={stats} clusters={clusters} />
          </aside>

          <section className="order-1 lg:order-2">
            <LiveFeed transactions={transactions} />
          </section>
        </div>
      </main>

      <footer className="border-t border-hairline bg-surface py-3 mt-auto">
        <div className="max-w-6xl mx-auto px-6 flex justify-between items-center text-[11px] text-muted mono">
          <div className="flex gap-4">
            <span>SYSTEM STATUS: ONLINE</span>
            <span>MODEL: {stats?.total > 0 ? "AUTO_DETECT" : "STANDBY"}</span>
          </div>
          <div>
            TAIL: {transactions[0]?.id ? transactions[0].id.substring(0, 16) : 'N/A'}
          </div>
        </div>
      </footer>
    </div>
  )
}
