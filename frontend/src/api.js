const BASE = 'http://localhost:8000/api'

async function req(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, opts)
  if (res.status === 429) throw new Error('RATE_LIMITED')
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return res.json()
}

export const api = {
  health: () => req('/health'),
  recentTransactions: (limit = 50) => req(`/transactions/recent?limit=${limit}`),
  statsSummary: () => req('/stats/summary'),
  clusters: () => req('/clusters'),
  simulateNormal: () => req('/simulate/normal', { method: 'POST' }),
  simulateAto: () => req('/simulate/ato', { method: 'POST' }),
  simulateBurst: (n = 15) => req(`/simulate/card_testing_burst?n=${n}`, { method: 'POST' }),
}
