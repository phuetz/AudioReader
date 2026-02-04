import { useEffect, useState } from 'react'
import { Activity } from 'lucide-react'
import { getHealth } from '../../api/endpoints'
import type { HealthResponse } from '../../api/types'

export default function TopBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {})
    const interval = setInterval(() => {
      getHealth().then(setHealth).catch(() => setHealth(null))
    }, 30_000)
    return () => clearInterval(interval)
  }, [])

  return (
    <header className="flex items-center justify-between px-6 h-14 bg-surface border-b border-border">
      <div />
      <div className="flex items-center gap-4">
        {/* Engine status */}
        <div className="flex items-center gap-2 text-xs text-secondary">
          <Activity className="w-3.5 h-3.5" />
          {health ? (
            <>
              {health.engines.kokoro && <span className="text-green">Kokoro</span>}
              {health.engines.edge_tts && <span className="text-cyan">Edge</span>}
              {health.engines.xtts && <span className="text-accent">XTTS</span>}
              <span className="text-muted">v{health.version}</span>
            </>
          ) : (
            <span className="text-red">API hors ligne</span>
          )}
        </div>
      </div>
    </header>
  )
}
