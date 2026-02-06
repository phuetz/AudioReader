import { useEffect, type ReactNode } from 'react'
import Sidebar, { MobileTabBar } from './Sidebar'
import TopBar from './TopBar'
import { useAudioPlayer } from '../../hooks/useAudioPlayer'
import { useGlobalSSE } from '../../hooks/useGlobalSSE'
import { useSettingsStore } from '../../stores/useSettingsStore'

export default function MainLayout({ children }: { children: ReactNode }) {
  // Initialize global audio player
  useAudioPlayer()

  // Initialize global SSE for real-time job updates
  useGlobalSSE()

  // Apply theme on mount
  const theme = useSettingsStore(s => s.theme)
  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light')
  }, [theme])

  return (
    <div className="flex h-screen bg-deep text-primary overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <TopBar />
        <main className="flex-1 overflow-y-auto p-4 md:p-6 pb-20 md:pb-6">
          {children}
        </main>
      </div>
      <MobileTabBar />
    </div>
  )
}
