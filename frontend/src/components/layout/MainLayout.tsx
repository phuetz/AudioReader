import type { ReactNode } from 'react'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import { useAudioPlayer } from '../../hooks/useAudioPlayer'

export default function MainLayout({ children }: { children: ReactNode }) {
  // Initialize global audio player
  useAudioPlayer()

  return (
    <div className="flex h-screen bg-deep text-primary overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <TopBar />
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
