import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import MainLayout from './components/layout/MainLayout'
import FloatingPlayer from './components/global/FloatingPlayer'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const QuickTextPage = lazy(() => import('./pages/QuickTextPage'))
const BookConversionPage = lazy(() => import('./pages/BookConversionPage'))
const CharactersPage = lazy(() => import('./pages/CharactersPage'))
const VoiceCloningPage = lazy(() => import('./pages/VoiceCloningPage'))
const PodcastPage = lazy(() => import('./pages/PodcastPage'))
const FilesPage = lazy(() => import('./pages/FilesPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin" />
    </div>
  )
}

export default function App() {
  return (
    <MainLayout>
      <Suspense fallback={<PageLoader />}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/text" element={<QuickTextPage />} />
          <Route path="/book" element={<BookConversionPage />} />
          <Route path="/characters" element={<CharactersPage />} />
          <Route path="/cloning" element={<VoiceCloningPage />} />
          <Route path="/podcast" element={<PodcastPage />} />
          <Route path="/files" element={<FilesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </Suspense>
      <FloatingPlayer />
    </MainLayout>
  )
}
