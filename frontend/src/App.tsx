import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import MainLayout from './components/layout/MainLayout'
import FloatingPlayer from './components/global/FloatingPlayer'
import SearchBar from './components/global/SearchBar'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const QuickTextPage = lazy(() => import('./pages/QuickTextPage'))
const BookConversionPage = lazy(() => import('./pages/BookConversionPage'))
const CharactersPage = lazy(() => import('./pages/CharactersPage'))
const VoiceCloningPage = lazy(() => import('./pages/VoiceCloningPage'))
const ACXAnalysisPage = lazy(() => import('./pages/ACXAnalysisPage'))
const CorrectionsPage = lazy(() => import('./pages/CorrectionsPage'))
const ProjectsPage = lazy(() => import('./pages/ProjectsPage'))
const PodcastPage = lazy(() => import('./pages/PodcastPage'))
const FilesPage = lazy(() => import('./pages/FilesPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))
const ReviewPage = lazy(() => import('./pages/ReviewPage'))
const QueuePage = lazy(() => import('./pages/QueuePage'))
const VoiceLabPage = lazy(() => import('./pages/VoiceLabPage'))

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
          <Route path="/acx" element={<ACXAnalysisPage />} />
          <Route path="/corrections" element={<CorrectionsPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/podcast" element={<PodcastPage />} />
          <Route path="/files" element={<FilesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/review/:jobId" element={<ReviewPage />} />
          <Route path="/queue" element={<QueuePage />} />
          <Route path="/voice-lab" element={<VoiceLabPage />} />
        </Routes>
      </Suspense>
      <FloatingPlayer />
      <SearchBar />
    </MainLayout>
  )
}
