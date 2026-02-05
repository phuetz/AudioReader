import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Type, BookOpen, Users, Mic, Shield, BookA, FolderKanban, Radio,
  FolderOpen, Settings, ChevronLeft, ChevronRight, Headphones,
} from 'lucide-react'
import { useState } from 'react'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/text', icon: Type, label: 'Texte' },
  { to: '/book', icon: BookOpen, label: 'Livre' },
  { to: '/characters', icon: Users, label: 'Personnages' },
  { to: '/cloning', icon: Mic, label: 'Clonage' },
  { to: '/acx', icon: Shield, label: 'ACX' },
  { to: '/corrections', icon: BookA, label: 'Corrections' },
  { to: '/projects', icon: FolderKanban, label: 'Projets' },
  { to: '/podcast', icon: Radio, label: 'Podcast' },
  { to: '/files', icon: FolderOpen, label: 'Fichiers' },
  { to: '/settings', icon: Settings, label: 'Paramètres' },
]

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={`flex flex-col bg-surface border-r border-border transition-all duration-200 ${
        collapsed ? 'w-16' : 'w-56'
      }`}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 px-4 h-14 border-b border-border">
        <Headphones className="w-6 h-6 text-accent shrink-0" />
        {!collapsed && <span className="font-semibold text-primary text-sm">AudioReader</span>}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'text-accent bg-accent/10 border-r-2 border-accent'
                  : 'text-secondary hover:text-primary hover:bg-panel'
              }`
            }
          >
            <Icon className="w-4.5 h-4.5 shrink-0" />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-center h-10 border-t border-border text-muted hover:text-primary transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  )
}
