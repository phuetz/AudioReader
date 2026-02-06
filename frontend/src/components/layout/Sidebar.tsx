import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Type, BookOpen, Users, Mic, Shield, BookA, FolderKanban, Radio,
  FolderOpen, Settings, ChevronLeft, ChevronRight, Headphones, ListOrdered, Sliders,
  Sun, Moon,
} from 'lucide-react'
import { useState } from 'react'
import { useSettingsStore } from '../../stores/useSettingsStore'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/text', icon: Type, label: 'Texte' },
  { to: '/book', icon: BookOpen, label: 'Livre' },
  { to: '/characters', icon: Users, label: 'Personnages' },
  { to: '/cloning', icon: Mic, label: 'Clonage' },
  { to: '/acx', icon: Shield, label: 'ACX' },
  { to: '/corrections', icon: BookA, label: 'Corrections' },
  { to: '/projects', icon: FolderKanban, label: 'Projets' },
  { to: '/queue', icon: ListOrdered, label: "File d'attente" },
  { to: '/voice-lab', icon: Sliders, label: 'Labo Voix' },
  { to: '/podcast', icon: Radio, label: 'Podcast' },
  { to: '/files', icon: FolderOpen, label: 'Fichiers' },
  { to: '/settings', icon: Settings, label: 'Paramètres' },
]

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const { theme, toggleTheme } = useSettingsStore()

  return (
    <aside
      className={`hidden md:flex flex-col bg-surface border-r border-border transition-all duration-200 ${
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

      {/* Theme toggle + collapse */}
      <div className="border-t border-border">
        <button
          onClick={toggleTheme}
          className="flex items-center gap-3 w-full px-4 py-2.5 text-sm text-muted hover:text-primary transition-colors"
          title={theme === 'dark' ? 'Mode clair' : 'Mode sombre'}
        >
          {theme === 'dark' ? <Sun className="w-4 h-4 shrink-0" /> : <Moon className="w-4 h-4 shrink-0" />}
          {!collapsed && <span>{theme === 'dark' ? 'Mode clair' : 'Mode sombre'}</span>}
        </button>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center w-full h-10 text-muted hover:text-primary transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </aside>
  )
}

/** Mobile bottom tab bar */
export function MobileTabBar() {
  const MOBILE_ITEMS = NAV_ITEMS.slice(0, 5) // Show first 5 items

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-surface border-t border-border flex justify-around py-1">
      {MOBILE_ITEMS.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `flex flex-col items-center gap-0.5 px-2 py-1.5 text-xs transition-colors ${
              isActive ? 'text-accent' : 'text-muted'
            }`
          }
        >
          <Icon className="w-5 h-5" />
          <span>{label}</span>
        </NavLink>
      ))}
    </nav>
  )
}
