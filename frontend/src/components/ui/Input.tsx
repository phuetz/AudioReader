import type { InputHTMLAttributes } from 'react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
}

export default function Input({ label, className = '', id, ...props }: InputProps) {
  const inputId = id || label?.toLowerCase().replace(/\s+/g, '-')
  return (
    <div className="space-y-1.5">
      {label && <label htmlFor={inputId} className="block text-xs font-medium text-secondary">{label}</label>}
      <input
        id={inputId}
        className={`w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm
          placeholder:text-muted focus:border-accent focus:ring-1 focus:ring-accent/30
          transition-colors ${className}`}
        {...props}
      />
    </div>
  )
}
