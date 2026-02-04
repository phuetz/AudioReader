import type { SelectHTMLAttributes } from 'react'

interface Option {
  value: string
  label: string
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  options: Option[]
}

export default function Select({ label, options, className = '', id, ...props }: SelectProps) {
  const selectId = id || label?.toLowerCase().replace(/\s+/g, '-')
  return (
    <div className="space-y-1.5">
      {label && <label htmlFor={selectId} className="block text-xs font-medium text-secondary">{label}</label>}
      <select
        id={selectId}
        className={`w-full px-3 py-2 rounded-lg bg-panel border border-border text-primary text-sm
          focus:border-accent focus:ring-1 focus:ring-accent/30 transition-colors cursor-pointer ${className}`}
        {...props}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}
