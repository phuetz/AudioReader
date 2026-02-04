import type { HTMLAttributes, ReactNode } from 'react'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string
  action?: ReactNode
}

export default function Card({ title, action, children, className = '', ...props }: CardProps) {
  return (
    <div className={`bg-surface border border-border rounded-xl ${className}`} {...props}>
      {(title || action) && (
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border">
          {title && <h3 className="text-sm font-medium text-primary">{title}</h3>}
          {action}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  )
}
