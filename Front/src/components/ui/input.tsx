import "./ui.css"
import * as React from "react"

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = "", ...props }, ref) => {
    return (
      <input ref={ref} className={`input ${className}`} {...props} />
    )
  }
)
Input.displayName = "Input"

export { Input } 