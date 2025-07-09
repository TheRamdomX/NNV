import "./ui.css"
import * as React from "react"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "secondary"
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = "", variant = "default", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={`button${variant === "secondary" ? " secondary" : ""} ${className}`}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button } 