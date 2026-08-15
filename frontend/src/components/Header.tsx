import "./Header.css";

interface HeaderProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export function Header({ theme, onToggleTheme }: HeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__brand">
          <div className="app-header__mark" aria-hidden="true">
            ED
          </div>
          <div>
            <h1 className="app-header__title">Avoidable ED Utilization Navigator</h1>
            <p className="app-header__subtitle">
              Care management &middot; pattern detection &amp; lower-acuity navigation
            </p>
          </div>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
        >
          {theme === "light" ? "🌙" : "☀️"}
          <span>{theme === "light" ? "Dark" : "Light"}</span>
        </button>
      </div>
    </header>
  );
}
