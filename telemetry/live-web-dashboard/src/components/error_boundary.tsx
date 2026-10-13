import { Component, type ErrorInfo, type ReactNode } from "react";

type ErrorBoundaryState = { failed: boolean };

export class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Dashboard render failed", error, info);
  }

  render() {
    if (this.state.failed) {
      return (
        <main className="hud-grid-bg grid min-h-full place-items-center bg-hud-bg p-4 text-hud-text">
          <section className="max-w-md rounded border border-sig-red/50 bg-hud-panel p-5">
            <h1 className="text-base font-semibold">Dashboard could not render</h1>
            <p className="mt-2 text-sm text-hud-muted">Reload the page. If the problem returns, verify the telemetry payload and browser console.</p>
            <button className="mt-4 rounded border border-hud-edge bg-hud-raise px-3 py-2 text-sm" onClick={() => window.location.reload()}>
              Reload dashboard
            </button>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}
