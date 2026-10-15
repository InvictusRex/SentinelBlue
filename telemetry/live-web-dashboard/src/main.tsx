import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./app";
import { ErrorBoundary } from "./components/error_boundary";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
