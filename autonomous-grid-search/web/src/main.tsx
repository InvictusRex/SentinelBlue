import React from "react";
import ReactDOM from "react-dom/client";
import { MissionPlanner } from "./components/mission_planner";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <div className="hud-grid-bg flex min-h-full flex-col bg-hud-bg font-sans text-hud-text">
      <MissionPlanner currentLatitude={0} currentLongitude={0} telemetryAvailable={false} />
    </div>
  </React.StrictMode>
);
