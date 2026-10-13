import { lazy, Suspense, useEffect, useState } from "react";
import { GaugeIcon, PlugZapIcon, SlidersHorizontalIcon, WavesIcon, WifiIcon } from "lucide-react";
import { AltitudeCard } from "./components/altitude_card";
import { AttitudeIndicator } from "./components/attitude_indicator";
import { BatteryBox } from "./components/battery_box";
import { FlightStatusCard } from "./components/flight_status_card";
import { LinkCard } from "./components/link_card";
import { NavSignalBox } from "./components/nav_signal_box";
import { MissionPlanner } from "./components/mission_planner";
import { SbcStatusCard } from "./components/sbc_status_card";
import { TopBar } from "./components/top_bar";
import { gyroSeries, motorSeries, pidSeries, vehicle } from "./data/telemetry";
import { useTelemetry } from "./hooks/use_telemetry";
import { telemetryEndpointFromInput } from "./lib/telemetry_endpoint";
import type { Telemetry } from "./types/telemetry";

const ChartPanel = lazy(() => import("./components/chart_panel").then((module) => ({ default: module.ChartPanel })));

function activeWarnings(t: Telemetry): string[] {
  const warnings: string[] = [];
  if (t.source.status === "offline") warnings.push("ESP JSON offline · armed state unknown");
  if (t.source.status === "stale") warnings.push(
    t.source.message.includes("NRF") ? "NRF packets stale · armed state unknown" : "ESP JSON response delayed · armed state unknown"
  );
  if (t.source.status === "online" && t.flight.armed === null) warnings.push("Armed state unavailable");
  if (t.flight.failsafe === true) warnings.push("Flight controller failsafe triggered");
  if (t.source.status === "online" && t.battery.total > 0 && t.battery.percent < 25) warnings.push("Battery low · RTL advised");
  if (t.source.status === "online" && t.link.loss > 5) warnings.push("Downlink packet loss");
  if (t.sbc.available && t.sbc.temp > 72) warnings.push("SBC thermal limit");
  if (t.source.status === "online" && t.nav.fix !== "NO DATA" && t.nav.hdop > 1.2) warnings.push("GPS HDOP degraded");
  return warnings;
}

function sourceTone(status: Telemetry["source"]["status"]): string {
  if (status === "online") return "bg-sig-green";
  if (status === "stale") return "bg-sig-amber";
  return "bg-sig-red";
}

const ESP_IP_STORAGE_KEY = "droneTelemetryEspIp";
const CELL_COUNT_STORAGE_KEY = "droneTelemetryCellCount";

function initialEspInput(): string {
  const fromQuery = new URLSearchParams(window.location.search).get("telemetryUrl");
  if (fromQuery) return fromQuery;
  return window.localStorage.getItem(ESP_IP_STORAGE_KEY) ?? "";
}

function initialCellCount(): number {
  const stored = Number(window.localStorage.getItem(CELL_COUNT_STORAGE_KEY));
  return Number.isInteger(stored) && stored >= 2 && stored <= 6 ? stored : 4;
}

function ConnectionPanel({
  value,
  cellCount,
  error,
  onChange,
  onCellCountChange,
  onSubmit
}: {
  value: string;
  cellCount: number;
  error: string;
  onChange: (value: string) => void;
  onCellCountChange: (value: number) => void;
  onSubmit: () => void;
}) {
  return (
    <div className="hud-grid-bg flex min-h-full w-full items-center justify-center bg-hud-bg p-4 font-sans text-hud-text">
      <form
        className="w-full max-w-[440px] rounded border border-hud-line bg-hud-panel p-4 shadow-[0_18px_60px_rgba(0,0,0,0.35)]"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <div className="mb-4 flex items-center gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded border border-teal-400/30 bg-teal-400/10 text-white">
            <WifiIcon size={18} />
          </span>
          <div className="min-w-0">
            <h1 className="text-sm font-semibold uppercase tracking-[0.08em] text-hud-text">Connect ESP32 Telemetry</h1>
            <p className="mt-1 text-xs leading-5 text-hud-muted">Enter the ESP32 address or a secure telemetry URL.</p>
          </div>
        </div>

        <label className="mb-2 block font-mono text-2xs uppercase tracking-[0.12em] text-hud-faint" htmlFor="esp-ip">
          ESP32 IP or telemetry URL
        </label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            id="esp-ip"
            autoFocus
            value={value}
            autoComplete="off"
            autoCapitalize="none"
            spellCheck={false}
            inputMode="url"
            onChange={(event) => onChange(event.target.value)}
            placeholder="10.160.142.17 or https://telemetry.example/api"
            className="h-10 min-w-0 flex-1 rounded border border-hud-line bg-hud-bg px-3 font-mono text-sm tabular text-hud-text outline-none transition focus:border-teal-300 focus:ring-2 focus:ring-teal-300/20"
          />
          <button
            type="submit"
            className="inline-flex h-10 items-center justify-center gap-2 rounded border border-teal-300/40 bg-teal-300/15 px-4 font-mono text-2xs font-semibold uppercase tracking-[0.12em] text-white transition hover:border-teal-200 hover:bg-teal-300/20 focus:outline-none focus:ring-2 focus:ring-teal-300/30"
          >
            <PlugZapIcon size={14} />
            Connect
          </button>
        </div>

        <label className="mt-3 flex items-center justify-between gap-3 font-mono text-xs uppercase tracking-[0.08em] text-hud-faint">
          Battery cells
          <select
            value={cellCount}
            onChange={(event) => onCellCountChange(Number(event.target.value))}
            className="h-9 rounded border border-hud-line bg-hud-bg px-3 text-sm text-hud-text outline-none focus:border-teal-300 focus:ring-2 focus:ring-teal-300/20"
          >
            {[2, 3, 4, 5, 6].map((count) => <option key={count} value={count}>{count}S</option>)}
          </select>
        </label>

        {error ? <p className="mt-3 font-mono text-2xs uppercase tracking-[0.08em] text-sig-red" role="alert">{error}</p> : null}

        <div className="mt-4 rounded border border-hud-line bg-hud-bg/70 p-3 font-mono text-2xs leading-5 text-hud-muted">
          <div>Example: 10.160.142.17</div>
          <div>Local: http://&lt;esp-ip&gt;/telemetry.json</div>
          <div>Hosted dashboard: use a complete HTTPS endpoint.</div>
        </div>
      </form>
    </div>
  );
}

export function App({ streaming = true }: { streaming?: boolean }) {
  const [live, setLive] = useState(streaming);
  const [espInput, setEspInput] = useState(initialEspInput);
  const [cellCount, setCellCount] = useState(initialCellCount);
  const [endpoint, setEndpoint] = useState("");
  const [connectError, setConnectError] = useState("");
  const [now, setNow] = useState(Date.now());
  const [view, setView] = useState<"telemetry" | "mission">("telemetry");
  const t = useTelemetry(live, endpoint, cellCount);
  const warnings = activeWarnings(t);
  const hasTelemetry = t.source.lastUpdated > 0;

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const connect = () => {
    try {
      const nextEndpoint = telemetryEndpointFromInput(espInput);
      if (nextEndpoint.length === 0) {
        setConnectError("Enter the ESP32 IP address from the OLED.");
        return;
      }
      setEndpoint(nextEndpoint);
      setLive(true);
      setConnectError("");
      window.localStorage.setItem(ESP_IP_STORAGE_KEY, espInput.trim());
      window.localStorage.setItem(CELL_COUNT_STORAGE_KEY, String(cellCount));
    } catch (error) {
      setConnectError(error instanceof Error ? error.message : "Use an IP, host, or HTTPS telemetry URL.");
    }
  };

  if (endpoint.length === 0) {
    return (
      <ConnectionPanel
        value={espInput}
        cellCount={cellCount}
        error={connectError}
        onChange={setEspInput}
        onCellCountChange={setCellCount}
        onSubmit={connect}
      />
    );
  }

  return (
    <div className="hud-grid-bg flex min-h-full w-full flex-col bg-hud-bg font-sans xl:h-full xl:min-h-0">
      <TopBar
        live={live}
        onToggleLive={() => setLive((value) => !value)}
        armed={t.flight.armed}
        mode={t.flight.mode}
        now={now}
        warnings={warnings}
      />

      <nav className="flex gap-1 border-b border-hud-line bg-hud-panel px-3 py-2" aria-label="Dashboard views">
        <button type="button" onClick={() => setView("telemetry")} className={`rounded px-3 py-1.5 text-xs font-semibold transition focus:outline-none focus:ring-2 focus:ring-teal-300/30 ${view === "telemetry" ? "bg-teal-300 text-[#04110f]" : "text-hud-muted hover:bg-white/5 hover:text-white"}`}>Telemetry</button>
        <button type="button" onClick={() => setView("mission")} className={`rounded px-3 py-1.5 text-xs font-semibold transition focus:outline-none focus:ring-2 focus:ring-teal-300/30 ${view === "mission" ? "bg-teal-300 text-[#04110f]" : "text-hud-muted hover:bg-white/5 hover:text-white"}`}>Mission planner</button>
      </nav>

      {view === "mission" ? (
        <MissionPlanner
          currentLatitude={t.nav.lat}
          currentLongitude={t.nav.lon}
          telemetryAvailable={hasTelemetry}
        />
      ) : (

      <main className="flex flex-1 flex-col gap-2 p-2.5 xl:min-h-0">
        {warnings.length > 0 ? (
          <div
            className={`flex items-start justify-between gap-3 rounded border px-3 py-2 font-mono text-xs uppercase tracking-[0.06em] ${
              t.source.status === "offline" ? "border-sig-red/50 bg-sig-red/10 text-sig-red" : "border-sig-amber/50 bg-sig-amber/10 text-sig-amber"
            }`}
            role="status"
            aria-live="polite"
          >
            <ul className="min-w-0 space-y-1">
              {warnings.map((warning) => <li key={warning}>{warning}</li>)}
            </ul>
            <span className="hidden min-w-0 truncate text-right sm:block">{t.source.endpoint}</span>
          </div>
        ) : null}

        <div className="grid flex-1 grid-cols-1 gap-2 xl:min-h-[900px] xl:grid-rows-[minmax(0,2fr)_repeat(3,minmax(0,1fr))]">
          <div className="grid grid-cols-1 gap-2 xl:min-h-0 xl:grid-cols-[minmax(400px,460px)_minmax(0,1fr)]">
            <AttitudeIndicator attitude={t.attitude} available={hasTelemetry} className="h-[360px] xl:h-full xl:min-h-0" />

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:h-full xl:min-h-0 xl:grid-cols-3 xl:grid-rows-2">
              <BatteryBox battery={t.battery} available={hasTelemetry} className="min-h-[156px] xl:h-full xl:min-h-0" />
              <NavSignalBox nav={t.nav} available={hasTelemetry} className="min-h-[156px] xl:h-full xl:min-h-0" />
              <AltitudeCard altitude={t.altitude} available={hasTelemetry} className="min-h-[156px] xl:h-full xl:min-h-0" />
              <LinkCard link={t.link} available={hasTelemetry} className="min-h-[156px] xl:h-full xl:min-h-0" />
              <FlightStatusCard flight={t.flight} className="min-h-[180px] xl:h-full xl:min-h-0" />
              <SbcStatusCard sbc={t.sbc} className="min-h-[180px] xl:h-full xl:min-h-0" />
            </div>
          </div>
          <Suspense fallback={<div className="h-[220px] animate-pulse rounded-md border border-hud-line bg-hud-panel xl:h-full" aria-label="Loading PID chart" />}>
            <ChartPanel
            label="PID Response"
            icon={<SlidersHorizontalIcon size={12} />}
            series={pidSeries}
            data={t.pid as unknown as Array<Record<string, number>>}
            domain={[-10, 10]}
            ticks={[-10, -5, 0, 5, 10]}
            unit="°"
            available={hasTelemetry}
            className="h-[220px] xl:h-full xl:min-h-0"
            footer={
              <>
                <span>error · degrees · 64 s window</span>
                <span>display rate 1 Hz</span>
              </>
            }
            />
          </Suspense>
          <Suspense fallback={<div className="h-[220px] animate-pulse rounded-md border border-hud-line bg-hud-panel xl:h-full" aria-label="Loading motor chart" />}>
            <ChartPanel
            label="Motor Power"
            icon={<GaugeIcon size={12} />}
            series={motorSeries}
            data={t.motors as unknown as Array<Record<string, number>>}
            domain={[0, 100]}
            ticks={[0, 20, 40, 60, 80, 100]}
            unit="%"
            available={hasTelemetry}
            className="h-[220px] xl:h-full xl:min-h-0"
            footer={
              <>
                <span>throttle out · quad-x</span>
                <span>
                  avg{" "}
                  {(
                    ((t.motors[t.motors.length - 1]?.m1 ?? 0) +
                      (t.motors[t.motors.length - 1]?.m2 ?? 0) +
                      (t.motors[t.motors.length - 1]?.m3 ?? 0) +
                      (t.motors[t.motors.length - 1]?.m4 ?? 0)) /
                    4
                  ).toFixed(1)}
                  %
                </span>
              </>
            }
            />
          </Suspense>
          <Suspense fallback={<div className="h-[220px] animate-pulse rounded-md border border-hud-line bg-hud-panel xl:h-full" aria-label="Loading gyroscope chart" />}>
            <ChartPanel
            label="Gyroscope · 3-Axis"
            icon={<WavesIcon size={12} />}
            series={gyroSeries}
            data={t.gyro as unknown as Array<Record<string, number>>}
            domain={[-40, 40]}
            ticks={[-40, -20, 0, 20, 40]}
            unit=""
            available={hasTelemetry}
            className="h-[220px] xl:h-full xl:min-h-0"
            footer={
              <>
                <span>deg/s · imu1 · filtered</span>
                <span>{vehicle.firmware}</span>
              </>
            }
            />
          </Suspense>
        </div>

        <footer className="flex items-center justify-between px-1 pb-1 font-mono text-2xs text-hud-faint">
          <span className="flex min-w-0 items-center gap-1.5 truncate">
            <span className={`h-1.5 w-1.5 rounded-full ${live ? sourceTone(t.source.status) : "bg-hud-edge"}`} aria-hidden="true" />
            {live ? `${t.source.status} · ${t.source.endpoint}` : "stream paused · last frame retained"}
            {t.source.lastUpdated > 0 ? ` · last JSON ${Math.max(0, Math.floor((now - t.source.lastUpdated) / 1000))}s ago` : " · no JSON received"}
          </span>
          <button
            type="button"
            className="shrink-0 rounded border border-hud-line px-2 py-1 uppercase tracking-[0.12em] text-hud-muted transition hover:border-teal-300/50 hover:text-white"
            onClick={() => {
              setLive(false);
              setEndpoint("");
            }}
          >
            change esp
          </button>
        </footer>
      </main>
      )}
    </div>
  );
}
