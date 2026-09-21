import { useState } from "react";
import { GaugeIcon, PlugZapIcon, SlidersHorizontalIcon, WavesIcon, WifiIcon } from "lucide-react";
import { AltitudeCard } from "./components/AltitudeCard";
import { AttitudeIndicator } from "./components/AttitudeIndicator";
import { BatteryBox } from "./components/BatteryBox";
import { ChartPanel } from "./components/ChartPanel";
import { FlightStatusCard } from "./components/FlightStatusCard";
import { LinkCard } from "./components/LinkCard";
import { NavSignalBox } from "./components/NavSignalBox";
import { SbcStatusCard } from "./components/SbcStatusCard";
import { TopBar } from "./components/TopBar";
import { gyroSeries, motorSeries, pidSeries, vehicle } from "./data/telemetry";
import { useTelemetry } from "./hooks/useTelemetry";
import type { Telemetry } from "./types/telemetry";

function activeWarning(t: Telemetry): string | null {
  if (t.source.status === "offline") return "esp json offline";
  if (t.source.status === "stale") return "nrf packets stale";
  if (t.battery.percent < 25) return "battery low · rtl advised";
  if (t.link.loss > 5) return "downlink packet loss";
  if (t.sbc.temp > 72) return "sbc thermal limit";
  if (t.nav.hdop > 1.2) return "gps hdop degraded";
  return null;
}

function sourceTone(status: Telemetry["source"]["status"]): string {
  if (status === "online") return "bg-sig-green";
  if (status === "stale") return "bg-sig-amber";
  return "bg-sig-red";
}

const ESP_IP_STORAGE_KEY = "droneTelemetryEspIp";

function initialEspInput(): string {
  const fromQuery = new URLSearchParams(window.location.search).get("telemetryUrl");
  if (fromQuery) {
    try {
      return new URL(fromQuery).host;
    } catch {
      return fromQuery.replace(/^https?:\/\//, "").replace(/\/telemetry\.json\/?$/, "");
    }
  }
  return window.localStorage.getItem(ESP_IP_STORAGE_KEY) ?? "";
}

function telemetryEndpointFromInput(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) return "";
  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  const url = new URL(withProtocol);
  if (url.pathname === "/" || url.pathname === "") {
    url.pathname = "/telemetry.json";
  }
  url.search = "";
  url.hash = "";
  return url.toString();
}

function ConnectionPanel({
  value,
  error,
  onChange,
  onSubmit
}: {
  value: string;
  error: string;
  onChange: (value: string) => void;
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
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded border border-teal-400/30 bg-teal-400/10 text-teal-300">
            <WifiIcon size={18} />
          </span>
          <div className="min-w-0">
            <h1 className="text-sm font-semibold uppercase tracking-[0.08em] text-hud-text">Connect ESP32 Telemetry</h1>
            <p className="mt-1 text-xs leading-5 text-hud-muted">Enter the IP shown on the ESP32 OLED.</p>
          </div>
        </div>

        <label className="mb-2 block font-mono text-2xs uppercase tracking-[0.12em] text-hud-faint" htmlFor="esp-ip">
          ESP32 IP address
        </label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            id="esp-ip"
            autoFocus
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder="10.160.142.17"
            className="h-10 min-w-0 flex-1 rounded border border-hud-line bg-hud-bg px-3 font-mono text-sm tabular text-hud-text outline-none transition focus:border-teal-300 focus:ring-2 focus:ring-teal-300/20"
          />
          <button
            type="submit"
            className="inline-flex h-10 items-center justify-center gap-2 rounded border border-teal-300/40 bg-teal-300/15 px-4 font-mono text-2xs font-semibold uppercase tracking-[0.12em] text-teal-200 transition hover:border-teal-200 hover:bg-teal-300/20 focus:outline-none focus:ring-2 focus:ring-teal-300/30"
          >
            <PlugZapIcon size={14} />
            Connect
          </button>
        </div>

        {error ? <p className="mt-3 font-mono text-2xs uppercase tracking-[0.08em] text-sig-red">{error}</p> : null}

        <div className="mt-4 rounded border border-hud-line bg-hud-bg/70 p-3 font-mono text-2xs leading-5 text-hud-muted">
          <div>Example: 10.160.142.17</div>
          <div>Dashboard will read: http://&lt;esp-ip&gt;/telemetry.json</div>
        </div>
      </form>
    </div>
  );
}

export function App({ streaming = true }: { streaming?: boolean }) {
  const [live, setLive] = useState(streaming);
  const [espInput, setEspInput] = useState(initialEspInput);
  const [endpoint, setEndpoint] = useState("");
  const [connectError, setConnectError] = useState("");
  const t = useTelemetry(live, endpoint);

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
    } catch {
      setConnectError("Use an IP or host like 10.160.142.17.");
    }
  };

  if (endpoint.length === 0) {
    return <ConnectionPanel value={espInput} error={connectError} onChange={setEspInput} onSubmit={connect} />;
  }

  return (
    <div className="hud-grid-bg flex min-h-full w-full flex-col bg-hud-bg font-sans">
      <TopBar
        live={live}
        onToggleLive={() => setLive((value) => !value)}
        armed={t.flight.armed}
        mode={t.flight.mode}
        clock={t.clock}
        warning={activeWarning(t)}
      />

      <main className="flex flex-1 flex-col gap-2 p-2.5">
        {t.source.status !== "online" ? (
          <div
            className={`flex items-center justify-between gap-3 rounded border px-3 py-2 font-mono text-2xs uppercase tracking-[0.08em] ${
              t.source.status === "offline"
                ? "border-sig-red/50 bg-sig-red/10 text-sig-red"
                : "border-sig-amber/50 bg-sig-amber/10 text-sig-amber"
            }`}
          >
            <span className="min-w-0 truncate">{t.source.message}</span>
            <span className="hidden min-w-0 truncate text-right sm:block">{t.source.endpoint}</span>
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-12">
          <BatteryBox battery={t.battery} className="min-h-[150px] xl:col-span-3 xl:h-[164px]" />
          <NavSignalBox nav={t.nav} className="min-h-[150px] xl:col-span-4 xl:h-[164px]" />
          <AltitudeCard altitude={t.altitude} className="min-h-[150px] xl:col-span-2 xl:h-[164px]" />
          <LinkCard link={t.link} className="min-h-[150px] xl:col-span-3 xl:h-[164px]" />
        </div>

        <div className="grid grid-cols-1 gap-2 lg:grid-cols-2 xl:grid-cols-12">
          <FlightStatusCard flight={t.flight} className="min-h-[178px] xl:col-span-4 xl:h-[178px]" />
          <SbcStatusCard sbc={t.sbc} className="min-h-[178px] xl:col-span-5 xl:h-[178px]" />
          <AttitudeIndicator attitude={t.attitude} className="min-h-[150px] lg:col-span-2 xl:col-span-3 xl:h-[178px]" />
        </div>

        <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
          <ChartPanel
            label="PID Response"
            icon={<SlidersHorizontalIcon size={12} />}
            series={pidSeries}
            data={t.pid as unknown as Array<Record<string, number>>}
            domain={[-10, 10]}
            ticks={[-10, -5, 0, 5, 10]}
            unit="°"
            className="h-[196px]"
            footer={
              <>
                <span>err · deg · 60 s window</span>
                <span>rate 50 hz</span>
              </>
            }
          />
          <ChartPanel
            label="Motor Power"
            icon={<GaugeIcon size={12} />}
            series={motorSeries}
            data={t.motors as unknown as Array<Record<string, number>>}
            domain={[40, 80]}
            ticks={[40, 50, 60, 70, 80]}
            unit="%"
            className="h-[196px]"
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
        </div>

        <ChartPanel
          label="Gyroscope · 3-Axis"
          icon={<WavesIcon size={12} />}
          series={gyroSeries}
          data={t.gyro as unknown as Array<Record<string, number>>}
          domain={[-40, 40]}
          ticks={[-40, -20, 0, 20, 40]}
          unit=""
          className="h-[170px]"
          footer={
            <>
              <span>deg/s · imu1 · filtered</span>
              <span>{vehicle.firmware}</span>
            </>
          }
        />

        <footer className="flex items-center justify-between px-1 pb-1 font-mono text-2xs text-hud-faint">
          <span className="flex items-center gap-1.5">
            <span className={`h-1.5 w-1.5 rounded-full ${live ? sourceTone(t.source.status) : "bg-hud-edge"}`} aria-hidden="true" />
            {live ? `${t.source.status} · ${t.source.endpoint}` : "stream held · last frame retained"}
          </span>
          <button
            type="button"
            className="hidden rounded border border-hud-line px-2 py-1 uppercase tracking-[0.12em] text-hud-muted transition hover:border-teal-300/50 hover:text-teal-200 sm:block"
            onClick={() => {
              setLive(false);
              setEndpoint("");
            }}
          >
            change esp
          </button>
        </footer>
      </main>
    </div>
  );
}
