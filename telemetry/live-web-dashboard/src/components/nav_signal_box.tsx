import { SatelliteDishIcon } from "lucide-react";
import { Chip, Panel, Stat } from "./panel";
import type { NavState } from "../types/telemetry";

function SignalBars({ rssi }: { rssi: number }) {
  const level = rssi >= 80 ? 4 : rssi >= 60 ? 3 : rssi >= 40 ? 2 : rssi > 0 ? 1 : 0;
  return (
    <div className="flex items-end gap-0.5" role="img" aria-label={`Radio signal ${rssi} percent`}>
      {[1, 2, 3, 4].map((i) => (
        <span
          key={i}
          className="w-1 rounded-sm"
          style={{ height: 3 + i * 2.5, backgroundColor: i <= level ? "#2dd4bf" : "#1d2733" }}
        />
      ))}
    </div>
  );
}

function Compass({ heading }: { heading: number }) {
  return (
    <div className="relative h-[58px] w-[58px] shrink-0 rounded-full border border-hud-line bg-hud-raise">
      {["N", "E", "S", "W"].map((d, i) => (
        <span
          key={d}
          className="absolute left-1/2 top-1 -translate-x-1/2 font-mono text-[10px] text-hud-faint"
          style={{ transform: `rotate(${i * 90}deg)`, transformOrigin: "50% 28px" }}
        >
          {d}
        </span>
      ))}
      <div className="absolute inset-0 transition-transform duration-200 ease-out" style={{ transform: `rotate(${heading}deg)` }}>
        <span className="absolute left-1/2 top-[9px] h-[18px] w-0.5 -translate-x-1/2 rounded-full bg-sig-cyan" />
        <span className="absolute left-1/2 top-1/2 h-[12px] w-0.5 -translate-x-1/2 rounded-full bg-hud-edge" />
      </div>
      <span className="tabular absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded bg-hud-panel px-1 font-mono text-[10px] font-semibold text-hud-text">
        {Math.round(heading)}°
      </span>
    </div>
  );
}

export function NavSignalBox({ nav, available, className }: { nav: NavState; available: boolean; className?: string }) {
  return (
    <Panel
      label="Navigation · Signal"
      icon={<SatelliteDishIcon size={12} />}
      meta={
        <>
          <Chip tone={available ? "cyan" : "amber"}>{available ? nav.fix : "NO DATA"}</Chip>
          <SignalBars rssi={available ? nav.rssi : 0} />
        </>
      }
      className={className}
      bodyClassName="flex items-center gap-3"
    >
      <div className="grid min-w-0 flex-1 grid-cols-3 gap-x-3 gap-y-1.5">
        <Stat label="Lat" value={available ? nav.lat.toFixed(6) : "--"} />
        <Stat label="Lon" value={available ? nav.lon.toFixed(6) : "--"} />
        <Stat label="Hdg" value={available ? `${Math.round(nav.heading)}°` : "--"} />
        <Stat label="Sats" value={available ? nav.sats : "--"} tone={available && nav.sats >= 12 ? "text-sig-green" : "text-sig-amber"} />
        <Stat label="HDOP" value={available ? nav.hdop.toFixed(2) : "--"} tone={available && nav.hdop < 1.1 ? "text-hud-text" : "text-sig-amber"} />
        <Stat label="RSSI" value={available ? nav.rssi : "--"} unit="%" />
      </div>
      <Compass heading={nav.heading} />
    </Panel>
  );
}
