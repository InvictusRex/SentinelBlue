import { MountainSnowIcon, TrendingDownIcon, TrendingUpIcon } from "lucide-react";
import { Chip, Panel } from "./panel";
import type { AltitudeState } from "../types/telemetry";

export function AltitudeCard({ altitude, available, className }: { altitude: AltitudeState; available: boolean; className?: string }) {
  const rising = altitude.vspeed >= 0;
  return (
    <Panel
      label="Altitude"
      icon={<MountainSnowIcon size={12} />}
      meta={<Chip tone={!available ? "amber" : rising ? "green" : "amber"}>{available ? rising ? "CLIMB" : "DESC" : "UNKNOWN"}</Chip>}
      className={className}
      bodyClassName="flex flex-col justify-between"
    >
      <div>
        <div className="text-2xs uppercase tracking-[0.1em] text-hud-faint">Relative</div>
        <div className="tabular font-mono text-[26px] font-semibold leading-7 text-hud-text">
          {available ? altitude.relative.toFixed(1) : "--"}
          <span className="ml-1 text-xs font-medium text-hud-faint">m</span>
        </div>
      </div>
      <dl className="space-y-1 border-t border-hud-line pt-1.5">
        <div className="flex items-center justify-between gap-2">
          <dt className="text-2xs uppercase tracking-[0.1em] text-hud-faint">MSL</dt>
          <dd className="tabular font-mono text-[12px] text-hud-dim">{available ? altitude.msl.toFixed(1) : "--"} m</dd>
        </div>
        <div className="flex items-center justify-between gap-2">
          <dt className="text-2xs uppercase tracking-[0.1em] text-hud-faint">V/S</dt>
          <dd className={`tabular flex items-center gap-1 font-mono text-[12px] ${rising ? "text-sig-green" : "text-sig-amber"}`}>
            {available ? rising ? <TrendingUpIcon size={11} /> : <TrendingDownIcon size={11} /> : null}
            {available ? altitude.vspeed.toFixed(1) : "--"} m/s
          </dd>
        </div>
      </dl>
    </Panel>
  );
}
