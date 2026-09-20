import { ActivityIcon, PauseIcon, PlayIcon, ShieldAlertIcon } from "lucide-react";
import { vehicle } from "../data/telemetry";
import { Chip } from "./Panel";

type TopBarProps = {
  live: boolean;
  onToggleLive: () => void;
  armed: boolean;
  mode: string;
  clock: number;
  warning: string | null;
};

function stamp(clock: number): string {
  const total = clock * 4;
  const h = String(Math.floor(total / 3600) % 24).padStart(2, "0");
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${h}:${m}:${s}Z`;
}

export function TopBar({ live, onToggleLive, armed, mode, clock, warning }: TopBarProps) {
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-hud-line bg-hud-panel px-3">
      <div className="flex items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded bg-sig-cyan/15 text-sig-cyan">
          <ActivityIcon size={13} />
        </span>
        <div className="leading-none">
          <div className="font-mono text-[12px] font-semibold tracking-wide text-hud-text">{vehicle.name}</div>
          <div className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.12em] text-hud-faint">{vehicle.linkPath}</div>
        </div>
      </div>

      <div className="hidden items-center gap-1.5 lg:flex">
        <Chip tone="blue">{mode}</Chip>
        <Chip tone={armed ? "red" : "neutral"}>{armed ? "ARMED" : "DISARMED"}</Chip>
        <Chip tone="neutral">{vehicle.sysId}</Chip>
      </div>

      {warning ? (
        <div className="ml-auto flex min-w-0 items-center gap-1.5 rounded border border-sig-amber/40 bg-sig-amber/10 px-2 py-1 text-sig-amber">
          <ShieldAlertIcon size={12} />
          <span className="truncate font-mono text-2xs font-semibold uppercase tracking-[0.08em]">{warning}</span>
        </div>
      ) : null}

      <div className={`flex items-center gap-2 ${warning ? "" : "ml-auto"}`}>
        <span className="tabular hidden font-mono text-[11px] text-hud-dim sm:block">{stamp(clock)}</span>
        <button
          type="button"
          onClick={onToggleLive}
          className="flex items-center gap-1.5 rounded border border-hud-edge bg-hud-raise px-2 py-1 font-mono text-2xs font-semibold uppercase tracking-[0.1em] text-hud-dim transition-colors duration-150 ease-out hover:border-sig-cyan/50 hover:text-sig-cyan focus:outline-none focus-visible:ring-1 focus-visible:ring-sig-cyan"
          aria-pressed={live}
        >
          {live ? <PauseIcon size={11} /> : <PlayIcon size={11} />}
          {live ? "Live" : "Hold"}
        </button>
      </div>
    </header>
  );
}
