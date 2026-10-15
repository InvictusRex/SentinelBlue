import { ActivityIcon, PauseIcon, PlayIcon, ShieldAlertIcon } from "lucide-react";
import { vehicle } from "../data/telemetry";
import { Chip } from "./panel";

type TopBarProps = {
  live: boolean;
  onToggleLive: () => void;
  armed: boolean | null;
  mode: string;
  now: number;
  warnings: string[];
};

function stamp(now: number): string {
  return new Date(now).toISOString().slice(11, 19) + "Z";
}

export function TopBar({ live, onToggleLive, armed, mode, now, warnings }: TopBarProps) {
  const armedLabel = armed === null ? "ARMED UNKNOWN" : armed ? "ARMED" : "DISARMED";
  const armedTone = armed === null ? "amber" : armed ? "red" : "neutral";

  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-hud-line bg-hud-panel px-3">
      <div className="flex items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded bg-sig-cyan/15 text-sig-cyan">
          <ActivityIcon size={13} />
        </span>
        <div className="leading-none">
          <div className="font-mono text-sm font-semibold tracking-wide text-hud-text">{vehicle.name}</div>
          <div className="mt-0.5 font-mono text-2xs uppercase tracking-[0.1em] text-hud-faint">{vehicle.linkPath}</div>
        </div>
      </div>

      <div className="hidden items-center gap-1.5 lg:flex">
        <Chip tone="blue">{mode}</Chip>
        <Chip tone={armedTone}>{armedLabel}</Chip>
        <Chip tone="neutral">{vehicle.sysId}</Chip>
      </div>

      {warnings.length > 0 ? (
        <div className="ml-auto hidden min-w-0 items-center gap-1.5 rounded border border-sig-amber/40 bg-sig-amber/10 px-2 py-1 text-sig-amber md:flex">
          <ShieldAlertIcon size={12} />
          <span className="truncate font-mono text-2xs font-semibold uppercase tracking-[0.08em]">
            {warnings[0]}{warnings.length > 1 ? ` +${warnings.length - 1}` : ""}
          </span>
        </div>
      ) : null}

      <div className={`flex items-center gap-2 ${warnings.length > 0 ? "ml-auto md:ml-0" : "ml-auto"}`}>
        <time className="tabular hidden font-mono text-2xs text-hud-dim sm:block" dateTime={new Date(now).toISOString()}>{stamp(now)}</time>
        <button
          type="button"
          onClick={onToggleLive}
          className="flex items-center gap-1.5 rounded border border-hud-edge bg-hud-raise px-2 py-1 font-mono text-2xs font-semibold uppercase tracking-[0.1em] text-hud-dim transition-colors duration-150 ease-out hover:border-sig-cyan/50 hover:text-white focus:outline-none focus-visible:ring-1 focus-visible:ring-sig-cyan"
          aria-pressed={live}
        >
          {live ? <PauseIcon size={11} /> : <PlayIcon size={11} />}
          {live ? "Pause" : "Resume"}
        </button>
      </div>
    </header>
  );
}
