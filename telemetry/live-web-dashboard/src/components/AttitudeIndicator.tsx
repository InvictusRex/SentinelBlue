import { CompassIcon } from "lucide-react";
import { Chip, Panel } from "./Panel";
import type { AttitudeState } from "../types/telemetry";
import { AttitudeScene } from "./AttitudeScene";

export function AttitudeIndicator({ attitude, className }: { attitude: AttitudeState; className?: string }) {
  const { roll, pitch, yaw } = attitude;

  return (
    <Panel
      label="Attitude · Heading"
      icon={<CompassIcon size={12} />}
      meta={<Chip tone="cyan">AHRS LOCK</Chip>}
      className={className}
      bodyClassName="flex items-center gap-3"
    >
      <AttitudeScene attitude={attitude} />

      <dl className="w-[84px] shrink-0 space-y-1.5">
        {[
          { label: "Roll", value: roll, color: "#4c8dff" },
          { label: "Pitch", value: pitch, color: "#2dd4bf" },
          { label: "Yaw", value: yaw, color: "#f5a524" }
        ].map((r) => (
          <div key={r.label} className="rounded border border-hud-line bg-hud-raise px-1.5 py-1">
            <dt className="flex items-center gap-1 text-2xs uppercase tracking-[0.1em] text-hud-faint">
              <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: r.color }} aria-hidden="true" />
              {r.label}
            </dt>
            <dd className="tabular font-mono text-[13px] font-medium leading-4 text-hud-text">{r.value.toFixed(1)}°</dd>
          </div>
        ))}
      </dl>
    </Panel>
  );
}
