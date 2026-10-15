import { lazy, Suspense } from "react";
import { CompassIcon } from "lucide-react";
import { Chip, Panel } from "./panel";
import type { AttitudeState } from "../types/telemetry";

const AttitudeScene = lazy(() => import("./attitude_scene").then((module) => ({ default: module.AttitudeScene })));

export function AttitudeIndicator({ attitude, available, className }: { attitude: AttitudeState; available: boolean; className?: string }) {
  const { roll, pitch, yaw } = attitude;

  return (
    <Panel
      label="Altitude · Heading"
      icon={<CompassIcon size={12} />}
      meta={<Chip tone={available ? "cyan" : "amber"}>{available ? "AHRS LOCK" : "NO DATA"}</Chip>}
      className={className}
      bodyClassName="flex flex-col gap-3"
    >
      {available ? (
        <Suspense fallback={<div className="min-h-[220px] flex-1 animate-pulse rounded bg-hud-raise" aria-label="Loading 3D attitude view" />}>
          <AttitudeScene attitude={attitude} />
        </Suspense>
      ) : (
        <div className="grid min-h-[220px] flex-1 place-items-center rounded bg-hud-raise font-mono text-xs uppercase tracking-[0.08em] text-hud-faint">
          No attitude telemetry
        </div>
      )}

      <dl className="grid shrink-0 grid-cols-3 gap-2">
        {[
          { label: "Roll", value: roll, color: "#4c8dff" },
          { label: "Pitch", value: pitch, color: "#2dd4bf" },
          { label: "Yaw", value: yaw, color: "#f5a524" }
        ].map((r) => (
          <div key={r.label} className="rounded border border-hud-line bg-hud-raise px-2 py-1.5">
            <dt className="flex items-center gap-1 text-2xs uppercase tracking-[0.1em] text-hud-faint">
              <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: r.color }} aria-hidden="true" />
              {r.label}
            </dt>
            <dd className="tabular font-mono text-sm font-medium text-hud-text">{available ? `${r.value.toFixed(1)}°` : "--"}</dd>
          </div>
        ))}
      </dl>
    </Panel>
  );
}
