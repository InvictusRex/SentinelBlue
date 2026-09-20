import { PlaneIcon } from "lucide-react";
import { Chip, Panel } from "./Panel";
import type { FlightState } from "../types/telemetry";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-hud-line/70 py-[6px] last:border-0">
      <span className="text-2xs uppercase tracking-[0.1em] text-hud-faint">{label}</span>
      <div className="flex min-w-0 items-center gap-1.5">{children}</div>
    </div>
  );
}

export function FlightStatusCard({ flight, className }: { flight: FlightState; className?: string }) {
  return (
    <Panel
      label="Flight Status"
      icon={<PlaneIcon size={12} />}
      meta={<Chip tone={flight.armed ? "red" : "neutral"}>{flight.armed ? "ARMED" : "DISARMED"}</Chip>}
      className={className}
      bodyClassName="flex flex-col justify-center gap-1"
    >
      <Row label="Mode">
        <span className="font-mono text-[13px] font-semibold tracking-wide text-sig-cyan">{flight.mode}</span>
      </Row>
      <Row label="Failsafe">
        <Chip tone={flight.failsafe ? "red" : "green"}>{flight.failsafe ? "TRIGGERED" : "CLEAR"}</Chip>
      </Row>
      <Row label="Mission">
        <span className="truncate font-mono text-[12px] text-hud-text">{flight.mission}</span>
        <span className="tabular font-mono text-2xs text-hud-faint">WP {flight.wp}</span>
      </Row>
      <Row label="Link">
        <Chip tone={flight.link === "OK" ? "green" : flight.link === "DEGRADED" ? "amber" : "red"}>
          MAVLINK {flight.link}
        </Chip>
      </Row>
    </Panel>
  );
}
