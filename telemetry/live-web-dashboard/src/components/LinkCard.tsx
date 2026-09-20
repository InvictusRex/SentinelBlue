import { RadioIcon } from "lucide-react";
import { Chip, Panel, Stat } from "./Panel";
import type { LinkState } from "../types/telemetry";

export function LinkCard({ link, className }: { link: LinkState; className?: string }) {
  const healthy = link.loss < 3 && link.rssi > -78;
  return (
    <Panel
      label="Downlink · ESP32"
      icon={<RadioIcon size={12} />}
      meta={<Chip tone={healthy ? "green" : "amber"}>{healthy ? "STABLE" : "DEGRADED"}</Chip>}
      className={className}
      bodyClassName="flex flex-col justify-between gap-2"
    >
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        <Stat label="RSSI" value={link.rssi} unit="dBm" />
        <Stat label="Pkt/s" value={Math.round(link.rate)} />
        <Stat label="Loss" value={`${link.loss.toFixed(1)}%`} tone={link.loss < 3 ? "text-hud-text" : "text-sig-amber"} />
        <Stat label="Latency" value={link.latency} unit="ms" />
      </div>
      <div className="flex h-7 items-end gap-[2px]" aria-hidden="true">
        {link.history.map((v, i) => (
          <span
            key={`${v}-${i}`}
            className="flex-1 rounded-sm"
            style={{
              height: `${Math.max(12, ((v - 55) / 45) * 100)}%`,
              backgroundColor: v > 85 ? "#2dd4bf" : v > 72 ? "#4c8dff" : "#f5a524",
              opacity: 0.35 + (i / link.history.length) * 0.65
            }}
          />
        ))}
      </div>
    </Panel>
  );
}
