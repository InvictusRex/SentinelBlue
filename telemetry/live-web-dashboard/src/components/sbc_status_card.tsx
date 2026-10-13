import { CpuIcon } from "lucide-react";
import { Bar, Chip, Panel } from "./panel";
import type { SbcState } from "../types/telemetry";

function formatUptime(ticks: number): string {
  const seconds = ticks;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${String(m).padStart(2, "0")}m`;
}

function MetricRow({ label, value, pct, color }: { label: string; value: string; pct: number; color: string }) {
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-2xs uppercase tracking-[0.1em] text-hud-faint">{label}</span>
        <span className="tabular font-mono text-[12px] text-hud-text">{value}</span>
      </div>
      <Bar value={pct} color={color} className="mt-1 h-1" />
    </div>
  );
}

export function SbcStatusCard({ sbc, className }: { sbc: SbcState; className?: string }) {
  const tempTone = sbc.temp > 70 ? "#f2555a" : sbc.temp > 62 ? "#f5a524" : "#3fcf8e";
  const [cpuUsage, memoryUsage, storageUsage] = sbc.load;

  return (
    <Panel
      label="SBC · Raspberry Pi 4B"
      icon={<CpuIcon size={12} />}
      meta={
        <>
          <Chip tone={sbc.available && sbc.temp > 70 ? "red" : sbc.available ? "green" : "amber"}>
            {sbc.available ? `${sbc.temp.toFixed(1)}°C` : "--°C"}
          </Chip>
          <Chip tone={sbc.available ? "neutral" : "amber"}>{sbc.available ? "ONLINE" : "UNKNOWN"}</Chip>
        </>
      }
      className={className}
      bodyClassName="flex flex-col justify-between gap-2"
    >
      <div className="grid grid-cols-2 gap-x-3 gap-y-2">
        <MetricRow label="CPU" value={sbc.available ? `${sbc.cpu.toFixed(0)}%` : "--"} pct={sbc.available ? sbc.cpu : 0} color={sbc.cpu > 80 ? "#f2555a" : "#4c8dff"} />
        <MetricRow label="Memory" value={sbc.available ? `${sbc.memPercent.toFixed(0)}%` : "--"} pct={sbc.available ? sbc.memPercent : 0} color="#2dd4bf" />
        <MetricRow label="Temp" value={sbc.available ? `${sbc.temp.toFixed(1)} °C` : "--"} pct={sbc.available ? ((sbc.temp - 30) / 55) * 100 : 0} color={tempTone} />
        <MetricRow label="Storage" value={sbc.available ? `${sbc.diskPercent.toFixed(0)}%` : "--"} pct={sbc.available ? sbc.diskPercent : 0} color="#9b8cff" />
      </div>

      <div className="flex items-end justify-between gap-2 border-t border-hud-line pt-2">
        <div className="flex items-end gap-1">
          {sbc.cores.length === 0 ? <span className="font-mono text-2xs text-hud-faint">per-core data unavailable</span> : sbc.cores.map((core, i) => {
            const height = core > 0 ? Math.max(8, core) : 0;

            return (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className="flex h-6 w-2.5 items-end rounded-sm bg-hud-raise">
                  <span
                    className="w-full rounded-sm transition-[height] duration-200 ease-out"
                    style={{ height: `${height}%`, backgroundColor: core > 85 ? "#f5a524" : "#4c8dff" }}
                  />
                </div>
                <span className="font-mono text-[10px] text-hud-faint">C{i}</span>
              </div>
            );
          })}
        </div>
        <div className="text-right">
          <div className="tabular font-mono text-[11px] text-hud-dim">
            {sbc.available ? `usage cpu ${cpuUsage.toFixed(0)}% mem ${memoryUsage.toFixed(0)}% disk ${storageUsage.toFixed(0)}%` : "usage unavailable"}
          </div>
          <div className="tabular font-mono text-2xs text-hud-faint">uptime {sbc.available ? formatUptime(sbc.uptime) : "--"}</div>
        </div>
      </div>
    </Panel>
  );
}
