import { BatteryMediumIcon } from "lucide-react";
import { Bar, Chip, Panel } from "./Panel";
import type { BatteryState } from "../types/telemetry";

function cellTone(v: number): string {
  if (v < 3.6) return "#f2555a";
  if (v < 3.8) return "#f5a524";
  return "#3fcf8e";
}

export function BatteryBox({ battery, className }: { battery: BatteryState; className?: string }) {
  const pack = battery.percent;
  const packColor = pack < 25 ? "#f2555a" : pack < 45 ? "#f5a524" : "#3fcf8e";
  const min = Math.min(...battery.cells.map((c) => c.volts));
  const max = Math.max(...battery.cells.map((c) => c.volts));
  const delta = (max - min) * 1000;

  return (
    <Panel
      label="Battery · 4S"
      icon={<BatteryMediumIcon size={12} />}
      meta={<Chip tone={pack < 25 ? "red" : pack < 45 ? "amber" : "green"}>{pack}%</Chip>}
      className={className}
      bodyClassName="flex flex-col gap-2"
    >
      <div className="flex items-end justify-between gap-2">
        <div className="min-w-0">
          <div className="tabular font-mono text-[26px] font-semibold leading-7 text-hud-text">
            {battery.total.toFixed(2)}
            <span className="ml-1 text-xs font-medium text-hud-faint">V</span>
          </div>
          <div className="tabular mt-0.5 font-mono text-2xs text-hud-faint">
            {battery.current.toFixed(1)} A · {battery.consumed} mAh
          </div>
        </div>
        <div className="w-[42%] shrink-0">
          <Bar value={pack} color={packColor} />
          <div className="tabular mt-1 text-right font-mono text-2xs text-hud-faint">Δ {delta.toFixed(0)} mV</div>
        </div>
      </div>

      <ul className="grid grid-cols-2 gap-x-2.5 gap-y-1">
        {battery.cells.map((cell) => (
          <li key={cell.label} className="flex items-center gap-1.5">
            <span className="w-4 font-mono text-2xs font-semibold text-hud-faint">{cell.label}</span>
            <Bar value={((cell.volts - 3.3) / (4.2 - 3.3)) * 100} color={cellTone(cell.volts)} className="h-1" />
            <span className="tabular w-9 shrink-0 text-right font-mono text-2xs text-hud-dim">
              {cell.volts.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}
