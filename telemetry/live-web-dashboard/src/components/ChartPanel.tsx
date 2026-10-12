import type { ReactNode } from "react";
import { ChartLegend, TelemetryChart } from "./TelemetryChart";
import { Panel } from "./Panel";
import type { SeriesDef } from "../types/telemetry";

type ChartPanelProps = {
  label: string;
  icon?: ReactNode;
  series: SeriesDef[];
  data: Array<Record<string, number>>;
  domain?: [number, number] | ["auto", "auto"];
  ticks?: number[];
  unit?: string;
  footer?: ReactNode;
  className?: string;
};

export function ChartPanel({ label, icon, series, data, domain, ticks, unit, footer, className = "" }: ChartPanelProps) {
  const latest = data[data.length - 1] ?? {};
  const values: Record<string, string> = {};
  series.forEach((s) => {
    const v = latest[s.key];
    values[s.key] = typeof v === "number" ? v.toFixed(1) : "--";
  });

  return (
    <Panel
      label={label}
      icon={icon}
      meta={<ChartLegend series={series} values={values} />}
      className={className}
      bodyClassName="flex flex-col gap-1 px-1.5 pb-1.5 pt-1.5"
    >
      <div className="min-h-0 flex-1">
        <TelemetryChart data={data} series={series} domain={domain} ticks={ticks} unit={unit} />
      </div>
      {footer ? <div className="flex items-center justify-between px-1 font-mono text-2xs text-hud-faint">{footer}</div> : null}
    </Panel>
  );
}
