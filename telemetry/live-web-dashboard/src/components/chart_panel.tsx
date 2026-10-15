import type { ReactNode } from "react";
import { ChartLegend, TelemetryChart } from "./telemetry_chart";
import { Panel } from "./panel";
import type { SeriesDef } from "../types/telemetry";

type ChartPanelProps = {
  label: string;
  icon?: ReactNode;
  series: SeriesDef[];
  data: Array<Record<string, number>>;
  domain?: [number, number] | ["auto", "auto"];
  ticks?: number[];
  unit?: string;
  available?: boolean;
  footer?: ReactNode;
  className?: string;
};

export function ChartPanel({ label, icon, series, data, domain, ticks, unit, available = true, footer, className = "" }: ChartPanelProps) {
  const latest = data[data.length - 1] ?? {};
  const values: Record<string, string> = {};
  series.forEach((s) => {
    const v = latest[s.key];
    values[s.key] = available && typeof v === "number" ? v.toFixed(1) : "--";
  });
  const summary = series.map((item) => `${item.label} ${values[item.key] ?? "unavailable"}${unit ?? ""}`).join(", ");

  return (
    <Panel
      label={label}
      icon={icon}
      meta={<ChartLegend series={series} values={values} />}
      className={className}
      bodyClassName="flex flex-col gap-1 px-1.5 pb-1.5 pt-1.5"
    >
      <div className="min-h-0 flex-1" role="img" aria-label={`${label} chart. Latest values: ${summary}.`}>
        {available
          ? <TelemetryChart data={data} series={series} domain={domain} ticks={ticks} unit={unit} />
          : <div className="grid h-full place-items-center font-mono text-xs uppercase tracking-[0.08em] text-hud-faint">No telemetry samples</div>}
      </div>
      {footer ? <div className="flex items-center justify-between px-1 font-mono text-2xs text-hud-faint">{footer}</div> : null}
    </Panel>
  );
}
