import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  YAxis
} from "recharts";
import type { SeriesDef } from "../types/telemetry";

type TelemetryChartProps = {
  data: Array<Record<string, number>>;
  series: SeriesDef[];
  domain?: [number, number] | ["auto", "auto"];
  ticks?: number[];
  unit?: string;
};

export function TelemetryChart({ data, series, domain = ["auto", "auto"], ticks, unit }: TelemetryChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="#18212c" strokeDasharray="2 4" vertical={false} />
        <YAxis
          domain={domain}
          ticks={ticks}
          width={34}
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#5c6a7a", fontSize: 9, fontFamily: "JetBrains Mono" }}
          tickFormatter={(v: number) => `${v}${unit ?? ""}`}
        />
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            stroke={s.color}
            strokeWidth={1.4}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ChartLegend({ series, values }: { series: SeriesDef[]; values?: Record<string, string> }) {
  return (
    <div className="flex items-center gap-2">
      {series.map((s) => (
        <span key={s.key} className="flex items-center gap-1">
          <span className="h-0.5 w-3 rounded-full" style={{ backgroundColor: s.color }} aria-hidden="true" />
          <span className="font-mono text-2xs uppercase tracking-[0.08em] text-hud-faint">{s.label}</span>
          {values ? <span className="tabular font-mono text-2xs text-hud-dim">{values[s.key]}</span> : null}
        </span>
      ))}
    </div>
  );
}
