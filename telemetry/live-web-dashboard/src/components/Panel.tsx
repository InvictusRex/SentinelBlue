import type { ReactNode } from "react";

type PanelProps = {
  label: string;
  icon?: ReactNode;
  meta?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
};

export function Panel({ label, icon, meta, className = "", bodyClassName = "", children }: PanelProps) {
  return (
    <section className={`flex min-w-0 flex-col rounded-md border border-hud-line bg-hud-panel ${className}`}>
      <header className="flex h-8 shrink-0 items-center gap-1.5 border-b border-hud-line px-2.5">
        {icon ? <span className="text-hud-faint">{icon}</span> : null}
        <h2 className="truncate text-2xs font-semibold uppercase tracking-[0.14em] text-hud-dim">
          {label}
        </h2>
        <div className="ml-auto flex min-w-0 items-center gap-1.5">{meta}</div>
      </header>
      <div className={`min-h-0 flex-1 px-2.5 py-2 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

type StatProps = {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: string;
};

export function Stat({ label, value, unit, tone = "text-hud-text" }: StatProps) {
  return (
    <div className="min-w-0">
      <div className="truncate text-2xs uppercase tracking-[0.1em] text-hud-faint">{label}</div>
      <div className={`tabular truncate font-mono text-[13px] font-medium leading-5 ${tone}`}>
        {value}
        {unit ? <span className="ml-0.5 text-2xs text-hud-faint">{unit}</span> : null}
      </div>
    </div>
  );
}

type ChipProps = {
  children: ReactNode;
  tone?: "neutral" | "green" | "amber" | "red" | "cyan" | "blue";
  className?: string;
};

const chipTones: Record<string, string> = {
  neutral: "border-hud-edge bg-hud-raise text-hud-dim",
  green: "border-sig-green/40 bg-sig-green/10 text-sig-green",
  amber: "border-sig-amber/40 bg-sig-amber/10 text-sig-amber",
  red: "border-sig-red/40 bg-sig-red/10 text-sig-red",
  cyan: "border-sig-cyan/40 bg-sig-cyan/10 text-sig-cyan",
  blue: "border-sig-blue/40 bg-sig-blue/10 text-sig-blue"
};

export function Chip({ children, tone = "neutral", className = "" }: ChipProps) {
  return (
    <span className={`whitespace-nowrap rounded border px-1.5 py-0.5 font-mono text-2xs font-semibold uppercase tracking-[0.08em] ${chipTones[tone]} ${className}`}>
      {children}
    </span>
  );
}

type BarProps = {
  value: number;
  color?: string;
  className?: string;
};

export function Bar({ value, color = "#2dd4bf", className = "" }: BarProps) {
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-sm bg-hud-raise ${className}`}>
      <div
        className="h-full rounded-sm transition-[width] duration-200 ease-out"
        style={{ width: `${Math.min(100, Math.max(0, value))}%`, backgroundColor: color }}
      />
    </div>
  );
}
