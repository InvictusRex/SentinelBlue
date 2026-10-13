import { useEffect, useMemo, useRef, useState } from "react";
import { CrosshairIcon, MapPinnedIcon, PlayIcon, RouteIcon, ShieldCheckIcon } from "lucide-react";

interface MissionCell {
  cell_id: number;
  row: number;
  column: number;
  center_latitude: number;
  center_longitude: number;
}

interface MissionPlan {
  center: { latitude: number; longitude: number };
  home: { latitude: number; longitude: number };
  radius_km: number;
  altitude_m: number;
  cell_count: number;
  rows: number;
  columns: number;
  cells: MissionCell[];
  route_cell_ids: number[];
  phases: string[];
  mission_items: Array<Record<string, number>>;
}

interface MissionResponse {
  status: string;
  plan: MissionPlan;
}

const API_STORAGE_KEY = "droneCompanionApi";
const VIEW_WIDTH = 720;
const VIEW_HEIGHT = 420;

function isPosition(latitude: number, longitude: number): boolean {
  return Number.isFinite(latitude) && Number.isFinite(longitude) &&
    latitude >= -90 && latitude <= 90 && longitude >= -180 && longitude <= 180 &&
    !(latitude === 0 && longitude === 0);
}

function normalizeApi(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) throw new Error("Enter the Raspberry Pi companion API address.");
  const withProtocol = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  const url = new URL(withProtocol);
  if (!/^https?:$/.test(url.protocol)) throw new Error("Use an HTTP or HTTPS companion API address.");
  return url.origin;
}

function factors(value: number): [number, number] {
  let rows = Math.floor(Math.sqrt(value));
  while (rows > 1 && value % rows !== 0) rows -= 1;
  return [rows, value / rows];
}

function routeIds(rows: number, columns: number): number[] {
  return Array.from({ length: rows }, (_, row) => {
    const ids = Array.from({ length: columns }, (_, column) => row * columns + column + 1);
    return row % 2 === 0 ? ids : ids.reverse();
  }).flat();
}

function coordinateOffset(
  originLatitude: number,
  originLongitude: number,
  northMetres: number,
  eastMetres: number
): [number, number] {
  const latitude = originLatitude + northMetres / 111_320;
  const longitudeScale = 111_320 * Math.max(0.01, Math.cos(originLatitude * Math.PI / 180));
  return [latitude, originLongitude + eastMetres / longitudeScale];
}

export function MissionPlanner({
  currentLatitude,
  currentLongitude,
  telemetryAvailable
}: {
  currentLatitude: number;
  currentLongitude: number;
  telemetryAvailable: boolean;
}) {
  const [apiInput, setApiInput] = useState(() => window.localStorage.getItem(API_STORAGE_KEY) ?? "localhost:8765");
  const [centerLatitude, setCenterLatitude] = useState("");
  const [centerLongitude, setCenterLongitude] = useState("");
  const [radiusKm, setRadiusKm] = useState(1);
  const [altitudeM, setAltitudeM] = useState(30);
  const [plan, setPlan] = useState<MissionPlan | null>(null);
  const [requestState, setRequestState] = useState<"idle" | "planning" | "ready" | "starting" | "running" | "error">("idle");
  const [message, setMessage] = useState("Select a point on the plot or enter coordinates.");
  const initializedFromTelemetry = useRef(false);

  const currentPositionValid = telemetryAvailable && isPosition(currentLatitude, currentLongitude);
  useEffect(() => {
    if (!initializedFromTelemetry.current && currentPositionValid) {
      setCenterLatitude(currentLatitude.toFixed(7));
      setCenterLongitude(currentLongitude.toFixed(7));
      initializedFromTelemetry.current = true;
    }
  }, [currentLatitude, currentLongitude, currentPositionValid]);

  const selectedLatitude = Number(centerLatitude);
  const selectedLongitude = Number(centerLongitude);
  const selectedPositionValid = isPosition(selectedLatitude, selectedLongitude);
  const estimatedCellCount = Math.max(1, Math.ceil(radiusKm * 4));
  const [estimatedRows, estimatedColumns] = useMemo(() => factors(estimatedCellCount), [estimatedCellCount]);
  const displayedRows = plan?.rows ?? estimatedRows;
  const displayedColumns = plan?.columns ?? estimatedColumns;
  const displayedRoute = plan?.route_cell_ids ?? routeIds(displayedRows, displayedColumns);

  const plotOrigin = useMemo(() => currentPositionValid
    ? { latitude: currentLatitude, longitude: currentLongitude }
    : selectedPositionValid
      ? { latitude: selectedLatitude, longitude: selectedLongitude }
      : null,
  [currentLatitude, currentLongitude, currentPositionValid, selectedLatitude, selectedLongitude, selectedPositionValid]);
  const extentMetres = Math.max(1_000, radiusKm * 1_600);

  const selectedPlotPoint = useMemo(() => {
    if (!plotOrigin || !selectedPositionValid) return null;
    const north = (selectedLatitude - plotOrigin.latitude) * 111_320;
    const east = (selectedLongitude - plotOrigin.longitude) * 111_320 *
      Math.max(0.01, Math.cos(plotOrigin.latitude * Math.PI / 180));
    return {
      x: VIEW_WIDTH / 2 + (east / extentMetres) * (VIEW_WIDTH / 2),
      y: VIEW_HEIGHT / 2 - (north / extentMetres) * (VIEW_HEIGHT / 2)
    };
  }, [extentMetres, plotOrigin, selectedLatitude, selectedLongitude, selectedPositionValid]);

  const searchBox = {
    width: (radiusKm * 1_000 / extentMetres) * VIEW_WIDTH,
    height: (radiusKm * 1_000 / extentMetres) * VIEW_HEIGHT
  };

  const selectPoint = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!plotOrigin) {
      setRequestState("error");
      setMessage("Enter valid coordinates before selecting a point on the plot.");
      return;
    }
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * VIEW_WIDTH;
    const y = ((event.clientY - bounds.top) / bounds.height) * VIEW_HEIGHT;
    const east = ((x - VIEW_WIDTH / 2) / (VIEW_WIDTH / 2)) * extentMetres;
    const north = ((VIEW_HEIGHT / 2 - y) / (VIEW_HEIGHT / 2)) * extentMetres;
    const [latitude, longitude] = coordinateOffset(plotOrigin.latitude, plotOrigin.longitude, north, east);
    setCenterLatitude(latitude.toFixed(7));
    setCenterLongitude(longitude.toFixed(7));
    setPlan(null);
    setRequestState("idle");
    setMessage("Search centre selected. Generate the route to review every cell.");
  };

  const requestPlan = async (path: "/api/mission/plan" | "/api/mission/simulate") => {
    if (!selectedPositionValid) {
      setRequestState("error");
      setMessage("Enter a valid latitude and longitude.");
      return;
    }
    const starting = path.endsWith("simulate");
    setRequestState(starting ? "starting" : "planning");
    setMessage(starting ? "Loading the mission into the simulator…" : "Generating the grid route on the Raspberry Pi…");
    try {
      const api = normalizeApi(apiInput);
      window.localStorage.setItem(API_STORAGE_KEY, apiInput.trim());
      const response = await fetch(`${api}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          center: { latitude: selectedLatitude, longitude: selectedLongitude },
          radius_km: radiusKm,
          altitude_m: altitudeM
        })
      });
      const body = await response.json() as MissionResponse | { error?: string };
      if (!response.ok || !("plan" in body)) {
        throw new Error("error" in body && body.error ? body.error : `Companion API returned ${response.status}`);
      }
      setPlan(body.plan);
      setRequestState(starting ? "running" : "ready");
      setMessage(starting
        ? "Simulation started. Follow mission progress in the telemetry view."
        : `${body.plan.cell_count} cells planned. Review the route before starting the simulator.`);
    } catch (error) {
      setRequestState("error");
      setMessage(error instanceof Error ? error.message : "The Raspberry Pi companion API could not be reached.");
    }
  };

  const boxLeft = (selectedPlotPoint?.x ?? VIEW_WIDTH / 2) - searchBox.width / 2;
  const boxTop = (selectedPlotPoint?.y ?? VIEW_HEIGHT / 2) - searchBox.height / 2;
  const cellWidth = searchBox.width / displayedColumns;
  const cellHeight = searchBox.height / displayedRows;
  const routePoints = displayedRoute.map((cellId) => {
    const index = cellId - 1;
    const row = Math.floor(index / displayedColumns);
    const column = index % displayedColumns;
    return `${boxLeft + column * cellWidth + cellWidth / 2},${boxTop + row * cellHeight + cellHeight / 2}`;
  }).join(" ");

  return (
    <main className="flex flex-1 flex-col gap-3 overflow-auto p-3 text-hud-text xl:min-h-0 xl:overflow-hidden">
      <div className="flex flex-col justify-between gap-3 border-b border-hud-line pb-3 lg:flex-row lg:items-end">
        <div>
          <h1 className="text-lg font-semibold tracking-[-0.02em]">Companion search planner</h1>
          <p className="mt-1 max-w-[72ch] text-sm leading-5 text-hud-muted">
            Choose the search centre, set radius and altitude, then inspect the complete takeoff, grid search, return and landing route.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded border border-sig-amber/40 bg-sig-amber/10 px-3 py-2 text-xs text-sig-amber">
          <ShieldCheckIcon size={16} />
          Simulation execution only · live Pixhawk control locked
        </div>
      </div>

      <div className="grid flex-1 gap-3 xl:min-h-0 xl:grid-cols-[minmax(0,1fr)_340px]">
        <section className="flex min-h-[480px] flex-col overflow-hidden rounded border border-hud-line bg-hud-panel xl:min-h-0">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-hud-line px-3 py-2">
            <span className="flex items-center gap-2 text-sm font-semibold"><MapPinnedIcon size={16} /> Search area</span>
            <span className="font-mono text-xs text-hud-muted">click plot to place centre · N ↑</span>
          </div>
          <svg
            className="min-h-[420px] flex-1 cursor-crosshair bg-[#070b10] focus:outline-none focus:ring-2 focus:ring-inset focus:ring-teal-300/50"
            viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
            role="application"
            aria-label="Mission coordinate plot. Click to select the search centre."
            tabIndex={0}
            onClick={selectPoint}
          >
            <defs>
              <pattern id="mission-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(94,234,212,0.10)" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width={VIEW_WIDTH} height={VIEW_HEIGHT} fill="url(#mission-grid)" />
            <line x1={VIEW_WIDTH / 2} y1="0" x2={VIEW_WIDTH / 2} y2={VIEW_HEIGHT} stroke="rgba(255,255,255,0.09)" />
            <line x1="0" y1={VIEW_HEIGHT / 2} x2={VIEW_WIDTH} y2={VIEW_HEIGHT / 2} stroke="rgba(255,255,255,0.09)" />

            {selectedPlotPoint ? (
              <g>
                <rect
                  x={boxLeft}
                  y={boxTop}
                  width={searchBox.width}
                  height={searchBox.height}
                  fill="rgba(45,212,191,0.08)"
                  stroke="#5eead4"
                  strokeWidth="2"
                />
                {Array.from({ length: displayedColumns - 1 }, (_, index) => (
                  <line key={`column-${index}`} x1={boxLeft + (index + 1) * cellWidth} y1={boxTop} x2={boxLeft + (index + 1) * cellWidth} y2={boxTop + searchBox.height} stroke="rgba(94,234,212,0.45)" />
                ))}
                {Array.from({ length: displayedRows - 1 }, (_, index) => (
                  <line key={`row-${index}`} x1={boxLeft} y1={boxTop + (index + 1) * cellHeight} x2={boxLeft + searchBox.width} y2={boxTop + (index + 1) * cellHeight} stroke="rgba(94,234,212,0.45)" />
                ))}
                <polyline points={routePoints} fill="none" stroke="#fbbf24" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                {displayedRoute.map((cellId) => {
                  const index = cellId - 1;
                  const row = Math.floor(index / displayedColumns);
                  const column = index % displayedColumns;
                  return <circle key={cellId} cx={boxLeft + column * cellWidth + cellWidth / 2} cy={boxTop + row * cellHeight + cellHeight / 2} r="4" fill="#fbbf24" />;
                })}
                <circle cx={selectedPlotPoint.x} cy={selectedPlotPoint.y} r="8" fill="#5eead4" stroke="#071014" strokeWidth="3" />
              </g>
            ) : null}

            {currentPositionValid ? (
              <g transform={`translate(${VIEW_WIDTH / 2} ${VIEW_HEIGHT / 2})`}>
                <circle r="11" fill="#60a5fa" stroke="#071014" strokeWidth="3" />
                <path d="M 0 -18 L 5 -9 L -5 -9 Z" fill="#60a5fa" />
                <text x="16" y="5" fill="#bfdbfe" fontSize="12" fontFamily="monospace">DRONE / HOME</text>
              </g>
            ) : null}
          </svg>
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-hud-line px-3 py-2 font-mono text-xs text-hud-muted">
            <span>{plotOrigin ? `${plotOrigin.latitude.toFixed(6)}, ${plotOrigin.longitude.toFixed(6)}` : "Waiting for a coordinate origin"}</span>
            <span>plot span ±{(extentMetres / 1000).toFixed(1)} km</span>
          </div>
        </section>

        <aside className="space-y-3 overflow-auto pr-0.5 xl:min-h-0">
          <form
            className="space-y-3 rounded border border-hud-line bg-hud-panel p-3"
            onSubmit={(event) => { event.preventDefault(); void requestPlan("/api/mission/plan"); }}
          >
            <div className="flex items-center gap-2 text-sm font-semibold"><CrosshairIcon size={16} /> Mission inputs</div>
            <label className="block text-xs text-hud-muted">
              Raspberry Pi companion API
              <input value={apiInput} onChange={(event) => setApiInput(event.target.value)} placeholder="192.168.1.50:8765" className="mt-1 h-9 w-full rounded border border-hud-line bg-hud-bg px-2 font-mono text-sm text-hud-text outline-none focus:border-teal-300 focus:ring-2 focus:ring-teal-300/20" />
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="text-xs text-hud-muted">Latitude
                <input value={centerLatitude} onChange={(event) => { setCenterLatitude(event.target.value); setPlan(null); }} inputMode="decimal" className="mt-1 h-9 w-full rounded border border-hud-line bg-hud-bg px-2 font-mono text-sm text-hud-text outline-none focus:border-teal-300" />
              </label>
              <label className="text-xs text-hud-muted">Longitude
                <input value={centerLongitude} onChange={(event) => { setCenterLongitude(event.target.value); setPlan(null); }} inputMode="decimal" className="mt-1 h-9 w-full rounded border border-hud-line bg-hud-bg px-2 font-mono text-sm text-hud-text outline-none focus:border-teal-300" />
              </label>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="text-xs text-hud-muted">Search radius · km
                <input type="number" min="0.25" max="10" step="0.25" value={radiusKm} onChange={(event) => { setRadiusKm(Number(event.target.value)); setPlan(null); }} className="mt-1 h-9 w-full rounded border border-hud-line bg-hud-bg px-2 font-mono text-sm text-hud-text outline-none focus:border-teal-300" />
              </label>
              <label className="text-xs text-hud-muted">Flight altitude · m
                <input type="number" min="5" max="120" step="1" value={altitudeM} onChange={(event) => { setAltitudeM(Number(event.target.value)); setPlan(null); }} className="mt-1 h-9 w-full rounded border border-hud-line bg-hud-bg px-2 font-mono text-sm text-hud-text outline-none focus:border-teal-300" />
              </label>
            </div>
            <div className="grid grid-cols-3 gap-2 border-y border-hud-line py-3 text-center">
              <div><div className="font-mono text-lg text-white">{estimatedCellCount}</div><div className="text-2xs uppercase text-hud-faint">cells</div></div>
              <div><div className="font-mono text-lg text-white">{estimatedRows} × {estimatedColumns}</div><div className="text-2xs uppercase text-hud-faint">grid</div></div>
              <div><div className="font-mono text-lg text-white">{(radiusKm * 2).toFixed(1)}</div><div className="text-2xs uppercase text-hud-faint">km side</div></div>
            </div>
            <button type="submit" disabled={requestState === "planning" || requestState === "starting"} className="inline-flex h-10 w-full items-center justify-center gap-2 rounded bg-teal-300 px-3 text-sm font-semibold text-[#04110f] transition hover:bg-teal-200 focus:outline-none focus:ring-2 focus:ring-teal-200/40 disabled:cursor-wait disabled:opacity-50">
              <RouteIcon size={16} /> {requestState === "planning" ? "Generating route…" : "Generate flight plan"}
            </button>
          </form>

          <section className="rounded border border-hud-line bg-hud-panel p-3" aria-live="polite">
            <div className={`text-sm leading-5 ${requestState === "error" ? "text-sig-red" : "text-hud-muted"}`}>{message}</div>
            {plan ? (
              <div className="mt-3 space-y-3">
                <ol className="grid grid-cols-2 gap-2 text-xs">
                  {plan.phases.map((phase, index) => (
                    <li key={phase} className="flex items-center gap-2 rounded bg-hud-bg px-2 py-2">
                      <span className="grid h-5 w-5 place-items-center rounded-full bg-teal-300/15 font-mono text-teal-200">{index + 1}</span>
                      <span className="capitalize">{phase.replace("_", " ")}</span>
                    </li>
                  ))}
                </ol>
                <div className="font-mono text-xs leading-5 text-hud-faint">
                  {plan.cell_count} search cells · {plan.mission_items.length} mission items · {plan.altitude_m} m AGL
                </div>
                <button type="button" onClick={() => void requestPlan("/api/mission/simulate")} disabled={requestState === "starting" || requestState === "running"} className="inline-flex h-10 w-full items-center justify-center gap-2 rounded border border-sig-amber/50 bg-sig-amber/10 text-sm font-semibold text-sig-amber transition hover:bg-sig-amber/15 focus:outline-none focus:ring-2 focus:ring-sig-amber/30 disabled:opacity-50">
                  <PlayIcon size={16} /> {requestState === "starting" ? "Starting simulator…" : requestState === "running" ? "Simulation running" : "Run in simulator"}
                </button>
              </div>
            ) : null}
          </section>
        </aside>
      </div>
    </main>
  );
}
