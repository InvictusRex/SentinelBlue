import { useEffect, useRef, useState } from "react";
import { CameraIcon, CrosshairIcon, DownloadIcon, MapPinnedIcon, PlayIcon, RouteIcon, ShieldAlertIcon, ShieldCheckIcon } from "lucide-react";
import { TerrainMissionMap } from "./terrain_mission_map";

interface MissionCell {
  cell_id: number; row: number; column: number;
  north_m: number; east_m: number; south_m: number; west_m: number;
  center_latitude: number; center_longitude: number;
}
interface SurveyLine {
  cell_id: number; line_index: number;
  start_latitude: number; start_longitude: number;
  end_latitude: number; end_longitude: number; length_m: number;
}
interface MissionPlan {
  center: { latitude: number; longitude: number };
  home: { latitude: number; longitude: number };
  radius_km: number; altitude_m: number; cruise_speed_m_s: number;
  cell_count: number; rows: number; columns: number;
  cells: MissionCell[]; survey_lines: SurveyLine[]; phases: string[];
  mission_items: Array<Record<string, number>>;
  camera: {
    horizontal_fov_deg: number; vertical_fov_deg: number;
    side_overlap_percent: number; forward_overlap_percent: number;
    footprint_width_m: number; footprint_height_m: number;
    line_spacing_m: number; trigger_distance_m: number;
  };
  estimate: {
    survey_line_count: number; survey_distance_m: number; total_distance_m: number;
    estimated_minutes: number; estimated_photos: number; mission_item_count: number;
    recommended_sorties: number; simulation_ready: boolean;
    blocking_reasons: string[]; warnings: string[];
  };
  terrain: {
    status: string; provider?: string; sample_count?: number;
    minimum_elevation_m?: number; maximum_elevation_m?: number;
    elevation_range_m?: number; message?: string;
    used_for_clearance: boolean; clearance_source?: string;
  };
}
interface MissionResponse { status: string; plan: MissionPlan }
interface ApiError { error?: string; reasons?: string[] }

const API_STORAGE_KEY = "droneCompanionApi";
const inputClass = "mt-1 h-9 w-full rounded border border-hud-line bg-hud-bg px-2 font-mono text-sm text-hud-text outline-none focus:border-teal-300";

function isPosition(latitude: number, longitude: number): boolean {
  return Number.isFinite(latitude) && Number.isFinite(longitude) &&
    latitude >= -90 && latitude <= 90 && longitude >= -180 && longitude <= 180 &&
    !(latitude === 0 && longitude === 0);
}
function normalizeApi(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) throw new Error("Enter the Raspberry Pi companion API address.");
  const url = new URL(/^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`);
  if (!/^https?:$/.test(url.protocol)) throw new Error("Use an HTTP or HTTPS companion API address.");
  return url.origin;
}
function compactDistance(metres: number): string {
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${metres.toFixed(1)} m`;
}
function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="rounded border border-hud-line bg-hud-bg px-2 py-2"><div className="font-mono text-sm text-white">{value}</div><div className="mt-0.5 text-2xs uppercase tracking-wide text-hud-faint">{label}</div></div>;
}
function missionPlannerFile(items: Array<Record<string, number>>): string {
  const rows = items.map((item) => {
    const coordinateFrame = item.frame === 6 ? 3 : item.frame;
    const latitude = item.frame === 6 ? item.x / 1e7 : item.x;
    const longitude = item.frame === 6 ? item.y / 1e7 : item.y;
    return [
      item.seq, item.current, coordinateFrame, item.command,
      item.param1, item.param2, item.param3, item.param4,
      latitude.toFixed(7), longitude.toFixed(7), item.z, item.autocontinue
    ].join("\t");
  });
  return `QGC WPL 110\n${rows.join("\n")}\n`;
}

export function MissionPlanner({ currentLatitude, currentLongitude, telemetryAvailable }: {
  currentLatitude: number; currentLongitude: number; telemetryAvailable: boolean;
}) {
  const [apiInput, setApiInput] = useState(() => window.localStorage.getItem(API_STORAGE_KEY) ?? "localhost:8765");
  const [centerLatitude, setCenterLatitude] = useState("");
  const [centerLongitude, setCenterLongitude] = useState("");
  const [radiusKm, setRadiusKm] = useState(0.025);
  const [altitudeM, setAltitudeM] = useState(5);
  const [horizontalFov, setHorizontalFov] = useState(78);
  const [verticalFov, setVerticalFov] = useState(60);
  const [sideOverlap, setSideOverlap] = useState(70);
  const [forwardOverlap, setForwardOverlap] = useState(75);
  const [cruiseSpeed, setCruiseSpeed] = useState(3);
  const [surfaceType, setSurfaceType] = useState("water");
  const [terrainEnabled, setTerrainEnabled] = useState(true);
  const [rangefinderConfirmed, setRangefinderConfirmed] = useState(false);
  const [plan, setPlan] = useState<MissionPlan | null>(null);
  const [requestState, setRequestState] = useState<"idle" | "planning" | "ready" | "starting" | "running" | "error">("idle");
  const [message, setMessage] = useState("Select a point on the terrain map or enter coordinates.");
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
  const needsRangefinder = altitudeM <= 15;
  const clearPlan = () => { setPlan(null); setRequestState("idle"); };

  const requestPlan = async (path: "/api/mission/plan" | "/api/mission/simulate") => {
    if (!selectedPositionValid) {
      setRequestState("error");
      setMessage("Enter a valid latitude and longitude or click the map.");
      return;
    }
    const starting = path.endsWith("simulate");
    setRequestState(starting ? "starting" : "planning");
    setMessage(starting ? "Loading the checked mission into the simulator..." : "Calculating camera coverage and terrain context...");
    try {
      const api = normalizeApi(apiInput);
      window.localStorage.setItem(API_STORAGE_KEY, apiInput.trim());
      const response = await fetch(`${api}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          center: { latitude: selectedLatitude, longitude: selectedLongitude },
          radius_km: radiusKm, altitude_m: altitudeM,
          camera_horizontal_fov_deg: horizontalFov, camera_vertical_fov_deg: verticalFov,
          side_overlap_percent: sideOverlap, forward_overlap_percent: forwardOverlap,
          cruise_speed_m_s: cruiseSpeed, surface_type: surfaceType,
          terrain_enabled: terrainEnabled
        })
      });
      const body = await response.json() as MissionResponse | ApiError;
      if (!response.ok || !("plan" in body)) {
        const details = "reasons" in body && body.reasons?.length ? ` ${body.reasons.join(" ")}` : "";
        throw new Error(`${"error" in body && body.error ? body.error : `Companion API returned ${response.status}`}${details}`);
      }
      setPlan(body.plan);
      setRequestState(starting ? "running" : "ready");
      setMessage(starting
        ? "Simulation started. Follow progress in the telemetry view."
        : body.plan.estimate.simulation_ready
          ? "Coverage plan generated. Confirm the rangefinder check before simulation."
          : `Plan generated, but it must be split into about ${body.plan.estimate.recommended_sorties} sorties.`);
    } catch (error) {
      setRequestState("error");
      setMessage(error instanceof Error ? error.message : "The Raspberry Pi companion API could not be reached.");
    }
  };

  const simulatorBlocked = !plan?.estimate.simulation_ready || (needsRangefinder && !rangefinderConfirmed);
  const terrainSummary = plan?.terrain.status === "available"
    ? `${plan.terrain.sample_count} DEM samples · ${plan.terrain.minimum_elevation_m}-${plan.terrain.maximum_elevation_m} m AMSL`
    : plan?.terrain.message ?? (plan?.terrain.status === "disabled" ? "Terrain sampling disabled" : "Terrain data unavailable");
  const downloadMission = () => {
    if (!plan?.mission_items.length) return;
    const blob = new Blob([missionPlannerFile(plan.mission_items)], { type: "text/plain;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "sentinelblue_camera_survey.waypoints";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <main className="flex flex-1 flex-col gap-3 overflow-auto p-3 text-hud-text xl:min-h-0 xl:overflow-hidden">
      <div className="flex flex-col justify-between gap-3 border-b border-hud-line pb-3 lg:flex-row lg:items-end">
        <div>
          <h1 className="text-lg font-semibold tracking-[-0.02em]">Camera coverage mission planner</h1>
          <p className="mt-1 max-w-[76ch] text-sm leading-5 text-hud-muted">Plan a complete lawnmower survey from the camera footprint, overlap and flight altitude. Every cell receives parallel coverage lines.</p>
        </div>
        <div className="flex items-center gap-2 rounded border border-sig-amber/40 bg-sig-amber/10 px-3 py-2 text-xs text-sig-amber"><ShieldCheckIcon size={16} />Simulation only · live Pixhawk control locked</div>
      </div>

      <div className="grid flex-1 gap-3 xl:min-h-0 xl:grid-cols-[minmax(0,1fr)_380px]">
        <section className="flex min-h-[500px] flex-col overflow-hidden rounded border border-hud-line bg-hud-panel xl:min-h-0">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-hud-line px-3 py-2">
            <span className="flex items-center gap-2 text-sm font-semibold"><MapPinnedIcon size={16} /> Terrain and coverage</span>
            <span className="font-mono text-xs text-hud-muted">click map to place search centre</span>
          </div>
          <TerrainMissionMap
            selected={selectedPositionValid ? { latitude: selectedLatitude, longitude: selectedLongitude } : null}
            plan={plan}
            onSelect={(latitude, longitude) => {
              setCenterLatitude(latitude.toFixed(7));
              setCenterLongitude(longitude.toFixed(7));
              clearPlan();
              setMessage("Search centre selected. Generate the plan to calculate coverage.");
            }}
          />
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-hud-line px-3 py-2 font-mono text-xs text-hud-muted">
            <span>{selectedPositionValid ? `${selectedLatitude.toFixed(6)}, ${selectedLongitude.toFixed(6)}` : "No search centre selected"}</span>
            <span>amber: flight lines · teal: cells · blue: home</span>
          </div>
        </section>

        <aside className="space-y-3 overflow-auto pr-0.5 xl:min-h-0">
          <form className="space-y-3 rounded border border-hud-line bg-hud-panel p-3" onSubmit={(event) => { event.preventDefault(); void requestPlan("/api/mission/plan"); }}>
            <div className="flex items-center gap-2 text-sm font-semibold"><CrosshairIcon size={16} /> Mission inputs</div>
            <label className="block text-xs text-hud-muted">Raspberry Pi companion API<input value={apiInput} onChange={(event) => setApiInput(event.target.value)} placeholder="192.168.1.50:8765" className={inputClass} /></label>
            <div className="grid grid-cols-2 gap-2">
              <label className="text-xs text-hud-muted">Latitude<input value={centerLatitude} onChange={(event) => { setCenterLatitude(event.target.value); clearPlan(); }} inputMode="decimal" className={inputClass} /></label>
              <label className="text-xs text-hud-muted">Longitude<input value={centerLongitude} onChange={(event) => { setCenterLongitude(event.target.value); clearPlan(); }} inputMode="decimal" className={inputClass} /></label>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <label className="text-xs text-hud-muted">Radius · km<input type="number" min="0.01" max="10" step="0.005" value={radiusKm} onChange={(event) => { setRadiusKm(Number(event.target.value)); clearPlan(); }} className={inputClass} /></label>
              <label className="text-xs text-hud-muted">Altitude · m<input type="number" min="5" max="120" step="1" value={altitudeM} onChange={(event) => { setAltitudeM(Number(event.target.value)); clearPlan(); }} className={inputClass} /></label>
              <label className="text-xs text-hud-muted">Speed · m/s<input type="number" min="0.5" max="15" step="0.5" value={cruiseSpeed} onChange={(event) => { setCruiseSpeed(Number(event.target.value)); clearPlan(); }} className={inputClass} /></label>
            </div>

            <div className="border-t border-hud-line pt-3">
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-hud-muted"><CameraIcon size={14} /> Downward camera</div>
              <div className="grid grid-cols-2 gap-2">
                <label className="text-xs text-hud-muted">Horizontal FOV · °<input type="number" min="10" max="170" value={horizontalFov} onChange={(event) => { setHorizontalFov(Number(event.target.value)); clearPlan(); }} className={inputClass} /></label>
                <label className="text-xs text-hud-muted">Vertical FOV · °<input type="number" min="10" max="170" value={verticalFov} onChange={(event) => { setVerticalFov(Number(event.target.value)); clearPlan(); }} className={inputClass} /></label>
                <label className="text-xs text-hud-muted">Side overlap · %<input type="number" min="0" max="95" value={sideOverlap} onChange={(event) => { setSideOverlap(Number(event.target.value)); clearPlan(); }} className={inputClass} /></label>
                <label className="text-xs text-hud-muted">Forward overlap · %<input type="number" min="0" max="95" value={forwardOverlap} onChange={(event) => { setForwardOverlap(Number(event.target.value)); clearPlan(); }} className={inputClass} /></label>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 border-t border-hud-line pt-3">
              <label className="text-xs text-hud-muted">Surface<select value={surfaceType} onChange={(event) => { setSurfaceType(event.target.value); clearPlan(); }} className={inputClass}><option value="water">Water</option><option value="land">Land</option></select></label>
              <label className="flex items-end gap-2 pb-2 text-xs text-hud-muted"><input type="checkbox" checked={terrainEnabled} onChange={(event) => { setTerrainEnabled(event.target.checked); clearPlan(); }} className="h-4 w-4 accent-teal-300" /> Sample terrain API</label>
            </div>
            <button type="submit" disabled={requestState === "planning" || requestState === "starting"} className="inline-flex h-10 w-full items-center justify-center gap-2 rounded bg-teal-300 px-3 text-sm font-semibold text-[#04110f] transition hover:bg-teal-200 disabled:cursor-wait disabled:opacity-50"><RouteIcon size={16} />{requestState === "planning" ? "Calculating coverage..." : "Generate coverage plan"}</button>
          </form>

          <section className="rounded border border-hud-line bg-hud-panel p-3" aria-live="polite">
            <div className={`text-sm leading-5 ${requestState === "error" ? "text-sig-red" : "text-hud-muted"}`}>{message}</div>
            {plan ? <div className="mt-3 space-y-3">
              <div className="grid grid-cols-3 gap-2">
                <Metric label="footprint" value={`${plan.camera.footprint_width_m} × ${plan.camera.footprint_height_m} m`} />
                <Metric label="line spacing" value={`${plan.camera.line_spacing_m} m`} />
                <Metric label="photo trigger" value={`${plan.camera.trigger_distance_m} m`} />
                <Metric label="flight lines" value={plan.estimate.survey_line_count} />
                <Metric label="route" value={compactDistance(plan.estimate.total_distance_m)} />
                <Metric label="duration" value={`${plan.estimate.estimated_minutes} min`} />
                <Metric label="photos" value={plan.estimate.estimated_photos.toLocaleString()} />
                <Metric label="mission items" value={plan.estimate.mission_item_count.toLocaleString()} />
                <Metric label="sorties needed" value={plan.estimate.recommended_sorties} />
              </div>
              <div className="rounded border border-hud-line bg-hud-bg p-2 text-xs leading-5 text-hud-muted"><span className="text-hud-text">Terrain:</span> {terrainSummary}<div className="text-hud-faint">DEM is map context only. Clearance at low altitude uses a downward rangefinder.</div></div>
              {plan.estimate.blocking_reasons.length ? <div className="rounded border border-sig-red/40 bg-sig-red/10 p-2 text-xs leading-5 text-sig-red"><div className="mb-1 flex items-center gap-2 font-semibold"><ShieldAlertIcon size={14} /> Execution blocked</div><ul className="list-disc pl-4">{plan.estimate.blocking_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div> : null}
              {plan.estimate.warnings.length ? <ul className="list-disc pl-4 text-xs leading-5 text-sig-amber">{plan.estimate.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
              {needsRangefinder ? <label className="flex gap-2 rounded border border-sig-amber/40 bg-sig-amber/10 p-2 text-xs leading-5 text-sig-amber"><input type="checkbox" checked={rangefinderConfirmed} onChange={(event) => setRangefinderConfirmed(event.target.checked)} className="mt-0.5 h-4 w-4 shrink-0 accent-amber-300" /><span>I confirm a downward rangefinder is installed, healthy and tested over this surface.</span></label> : null}
              <ol className="grid grid-cols-2 gap-2 text-xs">{plan.phases.map((phase, index) => <li key={phase} className="flex items-center gap-2 rounded bg-hud-bg px-2 py-2"><span className="grid h-5 w-5 place-items-center rounded-full bg-teal-300/15 font-mono text-teal-200">{index + 1}</span><span className="capitalize">{phase.replace(/_/g, " ")}</span></li>)}</ol>
              {plan.mission_items.length ? <button type="button" onClick={downloadMission} className="inline-flex h-10 w-full items-center justify-center gap-2 rounded border border-hud-line bg-hud-bg text-sm font-semibold text-hud-text transition hover:border-teal-300/60 hover:text-teal-200"><DownloadIcon size={16} />Download Mission Planner file</button> : null}
              <button type="button" onClick={() => void requestPlan("/api/mission/simulate")} disabled={requestState === "starting" || requestState === "running" || simulatorBlocked} title={simulatorBlocked ? "Resolve the safety and sortie checks first" : undefined} className="inline-flex h-10 w-full items-center justify-center gap-2 rounded border border-sig-amber/50 bg-sig-amber/10 text-sm font-semibold text-sig-amber transition hover:bg-sig-amber/15 disabled:cursor-not-allowed disabled:opacity-40"><PlayIcon size={16} />{requestState === "starting" ? "Starting simulator..." : requestState === "running" ? "Simulation running" : "Run checked plan in simulator"}</button>
            </div> : null}
          </section>
        </aside>
      </div>
    </main>
  );
}
