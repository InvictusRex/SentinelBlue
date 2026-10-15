import { useEffect, useMemo, useRef, useState } from "react";
import { DownloadIcon, MapPinnedIcon } from "lucide-react";
import { TerrainMissionMap } from "./terrain_mission_map";
import { downloadText, missionPlannerFile, radiusPreview, type Position, type SurveyPlan } from "../lib/survey_plan";

const inputClass = "mt-1 h-11 w-full rounded border border-hud-line bg-hud-bg px-3 font-mono text-sm text-hud-text focus:border-teal-300 focus:outline-none";
const buttonClass = "min-h-11 rounded border border-hud-line px-3 py-2 text-sm font-semibold hover:border-teal-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-teal-300 disabled:cursor-not-allowed disabled:opacity-40";
function apiOrigin(input: string) {
  const url = new URL(/^https?:\/\//i.test(input) ? input : `http://${input}`);
  if (!/^https?:$/.test(url.protocol) || url.username || url.password) throw new Error("Enter an HTTP or HTTPS companion API address.");
  return url.origin;
}

export function MissionPlanner({ currentLatitude, currentLongitude, telemetryAvailable }: {
  currentLatitude: number; currentLongitude: number; telemetryAvailable: boolean;
}) {
  const [api, setApi] = useState(() => { try { return localStorage.getItem("droneCompanionApi") ?? "localhost:8765"; } catch { return "localhost:8765"; } });
  const [center, setCenter] = useState<Position | null>(null);
  const [radius, setRadius] = useState(25);
  const [focus, setFocus] = useState<Position>({ latitude: 12.971598, longitude: 77.594562 });
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [fitVersion, setFitVersion] = useState(0);
  const [plan, setPlan] = useState<SurveyPlan | null>(null);
  const [sortie, setSortie] = useState(1);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState<"planning" | "starting" | null>(null);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("Click the map to choose the search centre.");
  const [failed, setFailed] = useState(false);
  const request = useRef<AbortController | null>(null);
  const version = useRef(0);
  const initialized = useRef(false);
  const vehicleValid = telemetryAvailable && Number.isFinite(currentLatitude) && Number.isFinite(currentLongitude) && Math.abs(currentLatitude) <= 85 && Math.abs(currentLongitude) <= 180 && (currentLatitude !== 0 || currentLongitude !== 0);
  useEffect(() => {
    if (!initialized.current && vehicleValid) {
      initialized.current = true; setFocus({ latitude: currentLatitude, longitude: currentLongitude });
    }
  }, [vehicleValid, currentLatitude, currentLongitude]);
  useEffect(() => () => request.current?.abort(), []);
  const preview = useMemo(() => radiusPreview(center, radius), [center, radius]);
  const boundary = plan?.boundary ?? preview;
  const invalidate = () => {
    version.current += 1; request.current?.abort(); setPlan(null); setConfirmed(false); setBusy(null); setFailed(false);
    setMessage("Search area changed. Send it to the Pi to generate the flight plan.");
  };
  const chooseCenter = (point: Position) => {
    if (busy === "starting") return;
    invalidate(); initialized.current = true; setCenter(point);
    setLatitude(point.latitude.toFixed(7)); setLongitude(point.longitude.toFixed(7));
  };
  const fail = (error: unknown) => { setFailed(true); setMessage(error instanceof Error ? error.message : "The companion API could not be reached."); };
  const generate = async () => {
    if (!center || !preview.length) return;
    invalidate(); const revision = version.current;
    const controller = new AbortController(); request.current = controller;
    const timeout = setTimeout(() => controller.abort(), 60000);
    setBusy("planning"); setMessage("Pi is calculating camera coverage, grid direction and battery sorties…");
    try {
      try { localStorage.setItem("droneCompanionApi", api); } catch { /* Storage is optional. */ }
      const response = await fetch(`${apiOrigin(api)}/api/mission/plan`, { method: "POST", signal: controller.signal, headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ search_area: { center, radius_m: radius } }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? `Planning failed (${response.status}).`);
      if (!body.plan?.search_area || !Array.isArray(body.plan.sorties)) throw new Error("Update the Pi companion API for point-and-radius planning.");
      if (revision !== version.current) return;
      setPlan(body.plan); setSortie(1); setFitVersion(value => value + 1);
      setMessage("Pi-generated plan ready. Review the route and select a flight sortie.");
    } catch (error) { if (revision === version.current) fail(controller.signal.aborted ? new Error("Planning timed out. Check the Pi connection or reduce the search radius.") : error); }
    finally { clearTimeout(timeout); if (revision === version.current) setBusy(null); }
  };
  const simulate = async () => {
    if (!plan || !confirmed) return;
    setBusy("starting"); setFailed(false);
    try {
      const response = await fetch(`${apiOrigin(api)}/api/mission/simulate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ plan_id: plan.plan_id, sortie_id: sortie, review_confirmed: true }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? "Simulator could not start.");
      setRunning(true); setMessage(`Sortie ${sortie} is running in simulation.`);
    } catch (error) { fail(error); } finally { setBusy(null); }
  };
  useEffect(() => {
    if (!running) return;
    const controller = new AbortController();
    const timer = setInterval(() => {
      fetch(`${apiOrigin(api)}/api/health`, { signal: controller.signal }).then(response => response.json()).then(body => {
        if (body.armed === false) { setRunning(false); setMessage("Simulator is disarmed. Review the next sortie before starting it."); }
      }).catch(() => { /* Connection loss must not enable another start. */ });
    }, 3000);
    return () => { clearInterval(timer); controller.abort(); };
  }, [api, running]);
  const selected = plan?.sorties.find(item => item.sortie_id === sortie);

  return <main className="flex flex-1 flex-col gap-3 overflow-auto p-3 text-hud-text xl:min-h-0 xl:overflow-hidden">
    <header className="flex flex-wrap items-end justify-between gap-2 border-b border-hud-line pb-3">
      <div><h1 className="text-xl font-semibold tracking-tight">Automatic search survey</h1><p className="mt-1 text-sm text-hud-muted">Choose a point and radius. The Pi plans the flight.</p></div>
      <span className="text-xs text-amber-200">Planning & simulation · live flight locked</span>
    </header>
    <div className="grid flex-1 gap-3 xl:min-h-0 xl:grid-cols-[minmax(0,1fr)_370px]">
      <section className="flex min-h-[500px] flex-col overflow-hidden rounded border border-hud-line bg-hud-panel xl:min-h-0" aria-label="Search area map">
        <div className="flex items-center justify-between gap-3 border-b border-hud-line p-3"><span className="text-sm text-hud-muted">Click the map to move the search centre.</span><button type="button" className={buttonClass} onClick={() => setFitVersion(value => value + 1)}>Fit area</button></div>
        <TerrainMissionMap boundary={boundary} home={plan?.home ?? null} center={center} plan={plan} sortie={sortie} mode="center" focus={focus} fitVersion={fitVersion} onClick={chooseCenter} onMoveVertex={() => {}} />
        <div className="flex flex-wrap justify-between gap-2 border-t border-hud-line px-3 py-2 text-xs text-hud-muted"><span>C: centre · H: Pi launch position · amber: survey · dashed: transfers</span><span>{center ? `${radius} m radius` : "No point selected"}</span></div>
      </section>
      <aside className="space-y-3 overflow-auto xl:min-h-0">
        <form onSubmit={event => { event.preventDefault(); void generate(); }} className="rounded border border-hud-line bg-hud-panel p-4">
          <fieldset disabled={busy === "starting"} className="min-w-0 space-y-4">
            <h2 className="flex items-center gap-2 text-sm font-semibold"><MapPinnedIcon size={16} />Search area</h2>
            <div className="text-xs leading-5 text-hud-muted">{center ? <><span className="block">Selected centre</span><span className="font-mono text-hud-text">{center.latitude.toFixed(7)}, {center.longitude.toFixed(7)}</span></> : "Select the centre on the map to begin."}</div>
            <label className="block text-sm text-hud-muted">Search radius · metres<input className={inputClass} type="number" required min="2" max="10000" step="1" value={radius} onChange={event => { invalidate(); setRadius(Number(event.target.value)); }} /><span className="mt-1 block text-xs">1,000 m = 1 km from the centre in every direction.</span></label>
            <p className="text-sm leading-6 text-hud-muted">The Pi uses its saved camera and aircraft settings to calculate spacing, direction, photo timing, takeoff, survey, return and landing. Large searches are split into battery sorties.</p>
            <button type="submit" disabled={busy !== null || !center || !preview.length || running} className="min-h-11 w-full rounded bg-teal-300 px-3 py-2 text-sm font-semibold text-[#04110f] hover:bg-teal-200 disabled:opacity-40">{busy === "planning" ? "Pi is planning…" : "Generate survey on Pi"}</button>
            <details><summary className="cursor-pointer text-xs text-hud-muted">Connection & coordinate entry</summary><div className="mt-3 space-y-3">
              <label className="block text-xs text-hud-muted">Companion API<input className={inputClass} disabled={running} value={api} onChange={event => { invalidate(); setApi(event.target.value); }} placeholder="192.168.1.50:8765" required /></label>
              <div className="grid grid-cols-2 gap-2"><label className="text-xs text-hud-muted">Latitude<input className={inputClass} value={latitude} onChange={event => setLatitude(event.target.value)} /></label><label className="text-xs text-hud-muted">Longitude<input className={inputClass} value={longitude} onChange={event => setLongitude(event.target.value)} /></label></div>
              <button type="button" className={buttonClass} onClick={() => { const point = { latitude: Number(latitude), longitude: Number(longitude) }; if (!latitude.trim() || !longitude.trim() || !Number.isFinite(point.latitude) || !Number.isFinite(point.longitude) || Math.abs(point.latitude) > 85 || Math.abs(point.longitude) > 180) { fail(new Error("Enter valid latitude (−85 to 85) and longitude (−180 to 180).")); return; } chooseCenter(point); setFocus(point); setFitVersion(value => value + 1); }}>Use this point</button>
            </div></details>
          </fieldset>
        </form>
        <section className="space-y-3 rounded border border-hud-line bg-hud-panel p-4" aria-label="Pi survey result">
          <p role={failed ? "alert" : "status"} className={`text-sm leading-5 ${failed ? "text-red-300" : "text-hud-muted"}`}>{message}</p>
          {plan && <>
            <h2 className="text-sm font-semibold">Pi flight plan</h2>
            <div className="grid grid-cols-2 gap-3 border-y border-hud-line py-3 text-sm"><div><strong className="block font-mono">{Math.round(plan.area_m2).toLocaleString()} m²</strong><span className="text-xs text-hud-muted">Search area</span></div><div><strong className="block font-mono">{plan.sorties.length}</strong><span className="text-xs text-hud-muted">Battery sorties</span></div><div><strong className="block font-mono">{plan.estimate.survey_line_count}</strong><span className="text-xs text-hud-muted">Flight lines</span></div><div><strong className="block font-mono">{plan.coverage_percent}%</strong><span className="text-xs text-hud-muted">Modelled coverage</span></div></div>
            <label className="block text-xs text-hud-muted">Flight sortie<select className={inputClass} disabled={busy === "starting"} value={sortie} onChange={event => { setSortie(Number(event.target.value)); setConfirmed(false); }}>{plan.sorties.map(item => <option key={item.sortie_id} value={item.sortie_id}>Sortie {item.sortie_id} · {item.estimated_minutes} min{item.ready ? "" : " · blocked"}</option>)}</select></label>
            {selected && <p className="text-xs leading-5 text-hud-muted">{(selected.distance_m / 1000).toFixed(2)} km · {selected.estimated_photos} photos · {selected.outside_boundary_m} m outside the search boundary on transfers/run-outs.</p>}
            {plan.estimate.blocking_reasons.length > 0 && <div className="border-l-2 border-red-400 pl-3 text-xs leading-5 text-red-200"><strong>Pi checks need attention</strong><ul className="list-disc pl-4">{plan.estimate.blocking_reasons.map(reason => <li key={reason}>{reason}</li>)}</ul></div>}
            <details><summary className="cursor-pointer text-xs text-hud-muted">Calculated settings & flight assumptions</summary><div className="mt-2 space-y-2 text-xs leading-5 text-hud-muted"><p>Pi profile: {plan.profile_name}. Height {plan.altitude_m} m · speed {plan.cruise_speed_m_s} m/s · gap {plan.camera.line_spacing_m} m · bearing {plan.grid_angle_deg}° · trigger every {plan.camera.trigger_distance_m} m. Budget {plan.sortie_budget_minutes} min per battery after reserve.</p><ul className="list-disc pl-4">{plan.estimate.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul><p>Terrain: {plan.terrain.message ?? plan.terrain.status}. Terrain data does not change flight altitude.</p></div></details>
            <p className="text-xs leading-5 text-amber-200">Altitude is relative to home. This does not automatically maintain clearance above water or terrain.</p>
            <button type="button" className={`${buttonClass} flex w-full items-center justify-center gap-2`} disabled={!selected?.ready} onClick={() => { try { if (selected) downloadText(`sentinelblue_survey_${sortie}.waypoints`, missionPlannerFile(selected.mission_items)); } catch (error) { fail(error); } }}><DownloadIcon size={15} />Export to Mission Planner</button>
            <p className="text-xs text-hud-muted">Mission Planner → PLAN → Load WP File.</p>
            <label className="flex gap-2 text-xs leading-5 text-hud-muted"><input className="mt-1 self-start" type="checkbox" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} /><span>I reviewed this route and its altitude assumptions for simulation.</span></label>
            <button type="button" className={`${buttonClass} w-full text-amber-200`} disabled={!selected?.ready || !confirmed || busy !== null || running} onClick={() => void simulate()}>{busy === "starting" ? "Starting…" : running ? "Simulation running" : "Simulate this sortie"}</button>
          </>}
        </section>
      </aside>
    </div>
  </main>;
}
