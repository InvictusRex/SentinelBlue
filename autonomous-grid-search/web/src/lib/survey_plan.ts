export interface Position { latitude: number; longitude: number }
export type MissionItem = Record<string, number>;
export interface SwarmSegment { start_s: number; end_s: number; start: Position; end: Position; phase: string; cell_id: number; sortie_id: number }
export interface SwarmPosition extends Position { drone_id: number; cell_id: number; phase: string }
export const droneColors = ["#2563eb", "#c026d3", "#15803d", "#ea580c", "#0891b2", "#7c3aed", "#be123c", "#854d0e"];
export interface SurveyPlan {
  search_area?: { center: Position; radius_m: number; shape?: string; side_m?: number }; profile_name?: string; cruise_speed_m_s: number;
  cell_count?: number; cell_side_m?: number; max_cell_side_m?: number;
  cells?: { cell_id: number; boundary: Position[]; center: Position; drone_id: number; line_count: number; coverage_percent: number; sortie_ids: number[] }[];
  swarm?: { simulation_only: boolean; drone_count: number; duration_s: number; warning: string;
    drones: { drone_id: number; cell_ids: number[]; sortie_ids: number[]; duration_s: number; segments: SwarmSegment[] }[] };
  plan_id: string; schema_version: number; home: Position; boundary: Position[];
  altitude_m: number; area_m2: number; coverage_percent: number; grid_angle_deg: number;
  sortie_budget_minutes: number;
  camera: { cross_track_m: number; along_track_m: number; line_spacing_m: number;
    trigger_distance_m: number; trigger_interval_s: number; gsd_width_cm: number; gsd_height_cm: number };
  survey_lines: { cell_id?: number; drone_id?: number; sortie_id: number; start_latitude: number; start_longitude: number; end_latitude: number; end_longitude: number }[];
  sorties: { cell_id?: number; drone_id?: number; scheduled_end_s?: number; sortie_id: number; line_count: number; route: Position[]; distance_m: number;
    estimated_minutes: number; estimated_photos: number; outside_boundary_m: number;
    mission_item_count: number; ready: boolean; blocking_reasons: string[]; mission_items: MissionItem[] }[];
  estimate: { survey_line_count: number; total_distance_m: number; estimated_minutes: number;
    estimated_photos: number; blocking_reasons: string[]; warnings: string[] };
  terrain: { status: string; message?: string; sample_count?: number; elevation_range_m?: number };
}

// Display preview only. The Pi computes the authoritative geodesic boundary.
export function squarePreview(center: Position | null, radius: number): Position[] {
  if (!center || !Number.isFinite(radius) || radius < 2 || radius > 10000) return [];
  const lat = center.latitude * Math.PI / 180, lon = center.longitude * Math.PI / 180;
  const distance = Math.SQRT2 * radius / 6371008.8;
  return [225, 135, 45, 315].map(degrees => {
    const bearing = degrees * Math.PI / 180;
    const latitude = Math.asin(Math.sin(lat) * Math.cos(distance) + Math.cos(lat) * Math.sin(distance) * Math.cos(bearing));
    const longitude = lon + Math.atan2(Math.sin(bearing) * Math.sin(distance) * Math.cos(lat), Math.cos(distance) - Math.sin(lat) * Math.sin(latitude));
    return { latitude: latitude * 180 / Math.PI, longitude: ((longitude * 180 / Math.PI + 540) % 360) - 180 };
  });
}

export function swarmPositions(swarm: SurveyPlan["swarm"], elapsed: number): SwarmPosition[] {
  return (swarm?.drones ?? []).flatMap(drone => {
    const segments = drone.segments;
    const segment = segments.find(item => elapsed < item.end_s) ?? segments[segments.length - 1];
    if (!segment) return [];
    const progress = Math.max(0, Math.min(1, (elapsed - segment.start_s) / Math.max(.001, segment.end_s - segment.start_s)));
    return [{ drone_id: drone.drone_id, cell_id: segment.cell_id, phase: elapsed >= drone.duration_s ? "complete" : segment.phase,
      latitude: segment.start.latitude + (segment.end.latitude - segment.start.latitude) * progress,
      longitude: segment.start.longitude + (segment.end.longitude - segment.start.longitude) * progress }];
  });
}

export function rectangle(a: Position, b: Position): Position[] {
  return [a, { latitude: a.latitude, longitude: b.longitude }, b, { latitude: b.latitude, longitude: a.longitude }];
}

export function parseBoundary(text: string): Position[] {
  const points = text.trim().split(/\n/).filter(line => line.trim()).map(line => {
    const pair = line.trim().split(/[,\s]+/);
    const [latitude, longitude] = pair.map(Number);
    if (pair.length !== 2 || !Number.isFinite(latitude) || !Number.isFinite(longitude) ||
        Math.abs(latitude) > 85 || Math.abs(longitude) > 180) throw new Error("Use one latitude, longitude pair per line (latitude −85 to 85).");
    return { latitude, longitude };
  });
  if (points.length < 3 || points.length > 64) throw new Error("Enter 3 to 64 boundary vertices.");
  return points;
}

export function missionPlannerFile(items: MissionItem[]): string {
  if (!items.length) throw new Error("No exportable mission items.");
  const rows = items.map(item => {
    const frame = item.frame === 6 ? 3 : item.frame;
    const lat = item.frame === 6 ? item.x / 1e7 : item.x;
    const lon = item.frame === 6 ? item.y / 1e7 : item.y;
    const values = [item.seq, item.current, frame, item.command, item.param1, item.param2,
      item.param3, item.param4, lat, lon, item.z, item.autocontinue];
    if (values.some(value => !Number.isFinite(value))) throw new Error("Invalid mission coordinates or command parameters.");
    return values.map((value, index) => index === 8 || index === 9 ? value.toFixed(7) : value).join("\t");
  });
  return `QGC WPL 110\n${rows.join("\n")}\n`;
}

export function downloadText(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
