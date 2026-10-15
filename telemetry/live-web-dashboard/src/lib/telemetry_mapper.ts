import { cellLabels } from "../data/telemetry";
import type { Telemetry } from "../types/telemetry";

export const SAMPLE_COUNT = 64;
export const POLL_INTERVAL_MS = 1000;
export const OFFLINE_AFTER_FAILURES = 3;

export type RawTelemetry = {
  connected?: boolean;
  battery_voltage?: number;
  battery_current?: number;
  battery_remaining?: number;
  battery_consumed_mah?: number;
  cell_voltages?: number[];
  altitude_relative?: number;
  altitude_msl?: number;
  climb_rate?: number;
  latitude?: number;
  longitude?: number;
  satellites?: number;
  gps_fix_type?: string;
  hdop?: number;
  rc_rssi?: number;
  radio_link_quality?: number;
  heading?: number;
  ground_speed?: number;
  armed?: boolean;
  failsafe?: boolean;
  flight_mode?: string;
  mission_state?: string;
  mission_current_seq?: number;
  mission_total_items?: number;
  attitude_roll?: number;
  attitude_pitch?: number;
  attitude_yaw?: number;
  error_roll?: number;
  error_pitch?: number;
  error_yaw?: number;
  motor_percent?: number[];
  gyro_x?: number;
  gyro_y?: number;
  gyro_z?: number;
  sbc?: {
    cpu_load_percent?: number;
    cpu_cores_percent?: number[];
    ram_used_mb?: number;
    ram_total_mb?: number;
    ram_percent?: number;
    cpu_temp_c?: number;
    disk_percent?: number;
    uptime_seconds?: number;
  };
};

const numericKeys = [
  "battery_voltage", "battery_current", "battery_remaining", "battery_consumed_mah", "altitude_relative",
  "altitude_msl", "climb_rate", "latitude", "longitude", "satellites", "hdop", "rc_rssi",
  "radio_link_quality", "heading", "ground_speed", "mission_current_seq", "mission_total_items",
  "attitude_roll", "attitude_pitch", "attitude_yaw", "error_roll", "error_pitch", "error_yaw",
  "gyro_x", "gyro_y", "gyro_z"
] as const;
const booleanKeys = ["connected", "armed", "failsafe"] as const;
const stringKeys = ["gps_fix_type", "flight_mode", "mission_state"] as const;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const numberOr = (value: unknown, fallback = 0) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

function finiteNumberArray(value: unknown, maxLength: number): number[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const values = value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
  return values.length === value.length ? values.slice(0, maxLength) : undefined;
}

export function parseRawTelemetry(value: unknown): RawTelemetry {
  if (!isRecord(value)) throw new Error("invalid telemetry payload");
  const parsed: RawTelemetry = {};
  let recognized = 0;

  for (const key of numericKeys) {
    const candidate = value[key];
    if (candidate === undefined) continue;
    if (typeof candidate !== "number" || !Number.isFinite(candidate)) throw new Error(`invalid ${key}`);
    (parsed as Record<string, unknown>)[key] = candidate;
    recognized += 1;
  }
  for (const key of booleanKeys) {
    const candidate = value[key];
    if (candidate === undefined) continue;
    if (typeof candidate !== "boolean") throw new Error(`invalid ${key}`);
    (parsed as Record<string, unknown>)[key] = candidate;
    recognized += 1;
  }
  for (const key of stringKeys) {
    const candidate = value[key];
    if (candidate === undefined) continue;
    if (typeof candidate !== "string") throw new Error(`invalid ${key}`);
    (parsed as Record<string, unknown>)[key] = candidate;
    recognized += 1;
  }

  for (const [key, maxLength] of [["cell_voltages", 6], ["motor_percent", 4]] as const) {
    if (value[key] === undefined) continue;
    const items = finiteNumberArray(value[key], maxLength);
    if (!items) throw new Error(`invalid ${key}`);
    parsed[key] = items;
    recognized += 1;
  }

  if (value.sbc !== undefined) {
    if (!isRecord(value.sbc)) throw new Error("invalid sbc");
    const sbc: NonNullable<RawTelemetry["sbc"]> = {};
    for (const key of ["cpu_load_percent", "ram_used_mb", "ram_total_mb", "ram_percent", "cpu_temp_c", "disk_percent", "uptime_seconds"] as const) {
      const candidate = value.sbc[key];
      if (candidate === undefined) continue;
      if (typeof candidate !== "number" || !Number.isFinite(candidate)) throw new Error(`invalid sbc.${key}`);
      sbc[key] = candidate;
    }
    const cores = finiteNumberArray(value.sbc.cpu_cores_percent, 32);
    if (value.sbc.cpu_cores_percent !== undefined && !cores) throw new Error("invalid sbc.cpu_cores_percent");
    if (cores) sbc.cpu_cores_percent = cores;
    parsed.sbc = sbc;
    recognized += 1;
  }

  if (recognized === 0) throw new Error("telemetry payload has no recognized fields");
  return parsed;
}

function zeroPid() {
  return Array.from({ length: SAMPLE_COUNT }, (_, t) => ({ t, roll: 0, pitch: 0, yaw: 0 }));
}

function zeroMotors() {
  return Array.from({ length: SAMPLE_COUNT }, (_, t) => ({ t, m1: 0, m2: 0, m3: 0, m4: 0 }));
}

function zeroGyro() {
  return Array.from({ length: SAMPLE_COUNT }, (_, t) => ({ t, x: 0, y: 0, z: 0 }));
}

export function createOfflineTelemetry(endpoint = "", message = "Enter ESP endpoint to start", cellCount = 4): Telemetry {
  return {
    clock: 0,
    source: { status: "offline", endpoint, message, lastUpdated: 0 },
    battery: {
      total: 0,
      percent: 0,
      current: 0,
      consumed: null,
      cells: cellLabels.slice(0, cellCount).map((label) => ({ label, volts: 0 }))
    },
    nav: { lat: 0, lon: 0, heading: 0, sats: 0, hdop: 99.9, rssi: 0, fix: "NO DATA", groundSpeed: 0 },
    altitude: { relative: 0, msl: 0, vspeed: 0, home: 0 },
    link: { rssi: 0, rate: 0, loss: 100, latency: 0, history: Array.from({ length: 28 }, () => 0) },
    flight: { mode: "UNKNOWN", armed: null, failsafe: null, mission: "TELEMETRY OFFLINE", wp: "-- / --", link: "LOST" },
    sbc: { available: false, cpu: 0, cores: [], memPercent: 0, temp: 0, uptime: 0, diskPercent: 0, load: [0, 0, 0] },
    attitude: { roll: 0, pitch: 0, yaw: 0 },
    pid: zeroPid(),
    motors: zeroMotors(),
    gyro: zeroGyro()
  };
}

export function reduceOfflineTelemetry(prev: Telemetry, endpoint: string, detail: string): Telemetry {
  const t = prev.clock + 1;
  return {
    ...prev,
    clock: t,
    source: { status: "offline", endpoint, message: `ESP JSON unavailable: ${detail}`, lastUpdated: prev.source.lastUpdated },
    nav: { ...prev.nav, fix: "NO DATA" },
    link: { ...prev.link, rate: 0, loss: 100, latency: 0, history: [...prev.link.history.slice(-27), 0] },
    flight: { ...prev.flight, mode: "UNKNOWN", armed: null, failsafe: null, mission: "TELEMETRY OFFLINE", link: "LOST" },
    sbc: { ...prev.sbc, available: false },
    pid: [...prev.pid.slice(-(SAMPLE_COUNT - 1)), { t, roll: 0, pitch: 0, yaw: 0 }],
    motors: [...prev.motors.slice(-(SAMPLE_COUNT - 1)), { t, m1: 0, m2: 0, m3: 0, m4: 0 }],
    gyro: [...prev.gyro.slice(-(SAMPLE_COUNT - 1)), { t, x: 0, y: 0, z: 0 }]
  };
}

export function reduceDegradedTelemetry(prev: Telemetry, endpoint: string, detail: string, failures: number): Telemetry {
  return {
    ...prev,
    source: {
      status: "stale",
      endpoint,
      message: `ESP JSON slow: ${detail} (${failures}/${OFFLINE_AFTER_FAILURES})`,
      lastUpdated: prev.source.lastUpdated
    }
  };
}

export function reduceLiveTelemetry(prev: Telemetry, raw: RawTelemetry, endpoint: string, latencyMs: number, cellCount: number): Telemetry {
  const t = prev.clock + 1;
  const nrfFresh = raw.connected !== false;
  const rawCells = raw.cell_voltages?.filter((cell) => cell > 0.5).slice(0, 6) ?? [];
  const fallbackCells = prev.battery.cells.slice(0, cellCount).map((cell) => cell.volts);
  const selectedCells = rawCells.length > 0 ? rawCells : fallbackCells;
  const cells = selectedCells.map((volts, index) => ({ label: cellLabels[index] ?? `${index + 1}S`, volts }));
  const total = numberOr(raw.battery_voltage, cells.reduce((sum, cell) => sum + cell.volts, 0));
  const percent = clamp(Math.round(numberOr(raw.battery_remaining, prev.battery.percent)), 0, 100);
  const heading = (numberOr(raw.heading, prev.nav.heading) + 360) % 360;
  const motors = raw.motor_percent ?? [];
  const lastMotors = prev.motors[prev.motors.length - 1] ?? { m1: 0, m2: 0, m3: 0, m4: 0 };
  const sbc = raw.sbc;
  const cpu = clamp(numberOr(sbc?.cpu_load_percent, prev.sbc.cpu), 0, 100);
  const ramPercent = clamp(numberOr(sbc?.ram_percent, prev.sbc.memPercent), 0, 100);
  const diskPercent = clamp(numberOr(sbc?.disk_percent, prev.sbc.diskPercent), 0, 100);
  const rssiPercent = clamp(Math.round(numberOr(raw.rc_rssi, prev.link.rssi)), 0, 100);
  const reportedQuality = raw.radio_link_quality === undefined ? null : clamp(raw.radio_link_quality, 0, 100);
  const linkRate = nrfFresh ? reportedQuality ?? rssiPercent : 0;
  const linkLoss = nrfFresh ? clamp(100 - linkRate, 0, 100) : 100;

  return {
    clock: t,
    source: {
      status: nrfFresh ? "online" : "stale",
      endpoint,
      message: nrfFresh ? "ESP JSON live" : "ESP JSON reachable, waiting for fresh NRF packets",
      lastUpdated: Date.now()
    },
    battery: {
      total,
      percent,
      current: numberOr(raw.battery_current, prev.battery.current),
      consumed: raw.battery_consumed_mah === undefined ? prev.battery.consumed : Math.max(0, Math.round(raw.battery_consumed_mah)),
      cells
    },
    nav: {
      lat: numberOr(raw.latitude, prev.nav.lat),
      lon: numberOr(raw.longitude, prev.nav.lon),
      heading,
      sats: clamp(Math.round(numberOr(raw.satellites, prev.nav.sats)), 0, 64),
      hdop: numberOr(raw.hdop, prev.nav.hdop),
      rssi: rssiPercent,
      fix: raw.gps_fix_type ?? prev.nav.fix,
      groundSpeed: nrfFresh ? numberOr(raw.ground_speed, prev.nav.groundSpeed) : 0
    },
    altitude: {
      relative: numberOr(raw.altitude_relative, prev.altitude.relative),
      msl: numberOr(raw.altitude_msl, prev.altitude.msl),
      vspeed: numberOr(raw.climb_rate, prev.altitude.vspeed),
      home: prev.altitude.home
    },
    link: { rssi: rssiPercent, rate: linkRate, loss: linkLoss, latency: latencyMs, history: [...prev.link.history.slice(-27), linkRate] },
    flight: {
      mode: nrfFresh ? raw.flight_mode ?? prev.flight.mode : "UNKNOWN",
      armed: nrfFresh ? raw.armed ?? prev.flight.armed : null,
      failsafe: nrfFresh ? raw.failsafe ?? null : null,
      mission: nrfFresh ? raw.mission_state ?? prev.flight.mission : "NRF LINK STALE",
      wp: `${Math.round(numberOr(raw.mission_current_seq, 0))} / ${Math.round(numberOr(raw.mission_total_items, 0))}`,
      link: nrfFresh ? "OK" : "LOST"
    },
    sbc: {
      available: sbc !== undefined,
      cpu,
      cores: sbc?.cpu_cores_percent?.map((value) => clamp(value, 0, 100)) ?? [],
      memPercent: ramPercent,
      temp: numberOr(sbc?.cpu_temp_c, prev.sbc.temp),
      uptime: numberOr(sbc?.uptime_seconds, prev.sbc.uptime),
      diskPercent,
      load: [cpu, ramPercent, diskPercent]
    },
    attitude: {
      roll: nrfFresh ? numberOr(raw.attitude_roll, prev.attitude.roll) : prev.attitude.roll,
      pitch: nrfFresh ? numberOr(raw.attitude_pitch, prev.attitude.pitch) : prev.attitude.pitch,
      yaw: nrfFresh ? numberOr(raw.attitude_yaw, heading) : prev.attitude.yaw
    },
    pid: [...prev.pid.slice(-(SAMPLE_COUNT - 1)), {
      t,
      roll: nrfFresh ? numberOr(raw.error_roll, 0) : 0,
      pitch: nrfFresh ? numberOr(raw.error_pitch, 0) : 0,
      yaw: nrfFresh ? numberOr(raw.error_yaw, 0) : 0
    }],
    motors: [...prev.motors.slice(-(SAMPLE_COUNT - 1)), {
      t,
      m1: nrfFresh ? numberOr(motors[0], lastMotors.m1) : 0,
      m2: nrfFresh ? numberOr(motors[1], lastMotors.m2) : 0,
      m3: nrfFresh ? numberOr(motors[2], lastMotors.m3) : 0,
      m4: nrfFresh ? numberOr(motors[3], lastMotors.m4) : 0
    }],
    gyro: [...prev.gyro.slice(-(SAMPLE_COUNT - 1)), {
      t,
      x: nrfFresh ? numberOr(raw.gyro_x, 0) : 0,
      y: nrfFresh ? numberOr(raw.gyro_y, 0) : 0,
      z: nrfFresh ? numberOr(raw.gyro_z, 0) : 0
    }]
  };
}
