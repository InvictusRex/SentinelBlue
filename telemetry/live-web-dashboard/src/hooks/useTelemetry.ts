import { useEffect, useState } from "react";
import { cellLabels } from "../data/telemetry";
import type { Telemetry } from "../types/telemetry";

const SAMPLES = 64;
const POLL_MS = 1000;
const REQUEST_TIMEOUT_MS = 2500;
const OFFLINE_AFTER_FAILURES = 3;
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const num = (value: unknown, fallback = 0) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);

type RawTelemetry = {
  connected?: boolean;
  battery_voltage?: number;
  battery_current?: number;
  battery_remaining?: number;
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
    ram_used_mb?: number;
    ram_total_mb?: number;
    ram_percent?: number;
    cpu_temp_c?: number;
    disk_percent?: number;
    uptime_seconds?: number;
  };
};

function zeroPid() {
  return Array.from({ length: SAMPLES }, (_, t) => ({ t, roll: 0, pitch: 0, yaw: 0 }));
}

function zeroMotors() {
  return Array.from({ length: SAMPLES }, (_, t) => ({ t, m1: 0, m2: 0, m3: 0, m4: 0 }));
}

function zeroGyro() {
  return Array.from({ length: SAMPLES }, (_, t) => ({ t, x: 0, y: 0, z: 0 }));
}

function offlineTelemetry(endpoint = "", message = "Enter ESP IP to start"): Telemetry {
  return {
    clock: 0,
    source: {
      status: "offline",
      endpoint,
      message,
      lastUpdated: 0
    },
    battery: {
      total: 0,
      percent: 0,
      current: 0,
      consumed: 0,
      cells: cellLabels.slice(0, 4).map((label) => ({ label, volts: 0 }))
    },
    nav: {
      lat: 0,
      lon: 0,
      heading: 0,
      sats: 0,
      hdop: 99.9,
      rssi: 0,
      fix: "NO JSON",
      groundSpeed: 0
    },
    altitude: {
      relative: 0,
      msl: 0,
      vspeed: 0,
      home: 0
    },
    link: {
      rssi: 0,
      rate: 0,
      loss: 100,
      latency: 0,
      history: Array.from({ length: 28 }, () => 0)
    },
    flight: {
      mode: "NO ESP JSON",
      armed: false,
      failsafe: true,
      mission: "ESP JSON OFFLINE",
      wp: "0 / 0",
      link: "LOST"
    },
    sbc: {
      cpu: 0,
      cores: [0, 0, 0, 0],
      memPercent: 0,
      temp: 0,
      uptime: 0,
      diskPercent: 0,
      load: [0, 0, 0]
    },
    attitude: { roll: 0, pitch: 0, yaw: 0 },
    pid: zeroPid(),
    motors: zeroMotors(),
    gyro: zeroGyro()
  };
}

function markJsonOffline(prev: Telemetry, endpoint: string, detail: string): Telemetry {
  const t = prev.clock + 1;

  return {
    ...prev,
    clock: t,
    source: {
      status: "offline",
      endpoint,
      message: `ESP JSON unavailable: ${detail}`,
      lastUpdated: prev.source.lastUpdated
    },
    nav: {
      ...prev.nav,
      fix: "NO JSON",
      rssi: 0,
      groundSpeed: 0
    },
    link: {
      rssi: 0,
      rate: 0,
      loss: 100,
      latency: 0,
      history: [...prev.link.history.slice(-27), 0]
    },
    flight: {
      ...prev.flight,
      mode: "NO ESP JSON",
      armed: false,
      failsafe: true,
      mission: "CHECK ESP WIFI JSON",
      link: "LOST"
    },
    pid: [...prev.pid.slice(-(SAMPLES - 1)), { t, roll: 0, pitch: 0, yaw: 0 }],
    motors: [...prev.motors.slice(-(SAMPLES - 1)), { t, m1: 0, m2: 0, m3: 0, m4: 0 }],
    gyro: [...prev.gyro.slice(-(SAMPLES - 1)), { t, x: 0, y: 0, z: 0 }]
  };
}

function markJsonDegraded(prev: Telemetry, endpoint: string, detail: string, failures: number): Telemetry {
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

function fromLive(prev: Telemetry, raw: RawTelemetry, endpoint: string, latencyMs: number): Telemetry {
  const t = prev.clock + 1;
  const nrfFresh = raw.connected !== false;
  const rawCells = Array.isArray(raw.cell_voltages) ? raw.cell_voltages.filter((cell) => cell > 0.5).slice(0, 6) : [];
  const cells = (rawCells.length > 0 ? rawCells : prev.battery.cells.map((cell) => cell.volts)).map((volts, i) => ({
    label: cellLabels[i] ?? `${i + 1}S`,
    volts
  }));
  const total = num(raw.battery_voltage, cells.reduce((sum, cell) => sum + cell.volts, 0));
  const percent = clamp(Math.round(num(raw.battery_remaining, prev.battery.percent)), 0, 100);
  const heading = (num(raw.heading, prev.nav.heading) + 360) % 360;
  const motors = Array.isArray(raw.motor_percent) ? raw.motor_percent : [];
  const lastMotors = prev.motors[prev.motors.length - 1] ?? { m1: 0, m2: 0, m3: 0, m4: 0 };
  const sbc = raw.sbc ?? {};
  const ramPercent = clamp(num(sbc.ram_percent, prev.sbc.memPercent), 0, 100);
  const diskPercent = clamp(num(sbc.disk_percent, prev.sbc.diskPercent), 0, 100);
  const rssiPercent = clamp(Math.round(num(raw.rc_rssi, 0)), 0, 100);
  const reportedQuality = num(raw.radio_link_quality, 0);
  const linkRate = nrfFresh ? clamp(reportedQuality > 0 ? reportedQuality : rssiPercent > 0 ? rssiPercent : 95, 0, 100) : 0;
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
      current: num(raw.battery_current, prev.battery.current),
      consumed: prev.battery.consumed,
      cells
    },
    nav: {
      lat: num(raw.latitude, prev.nav.lat),
      lon: num(raw.longitude, prev.nav.lon),
      heading,
      sats: clamp(Math.round(num(raw.satellites, prev.nav.sats)), 0, 32),
      hdop: num(raw.hdop, prev.nav.hdop),
      rssi: rssiPercent,
      fix: raw.gps_fix_type ?? prev.nav.fix,
      groundSpeed: nrfFresh ? num(raw.ground_speed, prev.nav.groundSpeed) : 0
    },
    altitude: {
      relative: num(raw.altitude_relative, prev.altitude.relative),
      msl: num(raw.altitude_msl, prev.altitude.msl),
      vspeed: num(raw.climb_rate, prev.altitude.vspeed),
      home: prev.altitude.home
    },
    link: {
      rssi: rssiPercent,
      rate: linkRate,
      loss: linkLoss,
      latency: latencyMs,
      history: [...prev.link.history.slice(-27), linkRate]
    },
    flight: {
      mode: nrfFresh ? raw.flight_mode ?? prev.flight.mode : "NO NRF",
      armed: nrfFresh ? raw.armed ?? prev.flight.armed : false,
      failsafe: !nrfFresh,
      mission: nrfFresh ? raw.mission_state ?? prev.flight.mission : "NRF LINK STALE",
      wp: `${Math.round(num(raw.mission_current_seq, 0))} / ${Math.round(num(raw.mission_total_items, 0))}`,
      link: nrfFresh ? "OK" : "LOST"
    },
    sbc: {
      cpu: clamp(num(sbc.cpu_load_percent, prev.sbc.cpu), 0, 100),
      cores: Array.from({ length: 4 }, () => clamp(num(sbc.cpu_load_percent, prev.sbc.cpu), 0, 100)),
      memPercent: ramPercent,
      temp: num(sbc.cpu_temp_c, prev.sbc.temp),
      uptime: num(sbc.uptime_seconds, prev.sbc.uptime),
      diskPercent,
      load: [clamp(num(sbc.cpu_load_percent, prev.sbc.cpu), 0, 100), ramPercent, diskPercent]
    },
    attitude: {
      roll: nrfFresh ? num(raw.attitude_roll, prev.attitude.roll) : 0,
      pitch: nrfFresh ? num(raw.attitude_pitch, prev.attitude.pitch) : 0,
      yaw: nrfFresh ? num(raw.attitude_yaw, heading) : heading
    },
    pid: [
      ...prev.pid.slice(-(SAMPLES - 1)),
      {
        t,
        roll: nrfFresh ? num(raw.error_roll, 0) : 0,
        pitch: nrfFresh ? num(raw.error_pitch, 0) : 0,
        yaw: nrfFresh ? num(raw.error_yaw, 0) : 0
      }
    ],
    motors: [
      ...prev.motors.slice(-(SAMPLES - 1)),
      {
        t,
        m1: nrfFresh ? num(motors[0], lastMotors.m1) : 0,
        m2: nrfFresh ? num(motors[1], lastMotors.m2) : 0,
        m3: nrfFresh ? num(motors[2], lastMotors.m3) : 0,
        m4: nrfFresh ? num(motors[3], lastMotors.m4) : 0
      }
    ],
    gyro: [
      ...prev.gyro.slice(-(SAMPLES - 1)),
      {
        t,
        x: nrfFresh ? num(raw.gyro_x, 0) : 0,
        y: nrfFresh ? num(raw.gyro_y, 0) : 0,
        z: nrfFresh ? num(raw.gyro_z, 0) : 0
      }
    ]
  };
}

function errorText(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") return "request timeout";
  if (error instanceof Error) return error.message;
  return "request failed";
}

export function useTelemetry(live: boolean, endpoint: string): Telemetry {
  const [state, setState] = useState<Telemetry>(() => offlineTelemetry(endpoint));

  useEffect(() => {
    setState(offlineTelemetry(endpoint, endpoint ? "Waiting for ESP JSON" : "Enter ESP IP to start"));
  }, [endpoint]);

  useEffect(() => {
    if (!live || endpoint.length === 0) return;
    let stopped = false;
    let failures = 0;
    let timer: number | undefined;

    const poll = async () => {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const startedAt = performance.now();
      try {
        const response = await fetch(endpoint, { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`http ${response.status}`);
        const raw = (await response.json()) as RawTelemetry;
        const latencyMs = Math.max(0, Math.round(performance.now() - startedAt));
        failures = 0;
        setState((prev) => fromLive(prev, raw, endpoint, latencyMs));
      } catch (error) {
        failures += 1;
        const detail = errorText(error);
        setState((prev) =>
          failures >= OFFLINE_AFTER_FAILURES
            ? markJsonOffline(prev, endpoint, detail)
            : markJsonDegraded(prev, endpoint, detail, failures)
        );
      } finally {
        window.clearTimeout(timeout);
        if (!stopped) {
          timer = window.setTimeout(() => void poll(), POLL_MS);
        }
      }
    };

    void poll();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [live, endpoint]);

  return state;
}
