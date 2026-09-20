import { useEffect, useState } from "react";
import { cellLabels } from "../data/telemetry";
import type { GyroPoint, MotorPoint, PidPoint, Telemetry } from "../types/telemetry";

const SAMPLES = 64;
const TICK_MS = 260;
const LIVE_ENDPOINT = "http://192.168.4.1/telemetry.json";

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const wobble = (amp: number) => (Math.random() - 0.5) * 2 * amp;
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

function getLiveEndpoint(): string {
  const configured = new URLSearchParams(window.location.search).get("telemetryUrl");
  return configured && configured.trim().length > 0 ? configured : LIVE_ENDPOINT;
}

function seedSeries(): {
  pid: PidPoint[];
  motors: MotorPoint[];
  gyro: GyroPoint[];
} {
  const pid: PidPoint[] = [];
  const motors: MotorPoint[] = [];
  const gyro: GyroPoint[] = [];

  for (let i = 0; i < SAMPLES; i += 1) {
    const p = i / 6;
    pid.push({
      t: i,
      roll: Math.sin(p) * 6 + wobble(1.4),
      pitch: Math.sin(p * 0.8 + 1.2) * 4.5 + wobble(1.2),
      yaw: Math.sin(p * 0.45 + 2.4) * 3 + wobble(0.8)
    });
    motors.push({
      t: i,
      m1: 58 + Math.sin(p * 0.9) * 5 + wobble(1.6),
      m2: 61 + Math.sin(p * 0.9 + 1.6) * 5 + wobble(1.6),
      m3: 56 + Math.sin(p * 0.9 + 3.1) * 5 + wobble(1.6),
      m4: 63 + Math.sin(p * 0.9 + 4.7) * 5 + wobble(1.6)
    });
    gyro.push({
      t: i,
      x: Math.sin(p * 1.4) * 22 + wobble(6),
      y: Math.sin(p * 1.1 + 0.9) * 17 + wobble(6),
      z: Math.sin(p * 0.7 + 2.2) * 11 + wobble(4)
    });
  }

  return { pid, motors, gyro };
}

function initial(): Telemetry {
  const series = seedSeries();
  const cells = [4.02, 3.98, 4.01, 3.95];

  return {
    clock: SAMPLES,
    battery: {
      total: cells.reduce((a, b) => a + b, 0),
      percent: 74,
      current: 18.4,
      consumed: 2140,
      cells: cells.map((volts, i) => ({ label: cellLabels[i], volts }))
    },
    nav: {
      lat: 12.971834,
      lon: 77.593122,
      heading: 142,
      sats: 14,
      hdop: 0.82,
      rssi: -63,
      fix: "RTK FLOAT",
      groundSpeed: 8.4
    },
    altitude: {
      relative: 62.4,
      msl: 982.7,
      vspeed: 0.6,
      home: 920.3
    },
    link: {
      rssi: -68,
      rate: 92,
      loss: 1.4,
      latency: 34,
      history: Array.from({ length: 28 }, () => 78 + Math.random() * 20)
    },
    flight: {
      mode: "AUTO",
      armed: true,
      failsafe: false,
      mission: "WAYPOINT NAV",
      wp: "7 / 12",
      link: "OK"
    },
    sbc: {
      cpu: 42,
      cores: [38, 51, 33, 46],
      memUsed: 2.9,
      memTotal: 8,
      temp: 58.2,
      uptime: 9412,
      diskUsed: 21.4,
      diskTotal: 64,
      load: [0.82, 0.71, 0.64]
    },
    attitude: { roll: 6.2, pitch: -3.4, yaw: 142 },
    pid: series.pid,
    motors: series.motors,
    gyro: series.gyro
  };
}

function step(prev: Telemetry): Telemetry {
  const t = prev.clock + 1;
  const p = t / 6;
  const cells = prev.battery.cells.map((cell) => ({
    ...cell,
    volts: clamp(cell.volts - 0.00035 + wobble(0.004), 3.2, 4.2)
  }));
  const total = cells.reduce((a, b) => a + b.volts, 0);
  const heading = (prev.nav.heading + wobble(1.1) + 360.15) % 360;
  const relativeAltitude = clamp(prev.altitude.relative + prev.altitude.vspeed * 0.26 + wobble(0.05), 0, 400);
  const verticalSpeed = clamp(prev.altitude.vspeed + wobble(0.16), -2.4, 2.4);
  const linkRate = clamp(prev.link.rate + wobble(3), 62, 100);

  return {
    clock: t,
    battery: {
      total,
      percent: clamp(Math.round(((total / 4 - 3.35) / (4.2 - 3.35)) * 100), 0, 100),
      current: clamp(prev.battery.current + wobble(0.6), 11, 27),
      consumed: prev.battery.consumed + 1,
      cells
    },
    nav: {
      ...prev.nav,
      lat: prev.nav.lat + 0.0000098,
      lon: prev.nav.lon + 0.0000074,
      heading,
      sats: clamp(Math.round(prev.nav.sats + wobble(0.4)), 11, 17),
      hdop: clamp(prev.nav.hdop + wobble(0.02), 0.6, 1.4),
      rssi: clamp(Math.round(prev.nav.rssi + wobble(0.9)), -78, -52),
      groundSpeed: clamp(prev.nav.groundSpeed + wobble(0.25), 3, 14)
    },
    altitude: {
      ...prev.altitude,
      relative: relativeAltitude,
      msl: prev.altitude.home + relativeAltitude,
      vspeed: verticalSpeed
    },
    link: {
      rssi: clamp(Math.round(prev.link.rssi + wobble(1.2)), -84, -54),
      rate: linkRate,
      loss: clamp(prev.link.loss + wobble(0.25), 0, 7.5),
      latency: clamp(Math.round(prev.link.latency + wobble(3)), 18, 96),
      history: [...prev.link.history.slice(-27), linkRate]
    },
    flight: prev.flight,
    sbc: {
      ...prev.sbc,
      cpu: clamp(prev.sbc.cpu + wobble(4), 18, 88),
      cores: prev.sbc.cores.map((core) => clamp(core + wobble(7), 8, 97)),
      memUsed: clamp(prev.sbc.memUsed + wobble(0.06), 1.8, 6.4),
      temp: clamp(prev.sbc.temp + wobble(0.5), 44, 76),
      uptime: prev.sbc.uptime + 1,
      diskUsed: prev.sbc.diskUsed,
      load: [
        clamp(prev.sbc.load[0] + wobble(0.06), 0.1, 3.2),
        prev.sbc.load[1],
        prev.sbc.load[2]
      ]
    },
    attitude: {
      roll: clamp(Math.sin(p * 0.7) * 14 + wobble(1.6), -32, 32),
      pitch: clamp(Math.sin(p * 0.5 + 1.1) * 9 + wobble(1.2), -24, 24),
      yaw: heading
    },
    pid: [
      ...prev.pid.slice(-(SAMPLES - 1)),
      {
        t,
        roll: Math.sin(p) * 6 + wobble(1.4),
        pitch: Math.sin(p * 0.8 + 1.2) * 4.5 + wobble(1.2),
        yaw: Math.sin(p * 0.45 + 2.4) * 3 + wobble(0.8)
      }
    ],
    motors: [
      ...prev.motors.slice(-(SAMPLES - 1)),
      {
        t,
        m1: 58 + Math.sin(p * 0.9) * 5 + wobble(1.8),
        m2: 61 + Math.sin(p * 0.9 + 1.6) * 5 + wobble(1.8),
        m3: 56 + Math.sin(p * 0.9 + 3.1) * 5 + wobble(1.8),
        m4: 63 + Math.sin(p * 0.9 + 4.7) * 5 + wobble(1.8)
      }
    ],
    gyro: [
      ...prev.gyro.slice(-(SAMPLES - 1)),
      {
        t,
        x: Math.sin(p * 1.4) * 22 + wobble(7),
        y: Math.sin(p * 1.1 + 0.9) * 17 + wobble(7),
        z: Math.sin(p * 0.7 + 2.2) * 11 + wobble(5)
      }
    ]
  };
}

function fromLive(prev: Telemetry, raw: RawTelemetry): Telemetry {
  const t = prev.clock + 1;
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
  const ramPercent = num(sbc.ram_percent, prev.sbc.memTotal > 0 ? (prev.sbc.memUsed / prev.sbc.memTotal) * 100 : 0);
  const memTotal = num(sbc.ram_total_mb, prev.sbc.memTotal * 1024) / 1024 || prev.sbc.memTotal;
  const memUsed = num(sbc.ram_used_mb, (ramPercent / 100) * memTotal * 1024) / 1024;

  return {
    clock: t,
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
      rssi: Math.round(num(raw.rc_rssi, prev.nav.rssi)),
      fix: raw.gps_fix_type ?? prev.nav.fix,
      groundSpeed: num(raw.ground_speed, prev.nav.groundSpeed)
    },
    altitude: {
      relative: num(raw.altitude_relative, prev.altitude.relative),
      msl: num(raw.altitude_msl, prev.altitude.msl),
      vspeed: num(raw.climb_rate, prev.altitude.vspeed),
      home: prev.altitude.home
    },
    link: {
      rssi: Math.round(num(raw.rc_rssi, prev.link.rssi)),
      rate: clamp(num(raw.radio_link_quality, raw.connected === false ? 0 : 95), 0, 100),
      loss: raw.connected === false ? 100 : 0,
      latency: prev.link.latency,
      history: [...prev.link.history.slice(-27), clamp(num(raw.radio_link_quality, raw.connected === false ? 0 : 95), 0, 100)]
    },
    flight: {
      mode: raw.flight_mode ?? prev.flight.mode,
      armed: raw.armed ?? prev.flight.armed,
      failsafe: raw.connected === false,
      mission: raw.mission_state ?? prev.flight.mission,
      wp: `${Math.round(num(raw.mission_current_seq, 0))} / ${Math.round(num(raw.mission_total_items, 0))}`,
      link: raw.connected === false ? "LOST" : "OK"
    },
    sbc: {
      cpu: clamp(num(sbc.cpu_load_percent, prev.sbc.cpu), 0, 100),
      cores: prev.sbc.cores,
      memUsed,
      memTotal,
      temp: num(sbc.cpu_temp_c, prev.sbc.temp),
      uptime: num(sbc.uptime_seconds, prev.sbc.uptime),
      diskUsed: clamp(num(sbc.disk_percent, prev.sbc.diskUsed), 0, 100),
      diskTotal: prev.sbc.diskTotal,
      load: prev.sbc.load
    },
    attitude: {
      roll: num(raw.attitude_roll, prev.attitude.roll),
      pitch: num(raw.attitude_pitch, prev.attitude.pitch),
      yaw: num(raw.attitude_yaw, heading)
    },
    pid: [
      ...prev.pid.slice(-(SAMPLES - 1)),
      {
        t,
        roll: num(raw.error_roll, 0),
        pitch: num(raw.error_pitch, 0),
        yaw: num(raw.error_yaw, 0)
      }
    ],
    motors: [
      ...prev.motors.slice(-(SAMPLES - 1)),
      {
        t,
        m1: num(motors[0], lastMotors.m1),
        m2: num(motors[1], lastMotors.m2),
        m3: num(motors[2], lastMotors.m3),
        m4: num(motors[3], lastMotors.m4)
      }
    ],
    gyro: [
      ...prev.gyro.slice(-(SAMPLES - 1)),
      {
        t,
        x: num(raw.gyro_x, 0),
        y: num(raw.gyro_y, 0),
        z: num(raw.gyro_z, 0)
      }
    ]
  };
}

export function useTelemetry(live: boolean): Telemetry {
  const [state, setState] = useState<Telemetry>(initial);

  useEffect(() => {
    if (!live) return;
    const endpoint = getLiveEndpoint();
    let failedPolls = 0;

    const poll = async () => {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 180);
      try {
        const response = await fetch(endpoint, { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`telemetry http ${response.status}`);
        const raw = (await response.json()) as RawTelemetry;
        failedPolls = 0;
        setState((prev) => fromLive(prev, raw));
      } catch {
        failedPolls += 1;
        if (failedPolls > 3) {
          setState(step);
        }
      } finally {
        window.clearTimeout(timeout);
      }
    };

    void poll();
    const id = window.setInterval(() => void poll(), TICK_MS);
    return () => window.clearInterval(id);
  }, [live]);

  return state;
}
