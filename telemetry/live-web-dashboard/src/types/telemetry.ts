export interface CellReading {
  label: string;
  volts: number;
}

export interface BatteryState {
  total: number;
  percent: number;
  current: number;
  consumed: number;
  cells: CellReading[];
}

export interface NavState {
  lat: number;
  lon: number;
  heading: number;
  sats: number;
  hdop: number;
  rssi: number;
  fix: string;
  groundSpeed: number;
}

export interface AltitudeState {
  relative: number;
  msl: number;
  vspeed: number;
  home: number;
}

export interface LinkState {
  rssi: number;
  rate: number;
  loss: number;
  latency: number;
  history: number[];
}

export type LinkQuality = "OK" | "DEGRADED" | "LOST";

export interface FlightState {
  mode: string;
  armed: boolean;
  failsafe: boolean;
  mission: string;
  wp: string;
  link: LinkQuality;
}

export interface SbcState {
  cpu: number;
  cores: number[];
  memPercent: number;
  temp: number;
  uptime: number;
  diskPercent: number;
  load: [number, number, number];
}

export interface AttitudeState {
  roll: number;
  pitch: number;
  yaw: number;
}

export interface PidPoint {
  t: number;
  roll: number;
  pitch: number;
  yaw: number;
}

export interface MotorPoint {
  t: number;
  m1: number;
  m2: number;
  m3: number;
  m4: number;
}

export interface GyroPoint {
  t: number;
  x: number;
  y: number;
  z: number;
}

export type TelemetrySourceStatus = "online" | "stale" | "offline";

export interface TelemetrySource {
  status: TelemetrySourceStatus;
  endpoint: string;
  message: string;
  lastUpdated: number;
}

export interface Telemetry {
  clock: number;
  source: TelemetrySource;
  battery: BatteryState;
  nav: NavState;
  altitude: AltitudeState;
  link: LinkState;
  flight: FlightState;
  sbc: SbcState;
  attitude: AttitudeState;
  pid: PidPoint[];
  motors: MotorPoint[];
  gyro: GyroPoint[];
}

export interface SeriesDef {
  key: string;
  label: string;
  color: string;
}
