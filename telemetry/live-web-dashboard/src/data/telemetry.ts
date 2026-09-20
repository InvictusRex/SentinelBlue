import type { SeriesDef } from "../types/telemetry";

export const vehicle = {
  name: "HX-4 / QUAD-X",
  sysId: "SYS 1 · COMP 1",
  firmware: "ArduCopter 4.5.7",
  linkPath: "MAVLINK · NRF24 · ESP32-WIFI"
};

export const cellLabels = ["1S", "2S", "3S", "4S"];

export const pidSeries: SeriesDef[] = [
  { key: "roll", label: "ROLL", color: "#4c8dff" },
  { key: "pitch", label: "PITCH", color: "#2dd4bf" },
  { key: "yaw", label: "YAW", color: "#f5a524" }
];

export const motorSeries: SeriesDef[] = [
  { key: "m1", label: "M1", color: "#2dd4bf" },
  { key: "m2", label: "M2", color: "#4c8dff" },
  { key: "m3", label: "M3", color: "#9b8cff" },
  { key: "m4", label: "M4", color: "#f472b6" }
];

export const gyroSeries: SeriesDef[] = [
  { key: "x", label: "GX", color: "#3fcf8e" },
  { key: "y", label: "GY", color: "#f5a524" },
  { key: "z", label: "GZ", color: "#f2555a" }
];
