import { describe, expect, it } from "vitest";
import {
  createOfflineTelemetry,
  parseRawTelemetry,
  reduceLiveTelemetry,
  reduceOfflineTelemetry
} from "./telemetry_mapper";

describe("telemetry payload validation", () => {
  it("accepts a valid partial telemetry payload", () => {
    expect(parseRawTelemetry({ connected: true, battery_voltage: 15.8 })).toEqual({
      connected: true,
      battery_voltage: 15.8
    });
  });

  it("rejects malformed and empty payloads", () => {
    expect(() => parseRawTelemetry({ battery_voltage: "15.8" })).toThrow("invalid battery_voltage");
    expect(() => parseRawTelemetry({ unknown: 1 })).toThrow("no recognized fields");
  });
});

describe("telemetry safety state", () => {
  it("marks armed and failsafe state unknown when telemetry is offline", () => {
    const previous = reduceLiveTelemetry(
      createOfflineTelemetry("http://esp/telemetry.json"),
      { connected: true, armed: true, attitude_roll: 12 },
      "http://esp/telemetry.json",
      10,
      4
    );
    const offline = reduceOfflineTelemetry(previous, "http://esp/telemetry.json", "request timeout");

    expect(offline.flight.armed).toBeNull();
    expect(offline.flight.failsafe).toBeNull();
    expect(offline.attitude.roll).toBe(12);
  });

  it("uses measured RSSI when link quality is absent", () => {
    const telemetry = reduceLiveTelemetry(
      createOfflineTelemetry(),
      { connected: true, rc_rssi: 42 },
      "http://esp/telemetry.json",
      8,
      4
    );

    expect(telemetry.link.rate).toBe(42);
    expect(telemetry.link.loss).toBe(58);
  });

  it("keeps unavailable consumption and CPU-core data explicit", () => {
    const telemetry = reduceLiveTelemetry(
      createOfflineTelemetry(),
      { connected: true, sbc: { cpu_load_percent: 35 } },
      "http://esp/telemetry.json",
      8,
      4
    );

    expect(telemetry.battery.consumed).toBeNull();
    expect(telemetry.sbc.cores).toEqual([]);
  });
});
