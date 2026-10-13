import { describe, expect, it } from "vitest";
import { telemetryEndpointFromInput } from "./telemetry_endpoint";

describe("telemetry endpoint normalization", () => {
  it("adds the local JSON path to an ESP address", () => {
    expect(telemetryEndpointFromInput("192.168.4.1", "http:")).toBe("http://192.168.4.1/telemetry.json");
  });

  it("preserves a complete secure API path", () => {
    expect(telemetryEndpointFromInput("https://telemetry.example/api/live?token=ignored", "https:"))
      .toBe("https://telemetry.example/api/live");
  });

  it("rejects insecure remote telemetry from an HTTPS dashboard", () => {
    expect(() => telemetryEndpointFromInput("http://192.168.4.1/telemetry.json", "https:"))
      .toThrow("Use an HTTPS telemetry URL");
  });
});
