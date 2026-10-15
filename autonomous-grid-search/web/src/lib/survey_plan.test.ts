import { describe, expect, it } from "vitest";
import { missionPlannerFile, parseBoundary, rectangle, squarePreview, swarmPositions } from "./survey_plan";

describe("survey geometry input and Mission Planner export", () => {
  it("previews a square using the centre-to-edge distance", () => {
    expect(squarePreview(null, 25)).toEqual([]);
    expect(squarePreview({ latitude: 0, longitude: 0 }, NaN)).toEqual([]);
    const square = squarePreview({ latitude: 0, longitude: 0 }, 1000);
    expect(square).toHaveLength(4);
    expect(square[0].latitude).toBeCloseTo(-0.008993, 5);
    expect(square[0].longitude).toBeCloseTo(-0.008993, 5);
    expect(square[2].latitude).toBeCloseTo(0.008993, 5);
    expect(square[2].longitude).toBeCloseTo(0.008993, 5);
  });
  it("interpolates the Pi swarm timeline and stops at its final position", () => {
    const swarm = { simulation_only: true, warning: "preview", drone_count: 1, duration_s: 10,
      drones: [{ drone_id: 1, cell_ids: [1], sortie_ids: [1], duration_s: 10, segments: [{ start_s: 0, end_s: 10, start: { latitude: 12, longitude: 77 }, end: { latitude: 12.001, longitude: 77.001 }, phase: "survey", cell_id: 1, sortie_id: 1 }] }] };
    expect(swarmPositions(swarm, 5)[0].latitude).toBeCloseTo(12.0005, 7);
    expect(swarmPositions(swarm, 20)[0]).toMatchObject({ latitude: 12.001, longitude: 77.001, phase: "complete" });
    expect(swarmPositions(undefined, 5)).toEqual([]);
  });
  it("converts integer global relative altitude into WPL degree coordinates", () => {
    const text = missionPlannerFile([{ seq: 1, current: 0, frame: 6, command: 16, param1: 1, param2: 0.1, param3: 0, param4: 90, x: 129715980, y: 775945620, z: 5, autocontinue: 1 },
      { seq: 2, current: 0, frame: 2, command: 206, param1: 1.443, param2: 0, param3: 1, param4: 0, x: 0, y: 0, z: 0, autocontinue: 1 }]);
    const rows = text.trim().split("\n");
    expect(rows[0]).toBe("QGC WPL 110");
    expect(rows[1].split("\t")).toEqual(["1", "0", "3", "16", "1", "0.1", "0", "90", "12.9715980", "77.5945620", "5", "1"]);
    expect(rows[2].split("\t")[3]).toBe("206");
    expect(() => missionPlannerFile([])).toThrow();
  });
  it("rejects invalid coordinate text and creates four rectangle corners", () => {
    expect(() => parseBoundary("12,77\n13,78\nNaN,79")).toThrow();
    expect(() => parseBoundary("12,77\n13,78")).toThrow();
    const corners = rectangle({ latitude: 12, longitude: 77 }, { latitude: 13, longitude: 78 });
    expect(corners).toHaveLength(4);
    expect(parseBoundary(corners.map(p => `${p.latitude},${p.longitude}`).join("\n"))).toEqual(corners);
  });
});
