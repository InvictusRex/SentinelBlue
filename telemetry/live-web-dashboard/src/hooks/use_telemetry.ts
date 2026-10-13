import { useEffect, useState } from "react";
import {
  createOfflineTelemetry,
  OFFLINE_AFTER_FAILURES,
  parseRawTelemetry,
  POLL_INTERVAL_MS,
  reduceDegradedTelemetry,
  reduceLiveTelemetry,
  reduceOfflineTelemetry
} from "../lib/telemetry_mapper";
import type { Telemetry } from "../types/telemetry";

const REQUEST_TIMEOUT_MS = 2500;

function errorText(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") return "request timeout";
  if (error instanceof Error) return error.message;
  return "request failed";
}

export function useTelemetry(live: boolean, endpoint: string, cellCount: number): Telemetry {
  const [state, setState] = useState<Telemetry>(() => createOfflineTelemetry(endpoint, undefined, cellCount));

  useEffect(() => {
    setState(createOfflineTelemetry(endpoint, endpoint ? "Waiting for telemetry JSON" : "Enter ESP endpoint to start", cellCount));
  }, [cellCount, endpoint]);

  useEffect(() => {
    if (!live || endpoint.length === 0) return;
    let stopped = false;
    let failures = 0;
    let timer: number | undefined;
    let activeController: AbortController | null = null;

    const poll = async () => {
      const controller = new AbortController();
      activeController = controller;
      const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const startedAt = performance.now();

      try {
        const response = await fetch(endpoint, { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const raw = parseRawTelemetry(await response.json());
        if (stopped) return;
        const latencyMs = Math.max(0, Math.round(performance.now() - startedAt));
        failures = 0;
        setState((previous) => reduceLiveTelemetry(previous, raw, endpoint, latencyMs, cellCount));
      } catch (error) {
        if (stopped) return;
        failures += 1;
        const detail = errorText(error);
        setState((previous) => failures >= OFFLINE_AFTER_FAILURES
          ? reduceOfflineTelemetry(previous, endpoint, detail)
          : reduceDegradedTelemetry(previous, endpoint, detail, failures));
      } finally {
        window.clearTimeout(timeout);
        if (activeController === controller) activeController = null;
        if (!stopped) timer = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
      }
    };

    void poll();
    return () => {
      stopped = true;
      activeController?.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [cellCount, endpoint, live]);

  return state;
}
