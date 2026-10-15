# Live Web Dashboard

The dashboard displays ESP32 drone telemetry through a React and Vite frontend. Docker and Nginx provide the production runtime.

## Local development

```bash
npm ci
npm run dev
```

Run all checks with:

```bash
npm run check
```

## Docker

```bash
docker compose up --build -d
```

Open `http://localhost:8080`. Nginx serves the production build, exposes `/healthz`, compresses text assets, and adds browser security headers.

## Telemetry endpoints

For local use, enter the ESP32 IP address. The dashboard appends `/telemetry.json` automatically. The browser and ESP32 must be reachable on the same network.

For a dashboard served over HTTPS, provide a complete HTTPS telemetry endpoint. Browsers block an HTTPS page from fetching the ESP32's plain HTTP address. Remote operation therefore requires this path:

```text
ESP32 or companion computer -> authenticated HTTPS or MQTT service -> HTTPS telemetry endpoint -> dashboard
```

The relay should authenticate the vehicle, encrypt traffic, restrict allowed browser origins, and return the same JSON fields as the ESP32 endpoint. Do not expose the ESP32 HTTP server directly to the internet.

## Telemetry behavior

- Payloads are validated before they update the display.
- Lost telemetry produces an unknown armed and failsafe state.
- Values that are not provided remain unavailable instead of being generated.
- The battery cell count can be selected from 2S through 6S.
- Charts display 64 samples collected at the dashboard's 1 Hz polling rate.


Autonomous grid-search planning has moved to [autonomous-grid-search](../../autonomous-grid-search/README.md).
