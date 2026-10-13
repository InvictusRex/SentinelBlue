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

## Camera coverage mission planner

Run the Raspberry Pi companion in simulation mode, then open the dashboard with
`?view=mission`. The planner:

- draws terrain and map context with MapLibre;
- divides the selected square into the requested number of search cells;
- calculates the downward camera footprint from altitude and horizontal and
  vertical field of view;
- creates parallel lawnmower lines in every cell using the requested side
  overlap;
- sets camera trigger distance from forward overlap;
- estimates route distance, duration, photographs, mission items and required
  sorties; and
- blocks simulator start when the mission exceeds the 500 item or 20 minute
  single sortie limits.

For a plan that fits the single sortie limits, select **Download Mission
Planner file**. In Mission Planner, open **PLAN**, select **Load WP File**, and
choose `sentinelblue_camera_survey.waypoints`. Review the route and vehicle
parameters before writing it to a flight controller.

The default map uses OpenStreetMap raster tiles and MapLibre demo terrain. Set
`VITE_MAPTILER_KEY` at build time to use the MapTiler hybrid map:

```bash
VITE_MAPTILER_KEY=your_key npm run build
```

The companion samples OpenTopoData SRTM90m by default. Set
`TERRAIN_API_URL` to another compatible OpenTopoData endpoint when running a
private terrain service.

Terrain elevation is planning context. A 5 metre flight over water requires a
tested downward rangefinder to measure surface clearance. The UI requires the
operator to acknowledge that sensor before simulation. Live Pixhawk mission
execution remains locked.
