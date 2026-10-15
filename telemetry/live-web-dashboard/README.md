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

## Auto-grid survey planner

Start the companion API on the Pi (or locally without hardware):

```bash
cd telemetry/handheld-nrf-monitor/pi-code
python -m pip install -r requirements.txt
python main.py --simulate --no-radio
```

Open the Docker dashboard at `http://localhost:8080/?view=mission`, or the Vite URL with `?view=mission`. Enter the Pi address and port 8765 in **Companion API**. Docker serves the website; the Python companion runs separately and must be reachable from the browser. Use a trusted network; the planning/simulation API has no authentication.

1. Click the map to select the **search centre**. Coordinate entry is available under **Connection & coordinate entry** for keyboard use.
2. Enter the **search radius in metres** (1,000 m = 1 km), then select **Generate survey on Pi**.
3. The browser sends only `search_area: {center: {latitude, longitude}, radius_m}`. The Pi reads its local `survey_config.json`, gets launch/home from fresh 3D vehicle GPS, creates a geodesic circular boundary and computes the grid, spacing, direction, camera triggering and battery sorties. The vehicle must be disarmed. Without usable GPS, the Pi can preview coverage but blocks export and simulation.
4. Review the returned route. Select a sortie and **Export to Mission Planner** to download `sentinelblue_survey_N.waypoints` (QGC WPL 110). In Mission Planner use **PLAN → Load WP File**. Export each battery sortie separately. Changes to the point or radius invalidate the old plan.
5. **Simulate this sortie** executes the exact cached plan after acknowledgement. Plans expire after 10 minutes or API restart. The API rejects replacement while the simulator is armed and rejects all live execution. Companion assist mode remains a readiness monitor.

### Configure the aircraft once on the Pi

Edit `telemetry/handheld-nrf-monitor/pi-code/survey_config.json` on the companion computer. Defaults are 5 m height, 2 m/s, 78° × 60° FOV, 70% side overlap, 75% forward overlap, 20 minute endurance and 20% reserve. These are starting values, not measured specifications for your hardware. Calibrate the camera, endurance and trigger interval for the actual aircraft before flight validation. The file is loaded for each planning request; the browser cannot override it through a point/radius request. Existing reviewed plans retain their original settings.

The Pi calculates spacing from its camera model, chooses an estimated shortest direction from 15° bearing candidates plus the longest boundary edge, and splits complete lines into sorties capped at 500 commands and the configured battery budget. Every sortie contains takeoff, speed, survey/camera commands, return and land. A 64-sided polygon circumscribes the requested circle so the approximation does not exclude the circular rim. Very large searches may exceed the 10,000 line / 100 sortie planning limits and must be divided into smaller search areas.

The geometry engine also retains its polygon planning API for integration; the operator-facing screen uses only point and radius.

### Camera and flight assumptions

Footprint = `2 × height × tan(FOV / 2)`. Maximum line gap = cross-track footprint × `(1 − side overlap)`. Trigger distance = along-track footprint × `(1 − forward overlap)`. Spacing tightens to fit the polygon extent. Crosshatch passes can have slightly different actual gaps; the displayed gap is the first pass. Modelled coverage uses flat, nadir, aligned footprints with distance-triggered photos. At least 99.5% modelled coverage is required to export; this is not measured image coverage.

A camera-rate check blocks intervals shorter than the entered camera minimum. Export includes `DO_CHANGE_SPEED` and `DO_SET_CAM_TRIGG_DIST`; configure and test the actual camera interface and yaw behaviour in ArduPilot before flight. The simulator models route, commanded speed and waypoint dwell; it does not simulate actual photos, wind, vehicle dynamics or camera latency.

For ArduCopter, configure face-next-waypoint yaw and a waypoint radius smaller than the line gap, and validate the route in ArduPilot SITL. Per-waypoint yaw and acceptance-radius fields do not configure these behaviours in Copter; see the [official mission command reference](https://docs.ardupilot.org/copter/docs/mission-command-list.html). The built-in simulator is not ArduPilot SITL.

Each line has a one-second dwell at each end. Estimates add the entered turn allowance per line and two minutes for takeoff/landing. They do not measure battery capacity, wind or power consumption. A line that cannot fit one sortie is blocked rather than truncated.

**Altitude is relative to home, not terrain-following.** Coverage assumes a flat surface at home elevation. A 5 m mission over water requires a validated downward sensor and flight-controller configuration; this planner does not configure them. DEM data is not water clearance. Run-outs and straight transfers may leave the boundary; the boundary is not a geofence. The planner does not provide obstacle avoidance, airspace checks or industrial flight certification.

### Maps and terrain

The default map uses OpenStreetMap raster tiles and MapLibre demo hillshade. Set `VITE_MAPTILER_KEY` at build time for MapTiler hybrid imagery. The map stays top-down for boundary accuracy.

Optional terrain context samples home, centre and boundary corners using OpenTopoData SRTM90m. Set `TERRAIN_API_URL` to another compatible endpoint. Missing tiles or terrain data do not change mission altitude. Coordinates remain editable when imagery is unavailable.

### Verification

Run `python -m unittest discover -s tests -v` from the Pi code directory and `npm run check` from the dashboard directory. Tests cover camera axes, sensor dimensions, coverage, invalid polygons, rotation, reverse entry, concave boundaries, crosshatch, camera rate, sortie limits, exact-plan execution, expiry, confirmation, live/armed guards, WPL coordinate frames and boundary parsing.
