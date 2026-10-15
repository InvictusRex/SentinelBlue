# Autonomous Grid Search

The Pi companion API plans square-cell camera surveys, and the standalone web app previews routes, battery sorties, and simulation. Live Pixhawk mission execution remains locked.

## Auto-grid survey planner

Start the companion API on the Pi, or run the simulator locally from the repository root:

```bash
cd telemetry/handheld-nrf-monitor/pi-code
python -m pip install -r ../../../autonomous-grid-search/backend/requirements.txt
python main.py --simulate --no-radio
```

In a second terminal, start the standalone webpage from the repository root:

```bash
cd autonomous-grid-search/web
npm ci
npm run dev
```

Open `http://127.0.0.1:5174/` and enter the Pi address and port 8765 in **Companion API**. The Python companion runs separately and must be reachable from the browser. Use a trusted network; the planning/simulation API has no authentication.

1. Click the map to select the **search centre**. Coordinate entry is available under **Connection & coordinate entry** for keyboard use.
2. Enter **centre to edge in metres**, then select **Generate survey on Pi**. The search area is a square: 50 m centre-to-edge means a 100 × 100 m square.
3. The browser sends only `search_area: {center: {latitude, longitude}, radius_m}`; the existing API field `radius_m` now represents square half-side. The Pi reads `survey_config.json`, gets launch/home from fresh 3D vehicle GPS, divides a square in a local metric projection into equal square cells, and computes a complete camera grid in every cell. The vehicle must be disarmed. Without usable GPS, the Pi previews coverage but blocks export and simulation.
4. Review the returned route. Select a sortie and **Export to Mission Planner** to download `sentinelblue_survey_N.waypoints` (QGC WPL 110). In Mission Planner use **PLAN → Load WP File**. Export each battery sortie separately. Changes to the point or radius invalidate the old plan.
5. **Play swarm preview** animates the Pi-generated assignments and timelines for virtual drones. Pause, reset or change playback speed; the display tracks completed cells and each drone's phase. This visual preview sends no vehicle commands. **Run single-drone simulator** executes the selected cached sortie after acknowledgement. Plans expire after 10 minutes or API restart. The API rejects replacement while the simulator is armed and rejects all live execution.

### Configure the aircraft once on the Pi

Edit `autonomous-grid-search/backend/survey_config.json` on the companion computer. Defaults are 5 m height, 2 m/s, 78° × 60° FOV, 70% side overlap, 75% forward overlap, 20 minute endurance and 20% reserve. These are starting values, not measured specifications for your hardware. Calibrate the camera, endurance and trigger interval for the actual aircraft before flight validation. The file is loaded for each planning request; the browser cannot override it through a point/radius request. Existing reviewed plans retain their original settings.

`max_cell_side_m` defaults to **50 m**, `swarm_drone_count` to **4**, `square_grid_angle_deg` to **0°** (north/south tracks), and `battery_swap_seconds` to **120 s** between a virtual drone's sorties. These values are configured on the Pi. Cell count per axis is `ceil(square_side / max_cell_side_m)`; the entire area is tiled by equal cells no larger than the maximum. For example, a 120 m square becomes nine 40 × 40 m cells, with no uncovered remainder strip.

The Pi calculates camera spacing and produces alternating parallel tracks within each cell. It assigns whole cells to the least-loaded virtual drone by accumulated estimated flight time. Every cell has one owner; its complete lines are split into battery sorties capped at 500 commands and the configured battery budget. Every sortie contains takeoff, speed, survey/camera commands, return and land. Planning is capped at 100 cells, 100 sorties and 10,000 lines; oversized requests return an error instead of dropping coverage.

**Swarm preview is a planning visualization, not live swarm control.** Timelines include takeoff/landing allowances, survey, transfers, turns and battery changes. Virtual drones share the same launch/home position. Shared launch and transit paths are not collision-deconflicted. Real simultaneous multi-drone flight remains disabled; this does not implement multi-vehicle MAVLink connections or formation control.

The geometry engine retains its polygon API for integration; the operator-facing screen uses only a centre and square extent. Exports remain separate by cell/drone/sortie, with the selected assignment shown in the sortie picker.

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

Run `python -m unittest discover -s tests -v` from `autonomous-grid-search/backend` and `npm run check` from `autonomous-grid-search/web`. Tests cover camera axes, sensor dimensions, coverage, invalid polygons, rotation, reverse entry, concave boundaries, crosshatch, camera rate, sortie limits, exact-plan execution, expiry, confirmation, live/armed guards, WPL coordinate frames and boundary parsing.
