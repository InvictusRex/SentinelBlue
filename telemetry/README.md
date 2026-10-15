# Telemetry

Telemetry contains the drone companion-computer code, handheld ground receiver firmware, and browser dashboard assets used by the SentinelBlue telemetry workflow.

## Directory Layout

- `handheld-nrf-monitor/` - Raspberry Pi and ESP32 code for collecting Pixhawk telemetry and sending the compact NRF24L01+ stream to the ESP32 receiver.
- `live-web-dashboard/` - Vite React dashboard source. Run it on a laptop/static host and connect it to the ESP32 `/telemetry.json` endpoint shown on the OLED.

## File Guide

### `handheld-nrf-monitor/esp-code/esp.ino`
ESP32 receiver firmware for the handheld unit. It listens for NRF24L01+ telemetry packets, validates frame checksums, decodes battery/GPS/altitude/attitude/SBC values, renders them on the SH1106 OLED, connects to Wi-Fi, and exposes the latest packet as JSON at `/telemetry.json`.

### `handheld-nrf-monitor/pi-code/main.py`
Entry point for the Raspberry Pi telemetry relay. It starts the MAVLink manager, SBC monitor, and NRF24L01+ transmitter. The laptop dashboard reads the ESP32 Wi-Fi JSON endpoint.

### `handheld-nrf-monitor/pi-code/mavlink_manager.py`
Manages the Pixhawk MAVLink connection or simulator and keeps the shared telemetry state used by the NRF transmitter.

### `handheld-nrf-monitor/pi-code/companion_mode.py`
Evaluates the companion computer state from Pixhawk link freshness, battery, GPS, and Raspberry Pi temperature. It is monitor-only: every state has command output disabled, including `assist_ready` and `failsafe`.

### `handheld-nrf-monitor/pi-code/radio_tx_module.py`
Packs selected telemetry into typed NRF24L01+ frames and broadcasts them to the ESP32 handheld receiver.

### `handheld-nrf-monitor/pi-code/sbc_monitor.py`
Reads Raspberry Pi CPU, memory, disk, thermal, and uptime metrics for dashboard telemetry.

### `live-web-dashboard/index.html`
Vite entrypoint used during development and build. Production/static deployments should serve the generated `dist/index.html`.

### `live-web-dashboard/src/`
React and TypeScript dashboard source for telemetry panels, charts, attitude view, and ESP JSON polling.

### `live-web-dashboard/package.json`
Defines the Vite development, build, and preview commands for the dashboard.

### `live-web-dashboard/dist/`
Generated production dashboard output from `npm run build`. This directory is ignored by Git and should be served from a laptop/static host.

## Running the Telemetry Relay

On the Raspberry Pi:

```bash
cd telemetry/handheld-nrf-monitor/pi-code
python3 main.py --port /dev/serial0 --baud 115200 --radio-rate 5
```

The default companion mode is `monitor`. To validate whether the system is
healthy enough for a future assist feature without sending flight commands:

```bash
python3 main.py --simulate --no-radio --companion-mode assist
```

Companion states are:

- `disabled` - state evaluation is disabled by configuration.
- `monitor` - telemetry and companion health are current.
- `assist_ready` - link, GPS, battery, and computer health checks passed. Flight command output remains disabled.
- `degraded` - one or more required inputs are unavailable or outside the configured limits.
- `failsafe` - a critical battery or temperature condition exists, or telemetry became stale while the aircraft was armed. This is a reported state; Pixhawk remains responsible for flight stabilization and configured vehicle failsafes.

## Autonomous Grid Search

Survey planning, the companion API, and the mission webpage are maintained in [autonomous-grid-search](../autonomous-grid-search/README.md). The Pi relay starts the planning API unless `--no-companion-api` is set. The telemetry dashboard displays ESP32 telemetry only.

## Running the Web Dashboard

From `telemetry/live-web-dashboard` on the laptop or static-host machine:

```bash
npm ci
npm run dev -- --host 0.0.0.0
```

Open the dashboard URL, select the battery cell count, and enter the ESP32 IP shown on the OLED. The dashboard reads:

```text
http://<esp-ip>/telemetry.json
```

For a production/static build:

```bash
npm run build
npm run preview -- --host 0.0.0.0
```

The Raspberry Pi does not serve the React dashboard in this telemetry flow.

## Running the Web Dashboard with Docker

Install Docker Desktop with its Linux container backend, then run this from
`telemetry/live-web-dashboard`:

```bash
docker compose up --build -d
```

The production build is served at `http://localhost:8080` on the Docker host.
Other devices on the same network can use `http://<docker-host-ip>:8080` if the
host firewall allows inbound TCP port 8080. Stop it with `docker compose down`.

The dashboard reads `http://<esp-ip>/telemetry.json` directly from each viewer's
browser during local use. A viewer must be able to reach the ESP32 on the same
network. An HTTPS deployment requires a complete HTTPS
telemetry endpoint because browsers block insecure HTTP telemetry requests from
secure pages. Remote access requires an authenticated HTTPS or MQTT relay between
the vehicle and dashboard; publishing Docker port 8080 does not make the ESP32
endpoint reachable over the Internet.

## ESP32 Wi-Fi Credentials

Do not commit Wi-Fi secrets into `esp.ino`. Provide them as build-time defines or through your local Arduino/PlatformIO environment:

```bash
-DDRONE_WIFI_STA_SSID=\"your-network\"
-DDRONE_WIFI_STA_PASSWORD=\"your-password\"
-DDRONE_WIFI_FALLBACK_AP_SSID=\"DroneTelemetryESP32-001\"
-DDRONE_WIFI_FALLBACK_AP_PASSWORD=\"per-device-strong-password\"
```

The fallback AP is disabled unless `DRONE_WIFI_FALLBACK_AP_PASSWORD` is set to a per-device password of at least 12 characters.
