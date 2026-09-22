# Telemetry

Telemetry contains the drone companion-computer code, handheld ground receiver firmware, and browser dashboard assets used by the SentinelBlue telemetry workflow.

## Directory Layout

- `handheld-nrf-monitor/` - Raspberry Pi and ESP32 code for collecting Pixhawk telemetry, sending the compact NRF24L01+ stream to the ESP32 receiver, and preserving the optional Pi mission-control API.
- `live-web-dashboard/` - Vite React dashboard source. Run it on a laptop/static host and connect it to the ESP32 `/telemetry.json` endpoint shown on the OLED.

## File Guide

### `handheld-nrf-monitor/esp-code/esp.ino`
ESP32 receiver firmware for the handheld unit. It listens for NRF24L01+ telemetry packets, validates frame checksums, decodes battery/GPS/altitude/attitude/SBC values, renders them on the SH1106 OLED, connects to Wi-Fi, and exposes the latest packet as JSON at `/telemetry.json`.

### `handheld-nrf-monitor/pi-code/main.py`
Entry point for the Raspberry Pi telemetry relay. It starts the MAVLink manager, SBC monitor, NRF24L01+ transmitter, and optional mission API/static server. The laptop dashboard reads the ESP32 Wi-Fi JSON endpoint; `--no-web` disables the optional Pi HTTP/API server.

### `handheld-nrf-monitor/pi-code/mavlink_manager.py`
Manages the Pixhawk MAVLink connection or simulator, keeps the shared telemetry state, sends vehicle commands, uploads missions, and tracks mission progress.

### `handheld-nrf-monitor/pi-code/radio_tx_module.py`
Packs selected telemetry into typed NRF24L01+ frames and broadcasts them to the ESP32 handheld receiver.

### `handheld-nrf-monitor/pi-code/sbc_monitor.py`
Reads Raspberry Pi CPU, memory, disk, thermal, and uptime metrics for dashboard telemetry.

### `handheld-nrf-monitor/pi-code/modules/grid_search_module.py`
Builds autonomous search missions, including grid, spiral, and sector patterns, and converts planned routes into MAVLink mission items.

### `handheld-nrf-monitor/pi-code/modules/web_dashboard_module.py`
Preserves the optional Pi HTTP/WebSocket server and `/api/mission/*` endpoints for existing mission-control workflows. When enabled and `live-web-dashboard/dist/` exists, it can also serve the built Vite dashboard assets.

### `handheld-nrf-monitor/pi-code/modules/__init__.py`
Marks the `modules` directory as a Python package.

### `live-web-dashboard/index.html`
Vite entrypoint used during development and build. Production/static deployments should serve the generated `dist/index.html`.

### `live-web-dashboard/src/`
React and TypeScript dashboard source for telemetry panels, charts, attitude view, and ESP JSON polling.

### `live-web-dashboard/package.json`
Defines the Vite development, build, and preview commands for the dashboard.

### `live-web-dashboard/dist/`
Generated production dashboard output from `npm run build`. This directory is ignored by Git and should be served from a laptop/static host, or by the optional Pi HTTP server when that legacy deployment path is needed.

## Running the Telemetry Relay

On the Raspberry Pi:

```bash
cd telemetry/handheld-nrf-monitor/pi-code
python3 main.py --port /dev/serial0 --baud 115200 --radio-rate 5
```

Add `--no-web` if only the MAVLink-to-NRF relay is required and the optional mission API/static server is not needed.

## Running the Web Dashboard

From `telemetry/live-web-dashboard` on the laptop or static-host machine:

```bash
npm ci
npm run dev -- --host 0.0.0.0
```

Open the dashboard URL and enter the ESP32 IP shown on the OLED. The dashboard reads:

```text
http://<esp-ip>/telemetry.json
```

For a production/static build:

```bash
npm run build
npm run preview -- --host 0.0.0.0
```

The optional Pi HTTP server can also serve `live-web-dashboard/dist/` after `npm run build`, but the telemetry data still comes from the ESP32 JSON endpoint entered in the dashboard.

## ESP32 Wi-Fi Credentials

Do not commit Wi-Fi secrets into `esp.ino`. Provide them as build-time defines or through your local Arduino/PlatformIO environment:

```bash
-DDRONE_WIFI_STA_SSID=\"your-network\"
-DDRONE_WIFI_STA_PASSWORD=\"your-password\"
-DDRONE_WIFI_FALLBACK_AP_SSID=\"DroneTelemetryESP32-001\"
-DDRONE_WIFI_FALLBACK_AP_PASSWORD=\"per-device-strong-password\"
```

The fallback AP is disabled unless `DRONE_WIFI_FALLBACK_AP_PASSWORD` is set to a per-device password of at least 12 characters.
