# Telemetry

Telemetry contains the drone companion-computer code, handheld ground receiver firmware, and browser dashboard assets used by the SentinelBlue telemetry workflow.

## Directory Layout

- `handheld-nrf-monitor/` - Raspberry Pi and ESP32 code for collecting Pixhawk telemetry and sending the compact NRF24L01+ stream to the handheld receiver.
- `live-web-dashboard/` - Vite React dashboard source. Build it before serving from the Raspberry Pi, or run it with Docker/Vite on a laptop.

## File Guide

### `handheld-nrf-monitor/esp_code/esp.ino`
ESP32 receiver firmware for the handheld unit. It listens for NRF24L01+ telemetry packets, validates the packet checksum, decodes battery/GPS/altitude/RSSI values, and renders them on the SH1106 OLED.

### `handheld-nrf-monitor/pi-code/main.py`
Entry point for the Raspberry Pi companion server. It starts the MAVLink manager, SBC monitor, web dashboard server, NRF24L01+ transmitter, and autonomous grid-search module.

### `handheld-nrf-monitor/pi-code/modules/mavlink_manager.py`
Manages the Pixhawk MAVLink connection or simulator, keeps the shared telemetry state, sends vehicle commands, uploads missions, and tracks mission progress.

### `handheld-nrf-monitor/pi-code/modules/radio_tx_module.py`
Packs selected telemetry into the 20-byte NRF24L01+ payload and broadcasts it to the ESP32 handheld receiver.

### `handheld-nrf-monitor/pi-code/modules/web_dashboard_module.py`
Runs the HTTP/WebSocket server, serves the built dashboard from `live-web-dashboard/dist/`, exposes telemetry APIs, and forwards mission-control requests to the grid-search module. If `dist/` is missing, it returns a build-required error instead of serving Vite source files.

### `handheld-nrf-monitor/pi-code/modules/grid_search_module.py`
Builds autonomous search missions, including grid, spiral, and sector patterns, and converts planned routes into MAVLink mission items.

### `handheld-nrf-monitor/pi-code/modules/sbc_monitor.py`
Reads Raspberry Pi CPU, memory, disk, thermal, and uptime metrics for dashboard telemetry.

### `handheld-nrf-monitor/pi-code/modules/__init__.py`
Marks the `modules` directory as a Python package.

### `live-web-dashboard/index.html`
Vite entrypoint used during development and build. The Pi server should not serve this source file directly in production.

### `live-web-dashboard/src/`
React and TypeScript dashboard source for telemetry panels, charts, attitude view, and ESP JSON polling.

### `live-web-dashboard/package.json`
Defines the Vite development, build, and preview commands for the dashboard.

### `live-web-dashboard/dist/`
Generated production dashboard output from `npm run build`. This directory is ignored by Git and must be created on the deployment target before the Pi web server can host the dashboard.

## Building the Web Dashboard for Pi Hosting

From `telemetry/live-web-dashboard`:

```bash
npm ci
npm run build
```

Then start the Raspberry Pi companion server. It will serve `telemetry/live-web-dashboard/dist/index.html` and the generated `dist/assets/*` files. For local development, run Vite instead:

```bash
npm run dev -- --host 0.0.0.0
```
