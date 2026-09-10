# Telemetry

Telemetry contains the drone companion-computer code, handheld ground receiver firmware, and browser dashboard assets used by the SentinelBlue telemetry workflow.

## Directory Layout

- `handheld-nrf-monitor/` - Raspberry Pi and ESP32 code for collecting Pixhawk telemetry and sending the compact NRF24L01+ stream to the handheld receiver.
- `live-web-dashboard/` - Static web dashboard files served by the Raspberry Pi companion computer.

## File Guide

### `handheld-nrf-monitor/esp_code/esp.ino`
ESP32 receiver firmware for the handheld unit. It listens for NRF24L01+ telemetry packets, validates the packet checksum, decodes battery/GPS/altitude/RSSI values, and renders them on the SH1106 OLED.

### `handheld-nrf-monitor/pi_code/main.py`
Entry point for the Raspberry Pi companion server. It starts the MAVLink manager, SBC monitor, web dashboard server, NRF24L01+ transmitter, and autonomous grid-search module.

### `handheld-nrf-monitor/pi_code/modules/mavlink_manager.py`
Manages the Pixhawk MAVLink connection or simulator, keeps the shared telemetry state, sends vehicle commands, uploads missions, and tracks mission progress.

### `handheld-nrf-monitor/pi_code/modules/radio_tx_module.py`
Packs selected telemetry into the 20-byte NRF24L01+ payload and broadcasts it to the ESP32 handheld receiver.

### `handheld-nrf-monitor/pi_code/modules/web_dashboard_module.py`
Runs the HTTP/WebSocket server, serves the dashboard from `live-web-dashboard/`, exposes telemetry APIs, and forwards mission-control requests to the grid-search module.

### `handheld-nrf-monitor/pi_code/modules/grid_search_module.py`
Builds autonomous search missions, including grid, spiral, and sector patterns, and converts planned routes into MAVLink mission items.

### `handheld-nrf-monitor/pi_code/modules/sbc_monitor.py`
Reads Raspberry Pi CPU, memory, disk, thermal, and uptime metrics for dashboard telemetry.

### `handheld-nrf-monitor/pi_code/modules/__init__.py`
Marks the `modules` directory as a Python package.

### `live-web-dashboard/index.html`
Dashboard markup for live telemetry, map tracking, mission planning, battery status, charts, motor outputs, and SBC diagnostics.

### `live-web-dashboard/style.css`
Dashboard styling for the mission-control layout, map panel, cards, modal controls, charts, and telemetry status states.

### `live-web-dashboard/app.js`
Dashboard client logic for WebSocket telemetry updates, Leaflet map rendering, chart updates, route previews, and mission-control API calls.
