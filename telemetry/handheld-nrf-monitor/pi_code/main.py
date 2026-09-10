#!/usr/bin/env python3
"""
======================================================================================
Project: Drone Telemetry Master Companion Server
File: pi/main.py

Description:
  Master multi-module supervisor running directly on the Raspberry Pi 4B.
  Coordinates all 3 core operational modules concurrently over a single MAVLink port:
  - Module 1: OLED / NRF24L01+ 20-byte compact radio transmitter (for ESP32 Ground Unit)
  - Module 2: Live Mission Control Web Dashboard & WebSocket server (port 8000)
  - Module 3: Automated Boustrophedon Grid Search autonomy controller
======================================================================================
"""

import os
import sys
import time
import signal
import argparse

# Add pi directory to module path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from modules.sbc_monitor import SBCMonitor
from modules.mavlink_manager import MAVLinkManager
from modules.radio_tx_module import RadioTXModule
from modules.web_dashboard_module import WebDashboardModule
from modules.grid_search_module import GridSearchModule

def main():
    parser = argparse.ArgumentParser(description="Drone Telemetry Master Companion Server")
    parser.add_argument("--port", default="/dev/serial0", help="Pixhawk serial port (default: /dev/serial0 or /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--web-port", type=int, default=8000, help="Web Dashboard HTTP Port (default: 8000)")
    parser.add_argument("--radio-rate", type=float, default=5.0, help="NRF24 broadcast rate in Hz (default: 5.0)")
    parser.add_argument("--no-radio", action="store_true", help="Disable NRF24 radio module")
    parser.add_argument("--simulate", action="store_true", help="Run simulated flight telemetry without hardware")
    args = parser.parse_args()

    print("=" * 76)
    print("        DRONE TELEMETRY MASTER COMPANION SERVER (RASPBERRY PI 4B)       ")
    print("=" * 76)
    print(f"[*] Flight Controller Port:  {args.port} @ {args.baud} baud")
    print(f"[*] Web Dashboard Port:      http://0.0.0.0:{args.web_port}")
    print(f"[*] Radio TX Broadcast Rate: {args.radio_rate} Hz ({'DISABLED' if args.no_radio else 'ENABLED'})")
    print(f"[*] Operation Mode:          {'SIMULATION' if args.simulate else 'LIVE PIXHAWK'}")
    print("----------------------------------------------------------------------------")

    # 1. SBC Monitor
    sbc_monitor = SBCMonitor()
    print("[+] Initialized SBC Hardware & Thermal Monitor.")

    # 2. Shared MAVLink Manager
    mav_manager = MAVLinkManager(port=args.port, baud=args.baud, simulate=args.simulate)
    mav_manager.start()

    # 3. Module 3: Grid Search Autonomy Engine
    grid_module = GridSearchModule(mav_manager)
    print("[✓] Module 3 (Grid Search Autonomy): Ready.")

    # 4. Module 2: Mission Control Web Dashboard & WebSocket Server
    dashboard_module = WebDashboardModule(
        mav_manager=mav_manager,
        sbc_monitor=sbc_monitor,
        grid_module=grid_module,
        port=args.web_port
    )
    dashboard_module.start()

    # 5. Module 1: NRF24 Radio Transmitter
    radio_module = RadioTXModule(
        mav_manager=mav_manager,
        rate_hz=args.radio_rate,
        enabled=not args.no_radio
    )
    radio_module.start()

    print("\n" + "=" * 76)
    print(f"  🚀 ALL 3 MODULES RUNNING CONCURRENTLY ON RASPBERRY PI!")
    print(f"  • Module 1 (Radio TX):        NRF24L01+ broadcast -> ESP32 Handheld OLED")
    print(f"  • Module 2 (Web Dashboard):   http://0.0.0.0:{args.web_port}")
    print(f"  • Module 3 (Grid Autonomy):   Boustrophedon generator & MAVLink controller")
    print("=" * 76 + "\n")
    print("[*] Press Ctrl+C to stop all services.\n")

    # Graceful exit handler
    def shutdown(signum, frame):
        print("\n[*] Stopping companion modules...")
        radio_module.stop()
        dashboard_module.stop()
        mav_manager.stop()
        print("[✓] All modules stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Supervisor keep-alive & live console status ticker (every 2s)
    last_print = 0
    while True:
        time.sleep(0.5)
        now = time.time()
        if now - last_print >= 2.0:
            last_print = now
            t = mav_manager.get_telemetry_snapshot()
            sbc = sbc_monitor.get_metrics_snapshot()
            conn_status = "✓ LIVE" if t.get("connected") else "! WAITING"
            v_bat = t.get('battery_voltage', 0.0)
            bat_str = f"{v_bat:5.2f}V" if v_bat > 0.5 else "USB 5V"
            print(f"[{conn_status}] Mode: {t.get('flight_mode'):<9} | "
                  f"Bat: {bat_str} | "
                  f"Roll: {t.get('attitude_roll', 0.0):+5.1f}° | "
                  f"Pitch: {t.get('attitude_pitch', 0.0):+5.1f}° | "
                  f"Hdg: {t.get('heading', 0.0):5.1f}° | "
                  f"Alt: {t.get('altitude_relative', 0.0):4.1f}m | "
                  f"Sats: {t.get('satellites', 0):2d} | "
                  f"CPU: {sbc.get('cpu_temp_c', 0.0):4.1f}°C")

if __name__ == "__main__":
    main()
