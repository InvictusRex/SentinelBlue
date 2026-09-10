#!/usr/bin/env python3
"""
======================================================================================
Module 2: Real-Time Mission Control Web Dashboard & Telemetry Hub
File: pi/modules/web_dashboard_module.py

Description:
  Serves the aerospace Mission Control Web UI directly from the Raspberry Pi:
  - Serves static assets (HTML/CSS/JS) on port 8000
  - Streams 10 Hz live JSON telemetry frames over WebSocket (/ws/telemetry)
  - Provides REST API (/api/telemetry)
  - Provides Grid Search mission trigger API (/api/mission/grid_search)
======================================================================================
"""

import os
import time
import json
import base64
import hashlib
import struct
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "web"))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def encode_ws_frame(message: str) -> bytes:
    """Encodes a text string into an unmasked RFC 6455 WebSocket frame."""
    data = message.encode('utf-8')
    length = len(data)
    frame = bytearray([0x81])

    if length <= 125:
        frame.append(length)
    elif length <= 65535:
        frame.append(126)
        frame.extend(struct.pack("!H", length))
    else:
        frame.append(127)
        frame.extend(struct.pack("!Q", length))

    frame.extend(data)
    return bytes(frame)

class WebDashboardModule:
    def __init__(self, mav_manager, sbc_monitor, grid_module=None, port: int = 8000):
        self.mav_manager = mav_manager
        self.sbc_monitor = sbc_monitor
        self.grid_module = grid_module
        self.port = port
        self.httpd = None
        self.server_thread = None
        self.ws_thread = None
        self.running = False
        self.connected_websockets = []
        self.ws_lock = threading.Lock()

    def get_full_telemetry(self) -> dict:
        t = self.mav_manager.get_telemetry_snapshot() if self.mav_manager else {}
        t["sbc"] = self.sbc_monitor.get_metrics_snapshot() if self.sbc_monitor else {}
        if self.grid_module:
            t["mission_progress"] = self.grid_module.get_status()
        return t

    def start(self):
        handler_factory = self._create_handler()
        server_address = ("0.0.0.0", self.port)
        self.httpd = ThreadedHTTPServer(server_address, handler_factory)
        self.running = True

        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()

        self.ws_thread = threading.Thread(target=self._ws_broadcaster, daemon=True)
        self.ws_thread.start()

        print(f"[✓] Module 2 (Web Dashboard): Mission Control live at http://0.0.0.0:{self.port}")

    def stop(self):
        self.running = False
        if self.httpd:
            self.httpd.shutdown()

    def _ws_broadcaster(self):
        while self.running:
            time.sleep(0.1) # 10 Hz
            with self.ws_lock:
                if not self.connected_websockets:
                    continue
                snapshot = self.get_full_telemetry()
                frame = encode_ws_frame(json.dumps(snapshot))
                dead_sockets = []
                for sock in self.connected_websockets:
                    try:
                        sock.sendall(frame)
                    except Exception:
                        dead_sockets.append(sock)
                for d in dead_sockets:
                    if d in self.connected_websockets:
                        self.connected_websockets.remove(d)

    def _create_handler(self):
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                # WebSocket upgrade
                if self.headers.get("Upgrade", "").lower() == "websocket":
                    self.handle_websocket()
                    return

                clean_path = self.path.split("?")[0]

                # REST API /api/telemetry
                if clean_path == "/api/telemetry":
                    payload = json.dumps(parent.get_full_telemetry())
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(payload.encode("utf-8"))
                    return

                # Mission Status
                if clean_path in ("/api/mission/status", "/api/mission/grid_search") and parent.grid_module:
                    payload = json.dumps(parent.grid_module.get_status())
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(payload.encode("utf-8"))
                    return

                # Static files
                if clean_path in ("/", "/index.html"):
                    target_file = os.path.join(WEB_DIR, "index.html")
                    content_type = "text/html; charset=utf-8"
                elif clean_path == "/style.css":
                    target_file = os.path.join(WEB_DIR, "style.css")
                    content_type = "text/css"
                elif clean_path == "/app.js":
                    target_file = os.path.join(WEB_DIR, "app.js")
                    content_type = "application/javascript"
                else:
                    self.send_error(404, "File Not Found")
                    return

                if os.path.exists(target_file):
                    with open(target_file, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_error(404, "Static file not found")

            def do_POST(self):
                clean_path = self.path.split("?")[0]
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
                params = json.loads(body) if body else {}

                # 1. Mission Preview
                if clean_path == "/api/mission/preview" and parent.grid_module:
                    snap = parent.mav_manager.get_telemetry_snapshot() if parent.mav_manager else {}
                    lat = params.get("lat") if (params.get("lat") is not None and params.get("lat") != 0) else snap.get("latitude", 12.971598)
                    lon = params.get("lon") if (params.get("lon") is not None and params.get("lon") != 0) else snap.get("longitude", 77.594562)

                    plan = parent.grid_module.plan_mission(
                        center_lat=lat,
                        center_lon=lon,
                        radius_m=float(params.get("radius", 50.0)),
                        altitude_m=float(params.get("altitude", 15.0)),
                        spacing_m=float(params.get("spacing", 10.0)),
                        speed_m_s=float(params.get("speed", 5.0)),
                        pattern=params.get("pattern", "grid"),
                        angle_deg=float(params.get("angle", 0.0)),
                        end_action=params.get("end_action", "RTL")
                    )
                    self._send_json(200, plan)
                    return

                # 2. Mission Upload
                elif clean_path in ("/api/mission/upload", "/api/mission/grid_search") and parent.grid_module:
                    res = parent.grid_module.trigger_mission_upload(
                        lat=params.get("lat"),
                        lon=params.get("lon"),
                        radius_m=float(params.get("radius", 50.0)),
                        altitude_m=float(params.get("altitude", 15.0)),
                        spacing_m=float(params.get("spacing", 10.0)),
                        speed_m_s=float(params.get("speed", 5.0)),
                        pattern=params.get("pattern", "grid"),
                        angle_deg=float(params.get("angle", 0.0)),
                        end_action=params.get("end_action", "RTL")
                    )
                    self._send_json(200, res)
                    return

                # 3. Mission Start (AUTO)
                elif clean_path == "/api/mission/start" and parent.grid_module:
                    res = parent.grid_module.start_mission(auto_arm=params.get("auto_arm", True))
                    self._send_json(200, res)
                    return

                # 4. Mission Pause (LOITER)
                elif clean_path == "/api/mission/pause" and parent.grid_module:
                    res = parent.grid_module.pause_mission()
                    self._send_json(200, res)
                    return

                # 5. Mission Resume (AUTO)
                elif clean_path == "/api/mission/resume" and parent.grid_module:
                    res = parent.grid_module.resume_mission()
                    self._send_json(200, res)
                    return

                # 6. Mission Abort (RTL)
                elif clean_path == "/api/mission/abort" and parent.grid_module:
                    res = parent.grid_module.abort_mission()
                    self._send_json(200, res)
                    return

                # 7. Mission Clear
                elif clean_path == "/api/mission/clear" and parent.grid_module:
                    res = parent.grid_module.clear_mission()
                    self._send_json(200, res)
                    return

                # 8. Flight Mode Change
                elif clean_path == "/api/mode" and parent.mav_manager:
                    mode = params.get("mode", "STABILIZE")
                    success = parent.mav_manager.set_flight_mode(mode)
                    self._send_json(200, {"status": "success" if success else "error", "mode": mode})
                    return

                # 9. Arm/Disarm
                elif clean_path == "/api/arm" and parent.mav_manager:
                    arm = params.get("arm", True)
                    success = parent.mav_manager.set_arm(arm)
                    self._send_json(200, {"status": "success" if success else "error", "armed": arm})
                    return

                self.send_error(404, "Endpoint not found.")

            def _send_json(self, status_code: int, data: dict):
                payload = json.dumps(data)
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))

            def handle_websocket(self):
                key = self.headers.get("Sec-WebSocket-Key")
                if not key:
                    self.send_error(400, "Missing Sec-WebSocket-Key")
                    return

                guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                accept = base64.b64encode(hashlib.sha1((key + guid).encode("utf-8")).digest()).decode("utf-8")

                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()

                sock = self.request
                with parent.ws_lock:
                    parent.connected_websockets.append(sock)

                try:
                    while True:
                        data = sock.recv(1024)
                        if not data:
                            break
                except Exception:
                    pass
                finally:
                    with parent.ws_lock:
                        if sock in parent.connected_websockets:
                            parent.connected_websockets.remove(sock)

        return Handler
