#!/usr/bin/env python3
"""HTTP API for mission preview and simulation-only execution."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from search_mission_planner import build_search_mission
from terrain_provider import sample_plan_terrain


MAX_REQUEST_BYTES = 64 * 1024


class CompanionAPIServer:
    def __init__(self, mav_manager, host: str = "0.0.0.0", port: int = 8765):
        self.mav_manager = mav_manager
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.server:
            return
        handler = _handler_for(self.mav_manager)
        self.server = ThreadingHTTPServer((self.host, self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        bound_port = self.server.server_address[1]
        print(f"[OK] Companion API: http://{self.host}:{bound_port} (live execution locked)")

    def stop(self) -> None:
        if not self.server:
            return
        self.server.shutdown()
        self.server.server_close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.server = None
        self.thread = None


def _handler_for(mav_manager):
    mission_lock = threading.Lock()

    class CompanionRequestHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors_headers()
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/health":
                telemetry = mav_manager.get_telemetry_snapshot()
                self._send_json(200, {
                    "status": "ok",
                    "simulation": bool(mav_manager.simulate),
                    "connected": bool(telemetry.get("connected", False)),
                    "live_execution_enabled": False,
                })
                return
            self._send_json(404, {"error": "endpoint not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in {"/api/mission/plan", "/api/mission/simulate"}:
                self._send_json(404, {"error": "endpoint not found"})
                return
            try:
                payload = self._read_json()
                plan = _plan_from_payload(payload, mav_manager.get_telemetry_snapshot())
                terrain = (
                    sample_plan_terrain(plan, str(payload.get("surface_type", "water")))
                    if payload.get("terrain_enabled", False)
                    else {"status": "disabled", "used_for_clearance": False}
                )
            except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return

            if path == "/api/mission/plan":
                plan_payload = plan.to_dict()
                plan_payload["terrain"] = terrain
                self._send_json(200, {"status": "planned", "plan": plan_payload})
                return

            if not mav_manager.simulate:
                self._send_json(403, {
                    "error": "live Pixhawk execution is locked; validate this mission in simulation first"
                })
                return

            if not plan.estimate.simulation_ready:
                self._send_json(422, {
                    "error": "mission exceeds single-sortie limits",
                    "reasons": list(plan.estimate.blocking_reasons),
                })
                return

            if not mission_lock.acquire(blocking=False):
                self._send_json(409, {"error": "another mission operation is in progress"})
                return
            try:
                if not mav_manager.clear_all_missions():
                    self._send_json(500, {"error": "could not clear the simulated mission"})
                    return
                if not mav_manager.upload_mission_items(list(plan.mission_items)):
                    self._send_json(500, {"error": "could not load the simulated mission"})
                    return
                if not mav_manager.set_arm(True) or not mav_manager.set_flight_mode("AUTO"):
                    self._send_json(500, {"error": "could not start the simulated mission"})
                    return
                plan_payload = plan.to_dict()
                plan_payload["terrain"] = terrain
                self._send_json(202, {"status": "simulation_started", "plan": plan_payload})
            finally:
                mission_lock.release()

        def _read_json(self) -> Dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError("request body must be between 1 byte and 64 KiB")
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self._cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def log_message(self, format_string: str, *args) -> None:
            print(f"[companion-api] {self.address_string()} {format_string % args}")

    return CompanionRequestHandler


def _plan_from_payload(payload: Dict[str, Any], telemetry: Dict[str, Any]):
    center = payload.get("center")
    if not isinstance(center, dict):
        raise ValueError("center must contain latitude and longitude")

    home_latitude = telemetry.get("latitude")
    home_longitude = telemetry.get("longitude")
    if not _valid_position(home_latitude, home_longitude):
        home_latitude = center.get("latitude")
        home_longitude = center.get("longitude")

    return build_search_mission(
        center_latitude=center.get("latitude"),
        center_longitude=center.get("longitude"),
        radius_km=payload.get("radius_km"),
        altitude_m=payload.get("altitude_m"),
        home_latitude=home_latitude,
        home_longitude=home_longitude,
        camera_horizontal_fov_deg=payload.get("camera_horizontal_fov_deg", 78.0),
        camera_vertical_fov_deg=payload.get("camera_vertical_fov_deg", 60.0),
        side_overlap_percent=payload.get("side_overlap_percent", 70.0),
        forward_overlap_percent=payload.get("forward_overlap_percent", 75.0),
        cruise_speed_m_s=payload.get("cruise_speed_m_s", 3.0),
    )


def _valid_position(latitude: Any, longitude: Any) -> bool:
    try:
        return -90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180 and not (
            float(latitude) == 0 and float(longitude) == 0
        )
    except (TypeError, ValueError):
        return False
