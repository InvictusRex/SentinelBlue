#!/usr/bin/env python3
"""
======================================================================================
Module 3: Automated Autonomous Mission Planner & Grid Search Autonomy Engine
File: pi/modules/grid_search_module.py

Description:
  Generates intelligent autonomous search missions:
  1. Boustrophedon (Lawnmower) Grid with arbitrary rotation angle
  2. Expanding Square / Archimedean Spiral Search
  3. Sector Search (SAR Pattern)

  Features:
  - Dynamic vertical takeoff from ANY starting point (MAV_CMD_NAV_TAKEOFF lat=0, lon=0)
  - Configurable flight speed (MAV_CMD_DO_CHANGE_SPEED)
  - End action: Return To Launch (RTL), Land, or Loiter
  - Route preview geometry, total path distance, and flight ETA calculation
  - Robust MAVLink 2 Mission Protocol uploader
======================================================================================
"""

import math
import time
import threading
from typing import List, Dict, Tuple, Any

EARTH_RADIUS = 6378137.0 # meters (WGS-84 equatorial)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS * c

def meter_offsets_to_latlon(center_lat: float, center_lon: float, dx: float, dy: float) -> Tuple[float, float]:
    """Converts local metric offsets (East=dx, North=dy) to latitude and longitude."""
    dlat = (dy / EARTH_RADIUS) * (180.0 / math.pi)
    dlon = (dx / (EARTH_RADIUS * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)
    return center_lat + dlat, center_lon + dlon

def rotate_point(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
    """Rotates local Cartesian point (x, y) clockwise by angle_deg degrees."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (x * cos_a + y * sin_a), (-x * sin_a + y * cos_a)


class GridSearchModule:
    """
    Autonomous Mission Planner & Search Grid Autonomy Controller for Pixhawk 2.4.8.
    """
    def __init__(self, mav_manager):
        self.mav_manager = mav_manager
        self.active_mission = False
        self.current_mission_data: Dict[str, Any] = {}
        self.mission_status = {
            "state": "IDLE", # IDLE, READY, UPLOADING, ACTIVE, PAUSED, COMPLETED, ABORTED, FAILED
            "pattern": "grid",
            "center_lat": 0.0,
            "center_lon": 0.0,
            "radius_m": 50.0,
            "altitude_m": 15.0,
            "spacing_m": 10.0,
            "speed_m_s": 5.0,
            "angle_deg": 0.0,
            "end_action": "RTL",
            "waypoint_count": 0,
            "current_wp_index": 0,
            "total_distance_m": 0.0,
            "estimated_duration_s": 0.0,
            "progress_percent": 0
        }
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------------------
    # 1. WAYPOINT GENERATION ALGORITHMS
    # ----------------------------------------------------------------------------------

    def generate_grid_waypoints(self, center_lat: float, center_lon: float,
                                radius_m: float = 50.0, altitude_m: float = 15.0,
                                spacing_m: float = 10.0, angle_deg: float = 0.0) -> List[Tuple[float, float, float]]:
        """
        Generates rotated Boustrophedon (lawnmower) serpentine grid waypoints clipped
        inside a circular search boundary of radius_m.
        """
        spacing_m = max(3.0, spacing_m)
        radius_m = max(5.0, radius_m)
        waypoints = []

        num_lanes = int((2.0 * radius_m) / spacing_m) + 1

        for i in range(num_lanes):
            # Local X position of the lane across diameter
            local_x = -radius_m + (i * spacing_m)

            # Circular boundary clipping for this lane
            if abs(local_x) >= radius_m:
                half_chord = 0.0
            else:
                half_chord = math.sqrt(radius_m**2 - local_x**2)

            if half_chord < 1.0:
                continue

            # Alternate direction for serpentine pattern (South->North, North->South)
            if i % 2 == 0:
                y_endpoints = [-half_chord, half_chord]
            else:
                y_endpoints = [half_chord, -half_chord]

            for local_y in y_endpoints:
                # Rotate local coordinate by grid angle
                rot_x, rot_y = rotate_point(local_x, local_y, angle_deg)
                wp_lat, wp_lon = meter_offsets_to_latlon(center_lat, center_lon, rot_x, rot_y)
                waypoints.append((round(wp_lat, 7), round(wp_lon, 7), round(altitude_m, 1)))

        return waypoints

    def generate_spiral_waypoints(self, center_lat: float, center_lon: float,
                                  radius_m: float = 50.0, altitude_m: float = 15.0,
                                  spacing_m: float = 10.0) -> List[Tuple[float, float, float]]:
        """
        Generates outward expanding Archimedean spiral / square search pattern from center point.
        """
        spacing_m = max(4.0, spacing_m)
        radius_m = max(5.0, radius_m)
        waypoints = []

        # Start at center datum
        waypoints.append((round(center_lat, 7), round(center_lon, 7), round(altitude_m, 1)))

        current_r = spacing_m
        step = 1
        x, y = 0.0, 0.0

        # Expanding box pattern (Right, Up, Left, Left, Down, Down, Right, Right, Right...)
        dx, dy = spacing_m, 0.0
        while current_r <= radius_m + (spacing_m / 2.0):
            for _ in range(2):
                for _ in range(step):
                    x += dx
                    y += dy
                    dist = math.sqrt(x**2 + y**2)
                    if dist <= radius_m:
                        wp_lat, wp_lon = meter_offsets_to_latlon(center_lat, center_lon, x, y)
                        waypoints.append((round(wp_lat, 7), round(wp_lon, 7), round(altitude_m, 1)))
                # Turn 90 degrees CCW
                dx, dy = -dy, dx
            step += 1
            current_r = max(abs(x), abs(y))

        return waypoints

    def generate_sector_waypoints(self, center_lat: float, center_lon: float,
                                  radius_m: float = 50.0, altitude_m: float = 15.0,
                                  angle_deg: float = 0.0) -> List[Tuple[float, float, float]]:
        """
        Generates 3-sector Search and Rescue (SAR) pattern radiating from datum with 120° cross-legs.
        """
        waypoints = []
        # Standard SAR Sector pattern (Center -> Leg 1 -> Cross 120° -> Center -> Leg 2 -> Cross -> Center -> Leg 3)
        angles = [angle_deg, angle_deg + 120.0, angle_deg + 240.0]

        for a in angles:
            # Outbound leg from center
            waypoints.append((round(center_lat, 7), round(center_lon, 7), round(altitude_m, 1)))
            rx, ry = rotate_point(0.0, radius_m, a)
            w1_lat, w1_lon = meter_offsets_to_latlon(center_lat, center_lon, rx, ry)
            waypoints.append((round(w1_lat, 7), round(w1_lon, 7), round(altitude_m, 1)))

            # Cross leg 120 degrees
            rx2, ry2 = rotate_point(0.0, radius_m, a + 120.0)
            w2_lat, w2_lon = meter_offsets_to_latlon(center_lat, center_lon, rx2, ry2)
            waypoints.append((round(w2_lat, 7), round(w2_lon, 7), round(altitude_m, 1)))

        # Final return to center
        waypoints.append((round(center_lat, 7), round(center_lon, 7), round(altitude_m, 1)))
        return waypoints

    # ----------------------------------------------------------------------------------
    # 2. MISSION CALCULATION & PREVIEW
    # ----------------------------------------------------------------------------------

    def calculate_mission_metrics(self, waypoints: List[Tuple[float, float, float]], speed_m_s: float = 5.0) -> Tuple[float, float]:
        """Computes total trajectory distance (meters) and estimated flight time (seconds)."""
        if not waypoints or len(waypoints) < 2:
            return 0.0, 0.0

        total_dist = 0.0
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            dist = haversine_distance(p1[0], p1[1], p2[0], p2[1])
            # Include 3D altitude delta
            dz = abs(p2[2] - p1[2])
            total_dist += math.sqrt(dist**2 + dz**2)

        speed_m_s = max(1.0, speed_m_s)
        estimated_time_s = total_dist / speed_m_s
        # Add 1.5s per turn / waypoint deceleration buffer
        estimated_time_s += len(waypoints) * 1.5

        return round(total_dist, 1), round(estimated_time_s, 0)

    def plan_mission(self, center_lat: float, center_lon: float,
                     radius_m: float = 50.0, altitude_m: float = 15.0,
                     spacing_m: float = 10.0, speed_m_s: float = 5.0,
                     pattern: str = "grid", angle_deg: float = 0.0,
                     end_action: str = "RTL") -> Dict[str, Any]:
        """
        Plans the complete autonomous mission and returns preview payload with metadata.
        """
        if pattern == "spiral":
            waypoints = self.generate_spiral_waypoints(center_lat, center_lon, radius_m, altitude_m, spacing_m)
        elif pattern == "sector":
            waypoints = self.generate_sector_waypoints(center_lat, center_lon, radius_m, altitude_m, angle_deg)
        else:
            pattern = "grid"
            waypoints = self.generate_grid_waypoints(center_lat, center_lon, radius_m, altitude_m, spacing_m, angle_deg)

        total_dist, est_duration = self.calculate_mission_metrics(waypoints, speed_m_s)

        # Build structured preview item list
        preview_items = []
        # Item 0: Dynamic Home
        preview_items.append({
            "seq": 0, "type": "HOME", "label": "Dynamic Home (Armed Position)",
            "lat": center_lat, "lon": center_lon, "alt": 0.0
        })
        # Item 1: Dynamic Takeoff (lat=0, lon=0)
        preview_items.append({
            "seq": 1, "type": "TAKEOFF", "label": f"Vertical Takeoff to {altitude_m}m AGL",
            "lat": 0.0, "lon": 0.0, "alt": altitude_m, "dynamic": True
        })
        # Item 2: Speed Command
        preview_items.append({
            "seq": 2, "type": "SPEED", "label": f"Set Cruise Speed ({speed_m_s} m/s)",
            "speed_m_s": speed_m_s
        })
        # Items 3..N: Survey Waypoints
        for idx, wp in enumerate(waypoints, start=3):
            preview_items.append({
                "seq": idx, "type": "WAYPOINT", "label": f"WP {idx - 2}",
                "lat": wp[0], "lon": wp[1], "alt": wp[2]
            })
        # Final Item: End Action
        preview_items.append({
            "seq": len(preview_items), "type": end_action.upper(),
            "label": f"End Action: {end_action.upper()}",
            "lat": center_lat, "lon": center_lon, "alt": 0.0
        })

        mission_payload = {
            "status": "ready",
            "pattern": pattern,
            "center": [center_lat, center_lon],
            "radius_m": radius_m,
            "altitude_m": altitude_m,
            "spacing_m": spacing_m,
            "speed_m_s": speed_m_s,
            "angle_deg": angle_deg,
            "end_action": end_action.upper(),
            "waypoint_count": len(waypoints),
            "total_items": len(preview_items),
            "total_distance_m": total_dist,
            "estimated_duration_s": est_duration,
            "waypoints": [[wp[0], wp[1], wp[2]] for wp in waypoints],
            "items": preview_items
        }

        with self._lock:
            self.current_mission_data = mission_payload

        return mission_payload

    # ----------------------------------------------------------------------------------
    # 3. MAVLINK MISSION COMPILER & UPLOADER
    # ----------------------------------------------------------------------------------

    def build_mavlink_mission_items(self, center_lat: float, center_lon: float,
                                    waypoints: List[Tuple[float, float, float]],
                                    takeoff_alt: float = 15.0, speed_m_s: float = 5.0,
                                    end_action: str = "RTL") -> List[Dict[str, Any]]:
        """
        Constructs standard MAVLink mission item dictionaries ready for serial transmission.
        """
        items = []

        # Sequence 0: Home Location (ArduPilot will overwrite on arm)
        items.append({
            "seq": 0,
            "frame": 0, # MAV_FRAME_GLOBAL
            "command": 16, # MAV_CMD_NAV_WAYPOINT
            "current": 0, "autocontinue": 1,
            "param1": 0, "param2": 0, "param3": 0, "param4": 0,
            "x": int(center_lat * 1e7), "y": int(center_lon * 1e7), "z": 0.0
        })

        # Sequence 1: Dynamic Vertical Takeoff (lat=0, lon=0 means takeoff at current spot!)
        items.append({
            "seq": 1,
            "frame": 3, # MAV_FRAME_GLOBAL_RELATIVE_ALT
            "command": 22, # MAV_CMD_NAV_TAKEOFF
            "current": 0, "autocontinue": 1,
            "param1": 0, "param2": 0, "param3": 0, "param4": 0,
            "x": 0, "y": 0, "z": float(takeoff_alt)
        })

        # Sequence 2: Set Cruise Speed
        items.append({
            "seq": 2,
            "frame": 3,
            "command": 178, # MAV_CMD_DO_CHANGE_SPEED
            "current": 0, "autocontinue": 1,
            "param1": 0, # 0 = Airspeed, 1 = Groundspeed
            "param2": float(speed_m_s), # Speed in m/s
            "param3": -1, "param4": 0,
            "x": 0, "y": 0, "z": 0.0
        })

        # Sequence 3..N: Search Grid Waypoints
        for idx, wp in enumerate(waypoints, start=3):
            items.append({
                "seq": idx,
                "frame": 3, # MAV_FRAME_GLOBAL_RELATIVE_ALT
                "command": 16, # MAV_CMD_NAV_WAYPOINT
                "current": 0, "autocontinue": 1,
                "param1": 0, # Hold time in sec
                "param2": 2.0, # Acceptance radius in meters
                "param3": 0, "param4": 0,
                "x": int(wp[0] * 1e7), "y": int(wp[1] * 1e7), "z": float(wp[2])
            })

        # Sequence N+1: End of Mission Action
        end_cmd = 20 # MAV_CMD_NAV_RETURN_TO_LAUNCH
        if end_action.upper() == "LAND":
            end_cmd = 21 # MAV_CMD_NAV_LAND
        elif end_action.upper() == "LOITER":
            end_cmd = 17 # MAV_CMD_NAV_LOITER_UNLIM

        items.append({
            "seq": len(items),
            "frame": 3,
            "command": end_cmd,
            "current": 0, "autocontinue": 1,
            "param1": 0, "param2": 0, "param3": 0, "param4": 0,
            "x": 0, "y": 0, "z": 0.0
        })

        return items

    def upload_mission_to_pixhawk(self, center_lat: float, center_lon: float,
                                  waypoints: List[Tuple[float, float, float]],
                                  takeoff_alt: float = 15.0, speed_m_s: float = 5.0,
                                  end_action: str = "RTL") -> bool:
        """
        Uploads compiled mission items to Pixhawk flight controller via MAVLinkManager.
        """
        items = self.build_mavlink_mission_items(center_lat, center_lon, waypoints, takeoff_alt, speed_m_s, end_action)

        if not self.mav_manager:
            print("[!] Mission Planner: MAVLinkManager not available.")
            return False

        print(f"[*] Mission Planner: Uploading {len(items)} items to Pixhawk (Takeoff + {len(waypoints)} WPs + {end_action})...")
        success = self.mav_manager.upload_mission_items(items)

        with self._lock:
            if success:
                self.mission_status.update({
                    "state": "UPLOADED",
                    "waypoint_count": len(waypoints),
                    "total_items": len(items),
                    "current_wp_index": 1,
                    "progress_percent": 0
                })
            else:
                self.mission_status["state"] = "UPLOAD_FAILED"

        return success

    # ----------------------------------------------------------------------------------
    # 4. HIGH-LEVEL API TRIGGER & STATUS
    # ----------------------------------------------------------------------------------

    def trigger_mission_upload(self, lat: float = None, lon: float = None,
                               radius_m: float = 50.0, altitude_m: float = 15.0,
                               spacing_m: float = 10.0, speed_m_s: float = 5.0,
                               pattern: str = "grid", angle_deg: float = 0.0,
                               end_action: str = "RTL") -> Dict[str, Any]:
        """
        Plans and immediately uploads mission to Pixhawk.
        """
        snap = self.mav_manager.get_telemetry_snapshot() if self.mav_manager else {}
        center_lat = lat if (lat is not None and lat != 0.0) else snap.get("latitude", 12.971598)
        center_lon = lon if (lon is not None and lon != 0.0) else snap.get("longitude", 77.594562)

        if center_lat == 0.0 or center_lon == 0.0:
            center_lat, center_lon = 12.971598, 77.594562

        # 1. Plan route
        plan = self.plan_mission(center_lat, center_lon, radius_m, altitude_m, spacing_m, speed_m_s, pattern, angle_deg, end_action)

        # 2. Upload to flight controller
        success = self.upload_mission_to_pixhawk(
            center_lat, center_lon, plan["waypoints"], altitude_m, speed_m_s, end_action
        )

        with self._lock:
            self.mission_status.update({
                "state": "READY" if success else "FAILED",
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_m": radius_m,
                "altitude_m": altitude_m,
                "spacing_m": spacing_m,
                "speed_m_s": speed_m_s,
                "pattern": pattern,
                "angle_deg": angle_deg,
                "end_action": end_action.upper(),
                "waypoint_count": plan["waypoint_count"],
                "total_distance_m": plan["total_distance_m"],
                "estimated_duration_s": plan["estimated_duration_s"]
            })

        return {
            "status": "success" if success else "error",
            "message": "Mission uploaded successfully to Pixhawk" if success else "Mission upload failed",
            "plan": plan
        }

    def start_mission(self, auto_arm: bool = False) -> Dict[str, Any]:
        """Commands Pixhawk into AUTO flight mode to begin mission."""
        if not self.mav_manager:
            return {"status": "error", "message": "No MAVLink connection"}

        if auto_arm:
            self.mav_manager.set_arm(True)
            time.sleep(0.5)

        success = self.mav_manager.set_flight_mode("AUTO")
        with self._lock:
            if success:
                self.mission_status["state"] = "ACTIVE"
        return {"status": "success" if success else "error", "mode": "AUTO"}

    def pause_mission(self) -> Dict[str, Any]:
        """Pauses mission by switching flight mode to LOITER (holds 3D position)."""
        if not self.mav_manager:
            return {"status": "error", "message": "No MAVLink connection"}

        success = self.mav_manager.set_flight_mode("LOITER")
        with self._lock:
            if success:
                self.mission_status["state"] = "PAUSED"
        return {"status": "success" if success else "error", "mode": "LOITER"}

    def resume_mission(self) -> Dict[str, Any]:
        """Resumes mission by returning flight mode to AUTO."""
        if not self.mav_manager:
            return {"status": "error", "message": "No MAVLink connection"}

        success = self.mav_manager.set_flight_mode("AUTO")
        with self._lock:
            if success:
                self.mission_status["state"] = "ACTIVE"
        return {"status": "success" if success else "error", "mode": "AUTO"}

    def abort_mission(self) -> Dict[str, Any]:
        """Aborts mission immediately by triggering Return to Launch (RTL)."""
        if not self.mav_manager:
            return {"status": "error", "message": "No MAVLink connection"}

        success = self.mav_manager.set_flight_mode("RTL")
        with self._lock:
            self.mission_status["state"] = "ABORTED"
        return {"status": "success" if success else "error", "mode": "RTL"}

    def clear_mission(self) -> Dict[str, Any]:
        """Clears all mission waypoints on Pixhawk."""
        if not self.mav_manager:
            return {"status": "error", "message": "No MAVLink connection"}

        success = self.mav_manager.clear_all_missions()
        with self._lock:
            self.mission_status["state"] = "IDLE"
            self.mission_status["waypoint_count"] = 0
            self.current_mission_data = {}
        return {"status": "success" if success else "error", "message": "Mission cleared"}

    def get_status(self) -> Dict[str, Any]:
        """Returns live mission execution telemetry status."""
        with self._lock:
            st = dict(self.mission_status)
            if self.current_mission_data:
                st["preview"] = self.current_mission_data
            return st

if __name__ == "__main__":
    from mavlink_manager import MAVLinkManager
    mgr = MAVLinkManager(simulate=True)
    mgr.start()
    grid = GridSearchModule(mgr)
    plan = grid.plan_mission(center_lat=12.971598, center_lon=77.594562, radius_m=40, spacing_m=10, angle_deg=30)
    print("Generated plan summary:")
    print(f"Waypoints: {plan['waypoint_count']} | Distance: {plan['total_distance_m']}m | Duration: {plan['estimated_duration_s']}s")
    mgr.stop()
