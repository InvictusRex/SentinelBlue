#!/usr/bin/env python3
"""
======================================================================================
Module: MAVLink Telemetry Manager & Shared State Hub
File: pi/tele/mavlink_manager.py

Description:
  Singleton MAVLink reader that connects to Pixhawk 2.4.8 on TELEM2 (/dev/serial0)
  or USB (/dev/ttyACM0 @ 115200 baud).
  
  Key Features:
  - Broadcasts 1 Hz Companion Heartbeats to keep ArduPilot active telemetry streaming alive
  - Requests all sensor & attitude streams at 10 Hz
  - Decodes ATTITUDE, AHRS, SYS_STATUS, GLOBAL_POSITION_INT, VFR_HUD, RAW_IMU, SERVO_OUTPUT
  - Thread-safe state store for WebSockets and Radio TX
======================================================================================
"""

import time
import math
import threading
from typing import Dict, Any, List, Optional

class MAVLinkManager:
    def __init__(self, port: str = "/dev/serial0", baud: int = 115200, simulate: bool = False):
        self.port = port
        self.baud = baud
        self.simulate = simulate
        self.running = False
        self.thread = None
        self.mav = None
        self._lock = threading.Lock()
        self._serial_cmd_lock = threading.Lock()

        # Simulation mission playback storage
        self.sim_mission_items: List[Dict[str, Any]] = []
        self.sim_mission_index: int = 0
        self.sim_current_lat: float = 12.971598
        self.sim_current_lon: float = 77.594562
        self.sim_current_alt: float = 0.0

        # Telemetry State Store (All 11 Categories + Mission Autonomy)
        self.state: Dict[str, Any] = {
            # 1. Total Battery & Power
            "battery_voltage": 0.0,
            "battery_current": 0.0,
            "battery_remaining": 0,
            
            # 2. Individual Cell Voltages (Cell 1 to 6)
            "cell_voltages": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "cell_delta_mv": 0,
            
            # 3. Altitude & Vertical Dynamics
            "altitude_relative": 0.0,
            "altitude_msl": 0.0,
            "climb_rate": 0.0,
            
            # 4. GPS & Positioning
            "latitude": 0.0,
            "longitude": 0.0,
            "satellites": 0,
            "gps_fix_type": "NO FIX",
            "hdop": 99.9,
            
            # 5. Signal Strength & Telemetry Link
            "rc_rssi": 0,
            "radio_link_quality": 0,
            
            # 6. Heading & Compass
            "heading": 0.0,
            "compass_status": "CALIBRATED",
            "ground_speed": 0.0,
            
            # 7. Flight Status & Mission Autonomy State
            "armed": False,
            "flight_mode": "DISCONNECTED",
            "system_status": "STANDBY",
            "mission_state": "STANDBY",
            "mission_progress_percent": 0,
            "mission_current_seq": 0,
            "mission_total_items": 0,
            "dist_to_target_wp_m": 0.0,
            "target_wp_lat": 0.0,
            "target_wp_lon": 0.0,
            
            # 8. 3-Axis Gyroscope & Accelerometer Rates
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 9.81,
            
            # 9. PID Attitude Tracking & Errors
            "attitude_roll": 0.0,
            "attitude_pitch": 0.0,
            "attitude_yaw": 0.0,
            "target_roll": 0.0,
            "target_pitch": 0.0,
            "target_yaw": 0.0,
            "error_roll": 0.0,
            "error_pitch": 0.0,
            "error_yaw": 0.0,
            
            # 10. Motor Power Outputs (Motors 1 to 4)
            "motor_pwm": [1000, 1000, 1000, 1000],
            "motor_percent": [0, 0, 0, 0],
            
            # Timing & Status
            "last_packet_timestamp": 0.0,
            "last_heartbeat_timestamp": 0.0,
            "last_message_type": "",
            "mavlink_status": "Waiting for serial port",
            "connected": False
        }

    def start(self):
        """Starts the MAVLink ingestion thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the ingestion loop."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.mav:
            try:
                self.mav.close()
            except Exception:
                pass

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Thread-safe copy of latest telemetry."""
        with self._lock:
            return dict(self.state)

    # ----------------------------------------------------------------------------------
    # MAVLINK COMMAND & MISSION METHODS
    # ----------------------------------------------------------------------------------

    def set_flight_mode(self, mode: str) -> bool:
        """Commands Pixhawk into requested flight mode (e.g. AUTO, LOITER, RTL, GUIDED)."""
        mode_upper = mode.upper()
        if self.simulate or not self.mav:
            print(f"[✓] Simulation: Flight mode switched to '{mode_upper}'")
            with self._lock:
                self.state["flight_mode"] = mode_upper
                if mode_upper == "AUTO":
                    self.state["armed"] = True
                    self.state["mission_state"] = "ACTIVE"
                    self.sim_mission_index = max(1, self.sim_mission_index)
                elif mode_upper == "LOITER":
                    self.state["mission_state"] = "PAUSED (LOITER)"
                elif mode_upper == "RTL":
                    self.state["mission_state"] = "RETURNING (RTL)"
            return True

        with self._serial_cmd_lock:
            try:
                from pymavlink import mavutil
                target_sys = getattr(self.mav, "target_system", 1) or 1
                mode_map = self.mav.mode_mapping()
                if mode_map and mode_upper in mode_map:
                    mode_id = mode_map[mode_upper]
                    self.mav.set_mode(mode_id)
                    with self._lock:
                        self.state["flight_mode"] = mode_upper
                    return True
                else:
                    custom_num = {
                        "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3,
                        "GUIDED": 4, "LOITER": 5, "RTL": 6, "CIRCLE": 7,
                        "LAND": 9, "POSHOLD": 16, "BRAKE": 17
                    }.get(mode_upper, 0)
                    self.mav.mav.command_long_send(
                        target_sys, 1,
                        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                        0,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        custom_num,
                        0, 0, 0, 0, 0
                    )
                    with self._lock:
                        self.state["flight_mode"] = mode_upper
                    return True
            except Exception as e:
                print(f"[!] Failed to set flight mode {mode}: {e}")
                return False

    def set_arm(self, arm: bool = True, force: bool = False) -> bool:
        """Arms or disarms the motors."""
        if self.simulate or not self.mav:
            print(f"[✓] Simulation: Motors {'ARMED' if arm else 'DISARMED'}")
            with self._lock:
                self.state["armed"] = arm
            return True

        with self._serial_cmd_lock:
            try:
                from pymavlink import mavutil
                target_sys = getattr(self.mav, "target_system", 1) or 1
                param1 = 1 if arm else 0
                param2 = 21196 if force else 0
                self.mav.mav.command_long_send(
                    target_sys, 1,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    param1, param2, 0, 0, 0, 0, 0
                )
                with self._lock:
                    self.state["armed"] = arm
                return True
            except Exception as e:
                print(f"[!] Arm command failed: {e}")
                return False

    def clear_all_missions(self) -> bool:
        """Clears all mission waypoints from flight controller."""
        if self.simulate or not self.mav:
            with self._lock:
                self.sim_mission_items = []
                self.sim_mission_index = 0
                self.state["mission_total_items"] = 0
                self.state["mission_current_seq"] = 0
                self.state["mission_progress_percent"] = 0
                self.state["mission_state"] = "STANDBY"
            return True

        with self._serial_cmd_lock:
            try:
                target_sys = getattr(self.mav, "target_system", 1) or 1
                target_comp = getattr(self.mav, "target_component", 1) or 1
                self.mav.mav.mission_clear_all_send(target_sys, target_comp)
                with self._lock:
                    self.state["mission_total_items"] = 0
                    self.state["mission_current_seq"] = 0
                    self.state["mission_progress_percent"] = 0
                    self.state["mission_state"] = "STANDBY"
                return True
            except Exception as e:
                print(f"[!] Mission clear error: {e}")
                return False

    def upload_mission_items(self, items: List[Dict[str, Any]]) -> bool:
        """Thread-safe mission upload to Pixhawk using MAVLink Mission Protocol."""
        if self.simulate or not self.mav:
            print(f"[✓] Simulation: Loaded {len(items)} mission items into simulator memory.")
            with self._lock:
                self.sim_mission_items = items
                self.sim_mission_index = 1
                self.state["mission_total_items"] = len(items)
                self.state["mission_current_seq"] = 1
                self.state["mission_progress_percent"] = 0
                self.state["mission_state"] = "READY (MISSION LOADED)"
            return True

        with self._serial_cmd_lock:
            try:
                from pymavlink import mavutil
                target_sys = getattr(self.mav, "target_system", 1) or 1
                target_comp = getattr(self.mav, "target_component", 1) or 1

                # 1. Clear existing mission
                self.mav.mav.mission_clear_all_send(target_sys, target_comp)
                time.sleep(0.1)

                # 2. Announce mission count
                total_count = len(items)
                self.mav.mav.mission_count_send(target_sys, target_comp, total_count)

                # 3. Transmit mission items upon request
                start_time = time.time()
                seq = 0
                while seq < total_count and (time.time() - start_time < 15.0):
                    msg = self.mav.recv_match(type=['MISSION_REQUEST_INT', 'MISSION_REQUEST', 'MISSION_ACK'], blocking=True, timeout=2.0)
                    if not msg:
                        continue
                    if msg.get_type() == 'MISSION_ACK':
                        if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                            with self._lock:
                                self.state["mission_total_items"] = total_count
                                self.state["mission_state"] = "READY (MISSION LOADED)"
                            return True
                        else:
                            print(f"[!] Mission upload rejected with ACK {msg.type}")
                            return False

                    req_seq = msg.seq
                    if 0 <= req_seq < total_count:
                        it = items[req_seq]
                        self.mav.mav.mission_item_int_send(
                            target_sys, target_comp,
                            it.get("seq", req_seq),
                            it.get("frame", mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT),
                            it.get("command", mavutil.mavlink.MAV_CMD_NAV_WAYPOINT),
                            it.get("current", 0),
                            it.get("autocontinue", 1),
                            it.get("param1", 0.0),
                            it.get("param2", 0.0),
                            it.get("param3", 0.0),
                            it.get("param4", 0.0),
                            int(it.get("x", 0)),
                            int(it.get("y", 0)),
                            float(it.get("z", 0.0))
                        )
                        seq = req_seq + 1

                ack = self.mav.recv_match(type='MISSION_ACK', blocking=True, timeout=3.0)
                if ack and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    with self._lock:
                        self.state["mission_total_items"] = total_count
                        self.state["mission_state"] = "READY (MISSION LOADED)"
                    return True
                return False

            except Exception as e:
                print(f"[!] Mission upload error: {e}")
                return False

    def _run_loop(self):
        if self.simulate:
            self._run_simulation()
        else:
            self._run_live_mavlink()

    def _send_companion_heartbeat(self, mavutil):
        """Sends periodic 1 Hz GCS heartbeat from Pi to Pixhawk to keep streams active."""
        if not self.mav:
            return
        try:
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0,
                mavutil.mavlink.MAV_STATE_ACTIVE
            )
        except Exception:
            pass

    def _request_all_streams(self, mavutil):
        """Requests individual telemetry streams, sets stream params, and modern message intervals."""
        if not self.mav:
            return

        target_sys = getattr(self.mav, "target_system", 1) or 1
        target_comp = 1  # ArduPilot Autopilot primary component is always 1

        try:
            # 1. Set ArduPilot Stream Parameters (SR0=USB, SR1=TELEM1, SR2=TELEM2)
            stream_params = {
                "SR0_EXTRA1": 10.0,   # ATTITUDE @ 10 Hz
                "SR0_EXTRA2": 10.0,   # VFR_HUD @ 10 Hz
                "SR0_EXTRA3": 10.0,   # AHRS / system status @ 10 Hz
                "SR0_RAW_SENS": 10.0, # RAW_IMU / SCALED_IMU @ 10 Hz
                "SR0_POSITION": 5.0,  # GPS & position @ 5 Hz
                "SR0_EXT_STAT": 2.0,  # SYS_STATUS & battery @ 2 Hz
                "SR0_RC_CHAN": 5.0,   # RC_CHANNELS & servos @ 5 Hz

                "SR1_EXTRA1": 10.0,
                "SR1_EXTRA2": 10.0,
                "SR1_EXTRA3": 10.0,
                "SR1_RAW_SENS": 10.0,
                "SR1_POSITION": 5.0,
                "SR1_EXT_STAT": 2.0,
                "SR1_RC_CHAN": 5.0,

                "SR2_EXTRA1": 10.0,
                "SR2_EXTRA2": 10.0,
                "SR2_EXTRA3": 10.0,
                "SR2_RAW_SENS": 10.0,
                "SR2_POSITION": 5.0,
                "SR2_EXT_STAT": 2.0,
                "SR2_RC_CHAN": 5.0,
            }
            for p_name, p_val in stream_params.items():
                try:
                    p_bytes = p_name.encode('utf-8')
                    self.mav.mav.param_set_send(
                        target_sys, target_comp,
                        p_bytes, float(p_val),
                        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                    )
                except Exception:
                    pass

            # 2. Legacy MAV_DATA_STREAM requests (targeting both Component 1 and Component 0)
            target_comps = [1, 0]
            if getattr(self.mav, "target_component", 0) not in target_comps:
                target_comps.append(self.mav.target_component)

            for comp_id in target_comps:
                for s in [
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
                    mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
                    mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
                    mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
                    mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                    mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
                    mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS
                ]:
                    self.mav.mav.request_data_stream_send(
                        target_sys, comp_id, s, 10, 1
                    )

            intervals = [
                ("MAVLINK_MSG_ID_ATTITUDE", 100000),
                ("MAVLINK_MSG_ID_ATTITUDE_TARGET", 100000),
                ("MAVLINK_MSG_ID_RAW_IMU", 100000),
                ("MAVLINK_MSG_ID_SCALED_IMU", 100000),
                ("MAVLINK_MSG_ID_HIGHRES_IMU", 100000),
                ("MAVLINK_MSG_ID_GLOBAL_POSITION_INT", 200000),
                ("MAVLINK_MSG_ID_GPS_RAW_INT", 200000),
                ("MAVLINK_MSG_ID_VFR_HUD", 200000),
                ("MAVLINK_MSG_ID_SYS_STATUS", 500000),
                ("MAVLINK_MSG_ID_BATTERY_STATUS", 500000),
                ("MAVLINK_MSG_ID_SERVO_OUTPUT_RAW", 200000),
                ("MAVLINK_MSG_ID_RC_CHANNELS", 200000),
                ("MAVLINK_MSG_ID_RADIO_STATUS", 500000),
                ("MAVLINK_MSG_ID_NAV_CONTROLLER_OUTPUT", 200000),
                ("MAVLINK_MSG_ID_MISSION_CURRENT", 500000),
            ]
            for msg_name, interval_us in intervals:
                msg_id = getattr(mavutil.mavlink, msg_name, None)
                if msg_id is None:
                    continue
                self.mav.mav.command_long_send(
                    target_sys, target_comp,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id, interval_us, 0, 0, 0, 0, 0
                )
        except Exception:
            pass

    def _set_connection_state(self, connected: bool, status: str):
        with self._lock:
            self.state["connected"] = connected
            self.state["mavlink_status"] = status
            if not connected:
                self.state["flight_mode"] = "NO HEARTBEAT" if "packet" in status.lower() else "DISCONNECTED"

    def _heading(self, value, scale: float = 1.0):
        try:
            return round((float(value) / scale) % 360.0, 2)
        except Exception:
            return None

    def _yaw_error(self, target: float, actual: float) -> float:
        return round((target - actual + 180.0) % 360.0 - 180.0, 2)

    def _update_attitude_errors(self):
        self.state["error_roll"] = round(self.state["target_roll"] - self.state["attitude_roll"], 2)
        self.state["error_pitch"] = round(self.state["target_pitch"] - self.state["attitude_pitch"], 2)
        self.state["error_yaw"] = self._yaw_error(self.state["target_yaw"], self.state["attitude_yaw"])

    def _mode_name(self, custom_mode) -> str:
        try:
            mode_map = self.mav.mode_mapping() if self.mav else {}
            inv_map = {v: k for k, v in mode_map.items()}
            return inv_map.get(custom_mode, f"MODE_{custom_mode}")
        except Exception:
            return f"MODE_{custom_mode}"

    def _handle_mavlink_message(self, msg, mavutil, now: Optional[float] = None) -> bool:
        msg_type = msg.get_type()
        if msg_type in ("BAD_DATA", "UNKNOWN"):
            return False

        now = now or time.time()
        with self._lock:
            self.state["last_packet_timestamp"] = now
            self.state["last_message_type"] = msg_type
            self.state["connected"] = True
            self.state["mavlink_status"] = "Receiving MAVLink telemetry"

            if msg_type == 'HEARTBEAT':
                self.state["last_heartbeat_timestamp"] = now
                self.state["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                self.state["flight_mode"] = self._mode_name(msg.custom_mode)
                sys_status_map = {
                    0: "UNINIT", 1: "BOOT", 2: "CALIBRATING", 3: "STANDBY",
                    4: "ACTIVE", 5: "CRITICAL", 6: "EMERGENCY", 7: "POWEROFF"
                }
                self.state["system_status"] = sys_status_map.get(msg.system_status, "ACTIVE")

            elif msg_type == 'SYS_STATUS':
                if 0 < msg.voltage_battery < 65535:
                    self.state["battery_voltage"] = round(msg.voltage_battery / 1000.0, 2)
                if msg.current_battery >= 0:
                    self.state["battery_current"] = round(msg.current_battery / 100.0, 1)
                if 0 <= msg.battery_remaining <= 100:
                    self.state["battery_remaining"] = msg.battery_remaining

            elif msg_type == 'BATTERY_STATUS':
                raw_cells = list(getattr(msg, "voltages", []) or [])[:6]
                valid_cells = []
                for value in raw_cells:
                    valid_cells.append(round(value / 1000.0, 3) if 500 <= value < 65535 else 0.0)
                valid_cells.extend([0.0] * (6 - len(valid_cells)))
                self.state["cell_voltages"] = valid_cells
                active_cells = [cell for cell in valid_cells if cell > 1.0]
                if active_cells:
                    self.state["cell_delta_mv"] = int(round((max(active_cells) - min(active_cells)) * 1000))
                    if self.state["battery_voltage"] <= 0.5:
                        self.state["battery_voltage"] = round(sum(active_cells), 2)
                if msg.current_battery >= 0:
                    self.state["battery_current"] = round(msg.current_battery / 100.0, 1)
                if 0 <= msg.battery_remaining <= 100:
                    self.state["battery_remaining"] = msg.battery_remaining

            elif msg_type == 'GLOBAL_POSITION_INT':
                self.state["latitude"] = round(msg.lat / 1e7, 7)
                self.state["longitude"] = round(msg.lon / 1e7, 7)
                self.state["altitude_relative"] = round(msg.relative_alt / 1000.0, 2)
                self.state["altitude_msl"] = round(msg.alt / 1000.0, 2)
                heading = self._heading(msg.hdg, 100.0) if msg.hdg != 65535 else None
                if heading is not None:
                    self.state["heading"] = heading

            elif msg_type == 'GPS_RAW_INT':
                if msg.satellites_visible != 255:
                    self.state["satellites"] = msg.satellites_visible
                fix_map = {0: "NO FIX", 1: "NO FIX", 2: "2D FIX", 3: "3D FIX", 4: "DGPS", 5: "RTK FLT", 6: "RTK FIX"}
                self.state["gps_fix_type"] = fix_map.get(msg.fix_type, "NO FIX")
                if 0 < msg.eph < 65535:
                    self.state["hdop"] = round(msg.eph / 100.0, 2)
                else:
                    self.state["hdop"] = 99.9

            elif msg_type == 'VFR_HUD':
                self.state["climb_rate"] = round(msg.climb, 2)
                self.state["ground_speed"] = round(msg.groundspeed, 2)
                heading = self._heading(msg.heading)
                if heading is not None:
                    self.state["heading"] = heading
                if hasattr(msg, 'alt') and msg.alt is not None:
                    self.state["altitude_msl"] = round(msg.alt, 2)
                    if self.state["altitude_relative"] == 0.0:
                        self.state["altitude_relative"] = round(msg.alt, 2)

            elif msg_type in ('ATTITUDE', 'AHRS2', 'AHRS3'):
                if hasattr(msg, 'roll'):
                    self.state["attitude_roll"] = round(math.degrees(msg.roll), 2)
                    self.state["attitude_pitch"] = round(math.degrees(msg.pitch), 2)
                    self.state["attitude_yaw"] = round(math.degrees(msg.yaw) % 360, 2)
                if hasattr(msg, 'rollspeed'):
                    self.state["gyro_x"] = round(math.degrees(msg.rollspeed), 2)
                    self.state["gyro_y"] = round(math.degrees(msg.pitchspeed), 2)
                    self.state["gyro_z"] = round(math.degrees(msg.yawspeed), 2)
                if self.state["heading"] == 0:
                    self.state["heading"] = self.state["attitude_yaw"]
                self._update_attitude_errors()

            elif msg_type == 'ATTITUDE_TARGET':
                try:
                    q = msg.q
                    sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
                    cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
                    self.state["target_roll"] = round(math.degrees(math.atan2(sinr_cosp, cosr_cosp)), 2)
                    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
                    self.state["target_pitch"] = round(math.degrees(math.asin(max(-1.0, min(1.0, sinp)))), 2)
                    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
                    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
                    self.state["target_yaw"] = round(math.degrees(math.atan2(siny_cosp, cosy_cosp)) % 360, 2)
                    self._update_attitude_errors()
                except Exception:
                    pass

            elif msg_type == 'NAV_CONTROLLER_OUTPUT':
                self.state["target_roll"] = round(msg.nav_roll, 2)
                self.state["target_pitch"] = round(msg.nav_pitch, 2)
                bearing = self._heading(msg.target_bearing)
                if bearing is not None:
                    self.state["target_yaw"] = bearing
                if hasattr(msg, "wp_dist"):
                    self.state["dist_to_target_wp_m"] = max(0.0, round(float(msg.wp_dist), 1))
                self._update_attitude_errors()

            elif msg_type == 'HIGHRES_IMU':
                self.state["accel_x"] = round(msg.xacc, 2)
                self.state["accel_y"] = round(msg.yacc, 2)
                self.state["accel_z"] = round(msg.zacc, 2)
                self.state["gyro_x"] = round(math.degrees(msg.xgyro), 2)
                self.state["gyro_y"] = round(math.degrees(msg.ygyro), 2)
                self.state["gyro_z"] = round(math.degrees(msg.zgyro), 2)

            elif msg_type in ('RAW_IMU', 'SCALED_IMU', 'SCALED_IMU2', 'SCALED_IMU3'):
                self.state["accel_x"] = round((msg.xacc / 1000.0) * 9.81, 2)
                self.state["accel_y"] = round((msg.yacc / 1000.0) * 9.81, 2)
                self.state["accel_z"] = round((msg.zacc / 1000.0) * 9.81, 2)
                if hasattr(msg, 'xgyro'):
                    self.state["gyro_x"] = round(math.degrees(msg.xgyro / 1000.0), 2)
                    self.state["gyro_y"] = round(math.degrees(msg.ygyro / 1000.0), 2)
                    self.state["gyro_z"] = round(math.degrees(msg.zgyro / 1000.0), 2)

            elif msg_type == 'SERVO_OUTPUT_RAW':
                pwms = [msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw]
                self.state["motor_pwm"] = pwms
                self.state["motor_percent"] = [max(0, min(100, int((p - 1000) / 10.0))) for p in pwms]

            elif msg_type == 'RC_CHANNELS':
                if 0 <= msg.rssi < 255:
                    self.state["rc_rssi"] = int((msg.rssi / 254.0) * 100)

            elif msg_type == 'RADIO_STATUS':
                if 0 <= msg.rssi < 255:
                    self.state["radio_link_quality"] = int((msg.rssi / 254.0) * 100)
                if 0 <= msg.remrssi < 255 and self.state["rc_rssi"] <= 0:
                    self.state["rc_rssi"] = int((msg.remrssi / 254.0) * 100)

            elif msg_type == 'MISSION_CURRENT':
                self.state["mission_current_seq"] = msg.seq
                total = getattr(msg, "total", 0) or self.state.get("mission_total_items", 0)
                if total > 0:
                    self.state["mission_total_items"] = total
                    self.state["mission_progress_percent"] = min(100, int((msg.seq / float(total)) * 100))
                self.state["mission_state"] = f"ACTIVE (WP {msg.seq}/{total})"

            elif msg_type == 'MISSION_ITEM_REACHED':
                self.state["mission_current_seq"] = msg.seq + 1
                self.state["mission_state"] = f"REACHED WP {msg.seq}"

        return True

    def _run_live_mavlink(self):
        try:
            from pymavlink import mavutil
        except ImportError:
            print("[!] ERROR: 'pymavlink' is not installed. Run: pip install pymavlink")
            self._set_connection_state(False, "pymavlink is not installed")
            return

        ports_to_try = [self.port, "/dev/ttyACM0", "/dev/serial0", "/dev/ttyAMA0", "/dev/ttyUSB0"]

        while self.running:
            mav_conn = None
            for p in ports_to_try:
                try:
                    mav_conn = mavutil.mavlink_connection(p, baud=self.baud)
                    print(f"[+] Opened serial port '{p}' at {self.baud} baud.")
                    break
                except Exception:
                    continue

            if not mav_conn:
                print(f"[!] Waiting for Pixhawk serial connection ({self.port}). Retrying in 2s...")
                self._set_connection_state(False, f"Waiting for Pixhawk serial port {self.port}")
                time.sleep(2.0)
                continue

            self.mav = mav_conn
            self._set_connection_state(False, "Serial port open, waiting for MAVLink heartbeat")

            print("[*] Waiting for MAVLink Heartbeat from Pixhawk...")
            try:
                hb = self.mav.wait_heartbeat(timeout=8)
                if hb:
                    tgt_sys = getattr(self.mav, "target_system", 1) or 1
                    tgt_comp = getattr(self.mav, "target_component", 1) or 1
                    print(f"[✓] Heartbeat received from Pixhawk (System: {tgt_sys}, Component: {tgt_comp})")
                    self._handle_mavlink_message(hb, mavutil)
                else:
                    self._set_connection_state(False, "Serial open but no MAVLink heartbeat")
            except Exception:
                self._set_connection_state(False, "Serial open but heartbeat wait failed")

            # Activate streams and send companion heartbeat
            self._send_companion_heartbeat(mavutil)
            self._request_all_streams(mavutil)

            last_heartbeat_time = time.time()
            last_stream_request_time = time.time()

            # Live Ingestion Loop
            while self.running:
                try:
                    now = time.time()

                    # 1. Send Companion Heartbeat at 1 Hz (CRITICAL for ArduPilot to stream data!)
                    if now - last_heartbeat_time >= 1.0:
                        last_heartbeat_time = now
                        self._send_companion_heartbeat(mavutil)

                    # 2. Re-request stream keep-alive every 3s
                    if now - last_stream_request_time >= 3.0:
                        last_stream_request_time = now
                        self._request_all_streams(mavutil)

                    with self._lock:
                        last_pkt = self.state.get("last_packet_timestamp", 0.0)
                        was_connected = self.state.get("connected", False)
                    if was_connected and last_pkt > 0 and (now - last_pkt > 10.0):
                        self._set_connection_state(False, "No MAVLink packets for 10 seconds")

                    packets_read = 0
                    while packets_read < 50:
                        msg = self.mav.recv_match(blocking=False)
                        if not msg:
                            break
                        packets_read += 1
                        try:
                            self._handle_mavlink_message(msg, mavutil, now)
                        except Exception as e:
                            print(f"[!] MAVLink decode error ({msg.get_type()}): {e}")

                    time.sleep(0.005) # Yield CPU

                except Exception as e:
                    print(f"[!] MAVLink read error: {e}")
                    time.sleep(0.1)
                    break

    def _run_simulation(self):
        """Dynamic simulation loop supporting both manual flight and autonomous mission execution."""
        print("[+] Starting high-fidelity telemetry & mission autonomy simulator (10 Hz)...")
        t = 0.0
        base_lat = 12.971598
        base_lon = 77.594562
        base_voltage = 12.58

        self.sim_current_lat = base_lat
        self.sim_current_lon = base_lon
        self.sim_current_alt = 0.0
        self.sim_mission_index = 1

        EARTH_R = 6378137.0

        while self.running:
            time.sleep(0.1)
            t += 0.1

            is_auto = (self.state.get("flight_mode") == "AUTO")
            has_mission = len(self.sim_mission_items) > 0

            # Default parameters
            sim_lat = self.sim_current_lat
            sim_lon = self.sim_current_lon
            sim_alt = self.sim_current_alt
            sim_heading = self.state.get("heading", 0.0)
            roll_actual = 0.0
            pitch_actual = 0.0
            climb_rate = 0.0
            mission_state = self.state.get("mission_state", "STANDBY")
            progress_pct = self.state.get("mission_progress_percent", 0)

            # --- AUTONOMOUS MISSION FLIGHT SIMULATOR ---
            if is_auto and has_mission and self.sim_mission_index < len(self.sim_mission_items):
                item = self.sim_mission_items[self.sim_mission_index]
                cmd = item.get("command", 16)
                total_wps = len(self.sim_mission_items)
                progress_pct = min(100, int((self.sim_mission_index / float(total_wps)) * 100))

                # 1. Takeoff Command (MAV_CMD_NAV_TAKEOFF = 22)
                if cmd == 22:
                    target_alt = item.get("z", 15.0)
                    if sim_alt < target_alt - 0.2:
                        sim_alt += 0.4 # Climb at 4 m/s
                        climb_rate = 4.0
                        pitch_actual = 1.5
                        mission_state = f"TAKEOFF ({sim_alt:.1f}m / {target_alt:.0f}m)"
                    else:
                        sim_alt = target_alt
                        climb_rate = 0.0
                        self.sim_mission_index += 1
                        mission_state = f"CRUISE -> WP 1"

                # 2. Speed Command (MAV_CMD_DO_CHANGE_SPEED = 178)
                elif cmd == 178:
                    self.sim_mission_index += 1

                # 3. Waypoint Command (MAV_CMD_NAV_WAYPOINT = 16)
                elif cmd == 16:
                    target_lat = item.get("x", 0) / 1e7
                    target_lon = item.get("y", 0) / 1e7
                    target_alt = item.get("z", 15.0)

                    # Calculate distance and bearing to waypoint
                    dlat = (target_lat - sim_lat) * (math.pi / 180.0) * EARTH_R
                    dlon = (target_lon - sim_lon) * (math.pi / 180.0) * EARTH_R * math.cos(math.radians(sim_lat))
                    dist = math.sqrt(dlat**2 + dlon**2)

                    target_bearing = (math.degrees(math.atan2(dlon, dlat)) + 360.0) % 360.0
                    heading_err = (target_bearing - sim_heading + 180.0) % 360.0 - 180.0
                    sim_heading = (sim_heading + min(max(heading_err * 0.3, -12.0), 12.0)) % 360.0

                    roll_actual = round(min(max(heading_err * 0.5, -20.0), 20.0), 2)
                    pitch_actual = -3.5 # Forward acceleration pitch

                    # Step towards target at 5.0 m/s (0.5m per 0.1s tick)
                    step_m = min(0.5, dist)
                    if dist > 0.01:
                        sim_lat += (dlat / dist) * (step_m / EARTH_R) * (180.0 / math.pi)
                        sim_lon += (dlon / dist) * (step_m / (EARTH_R * math.cos(math.radians(sim_lat)))) * (180.0 / math.pi)

                    mission_state = f"NAV WP {self.sim_mission_index}/{total_wps - 1} ({dist:.1f}m)"

                    # Reached Waypoint Acceptance Radius (2.0m)
                    if dist <= 2.0:
                        self.sim_mission_index += 1

                # 4. Return to Launch / Land (MAV_CMD_NAV_RETURN_TO_LAUNCH = 20, MAV_CMD_NAV_LAND = 21)
                elif cmd in (20, 21):
                    dlat = (base_lat - sim_lat) * (math.pi / 180.0) * EARTH_R
                    dlon = (base_lon - sim_lon) * (math.pi / 180.0) * EARTH_R * math.cos(math.radians(sim_lat))
                    dist = math.sqrt(dlat**2 + dlon**2)

                    if dist > 2.0:
                        target_bearing = (math.degrees(math.atan2(dlon, dlat)) + 360.0) % 360.0
                        sim_heading = target_bearing
                        step_m = min(0.5, dist)
                        sim_lat += (dlat / dist) * (step_m / EARTH_R) * (180.0 / math.pi)
                        sim_lon += (dlon / dist) * (step_m / (EARTH_R * math.cos(math.radians(sim_lat)))) * (180.0 / math.pi)
                        mission_state = f"RTL RETURN ({dist:.1f}m)"
                    else:
                        if sim_alt > 0.3:
                            sim_alt -= 0.3 # Descend at 3 m/s
                            climb_rate = -3.0
                            mission_state = f"RTL LANDING ({sim_alt:.1f}m)"
                        else:
                            sim_alt = 0.0
                            climb_rate = 0.0
                            self.sim_mission_index += 1
                            mission_state = "MISSION COMPLETED"
                            self.state["flight_mode"] = "STABILIZE"
                            self.state["armed"] = False

                self.sim_current_lat = sim_lat
                self.sim_current_lon = sim_lon
                self.sim_current_alt = sim_alt

            elif is_auto and has_mission and self.sim_mission_index >= len(self.sim_mission_items):
                mission_state = "MISSION COMPLETED"
                progress_pct = 100

            else:
                # Normal Loiter / Telemetry Simulation Mode
                sim_alt = round(max(0.0, 15.0 + 3.0 * math.sin(t * 0.2)), 1)
                sim_lat = round(base_lat + 0.0004 * math.sin(t * 0.1), 7)
                sim_lon = round(base_lon + 0.0004 * math.cos(t * 0.1), 7)
                sim_heading = round((t * 12.0) % 360, 1)
                roll_actual = round(3.5 * math.sin(t * 0.8), 2)
                pitch_actual = round(2.0 * math.cos(t * 0.5), 2)

            gyro_x = round(5.0 * math.cos(t * 1.5), 2)
            gyro_y = round(4.0 * math.sin(t * 1.2), 2)
            gyro_z = round(2.5 * math.sin(t * 0.4), 2)

            accel_x = round(0.4 * math.sin(t * 1.5), 2)
            accel_y = round(0.3 * math.cos(t * 1.2), 2)
            accel_z = round(9.81 + 0.5 * math.sin(t * 2.0), 2)

            sim_voltage = max(10.8, round(base_voltage - (t * 0.002), 2))
            c1 = round((sim_voltage / 3.0) + 0.012, 3)
            c2 = round((sim_voltage / 3.0) - 0.008, 3)
            c3 = round((sim_voltage / 3.0) + 0.002, 3)
            cell_delta = int((max(c1, c2, c3) - min(c1, c2, c3)) * 1000)

            base_pwm = 1580
            m1 = int(base_pwm + 45 * math.sin(t * 2.0))
            m2 = int(base_pwm - 35 * math.sin(t * 2.0))
            m3 = int(base_pwm + 40 * math.cos(t * 2.0))
            m4 = int(base_pwm - 30 * math.cos(t * 2.0))

            with self._lock:
                self.state.update({
                    "battery_voltage": sim_voltage,
                    "battery_current": round(14.5 + 2.5 * math.sin(t * 0.5), 1),
                    "battery_remaining": max(10, int(88 - (t * 0.05))),
                    "cell_voltages": [c1, c2, c3, 0.0, 0.0, 0.0],
                    "cell_delta_mv": cell_delta,
                    "altitude_relative": round(sim_alt, 1),
                    "altitude_msl": round(sim_alt + 920.0, 1),
                    "climb_rate": round(climb_rate, 1),
                    "latitude": round(sim_lat, 7),
                    "longitude": round(sim_lon, 7),
                    "satellites": 16,
                    "gps_fix_type": "3D FIX",
                    "hdop": 0.85,
                    "rc_rssi": int(92 + 5 * math.sin(t * 0.1)),
                    "radio_link_quality": 98,
                    "heading": round(sim_heading, 1),
                    "ground_speed": 5.0 if is_auto and has_mission else 2.0,
                    "mission_state": mission_state,
                    "mission_progress_percent": progress_pct,
                    "mission_current_seq": self.sim_mission_index,
                    "mission_total_items": len(self.sim_mission_items),
                    "gyro_x": gyro_x,
                    "gyro_y": gyro_y,
                    "gyro_z": gyro_z,
                    "accel_x": accel_x,
                    "accel_y": accel_y,
                    "accel_z": accel_z,
                    "attitude_roll": roll_actual,
                    "attitude_pitch": pitch_actual,
                    "attitude_yaw": round(sim_heading, 1),
                    "target_roll": 0.0,
                    "target_pitch": 0.0,
                    "target_yaw": round(sim_heading, 1),
                    "error_roll": round(-roll_actual, 2),
                    "error_pitch": round(-pitch_actual, 2),
                    "error_yaw": 0.0,
                    "motor_pwm": [m1, m2, m3, m4],
                    "motor_percent": [int((m - 1000) / 10.0) for m in [m1, m2, m3, m4]],
                    "last_packet_timestamp": time.time(),
                    "last_heartbeat_timestamp": time.time(),
                    "last_message_type": "SIMULATION",
                    "mavlink_status": "Simulation telemetry running",
                    "connected": True
                })
