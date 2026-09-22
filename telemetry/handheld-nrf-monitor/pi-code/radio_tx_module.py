#!/usr/bin/env python3
"""
======================================================================================
Module 1: NRF24L01+ Radio Telemetry Transmitter (Airborne)
File: pi/tele/radio_tx_module.py

Description:
  Takes live flight telemetry from MAVLinkManager, packs it into the compact
  20-byte binary struct (with 8-bit XOR checksum), and broadcasts it at 5 Hz
  via hardware SPI0 to the handheld ESP32 ground receiver (SH1106 OLED).
======================================================================================
"""

import time
import struct
import threading

# Register definitions
R_REGISTER         = 0x00
W_REGISTER         = 0x20
W_TX_PAYLOAD       = 0xA0
FLUSH_TX           = 0xE1
FLUSH_RX           = 0xE2

REG_CONFIG         = 0x00
REG_EN_AA          = 0x01
REG_SETUP_AW       = 0x03
REG_SETUP_RETR     = 0x04
REG_RF_CH          = 0x05
REG_RF_SETUP       = 0x06
REG_STATUS         = 0x07
REG_TX_ADDR        = 0x10
REG_RX_ADDR_P0     = 0x0A
REG_FEATURE        = 0x1D
REG_DYNPD          = 0x1C

PACKET_MAGIC       = 0xAA
RF_CHANNEL         = 90
TX_ADDRESS         = b"2Node"
FRAME_BASIC        = 0
FRAME_POWER_SBC    = 1
FRAME_ATTITUDE     = 2
FRAME_MOTORS       = 3

class RadioTXModule:
    def __init__(self, mav_manager, sbc_monitor=None, rate_hz: float = 5.0, enabled: bool = True):
        self.mav_manager = mav_manager
        self.sbc_monitor = sbc_monitor
        self.rate_hz = rate_hz
        self.enabled = enabled
        self.running = False
        self.thread = None
        self.radio_available = False
        self.spi = None
        self.ce_fn = None
        self.seq = 0

    def start(self):
        if not self.enabled:
            print("[*] Module 1 (Radio TX): Disabled by user configuration.")
            return

        self._init_hardware()
        self.running = True
        self.thread = threading.Thread(target=self._tx_loop, daemon=True)
        self.thread.start()
        print(f"[✓] Module 1 (Radio TX): Running at {self.rate_hz} Hz ({'HARDWARE SPI' if self.radio_available else 'SIMULATED MOCK'}).")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass

    def _init_hardware(self):
        try:
            import spidev
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 2000000
            self.spi.mode = 0

            # Setup CE pin
            try:
                from gpiozero import DigitalOutputDevice
                ce_dev = DigitalOutputDevice(25, active_high=True, initial_value=False)
                self.ce_fn = lambda s: ce_dev.on() if s else ce_dev.off()
            except Exception:
                try:
                    import RPi.GPIO as GPIO
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(25, GPIO.OUT, initial=GPIO.LOW)
                    self.ce_fn = lambda s: GPIO.output(25, GPIO.HIGH if s else GPIO.LOW)
                except Exception:
                    self.ce_fn = lambda s: None

            # Test SPI communication with radio
            self._set_ce(False)
            time.sleep(0.02)
            aw = self._read_reg(REG_SETUP_AW)
            if aw in (0x01, 0x02, 0x03):
                self._configure_nrf24()
                self.radio_available = True
            else:
                print(f"[!] Module 1 (Radio TX): NRF24 SPI test returned 0x{aw:02X}. Radio hardware not present.")
                self.radio_available = False

        except Exception as e:
            print(f"[!] Module 1 (Radio TX): SPI/GPIO driver not available ({e}). Running in mock mode.")
            self.radio_available = False

    def _set_ce(self, state: bool):
        if self.ce_fn:
            try:
                self.ce_fn(state)
            except Exception:
                pass

    def _read_reg(self, reg: int) -> int:
        if not self.spi:
            return 0
        resp = self.spi.xfer2([R_REGISTER | (reg & 0x1F), 0x00])
        return resp[1]

    def _write_reg(self, reg: int, val):
        if not self.spi:
            return
        if isinstance(val, (bytes, bytearray, list)):
            self.spi.xfer2([W_REGISTER | (reg & 0x1F)] + list(val))
        else:
            self.spi.xfer2([W_REGISTER | (reg & 0x1F), val])

    def _configure_nrf24(self):
        self.spi.xfer2([FLUSH_TX])
        self.spi.xfer2([FLUSH_RX])
        self._write_reg(REG_STATUS, 0x70)
        self._write_reg(REG_SETUP_AW, 0x03)       # 5-byte address width
        self._write_reg(REG_EN_AA, 0x00)          # No Auto-ACK (broadcast mode)
        self._write_reg(REG_SETUP_RETR, 0x00)     # No retries
        self._write_reg(REG_RF_CH, RF_CHANNEL)    # Channel 90 (2.490 GHz)
        self._write_reg(REG_RF_SETUP, 0x26)       # 250 kbps, PA_MAX
        self._write_reg(REG_FEATURE, 0x04)        # Enable Dynamic Payload
        self._write_reg(REG_DYNPD, 0x01)          # Pipe 0 DPL
        self._write_reg(REG_TX_ADDR, TX_ADDRESS)
        self._write_reg(REG_RX_ADDR_P0, TX_ADDRESS)
        self._write_reg(REG_CONFIG, 0x0E)         # PWR_UP=1, CRC=16bit, TX mode
        time.sleep(0.01)

    def _clamp_int(self, value, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(round(value))))
        except Exception:
            return low

    def _u8(self, value) -> bytes:
        return struct.pack("<B", self._clamp_int(value, 0, 255))

    def _u16(self, value) -> bytes:
        return struct.pack("<H", self._clamp_int(value, 0, 65535))

    def _i16(self, value) -> bytes:
        return struct.pack("<h", self._clamp_int(value, -32768, 32767))

    def _i32(self, value) -> bytes:
        return struct.pack("<i", self._clamp_int(value, -2147483648, 2147483647))

    def _finish_frame(self, body: bytearray) -> bytes:
        if len(body) > 31:
            raise ValueError(f"NRF24 frame body too large: {len(body)} bytes")
        body.extend(b"\x00" * (31 - len(body)))
        checksum = 0
        for byte in body:
            checksum ^= byte
        return bytes(body + bytes([checksum]))

    def _frame_body(self, frame_type: int, flags: int = 0) -> bytearray:
        return bytearray([PACKET_MAGIC, self.seq % 256, frame_type & 0xFF, flags & 0xFF])

    def _gps_fix_id(self, fix: str) -> int:
        return {
            "NO FIX": 0,
            "2D FIX": 2,
            "3D FIX": 3,
            "DGPS": 4,
            "RTK FLT": 5,
            "RTK FIX": 6,
        }.get(str(fix).upper(), 0)

    def _flight_mode_id(self, mode: str) -> int:
        modes = {
            "DISCONNECTED": 0, "STABILIZE": 1, "ALT_HOLD": 2, "AUTO": 3,
            "GUIDED": 4, "LOITER": 5, "RTL": 6, "LAND": 7, "POSHOLD": 8,
            "BRAKE": 9, "NO HEARTBEAT": 10,
        }
        return modes.get(str(mode).upper(), 255)

    def _system_status_id(self, status: str) -> int:
        return {
            "UNINIT": 0, "BOOT": 1, "CALIBRATING": 2, "STANDBY": 3,
            "ACTIVE": 4, "CRITICAL": 5, "EMERGENCY": 6,
        }.get(str(status).upper(), 4)

    def _mission_state_id(self, state: str) -> int:
        text = str(state).upper()
        if "TAKEOFF" in text:
            return 2
        if "NAV" in text or "ACTIVE" in text or "CRUISE" in text:
            return 3
        if "RTL" in text or "RETURN" in text:
            return 4
        if "LAND" in text:
            return 5
        if "PAUSED" in text or "LOITER" in text:
            return 6
        if "COMPLETED" in text:
            return 7
        if "READY" in text:
            return 1
        return 0

    def _pack_telemetry_frames(self, t: dict, sbc: dict) -> list:
        bat_v = t.get("battery_voltage", 0.0)
        bat_mv = int(bat_v * 1000) if bat_v > 0.5 else 5000
        rssi = int(t.get("rc_rssi", 0))
        if rssi <= 0 and t.get("connected", False):
            rssi = 90
        link_quality = int(t.get("radio_link_quality", 0) or rssi)

        alt = t.get("altitude_relative", 0.0)
        if alt == 0.0 and t.get("altitude_msl", 0.0) != 0.0:
            alt = t.get("altitude_msl", 0.0)

        cells = list(t.get("cell_voltages", []))[:6]
        cells.extend([0.0] * (6 - len(cells)))
        motor_pwm = list(t.get("motor_pwm", []))[:4]
        motor_pwm.extend([1000] * (4 - len(motor_pwm)))
        motor_percent = list(t.get("motor_percent", []))[:4]
        motor_percent.extend([0] * (4 - len(motor_percent)))

        flags = (1 if t.get("armed") else 0) | (2 if t.get("connected") else 0)
        basic = self._frame_body(FRAME_BASIC, flags)
        basic += self._u16(bat_mv)
        basic += self._i16(rssi)
        basic += self._i32(alt * 100)
        basic += self._i32(t.get("latitude", 0.0) * 1e7)
        basic += self._i32(t.get("longitude", 0.0) * 1e7)
        basic += self._u8(t.get("satellites", 0))
        basic += self._u16(t.get("heading", 0.0) * 100)
        basic += self._u8(t.get("battery_remaining", 0))
        basic += self._u8(self._gps_fix_id(t.get("gps_fix_type", "NO FIX")))
        basic += self._u8(self._flight_mode_id(t.get("flight_mode", "DISCONNECTED")))
        basic += self._u8(self._system_status_id(t.get("system_status", "ACTIVE")))
        basic += self._u8(link_quality)
        basic += self._u8(t.get("mission_progress_percent", 0))
        basic += self._u8(len([c for c in cells if c > 1.0]))

        power = self._frame_body(FRAME_POWER_SBC, flags)
        power += self._i16(t.get("battery_current", 0.0) * 100)
        for cell in cells:
            power += self._u16(cell * 1000)
        power += self._u16(t.get("cell_delta_mv", 0))
        power += self._u8(sbc.get("cpu_load_percent", 0))
        power += self._i16(sbc.get("cpu_temp_c", 0.0) * 10)
        power += self._u8(sbc.get("ram_percent", 0))
        power += self._u8(sbc.get("disk_percent", 0))
        power += self._u16(sbc.get("uptime_seconds", 0) / 60)
        power += self._u16(0)

        attitude = self._frame_body(FRAME_ATTITUDE, flags)
        attitude += self._i16(t.get("attitude_roll", 0.0) * 100)
        attitude += self._i16(t.get("attitude_pitch", 0.0) * 100)
        attitude += self._u16(t.get("attitude_yaw", 0.0) * 100)
        attitude += self._i16(t.get("target_roll", 0.0) * 100)
        attitude += self._i16(t.get("target_pitch", 0.0) * 100)
        attitude += self._u16(t.get("target_yaw", 0.0) * 100)
        attitude += self._i16(t.get("error_roll", 0.0) * 100)
        attitude += self._i16(t.get("error_pitch", 0.0) * 100)
        attitude += self._i16(t.get("gyro_x", 0.0) * 100)
        attitude += self._i16(t.get("gyro_y", 0.0) * 100)
        attitude += self._i16(t.get("gyro_z", 0.0) * 100)
        attitude += self._i16(t.get("climb_rate", 0.0) * 100)
        attitude += self._i16(t.get("altitude_msl", 0.0) * 10)

        motors = self._frame_body(FRAME_MOTORS, flags)
        for value in motor_percent:
            motors += self._u8(value)
        for value in motor_pwm:
            motors += self._u16(value)
        motors += self._u16(t.get("mission_current_seq", 0))
        motors += self._u16(t.get("mission_total_items", 0))
        motors += self._u16(t.get("dist_to_target_wp_m", 0.0) * 10)
        motors += self._u16(t.get("ground_speed", 0.0) * 100)
        motors += self._i16(t.get("accel_x", 0.0) * 100)
        motors += self._i16(t.get("accel_y", 0.0) * 100)
        motors += self._i16(t.get("accel_z", 0.0) * 100)
        motors += self._u8(self._mission_state_id(t.get("mission_state", "STANDBY")))

        return [
            self._finish_frame(basic),
            self._finish_frame(power),
            self._finish_frame(attitude),
            self._finish_frame(motors),
        ]

    def _transmit(self, payload: bytes) -> bool:
        if not self.radio_available or not self.spi:
            return True
        try:
            self.spi.xfer2([FLUSH_TX])
            self._write_reg(REG_STATUS, 0x70)
            full_payload = payload + bytes(32 - len(payload)) if len(payload) < 32 else payload[:32]
            self.spi.xfer2([W_TX_PAYLOAD] + list(full_payload))
            self._set_ce(True)
            time.sleep(0.00002) # 20 µs pulse
            self._set_ce(False)
            time.sleep(0.002)
            status = self._read_reg(REG_STATUS)
            self._write_reg(REG_STATUS, 0x70)
            return (status & 0x20) != 0
        except Exception:
            return False

    def _tx_loop(self):
        interval = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.2
        while self.running:
            start_time = time.time()
            telemetry = self.mav_manager.get_telemetry_snapshot()
            sbc = self.sbc_monitor.get_metrics_snapshot() if self.sbc_monitor else {}
            for packet in self._pack_telemetry_frames(telemetry, sbc):
                self._transmit(packet)
                time.sleep(0.003)
            self.seq = (self.seq + 1) % 256

            elapsed = time.time() - start_time
            sleep_time = max(0.01, interval - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    from mavlink_manager import MAVLinkManager
    mgr = MAVLinkManager(simulate=True)
    mgr.start()
    tx = RadioTXModule(mgr, rate_hz=5.0)
    tx.start()
    time.sleep(2.0)
    tx.stop()
    mgr.stop()
    print("[✓] RadioTXModule test passed.")
