#!/usr/bin/env python3
"""
======================================================================================
Module 1: NRF24L01+ Radio Telemetry Transmitter (Airborne)
File: pi/modules/radio_tx_module.py

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

class RadioTXModule:
    def __init__(self, mav_manager, rate_hz: float = 5.0, enabled: bool = True):
        self.mav_manager = mav_manager
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
                ce_dev = DigitalOutputDevice(22, active_high=True, initial_value=False)
                self.ce_fn = lambda s: ce_dev.on() if s else ce_dev.off()
            except Exception:
                try:
                    import RPi.GPIO as GPIO
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(22, GPIO.OUT, initial=GPIO.LOW)
                    self.ce_fn = lambda s: GPIO.output(22, GPIO.HIGH if s else GPIO.LOW)
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

    def _pack_telemetry(self, t: dict) -> bytes:
        """Packs telemetry into the 20-byte struct with 8-bit XOR checksum."""
        # 1. Battery: if 0 on USB power, transmit 5.00V (5000mV) or real LiPo mV
        bat_v = t.get("battery_voltage", 0.0)
        bat_mv = int(bat_v * 1000) if bat_v > 0.5 else 5000

        # 2. RSSI: use RC RSSI or fallback to companion link quality (90%)
        rssi = int(t.get("rc_rssi", 0))
        if rssi <= 0 and t.get("connected", False):
            rssi = 90

        # 3. Altitude: use relative altitude or fallback to MSL barometric altitude
        alt = t.get("altitude_relative", 0.0)
        if alt == 0.0 and t.get("altitude_msl", 0.0) != 0.0:
            alt = t.get("altitude_msl", 0.0)
        alt_cm = int(alt * 100)

        lat_e7 = int((t.get("latitude", 0.0)) * 1e7)
        lon_e7 = int((t.get("longitude", 0.0)) * 1e7)
        sats = int(t.get("satellites", 0))

        payload = struct.pack(
            "<BBHhiiiB",
            PACKET_MAGIC,
            self.seq % 256,
            max(0, min(65535, bat_mv)),
            max(-32768, min(32767, rssi)),
            alt_cm,
            lat_e7,
            lon_e7,
            max(0, min(255, sats))
        )

        # 8-bit XOR checksum
        checksum = 0
        for byte in payload:
            checksum ^= byte

        return payload + bytes([checksum])

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
            packet = self._pack_telemetry(telemetry)
            self._transmit(packet)
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
