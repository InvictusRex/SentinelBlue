/*
 * ======================================================================================
 * Project: ESP32 Drone Telemetry Receiver (Ground Unit)
 * File: esp/esp.ino
 *
 * Hardware Modules:
 *   - ESP32 Development Board (ESP-WROOM-32)
 *   - 1.3-inch I2C OLED Display (JMD1.3A / SH1106 Driver, 128x64)
 *   - NRF24L01+ PA + LNA Transceiver Module (with external antenna)
 *   - HW-200 3.3V Base Adapter Board (powered via ESP32 VIN/5V)
 *
 * Hardware Pinout Connections:
 *   [NRF24L01+ via HW-200 Adapter -> ESP32 Hardware VSPI]
 *     - VCC  -> ESP32 VIN (5V) [Regulated to stable 3.3V by HW-200 AMS1117]
 *     - GND  -> ESP32 GND
 *     - CE   -> GPIO 4
 *     - CSN  -> GPIO 5
 *     - SCK  -> GPIO 18 (VSPI SCK)
 *     - MOSI -> GPIO 23 (VSPI MOSI)
 *     - MISO -> GPIO 19 (VSPI MISO)
 *
 *   [1.3" SH1106 OLED Display -> ESP32 Hardware I2C]
 *     - VCC  -> ESP32 3.3V (or 5V)
 *     - GND  -> ESP32 GND
 *     - SCL  -> GPIO 22 (I2C Clock)
 *     - SDA  -> GPIO 21 (I2C Data)
 *
 * Serial Monitor Baud Rate: 115200
 * ======================================================================================
 */

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <RF24.h>
#include <U8g2lib.h>

// ======================================================================================
// PIN DEFINITIONS
// ======================================================================================
#define PIN_NRF_CE    4
#define PIN_NRF_CSN   5
#define PIN_NRF_SCK   18
#define PIN_NRF_MISO  19
#define PIN_NRF_MOSI  23

#define PIN_OLED_SDA  21
#define PIN_OLED_SCL  22

// ======================================================================================
// TELEMETRY PACKET STRUCTURE (20 Bytes Packed Binary Struct)
// ======================================================================================
#define PACKET_MAGIC 0xAA
#define FRAME_BASIC 0
#define FRAME_POWER_SBC 1
#define FRAME_ATTITUDE 2
#define FRAME_MOTORS 3

const char* WIFI_STA_SSID = "gamma";
const char* WIFI_STA_PASSWORD = "gammared";
const char* WIFI_HOSTNAME = "drone-esp32";
const char* WIFI_FALLBACK_AP_SSID = "DroneTelemetryESP32";
const char* WIFI_FALLBACK_AP_PASSWORD = "drone12345";
const uint16_t HTTP_PORT = 80;

struct __attribute__((packed)) TelemetryPacket {
  uint8_t  magic;       // Byte 0:     0xAA Sync Header
  uint8_t  seq;         // Byte 1:     Packet sequence counter (0-255)
  uint16_t bat_mv;      // Bytes 2-3:  Battery Voltage in millivolts (e.g. 12580 = 12.58V)
  int16_t  rssi;        // Bytes 4-5:  FlySky Receiver RSSI (e.g. 85 % or -65 dBm)
  int32_t  alt_cm;      // Bytes 6-9:  Altitude in centimeters (e.g. 4520 = 45.20m)
  int32_t  lat_e7;      // Bytes 10-13: Latitude * 10^7 (e.g. 123456780 = 12.3456780 deg)
  int32_t  lon_e7;      // Bytes 14-17: Longitude * 10^7 (e.g. 771234560 = 77.1234560 deg)
  uint8_t  satellites;  // Byte 18:    GPS Satellite count (e.g. 14)
  uint8_t  checksum;    // Byte 19:    8-bit XOR Checksum
};

struct __attribute__((packed)) BasicFrame {
  uint8_t magic;
  uint8_t seq;
  uint8_t frameType;
  uint8_t flags;
  uint16_t bat_mv;
  int16_t rssi;
  int32_t alt_cm;
  int32_t lat_e7;
  int32_t lon_e7;
  uint8_t satellites;
  uint16_t heading_cd;
  uint8_t battery_remaining;
  uint8_t gps_fix;
  uint8_t flight_mode;
  uint8_t system_status;
  uint8_t link_quality;
  uint8_t mission_progress;
  uint8_t cell_count;
  uint8_t pad;
  uint8_t checksum;
};

struct __attribute__((packed)) PowerSbcFrame {
  uint8_t magic;
  uint8_t seq;
  uint8_t frameType;
  uint8_t flags;
  int16_t battery_current_ca;
  uint16_t cell_mv[6];
  uint16_t cell_delta_mv;
  uint8_t cpu_load_percent;
  int16_t cpu_temp_dc;
  uint8_t ram_percent;
  uint8_t disk_percent;
  uint16_t uptime_minutes;
  uint16_t reserved;
  uint8_t checksum;
};

struct __attribute__((packed)) AttitudeFrame {
  uint8_t magic;
  uint8_t seq;
  uint8_t frameType;
  uint8_t flags;
  int16_t roll_cd;
  int16_t pitch_cd;
  uint16_t yaw_cd;
  int16_t target_roll_cd;
  int16_t target_pitch_cd;
  uint16_t target_yaw_cd;
  int16_t error_roll_cd;
  int16_t error_pitch_cd;
  int16_t gyro_x_cdps;
  int16_t gyro_y_cdps;
  int16_t gyro_z_cdps;
  int16_t climb_cms;
  int16_t altitude_msl_dm;
  uint8_t pad;
  uint8_t checksum;
};

struct __attribute__((packed)) MotorsFrame {
  uint8_t magic;
  uint8_t seq;
  uint8_t frameType;
  uint8_t flags;
  uint8_t motor_percent[4];
  uint16_t motor_pwm[4];
  uint16_t mission_current_seq;
  uint16_t mission_total_items;
  uint16_t dist_to_target_wp_dm;
  uint16_t ground_speed_cms;
  int16_t accel_x_centi;
  int16_t accel_y_centi;
  int16_t accel_z_centi;
  uint8_t mission_state;
  uint8_t checksum;
};

// ======================================================================================
// DISPLAY & RADIO INSTANTIATION
// ======================================================================================
// SH1106 128x64 Full Buffer Hardware I2C (Address 0x3C)
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

// NRF24L01 Radio on VSPI (CE=4, CSN=5)
RF24 radio(PIN_NRF_CE, PIN_NRF_CSN);
WebServer server(HTTP_PORT);

// Radio Configuration
const byte rfAddresses[][6] = {"1Node", "2Node"};
const uint8_t RF_CHANNEL = 90; // 2.490 GHz

// ======================================================================================
// STATE VARIABLES
// ======================================================================================
float g_batVoltage = 0.0;    // Volts
float g_batteryCurrent = 0.0;
uint8_t g_batteryRemaining = 0;
float g_cellVoltages[6] = {0, 0, 0, 0, 0, 0};
uint16_t g_cellDeltaMv = 0;
uint8_t g_cellCount = 0;
int16_t g_rssiVal = 0;       // %
uint8_t g_linkQuality = 0;
float g_altitude = 0.0;      // Meters
float g_altitudeMsl = 0.0;
float g_climbRate = 0.0;
double g_latitude = 0.0;     // Degrees
double g_longitude = 0.0;    // Degrees
uint8_t g_satellites = 0;    // Count
uint8_t g_gpsFix = 0;
float g_heading = 0.0;
float g_groundSpeed = 0.0;
uint8_t g_lastSeq = 0;       // Sequence
uint8_t g_flightMode = 0;
uint8_t g_systemStatus = 4;
uint8_t g_missionState = 0;
uint8_t g_missionProgress = 0;
uint16_t g_missionCurrentSeq = 0;
uint16_t g_missionTotalItems = 0;
bool g_armed = false;
bool g_piMavlinkConnected = false;
float g_roll = 0.0;
float g_pitch = 0.0;
float g_yaw = 0.0;
float g_targetRoll = 0.0;
float g_targetPitch = 0.0;
float g_targetYaw = 0.0;
float g_errorRoll = 0.0;
float g_errorPitch = 0.0;
float g_gyroX = 0.0;
float g_gyroY = 0.0;
float g_gyroZ = 0.0;
float g_accelX = 0.0;
float g_accelY = 0.0;
float g_accelZ = 0.0;
uint8_t g_motorPercent[4] = {0, 0, 0, 0};
uint16_t g_motorPwm[4] = {1000, 1000, 1000, 1000};
uint8_t g_cpuLoad = 0;
float g_cpuTemp = 0.0;
uint8_t g_ramPercent = 0;
uint8_t g_diskPercent = 0;
uint32_t g_sbcUptime = 0;

bool g_hasReceivedData = false;
bool g_heartbeatState = false;
bool g_radioHardwareOk = false;
bool g_wifiApMode = false;
String g_wifiIpText = "0.0.0.0";
unsigned long g_lastPacketTime = 0;
const unsigned long DISCONNECT_TIMEOUT_MS = 10000; // 10 seconds timeout

// Display Refresh Timer
unsigned long g_lastDisplayRefresh = 0;
const unsigned long DISPLAY_REFRESH_INTERVAL_MS = 60; // ~16 FPS

// ======================================================================================
// CHECKSUM HELPER
// ======================================================================================
uint8_t calculateChecksum(const uint8_t* buffer, size_t length) {
  uint8_t c = 0;
  for (size_t i = 0; i < length; i++) {
    c ^= buffer[i];
  }
  return c;
}

bool validateFrameChecksum(const uint8_t* buffer, size_t size) {
  if (size != 32) return false;
  return buffer[31] == calculateChecksum(buffer, 31);
}

const char* gpsFixLabel(uint8_t value) {
  switch (value) {
    case 2: return "2D FIX";
    case 3: return "3D FIX";
    case 4: return "DGPS";
    case 5: return "RTK FLT";
    case 6: return "RTK FIX";
    default: return "NO FIX";
  }
}

const char* flightModeLabel(uint8_t value) {
  switch (value) {
    case 1: return "STABILIZE";
    case 2: return "ALT_HOLD";
    case 3: return "AUTO";
    case 4: return "GUIDED";
    case 5: return "LOITER";
    case 6: return "RTL";
    case 7: return "LAND";
    case 8: return "POSHOLD";
    case 9: return "BRAKE";
    case 10: return "NO HEARTBEAT";
    case 255: return "CUSTOM";
    default: return "DISCONNECTED";
  }
}

const char* systemStatusLabel(uint8_t value) {
  switch (value) {
    case 0: return "UNINIT";
    case 1: return "BOOT";
    case 2: return "CALIBRATING";
    case 3: return "STANDBY";
    case 5: return "CRITICAL";
    case 6: return "EMERGENCY";
    default: return "ACTIVE";
  }
}

const char* missionStateLabel(uint8_t value) {
  switch (value) {
    case 1: return "READY";
    case 2: return "TAKEOFF";
    case 3: return "WAYPOINT NAV";
    case 4: return "RTL";
    case 5: return "LANDING";
    case 6: return "PAUSED";
    case 7: return "MISSION COMPLETE";
    default: return "STANDBY";
  }
}

// ======================================================================================
// PACKET PROCESSORS
// ======================================================================================
bool processTelemetryFrame(const uint8_t* buffer, size_t size, bool* shouldPrint) {
  if (!validateFrameChecksum(buffer, size)) return false;

  uint8_t frameType = buffer[2];
  *shouldPrint = false;

  if (frameType == FRAME_BASIC) {
    const BasicFrame* frame = (const BasicFrame*)buffer;
    g_batVoltage = frame->bat_mv / 1000.0f;
    g_rssiVal = frame->rssi;
    g_altitude = frame->alt_cm / 100.0f;
    g_latitude = frame->lat_e7 / 10000000.0;
    g_longitude = frame->lon_e7 / 10000000.0;
    g_satellites = frame->satellites;
    g_heading = frame->heading_cd / 100.0f;
    g_batteryRemaining = frame->battery_remaining;
    g_gpsFix = frame->gps_fix;
    g_flightMode = frame->flight_mode;
    g_systemStatus = frame->system_status;
    g_linkQuality = frame->link_quality;
    g_missionProgress = frame->mission_progress;
    g_cellCount = frame->cell_count;
    g_lastSeq = frame->seq;
    g_armed = (frame->flags & 0x01) != 0;
    g_piMavlinkConnected = (frame->flags & 0x02) != 0;
    *shouldPrint = true;
    return true;
  }

  if (frameType == FRAME_POWER_SBC) {
    const PowerSbcFrame* frame = (const PowerSbcFrame*)buffer;
    g_batteryCurrent = frame->battery_current_ca / 100.0f;
    for (uint8_t i = 0; i < 6; i++) {
      g_cellVoltages[i] = frame->cell_mv[i] / 1000.0f;
    }
    g_cellDeltaMv = frame->cell_delta_mv;
    g_cpuLoad = frame->cpu_load_percent;
    g_cpuTemp = frame->cpu_temp_dc / 10.0f;
    g_ramPercent = frame->ram_percent;
    g_diskPercent = frame->disk_percent;
    g_sbcUptime = (uint32_t)frame->uptime_minutes * 60UL;
    return true;
  }

  if (frameType == FRAME_ATTITUDE) {
    const AttitudeFrame* frame = (const AttitudeFrame*)buffer;
    g_roll = frame->roll_cd / 100.0f;
    g_pitch = frame->pitch_cd / 100.0f;
    g_yaw = frame->yaw_cd / 100.0f;
    g_targetRoll = frame->target_roll_cd / 100.0f;
    g_targetPitch = frame->target_pitch_cd / 100.0f;
    g_targetYaw = frame->target_yaw_cd / 100.0f;
    g_errorRoll = frame->error_roll_cd / 100.0f;
    g_errorPitch = frame->error_pitch_cd / 100.0f;
    g_gyroX = frame->gyro_x_cdps / 100.0f;
    g_gyroY = frame->gyro_y_cdps / 100.0f;
    g_gyroZ = frame->gyro_z_cdps / 100.0f;
    g_climbRate = frame->climb_cms / 100.0f;
    g_altitudeMsl = frame->altitude_msl_dm / 10.0f;
    return true;
  }

  if (frameType == FRAME_MOTORS) {
    const MotorsFrame* frame = (const MotorsFrame*)buffer;
    for (uint8_t i = 0; i < 4; i++) {
      g_motorPercent[i] = frame->motor_percent[i];
      g_motorPwm[i] = frame->motor_pwm[i];
    }
    g_missionCurrentSeq = frame->mission_current_seq;
    g_missionTotalItems = frame->mission_total_items;
    g_groundSpeed = frame->ground_speed_cms / 100.0f;
    g_accelX = frame->accel_x_centi / 100.0f;
    g_accelY = frame->accel_y_centi / 100.0f;
    g_accelZ = frame->accel_z_centi / 100.0f;
    g_missionState = frame->mission_state;
    return true;
  }

  return false;
}

bool processBinaryPacket(const uint8_t* buffer, size_t size) {
  if (size < sizeof(TelemetryPacket)) return false;

  const TelemetryPacket* pkt = (const TelemetryPacket*)buffer;
  if (pkt->magic != PACKET_MAGIC) return false;

  // Validate XOR Checksum over bytes 0 to 18
  uint8_t expectedChecksum = calculateChecksum(buffer, sizeof(TelemetryPacket) - 1);
  if (pkt->checksum != expectedChecksum) {
    Serial.println(F("[NRF24] Corrupted packet dropped (Checksum mismatch)."));
    return false;
  }

  // Unpack fixed-point integers into engineering units
  g_batVoltage = pkt->bat_mv / 1000.0f;
  g_rssiVal    = pkt->rssi;
  g_altitude   = pkt->alt_cm / 100.0f;
  g_latitude   = pkt->lat_e7 / 10000000.0;
  g_longitude  = pkt->lon_e7 / 10000000.0;
  g_satellites = pkt->satellites;
  g_lastSeq    = pkt->seq;
  g_piMavlinkConnected = true;

  return true;
}

// Fallback CSV Parser
bool processCsvPacket(const char* text) {
  char temp[33];
  strncpy(temp, text, sizeof(temp));
  temp[sizeof(temp) - 1] = '\0';

  char* token = strtok(temp, ",");
  if (!token) return false;
  g_batVoltage = atof(token);

  token = strtok(NULL, ",");
  if (token) g_rssiVal = atoi(token);

  token = strtok(NULL, ",");
  if (token) g_altitude = atof(token);

  token = strtok(NULL, ",");
  if (token) g_latitude = atof(token);

  token = strtok(NULL, ",");
  if (token) g_longitude = atof(token);

  return true;
}

// ======================================================================================
// OLED DRAWING FUNCTIONS
// ======================================================================================
void drawHardwareErrorScreen() {
  u8g2.clearBuffer();
  u8g2.drawRFrame(0, 0, 128, 64, 3);

  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.drawBox(2, 2, 124, 15);
  u8g2.setDrawColor(0);
  u8g2.setCursor(12, 13);
  u8g2.print(F("! HARDWARE ERROR !"));
  u8g2.setDrawColor(1);

  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.setCursor(14, 34);
  u8g2.print(F("NRF24 Not Found"));

  u8g2.setFont(u8g2_font_5x8_tr);
  u8g2.setCursor(12, 50);
  u8g2.print(F("Check SPI & 5V Power"));
  u8g2.sendBuffer();
}

void drawDisconnectScreen(unsigned long elapsedMs) {
  u8g2.clearBuffer();

  // Outer rounded border frame
  u8g2.drawRFrame(0, 0, 128, 64, 3);

  // Warning Header Banner
  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.drawBox(2, 2, 124, 15);
  u8g2.setDrawColor(0);
  u8g2.setCursor(14, 13);
  u8g2.print(F("! NO CONNECTION !"));
  u8g2.setDrawColor(1);

  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.setCursor(16, 32);
  u8g2.print(F("Telemetry Lost"));

  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(10, 46);
  if (!g_hasReceivedData) {
    u8g2.print(F("Waiting NRF link"));
  } else {
    unsigned long elapsedSec = elapsedMs / 1000;
    u8g2.print(F("Lost "));
    u8g2.print(elapsedSec);
    u8g2.print(F("s"));
  }

  u8g2.setCursor(10, 59);
  u8g2.print(F("IP:"));
  u8g2.print(g_wifiIpText);

  u8g2.sendBuffer();
}

void drawWifiStatusScreen(const __FlashStringHelper* title, const char* ssid, const String& ipText) {
  u8g2.clearBuffer();
  u8g2.drawRFrame(0, 0, 128, 64, 4);
  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(10, 14);
  u8g2.print(title);
  u8g2.setCursor(10, 30);
  u8g2.print(F("SSID:"));
  u8g2.print(ssid);
  u8g2.setCursor(10, 46);
  u8g2.print(F("IP:"));
  u8g2.print(ipText);
  u8g2.setCursor(10, 59);
  u8g2.print(F("/telemetry.json"));
  u8g2.sendBuffer();
}

void drawTelemetryScreen() {
  u8g2.clearBuffer();

  // LEVEL 1: Battery Voltage & Satellites / Heartbeat Indicator
  u8g2.setFont(u8g2_font_7x14B_tr);
  u8g2.setCursor(2, 13);
  u8g2.print(F("BAT:"));
  u8g2.print(g_batVoltage, 2);
  u8g2.print(F("V"));

  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(76, 12);
  u8g2.print(F("SAT:"));
  u8g2.print(g_satellites);

  // Heartbeat Dot
  if (g_heartbeatState) {
    u8g2.drawDisc(122, 9, 3);
  } else {
    u8g2.drawCircle(122, 9, 3);
  }

  u8g2.drawHLine(0, 16, 128);

  // LEVEL 2: FlySky RSSI & Altitude in Meters
  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(2, 29);
  u8g2.print(F("RSSI:"));
  u8g2.print(g_rssiVal);
  u8g2.print(F("%"));

  u8g2.setCursor(68, 29);
  u8g2.print(F("ALT:"));
  u8g2.print(g_altitude, 1);
  u8g2.print(F("m"));

  u8g2.drawHLine(0, 33, 128);

  // LEVEL 3: High-Precision GPS Coordinates
  u8g2.setFont(u8g2_font_5x8_tr);

  u8g2.setCursor(2, 45);
  u8g2.print(F("LAT: "));
  if (g_latitude != 0.0) {
    u8g2.print(g_latitude, 6);
  } else {
    u8g2.print(F("NO FIX"));
  }

  u8g2.setCursor(2, 58);
  u8g2.print(F("LON: "));
  if (g_longitude != 0.0) {
    u8g2.print(g_longitude, 6);
  } else {
    u8g2.print(F("NO FIX"));
  }

  u8g2.sendBuffer();
}

void addCorsHeaders() {
  server.sendHeader(F("Access-Control-Allow-Origin"), F("*"));
  server.sendHeader(F("Access-Control-Allow-Methods"), F("GET, OPTIONS"));
  server.sendHeader(F("Access-Control-Allow-Headers"), F("Content-Type"));
}

String makeTelemetryJson() {
  bool radioFresh = g_hasReceivedData && (millis() - g_lastPacketTime < DISCONNECT_TIMEOUT_MS);
  String json;
  json.reserve(1800);

  json += F("{\"source\":\"esp32-nrf\",\"updated_ms\":");
  json += millis();
  json += F(",\"connected\":");
  json += (radioFresh ? F("true") : F("false"));
  json += F(",\"mavlink_connected\":");
  json += (g_piMavlinkConnected ? F("true") : F("false"));
  json += F(",\"seq\":");
  json += String((int)g_lastSeq);
  json += F(",\"battery_voltage\":");
  json += String(g_batVoltage, 2);
  json += F(",\"battery_current\":");
  json += String(g_batteryCurrent, 2);
  json += F(",\"battery_remaining\":");
  json += String((int)g_batteryRemaining);
  json += F(",\"cell_voltages\":[");
  for (uint8_t i = 0; i < 6; i++) {
    if (i) json += ',';
    json += String(g_cellVoltages[i], 3);
  }
  json += F("],\"cell_delta_mv\":");
  json += g_cellDeltaMv;
  json += F(",\"altitude_relative\":");
  json += String(g_altitude, 2);
  json += F(",\"altitude_msl\":");
  json += String(g_altitudeMsl, 2);
  json += F(",\"climb_rate\":");
  json += String(g_climbRate, 2);
  json += F(",\"latitude\":");
  json += String(g_latitude, 7);
  json += F(",\"longitude\":");
  json += String(g_longitude, 7);
  json += F(",\"satellites\":");
  json += String((int)g_satellites);
  json += F(",\"gps_fix_type\":\"");
  json += gpsFixLabel(g_gpsFix);
  json += F("\",\"rc_rssi\":");
  json += g_rssiVal;
  json += F(",\"radio_link_quality\":");
  json += String((int)g_linkQuality);
  json += F(",\"heading\":");
  json += String(g_heading, 2);
  json += F(",\"ground_speed\":");
  json += String(g_groundSpeed, 2);
  json += F(",\"armed\":");
  json += (g_armed ? F("true") : F("false"));
  json += F(",\"flight_mode\":\"");
  json += flightModeLabel(g_flightMode);
  json += F("\",\"system_status\":\"");
  json += systemStatusLabel(g_systemStatus);
  json += F("\",\"mission_state\":\"");
  json += missionStateLabel(g_missionState);
  json += F("\",\"mission_progress_percent\":");
  json += String((int)g_missionProgress);
  json += F(",\"mission_current_seq\":");
  json += g_missionCurrentSeq;
  json += F(",\"mission_total_items\":");
  json += g_missionTotalItems;
  json += F(",\"attitude_roll\":");
  json += String(g_roll, 2);
  json += F(",\"attitude_pitch\":");
  json += String(g_pitch, 2);
  json += F(",\"attitude_yaw\":");
  json += String(g_yaw, 2);
  json += F(",\"target_roll\":");
  json += String(g_targetRoll, 2);
  json += F(",\"target_pitch\":");
  json += String(g_targetPitch, 2);
  json += F(",\"target_yaw\":");
  json += String(g_targetYaw, 2);
  json += F(",\"error_roll\":");
  json += String(g_errorRoll, 2);
  json += F(",\"error_pitch\":");
  json += String(g_errorPitch, 2);
  json += F(",\"gyro_x\":");
  json += String(g_gyroX, 2);
  json += F(",\"gyro_y\":");
  json += String(g_gyroY, 2);
  json += F(",\"gyro_z\":");
  json += String(g_gyroZ, 2);
  json += F(",\"accel_x\":");
  json += String(g_accelX, 2);
  json += F(",\"accel_y\":");
  json += String(g_accelY, 2);
  json += F(",\"accel_z\":");
  json += String(g_accelZ, 2);
  json += F(",\"motor_percent\":[");
  for (uint8_t i = 0; i < 4; i++) {
    if (i) json += ',';
    json += String((int)g_motorPercent[i]);
  }
  json += F("],\"motor_pwm\":[");
  for (uint8_t i = 0; i < 4; i++) {
    if (i) json += ',';
    json += g_motorPwm[i];
  }
  json += F("],\"sbc\":{\"cpu_load_percent\":");
  json += String((int)g_cpuLoad);
  json += F(",\"cpu_temp_c\":");
  json += String(g_cpuTemp, 1);
  json += F(",\"ram_percent\":");
  json += String((int)g_ramPercent);
  json += F(",\"disk_percent\":");
  json += String((int)g_diskPercent);
  json += F(",\"uptime_seconds\":");
  json += g_sbcUptime;
  json += F("}}");

  return json;
}

void handleTelemetryJson() {
  addCorsHeaders();
  server.send(200, F("application/json"), makeTelemetryJson());
}

void handleRoot() {
  addCorsHeaders();
  String message = F("ESP32 drone telemetry JSON: http://");
  message += g_wifiIpText;
  message += F("/telemetry.json");
  server.send(200, F("text/plain"), message);
}

void handleOptions() {
  addCorsHeaders();
  server.send(204);
}

bool hasStationCredentials() {
  return strlen(WIFI_STA_SSID) > 0 &&
         strcmp(WIFI_STA_SSID, "YOUR_WIFI_SSID") != 0 &&
         strcmp(WIFI_STA_PASSWORD, "YOUR_WIFI_PASSWORD") != 0;
}

void startFallbackAccessPoint() {
  g_wifiApMode = true;
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_FALLBACK_AP_SSID, WIFI_FALLBACK_AP_PASSWORD);
  g_wifiIpText = WiFi.softAPIP().toString();

  Serial.print(F("[WIFI] Fallback AP started: "));
  Serial.print(WIFI_FALLBACK_AP_SSID);
  Serial.print(F(" / "));
  Serial.println(g_wifiIpText);

  drawWifiStatusScreen(F("WIFI AP MODE"), WIFI_FALLBACK_AP_SSID, g_wifiIpText);
}

void setupWifiApi() {
  g_wifiApMode = false;
  WiFi.setHostname(WIFI_HOSTNAME);

  if (hasStationCredentials()) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_STA_SSID, WIFI_STA_PASSWORD);
    drawWifiStatusScreen(F("WIFI CONNECTING"), WIFI_STA_SSID, F("..."));

    Serial.print(F("[WIFI] Connecting to "));
    Serial.println(WIFI_STA_SSID);

    unsigned long startMs = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - startMs < 15000) {
      delay(300);
      Serial.print('.');
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
      g_wifiIpText = WiFi.localIP().toString();
      Serial.print(F("[WIFI] Connected: "));
      Serial.print(WIFI_STA_SSID);
      Serial.print(F(" / "));
      Serial.println(g_wifiIpText);
      drawWifiStatusScreen(F("WIFI CONNECTED"), WIFI_STA_SSID, g_wifiIpText);
    } else {
      Serial.println(F("[WIFI] Station connection failed. Starting fallback AP."));
      startFallbackAccessPoint();
    }
  } else {
    Serial.println(F("[WIFI] Station credentials not set. Starting fallback AP."));
    startFallbackAccessPoint();
  }

  server.on("/", HTTP_GET, handleRoot);
  server.on("/telemetry.json", HTTP_GET, handleTelemetryJson);
  server.on("/api/telemetry", HTTP_GET, handleTelemetryJson);
  server.on("/telemetry.json", HTTP_OPTIONS, handleOptions);
  server.on("/api/telemetry", HTTP_OPTIONS, handleOptions);
  server.onNotFound(handleRoot);
  server.begin();

  if (MDNS.begin(WIFI_HOSTNAME)) {
    MDNS.addService("http", "tcp", HTTP_PORT);
    Serial.print(F("[MDNS] http://"));
    Serial.print(WIFI_HOSTNAME);
    Serial.println(F(".local/telemetry.json"));
  }

  Serial.print(F("[HTTP] Telemetry JSON: http://"));
  Serial.print(g_wifiIpText);
  Serial.println(F("/telemetry.json"));
}

// ======================================================================================
// ARDUINO SETUP
// ======================================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println(F("\n\n=============================================="));
  Serial.println(F("    ESP32 Drone Telemetry Receiver Booting    "));
  Serial.println(F("=============================================="));

  // 1. Initialize Hardware I2C and SH1106 OLED
  Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL);
  u8g2.begin();
  u8g2.setContrast(255);

  // Splash Screen
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.drawRFrame(0, 0, 128, 64, 4);
  u8g2.setCursor(12, 25);
  u8g2.print(F("DRONE TELEMETRY"));
  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(22, 45);
  u8g2.print(F("Initializing..."));
  u8g2.sendBuffer();

  setupWifiApi();

  // 2. Initialize VSPI Bus
  SPI.begin(PIN_NRF_SCK, PIN_NRF_MISO, PIN_NRF_MOSI, -1);
  pinMode(PIN_NRF_CE, OUTPUT);
  pinMode(PIN_NRF_CSN, OUTPUT);
  digitalWrite(PIN_NRF_CSN, HIGH);

  // 3. Initialize NRF24L01+ Radio
  if (!radio.begin()) {
    Serial.println(F("[ERROR] NRF24L01 hardware not detected! Check SPI wiring & HW-200 5V power."));
    g_radioHardwareOk = false;
    drawHardwareErrorScreen();
  } else {
    g_radioHardwareOk = true;
    radio.setChannel(RF_CHANNEL);
    radio.setDataRate(RF24_250KBPS);      // Long range mode (250 kbps)
    radio.setPALevel(RF24_PA_HIGH);
    radio.setCRCLength(RF24_CRC_16);
    radio.setAutoAck(false);
    radio.enableDynamicPayloads();

    radio.openReadingPipe(1, rfAddresses[1]); // "2Node"
    radio.openWritingPipe(rfAddresses[0]);
    radio.startListening();

    Serial.print(F("[NRF24] Radio initialized successfully on Channel "));
    Serial.println(RF_CHANNEL);
  }
}

// ======================================================================================
// MAIN ARDUINO LOOP
// ======================================================================================
void loop() {
  server.handleClient();

  if (!g_radioHardwareOk) {
    delay(500);
    return;
  }

  // 1. Non-blocking Radio Packet Polling
  while (radio.available()) {
    uint8_t rawPayload[32] = {0};
    uint8_t payloadSize = radio.getDynamicPayloadSize();
    if (payloadSize == 0 || payloadSize > 32) {
      payloadSize = sizeof(TelemetryPacket);
    }

    radio.read(&rawPayload, payloadSize);

    bool packetDecoded = false;
    bool shouldPrintTelemetry = false;

    if (rawPayload[0] == PACKET_MAGIC && payloadSize == 32) {
      packetDecoded = processTelemetryFrame(rawPayload, payloadSize, &shouldPrintTelemetry);
      if (!packetDecoded) {
        Serial.println(F("[NRF24] Corrupted packet dropped (Frame checksum mismatch)."));
      }
    } else if (rawPayload[0] == PACKET_MAGIC) {
      packetDecoded = processBinaryPacket(rawPayload, payloadSize);
      shouldPrintTelemetry = packetDecoded;
    } else {
      packetDecoded = processCsvPacket((const char*)rawPayload);
      shouldPrintTelemetry = packetDecoded;
    }

    if (packetDecoded) {
      g_hasReceivedData = true;
      g_lastPacketTime = millis();
      g_heartbeatState = !g_heartbeatState;

      // Safe serial printing
      if (shouldPrintTelemetry) {
        Serial.print(F("[RX] BAT: "));
        Serial.print(g_batVoltage, 2);
        Serial.print(F("V | RSSI: "));
        Serial.print(g_rssiVal);
        Serial.print(F("% | ALT: "));
        Serial.print(g_altitude, 1);
        Serial.print(F("m | LAT: "));
        Serial.print(g_latitude, 6);
        Serial.print(F(" | LON: "));
        Serial.print(g_longitude, 6);
        Serial.print(F(" | SATS: "));
        Serial.println(g_satellites);
      }
    }
  }

  // 2. Non-blocking OLED Screen Refresh
  unsigned long currentMillis = millis();
  if (currentMillis - g_lastDisplayRefresh >= DISPLAY_REFRESH_INTERVAL_MS) {
    g_lastDisplayRefresh = currentMillis;

    unsigned long timeSinceLastPacket = currentMillis - g_lastPacketTime;
    if (!g_hasReceivedData || timeSinceLastPacket >= DISCONNECT_TIMEOUT_MS) {
      drawDisconnectScreen(timeSinceLastPacket);
    } else {
      drawTelemetryScreen();
    }
  }
}
