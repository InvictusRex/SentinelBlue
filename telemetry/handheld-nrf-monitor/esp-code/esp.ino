#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <RF24.h>
#include <U8g2lib.h>

#define PIN_NRF_CE    4
#define PIN_NRF_CSN   5
#define PIN_NRF_SCK   18
#define PIN_NRF_MISO  19
#define PIN_NRF_MOSI  23

#define PIN_OLED_SDA  21
#define PIN_OLED_SCL  22

#define PACKET_MAGIC 0xAA

struct __attribute__((packed)) TelemetryPacket {
  uint8_t  magic;
  uint8_t  seq;
  uint16_t bat_mv;
  int16_t  rssi;
  int32_t  alt_cm;
  int32_t  lat_e7;
  int32_t  lon_e7;
  uint8_t  satellites;
  uint8_t  checksum;
};

U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0,  U8X8_PIN_NONE);

RF24 radio(PIN_NRF_CE, PIN_NRF_CSN);

const byte rfAddresses[][6] = {"1Node", "2Node"};
const uint8_t RF_CHANNEL = 90;

float g_batVoltage = 0.0;
int16_t g_rssiVal = 0;
float g_altitude = 0.0;
double g_latitude = 0.0;
double g_longitude = 0.0;
uint8_t g_satellites = 0;
uint8_t g_lastSeq = 0;

bool g_hasReceivedData = false;
bool g_heartbeatState = false;
bool g_radioHardwareOk = false;
unsigned long g_lastPacketTime = 0;
const unsigned long DISCONNECT_TIMEOUT_MS = 10000;

unsigned long g_lastDisplayRefresh = 0;
const unsigned long DISPLAY_REFRESH_INTERVAL_MS = 60;

uint8_t calculateChecksum(const uint8_t* buffer, size_t length) {
  uint8_t c = 0;
  for (size_t i = 0; i < length; i++) {
    c ^= buffer[i];
  }
  return c;
}

bool processBinaryPacket(const uint8_t* buffer, size_t size) {
  if (size < sizeof(TelemetryPacket)) return false;

  const TelemetryPacket* pkt = (const TelemetryPacket*)buffer;
  if (pkt->magic != PACKET_MAGIC) return false;

  uint8_t expectedChecksum = calculateChecksum(buffer, sizeof(TelemetryPacket) - 1);
  if (pkt->checksum != expectedChecksum) {
    Serial.println(F("[NRF24] Corrupted packet dropped (Checksum mismatch)."));
    return false;
  }

  g_batVoltage = pkt->bat_mv / 1000.0f;
  g_rssiVal    = pkt->rssi;
  g_altitude   = pkt->alt_cm / 100.0f;
  g_latitude   = pkt->lat_e7 / 10000000.0;
  g_longitude  = pkt->lon_e7 / 10000000.0;
  g_satellites = pkt->satellites;
  g_lastSeq    = pkt->seq;

  return true;
}

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

  u8g2.drawRFrame(0, 0, 128, 64, 3);

  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.drawBox(2, 2, 124, 15);
  u8g2.setDrawColor(0);
  u8g2.setCursor(14, 13);
  u8g2.print(F("! NO CONNECTION !"));
  u8g2.setDrawColor(1);

  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.setCursor(16, 34);
  u8g2.print(F("Telemetry Lost"));

  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(14, 50);
  if (!g_hasReceivedData) {
    u8g2.print(F("Waiting for link..."));
  } else {
    unsigned long elapsedSec = elapsedMs / 1000;
    u8g2.print(F("Lost: "));
    u8g2.print(elapsedSec);
    u8g2.print(F("s ago"));
  }

  u8g2.sendBuffer();
}

void drawTelemetryScreen() {
  u8g2.clearBuffer();

  u8g2.setFont(u8g2_font_7x14B_tr);
  u8g2.setCursor(2, 13);
  u8g2.print(F("BAT:"));
  u8g2.print(g_batVoltage, 2);
  u8g2.print(F("V"));

  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(76, 12);
  u8g2.print(F("SAT:"));
  u8g2.print(g_satellites);

  if (g_heartbeatState) {
    u8g2.drawDisc(122, 9, 3);
  } else {
    u8g2.drawCircle(122, 9, 3);
  }

  u8g2.drawHLine(0, 16, 128);

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

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println(F("\n\n=============================================="));
  Serial.println(F("    ESP32 Drone Telemetry Receiver Booting    "));
  Serial.println(F("=============================================="));

  Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL);
  u8g2.begin();
  u8g2.setContrast(255);

  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.drawRFrame(0, 0, 128, 64, 4);
  u8g2.setCursor(12, 25);
  u8g2.print(F("DRONE TELEMETRY"));
  u8g2.setFont(u8g2_font_6x12_tf);
  u8g2.setCursor(22, 45);
  u8g2.print(F("Initializing..."));
  u8g2.sendBuffer();

  SPI.begin(PIN_NRF_SCK, PIN_NRF_MISO, PIN_NRF_MOSI, -1);
  pinMode(PIN_NRF_CE, OUTPUT);
  pinMode(PIN_NRF_CSN, OUTPUT);
  digitalWrite(PIN_NRF_CSN, HIGH);

  if (!radio.begin()) {
    Serial.println(F("[ERROR] NRF24L01 hardware not detected! Check SPI wiring & HW-200 5V power."));
    g_radioHardwareOk = false;
    drawHardwareErrorScreen();
  } else {
    g_radioHardwareOk = true;
    radio.setChannel(RF_CHANNEL);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_HIGH);
    radio.setCRCLength(RF24_CRC_16);
    radio.setAutoAck(false);
    radio.enableDynamicPayloads();

    radio.openReadingPipe(1, rfAddresses[1]);
    radio.openWritingPipe(rfAddresses[0]);
    radio.startListening();

    Serial.print(F("[NRF24] Radio initialized successfully on Channel "));
    Serial.println(RF_CHANNEL);
  }
}

void loop() {
  if (!g_radioHardwareOk) {
    delay(500);
    return;
  }

  while (radio.available()) {
    uint8_t rawPayload[32] = {0};
    uint8_t payloadSize = radio.getDynamicPayloadSize();
    if (payloadSize == 0 || payloadSize > 32) {
      payloadSize = sizeof(TelemetryPacket);
    }

    radio.read(&rawPayload, payloadSize);

    bool packetDecoded = false;

    if (rawPayload[0] == PACKET_MAGIC) {
      packetDecoded = processBinaryPacket(rawPayload, payloadSize);
    } else {
      packetDecoded = processCsvPacket((const char*)rawPayload);
    }

    if (packetDecoded) {
      g_hasReceivedData = true;
      g_lastPacketTime = millis();
      g_heartbeatState = !g_heartbeatState;

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
