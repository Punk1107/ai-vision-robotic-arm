/*
 * arm_firmware.ino — AI Robotic Arm Servo Controller
 * ====================================================
 * Receives newline-delimited JSON commands from Python over USB Serial.
 * Controls 5 servos (Base, Shoulder, Elbow, Wrist, Gripper) via PWM.
 *
 * Protocol:
 *   {"cmd": "move",   "angles": [θ1, θ2, θ3, θ4, θ5], "speed": 50}
 *   {"cmd": "home"}
 *   {"cmd": "grip",   "value": 90}
 *   {"cmd": "estop"}
 *
 * Response:
 *   {"status": "ok"}
 *   {"status": "error", "msg": "..."}
 *
 * Wiring (Arduino Uno / Nano):
 *   D3  → Base servo     (θ1)
 *   D5  → Shoulder servo (θ2)
 *   D6  → Elbow servo    (θ3)
 *   D9  → Wrist servo    (θ4)
 *   D10 → Gripper servo  (θ5)
 */

#include <Servo.h>
#include <ArduinoJson.h>   // v6 — install via Library Manager

// ── Pin assignments ───────────────────────────────────────────────────────────
const int PIN_BASE     = 3;
const int PIN_SHOULDER = 5;
const int PIN_ELBOW    = 6;
const int PIN_WRIST    = 9;
const int PIN_GRIPPER  = 10;

// ── Servo objects ─────────────────────────────────────────────────────────────
Servo servoBase, servoShoulder, servoElbow, servoWrist, servoGripper;
Servo* servos[5] = {
  &servoBase, &servoShoulder, &servoElbow, &servoWrist, &servoGripper
};

// ── Joint limits (degrees) ────────────────────────────────────────────────────
const int LIMIT_MIN[5] = {  0,   0,   0, -90,  0 };
const int LIMIT_MAX[5] = {180, 150, 150,  90, 90 };

// ── Home angles ───────────────────────────────────────────────────────────────
const int HOME[5] = { 90, 90, 0, 90, 0 };

// ── Current angles ────────────────────────────────────────────────────────────
float currentAngles[5] = { 90, 90, 0, 90, 0 };

// ── Serial buffer ─────────────────────────────────────────────────────────────
String inputBuffer = "";
bool   newCommand  = false;

// ── E-stop flag ───────────────────────────────────────────────────────────────
bool eStop = false;

// =============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  servoBase.attach(PIN_BASE);
  servoShoulder.attach(PIN_SHOULDER);
  servoElbow.attach(PIN_ELBOW);
  servoWrist.attach(PIN_WRIST);
  servoGripper.attach(PIN_GRIPPER);

  goHome(30);   // move to home at startup
  Serial.println("{\"status\":\"ok\",\"msg\":\"arm_ready\"}");
}

// =============================================================================
void loop() {
  // Read serial line
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      newCommand = true;
    } else {
      inputBuffer += c;
    }
  }

  if (newCommand) {
    newCommand = false;
    processCommand(inputBuffer);
    inputBuffer = "";
  }
}

// =============================================================================
void processCommand(const String& raw) {
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, raw);

  if (err) {
    sendError("JSON parse failed");
    return;
  }

  const char* cmd = doc["cmd"];
  if (!cmd) { sendError("missing 'cmd'"); return; }

  // ── move ──────────────────────────────────────────────────────────────────
  if (strcmp(cmd, "move") == 0) {
    if (eStop) { sendError("estop active — send 'home' to reset"); return; }

    JsonArray angles = doc["angles"];
    if (angles.isNull() || angles.size() < 5) {
      sendError("'angles' must have 5 elements");
      return;
    }

    int speed = doc["speed"] | 50;
    int delayMs = map(speed, 0, 100, 30, 2);  // higher speed → less delay

    float targets[5];
    for (int i = 0; i < 5; i++) {
      targets[i] = constrain((float)angles[i], LIMIT_MIN[i], LIMIT_MAX[i]);
    }
    sweepTo(targets, delayMs);
    sendOk();

  // ── home ──────────────────────────────────────────────────────────────────
  } else if (strcmp(cmd, "home") == 0) {
    eStop = false;
    goHome(20);
    sendOk();

  // ── grip ──────────────────────────────────────────────────────────────────
  } else if (strcmp(cmd, "grip") == 0) {
    int val = constrain((int)doc["value"], 0, 90);
    servoGripper.write(val);
    currentAngles[4] = val;
    sendOk();

  // ── estop ─────────────────────────────────────────────────────────────────
  } else if (strcmp(cmd, "estop") == 0) {
    eStop = true;
    // Detach servos to remove holding torque
    for (int i = 0; i < 5; i++) servos[i]->detach();
    sendOk();

  } else {
    sendError("unknown command");
  }
}

// =============================================================================
// Smooth interpolated move
void sweepTo(float targets[5], int stepDelayMs) {
  const int STEPS = 50;
  float starts[5];
  for (int i = 0; i < 5; i++) starts[i] = currentAngles[i];

  for (int s = 1; s <= STEPS; s++) {
    for (int i = 0; i < 5; i++) {
      float angle = starts[i] + (targets[i] - starts[i]) * s / STEPS;
      servos[i]->write((int)angle);
    }
    delay(stepDelayMs);
  }

  for (int i = 0; i < 5; i++) currentAngles[i] = targets[i];
}

// =============================================================================
void goHome(int stepDelayMs) {
  float h[5];
  for (int i = 0; i < 5; i++) h[i] = HOME[i];

  // Re-attach if detached by estop
  servoBase.attach(PIN_BASE);
  servoShoulder.attach(PIN_SHOULDER);
  servoElbow.attach(PIN_ELBOW);
  servoWrist.attach(PIN_WRIST);
  servoGripper.attach(PIN_GRIPPER);

  sweepTo(h, stepDelayMs);
}

// =============================================================================
void sendOk() {
  Serial.println("{\"status\":\"ok\"}");
}

void sendError(const char* msg) {
  StaticJsonDocument<128> doc;
  doc["status"] = "error";
  doc["msg"]    = msg;
  serializeJson(doc, Serial);
  Serial.println();
}
