/*
 * arm_firmware.ino — AI Robotic Arm Servo Controller (FreeRTOS Version)
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
#include <ArduinoJson.h>      // v6 — install via Library Manager
#include <Arduino_FreeRTOS.h> // FreeRTOS Library for Arduino
#include <queue.h>

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

// ── E-stop flag ───────────────────────────────────────────────────────────────
volatile bool eStop = false;

// ── FreeRTOS Data Structures ──────────────────────────────────────────────────
// Command structure to pass between tasks
struct ArmCommand {
  char cmd[16];
  float angles[5];
  int speed;
  int gripValue;
};

QueueHandle_t commandQueue;

// Task Handles
TaskHandle_t TaskSerialHandle;
TaskHandle_t TaskServoHandle;

// Function Prototypes
void TaskSerialRead(void *pvParameters);
void TaskServoControl(void *pvParameters);
void sweepTo(float targets[5], int stepDelayMs);
void goHome(int stepDelayMs);
void sendOk();
void sendError(const char* msg);

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
  
  // Create a queue capable of containing 5 commands
  commandQueue = xQueueCreate(5, sizeof(ArmCommand));

  if (commandQueue != NULL) {
    // Create Tasks
    xTaskCreate(TaskSerialRead,   "SerialRead",   256, NULL, 2, &TaskSerialHandle);
    xTaskCreate(TaskServoControl, "ServoControl", 256, NULL, 1, &TaskServoHandle);
  } else {
    Serial.println("{\"status\":\"error\",\"msg\":\"Queue creation failed\"}");
  }

  Serial.println("{\"status\":\"ok\",\"msg\":\"arm_ready\"}");
  // The FreeRTOS scheduler starts automatically in the Arduino port
}

// =============================================================================
void loop() {
  // Empty. Execution is in the RTOS Tasks.
}

// =============================================================================
// TASK 1: Read JSON from Serial
// =============================================================================
void TaskSerialRead(void *pvParameters) {
  (void) pvParameters;
  String inputBuffer = "";
  
  for (;;) {
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\n') {
        StaticJsonDocument<256> doc;
        DeserializationError err = deserializeJson(doc, inputBuffer);
        inputBuffer = "";

        if (err) {
          sendError("JSON parse failed");
          continue;
        }

        const char* cmd = doc["cmd"];
        if (!cmd) { sendError("missing 'cmd'"); continue; }

        ArmCommand newCmd;
        strncpy(newCmd.cmd, cmd, sizeof(newCmd.cmd) - 1);
        newCmd.cmd[sizeof(newCmd.cmd)-1] = '\0';

        if (strcmp(cmd, "estop") == 0) {
          eStop = true;
          // Bypass queue, detach servos immediately
          for (int i = 0; i < 5; i++) servos[i]->detach();
          sendOk();
          // Clear queue
          xQueueReset(commandQueue);
        } 
        else if (strcmp(cmd, "move") == 0) {
          if (eStop) { sendError("estop active"); continue; }
          JsonArray angles = doc["angles"];
          if (angles.isNull() || angles.size() < 5) {
            sendError("'angles' must have 5 elements");
            continue;
          }
          for (int i = 0; i < 5; i++) newCmd.angles[i] = angles[i];
          newCmd.speed = doc["speed"] | 50;
          xQueueSend(commandQueue, &newCmd, portMAX_DELAY);
        }
        else if (strcmp(cmd, "home") == 0) {
          eStop = false;
          xQueueSend(commandQueue, &newCmd, portMAX_DELAY);
        }
        else if (strcmp(cmd, "grip") == 0) {
          if (eStop) { sendError("estop active"); continue; }
          newCmd.gripValue = constrain((int)doc["value"], 0, 90);
          xQueueSend(commandQueue, &newCmd, portMAX_DELAY);
        }
        else {
          sendError("unknown command");
        }
      } else {
        inputBuffer += c;
      }
    }
    // Yield to let other tasks run
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

// =============================================================================
// TASK 2: Control Servos smoothly
// =============================================================================
void TaskServoControl(void *pvParameters) {
  (void) pvParameters;
  ArmCommand rxCmd;

  for (;;) {
    if (xQueueReceive(commandQueue, &rxCmd, portMAX_DELAY) == pdPASS) {
      if (eStop) continue; // Ignore commands if eStop is active

      if (strcmp(rxCmd.cmd, "move") == 0) {
        float targets[5];
        for (int i = 0; i < 5; i++) {
          targets[i] = constrain(rxCmd.angles[i], LIMIT_MIN[i], LIMIT_MAX[i]);
        }
        int delayMs = map(rxCmd.speed, 0, 100, 30, 2);
        sweepTo(targets, delayMs);
        sendOk();
      }
      else if (strcmp(rxCmd.cmd, "home") == 0) {
        goHome(20);
        sendOk();
      }
      else if (strcmp(rxCmd.cmd, "grip") == 0) {
        servoGripper.write(rxCmd.gripValue);
        currentAngles[4] = rxCmd.gripValue;
        sendOk();
      }
    }
  }
}

// =============================================================================
// Smooth interpolated move
void sweepTo(float targets[5], int stepDelayMs) {
  const int STEPS = 50;
  float starts[5];
  for (int i = 0; i < 5; i++) starts[i] = currentAngles[i];

  for (int s = 1; s <= STEPS; s++) {
    if (eStop) return; // Abort interpolation if eStop triggered
    for (int i = 0; i < 5; i++) {
      float angle = starts[i] + (targets[i] - starts[i]) * s / STEPS;
      servos[i]->write((int)angle);
    }
    vTaskDelay(stepDelayMs / portTICK_PERIOD_MS);
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
