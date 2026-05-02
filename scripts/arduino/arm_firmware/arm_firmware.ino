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
 *
 * Fixes applied (v1.1):
 *   [FIX-1] sweepTo(): clamp stepDelayMs to ≥1 to prevent vTaskDelay(0)
 *           which causes FreeRTOS to spin and starve the serial task.
 *   [FIX-2] speed → delayMs: constrain speed to [0,100] before map()
 *           to prevent out-of-range values corrupting the delay.
 *   [FIX-3] Replaced Arduino String in TaskSerialRead with a fixed-size
 *           char buffer to eliminate heap fragmentation on AVR MCUs.
 *   [FIX-4] "arm_ready" message is now sent only when queue creation
 *           succeeds, preventing Python from thinking the arm is ready
 *           when the FreeRTOS queue failed to allocate.
 */

#include <ArduinoJson.h>      // v6 — install via Library Manager
#include <Arduino_FreeRTOS.h> // FreeRTOS Library for Arduino
#include <Servo.h>
#include <queue.h>

// ── Pin assignments
// ───────────────────────────────────────────────────────────
const int PIN_BASE = 3;
const int PIN_SHOULDER = 5;
const int PIN_ELBOW = 6;
const int PIN_WRIST = 9;
const int PIN_GRIPPER = 10;

// ── Servo objects
// ─────────────────────────────────────────────────────────────
Servo servoBase, servoShoulder, servoElbow, servoWrist, servoGripper;
Servo *servos[5] = {&servoBase, &servoShoulder, &servoElbow, &servoWrist,
                    &servoGripper};

// ── Joint limits (degrees) — servo hardware range ────────────────────────────
// Note: Python IK uses signed angles (-90..+90 for base/wrist).
// The firmware receives angles already mapped to servo range [0..180].
const int LIMIT_MIN[5] = {0, 0, 0, 0, 0};
const int LIMIT_MAX[5] = {180, 150, 150, 180, 90};

// ── Home angles
// ───────────────────────────────────────────────────────────────
const int HOME[5] = {90, 90, 0, 90, 0};

// ── Current angles
// ────────────────────────────────────────────────────────────
float currentAngles[5] = {90, 90, 0, 90, 0};

// ── E-stop flag
// ───────────────────────────────────────────────────────────────
volatile bool eStop = false;

// ── FreeRTOS Data Structures
// ──────────────────────────────────────────────────
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

// [FIX-3] Serial input buffer: fixed-size char array instead of Arduino String
// to prevent heap fragmentation on AVR MCUs under FreeRTOS.
#define SERIAL_BUF_LEN 128

// Function Prototypes
void TaskSerialRead(void *pvParameters);
void TaskServoControl(void *pvParameters);
void sweepTo(float targets[5], int stepDelayMs);
void goHome(int stepDelayMs);
void sendOk();
void sendError(const char *msg);

// =============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }

  servoBase.attach(PIN_BASE);
  servoShoulder.attach(PIN_SHOULDER);
  servoElbow.attach(PIN_ELBOW);
  servoWrist.attach(PIN_WRIST);
  servoGripper.attach(PIN_GRIPPER);

  goHome(30); // move to home at startup

  // Create a queue capable of containing 5 commands
  commandQueue = xQueueCreate(5, sizeof(ArmCommand));

  // [FIX-4] Only report ready AFTER confirming queue allocation succeeded.
  // Previously, "arm_ready" was sent unconditionally, which could mislead
  // the Python host into thinking the arm is operational when it is not.
  if (commandQueue != NULL) {
    xTaskCreate(TaskSerialRead, "SerialRead", 256, NULL, 2, &TaskSerialHandle);
    xTaskCreate(TaskServoControl, "ServoControl", 256, NULL, 1,
                &TaskServoHandle);
    Serial.println("{\"status\":\"ok\",\"msg\":\"arm_ready\"}");
  } else {
    Serial.println("{\"status\":\"error\",\"msg\":\"Queue creation failed — "
                   "insufficient heap\"}");
  }

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
  (void)pvParameters;

  // [FIX-3] Use a fixed char buffer instead of Arduino String to prevent
  // heap fragmentation. Stack-allocated inside the task is safe.
  char inputBuffer[SERIAL_BUF_LEN];
  uint8_t bufIdx = 0;
  memset(inputBuffer, 0, sizeof(inputBuffer));

  for (;;) {
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\n') {
        inputBuffer[bufIdx] = '\0'; // null-terminate

        StaticJsonDocument<256> doc;
        DeserializationError err = deserializeJson(doc, inputBuffer);

        // Clear buffer for next message
        bufIdx = 0;
        memset(inputBuffer, 0, sizeof(inputBuffer));

        if (err) {
          sendError("JSON parse failed");
          continue;
        }

        const char *cmd = doc["cmd"];
        if (!cmd) {
          sendError("missing 'cmd'");
          continue;
        }

        ArmCommand newCmd;
        memset(&newCmd, 0, sizeof(newCmd));
        strncpy(newCmd.cmd, cmd, sizeof(newCmd.cmd) - 1);
        newCmd.cmd[sizeof(newCmd.cmd) - 1] = '\0';

        if (strcmp(cmd, "estop") == 0) {
          eStop = true;
          for (int i = 0; i < 5; i++)
            servos[i]->detach();
          sendOk();
          xQueueReset(commandQueue);
        } else if (strcmp(cmd, "move") == 0) {
          if (eStop) {
            sendError("estop active");
            continue;
          }
          JsonArray angles = doc["angles"];
          if (angles.isNull() || angles.size() < 5) {
            sendError("'angles' must have 5 elements");
            continue;
          }
          for (int i = 0; i < 5; i++)
            newCmd.angles[i] = angles[i].as<float>();
          // [FIX-2] Constrain speed to valid range before storing
          newCmd.speed = constrain(doc["speed"].as<int>(), 0, 100);
          if (newCmd.speed == 0)
            newCmd.speed = 50; // default if omitted
          xQueueSend(commandQueue, &newCmd, portMAX_DELAY);
        } else if (strcmp(cmd, "home") == 0) {
          eStop = false;
          xQueueSend(commandQueue, &newCmd, portMAX_DELAY);
        } else if (strcmp(cmd, "grip") == 0) {
          if (eStop) {
            sendError("estop active");
            continue;
          }
          newCmd.gripValue = constrain(doc["value"].as<int>(), 0, 90);
          xQueueSend(commandQueue, &newCmd, portMAX_DELAY);
        } else {
          sendError("unknown command");
        }
      } else {
        // [FIX-3] Guard against buffer overflow
        if (bufIdx < SERIAL_BUF_LEN - 1) {
          inputBuffer[bufIdx++] = c;
        } else {
          // Buffer overflow — discard current message and reset
          bufIdx = 0;
          memset(inputBuffer, 0, sizeof(inputBuffer));
          sendError("input buffer overflow");
        }
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
  (void)pvParameters;
  ArmCommand rxCmd;

  for (;;) {
    if (xQueueReceive(commandQueue, &rxCmd, portMAX_DELAY) == pdPASS) {
      if (eStop)
        continue;

      if (strcmp(rxCmd.cmd, "move") == 0) {
        float targets[5];
        for (int i = 0; i < 5; i++) {
          targets[i] = constrain(rxCmd.angles[i], LIMIT_MIN[i], LIMIT_MAX[i]);
        }
        // [FIX-2] Constrain speed before mapping to prevent out-of-range delay
        int safeSpeed = constrain(rxCmd.speed, 0, 100);
        int delayMs = map(safeSpeed, 0, 100, 30, 2);
        sweepTo(targets, delayMs);
        sendOk();
      } else if (strcmp(rxCmd.cmd, "home") == 0) {
        goHome(20);
        sendOk();
      } else if (strcmp(rxCmd.cmd, "grip") == 0) {
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
  for (int i = 0; i < 5; i++)
    starts[i] = currentAngles[i];

  for (int s = 1; s <= STEPS; s++) {
    if (eStop)
      return;
    for (int i = 0; i < 5; i++) {
      float angle = starts[i] + (targets[i] - starts[i]) * s / STEPS;
      servos[i]->write((int)angle);
    }
    // [FIX-1] Clamp delay to minimum 1 tick to prevent vTaskDelay(0).
    // vTaskDelay(0) does NOT yield — it causes the servo task to spin and
    // starve the serial read task, making the arm unresponsive to estop.
    TickType_t ticks = (TickType_t)stepDelayMs / portTICK_PERIOD_MS;
    if (ticks == 0)
      ticks = 1;
    vTaskDelay(ticks);
  }

  for (int i = 0; i < 5; i++)
    currentAngles[i] = targets[i];
}

// =============================================================================
void goHome(int stepDelayMs) {
  float h[5];
  for (int i = 0; i < 5; i++)
    h[i] = HOME[i];

  // Re-attach if detached by estop
  servoBase.attach(PIN_BASE);
  servoShoulder.attach(PIN_SHOULDER);
  servoElbow.attach(PIN_ELBOW);
  servoWrist.attach(PIN_WRIST);
  servoGripper.attach(PIN_GRIPPER);

  sweepTo(h, stepDelayMs);
}

// =============================================================================
void sendOk() { Serial.println("{\"status\":\"ok\"}"); }

void sendError(const char *msg) {
  StaticJsonDocument<128> doc;
  doc["status"] = "error";
  doc["msg"] = msg;
  serializeJson(doc, Serial);
  Serial.println();
}
