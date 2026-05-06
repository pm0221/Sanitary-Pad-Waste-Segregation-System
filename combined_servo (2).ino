/*
================================================================================
  AHP WASTE SEGREGATION — SERVO CONTROL (FINAL)
================================================================================
  BEHAVIOUR:
    Python sends "1" → SANITARY PAD detected
                     → Flaps rotate LEFT to 45 degrees
                     → Hold 3 seconds
                     → Return to REST (90 degrees)

    Python sends "0" → Nothing detected → Flaps stay at REST

  REST POSITION: Both flaps at 90 degrees (center, parallel)

  WIRING:
    Flap servo 1 signal → Pin 9
    Flap servo 2 signal → Pin 10
    Both red wires      → 5V
    Both black wires    → GND

  BAUD RATE: 9600
================================================================================
*/

#include <Servo.h>

#define FLAP1_PIN     9
#define FLAP2_PIN     10
#define ANGLE_REST    90      // center rest position
#define ANGLE_OPEN    45      // 45 degrees LEFT — sanitary pad bin
#define HOLD_TIME     3000    // hold open for 3 seconds
#define STEP_DELAY    10      // ms between each step
#define STEP_SIZE     3       // degrees per step

Servo flap1;
Servo flap2;
bool busy = false;


void setup() {
  Serial.begin(9600);
  flap1.attach(FLAP1_PIN);
  flap2.attach(FLAP2_PIN);

  // Start at rest position
  flap1.write(ANGLE_REST);
  flap2.write(ANGLE_REST);
  delay(500);

  Serial.println("========================================");
  Serial.println("  AHP WASTE SEGREGATION SYSTEM ONLINE  ");
  Serial.println("  Flaps at REST: 90 degrees             ");
  Serial.println("  Waiting for detection...              ");
  Serial.println("========================================");
}


void loop() {
  // Listen for Python signal
  if (Serial.available() > 0 && !busy) {
    char input = Serial.read();

    if (input == '1') {
      // SANITARY PAD DETECTED — move flaps
      Serial.println("");
      Serial.println("╔══════════════════════════════════════╗");
      Serial.println("║  DETECTION : SANITARY PAD            ║");
      Serial.println("║  DIRECTION : LEFT                    ║");
      Serial.println("║  ACTION    : Rotating to 45 degrees  ║");
      Serial.println("╚══════════════════════════════════════╝");
      moveFlaps();
    }
    else if (input == '0') {
      Serial.println("[INFO] No sanitary pad — flaps stay at REST (90 deg)");
    }
  }
}


void moveFlaps() {
  busy = true;

  // ── STEP 1: Rotate LEFT from 90 to 45 degrees ──────────────
  Serial.println("----------------------------------------");
  Serial.println(">> ROTATING LEFT (90 -> 45 degrees)");
  Serial.println("----------------------------------------");

  for (int a = ANGLE_REST; a >= ANGLE_OPEN; a -= STEP_SIZE) {
    flap1.write(a);
    flap2.write(a);
    Serial.print("  LEFT << ");
    Serial.print(a);
    Serial.println(" deg");
    delay(STEP_DELAY);
  }

  // Land exactly at 45
  flap1.write(ANGLE_OPEN);
  flap2.write(ANGLE_OPEN);

  Serial.println("----------------------------------------");
  Serial.println("  FLAP OPEN : 45 deg — SANITARY PAD BIN");
  Serial.println("----------------------------------------");

  // ── STEP 2: Hold open for 3 seconds ────────────────────────
  for (int i = HOLD_TIME / 1000; i > 0; i--) {
    Serial.print("  HOLDING... ");
    Serial.print(i);
    Serial.println("s");
    delay(1000);
  }

  // ── STEP 3: Return RIGHT from 45 back to 90 ────────────────
  Serial.println(">> RETURNING TO REST (45 -> 90 degrees)");

  for (int a = ANGLE_OPEN; a <= ANGLE_REST; a += STEP_SIZE) {
    flap1.write(a);
    flap2.write(a);
    Serial.print("  RIGHT >> ");
    Serial.print(a);
    Serial.println(" deg");
    delay(STEP_DELAY);
  }

  // Land exactly at 90
  flap1.write(ANGLE_REST);
  flap2.write(ANGLE_REST);

  Serial.println("========================================");
  Serial.println("  FLAP RESET : 90 deg — REST POSITION  ");
  Serial.println("  READY FOR NEXT OBJECT                 ");
  Serial.println("========================================");
  Serial.println("");

  busy = false;
}
