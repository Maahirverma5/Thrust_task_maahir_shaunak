import cv2 as cv
import mediapipe as mp
import random
import math
import time
import pygame
from collections import deque

# ===============================
# CONFIGURATION
# ===============================
WIDTH, HEIGHT = 1280, 720

MAX_BUBBLES = 5
MIN_RADIUS = 22
MAX_RADIUS = 45

MAX_LEVEL = 5
START_LIVES = 10
SPAWN_INTERVAL = 1.2

# Trail & slicing
TRAIL_LENGTH = 18
TRAIL_BASE_THICKNESS = 10
SMOOTHING_ALPHA = 0.35
MIN_SLICE_DISTANCE = 25

# Effects
PARTICLE_COUNT = 18
PARTICLE_LIFE = 15
FLASH_DURATION = 6

# ===============================
# AUDIO SETUP
# ===============================
pygame.mixer.init()
bubble_sound = pygame.mixer.Sound("bubblepop.mp3")
bomb_sound = pygame.mixer.Sound("bomb.mp3")
bubble_sound.set_volume(0.6)
bomb_sound.set_volume(0.8)

# ===============================
# MEDIAPIPE HANDS
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.70
)

# ===============================
# LINE–CIRCLE INTERSECTION
# ===============================
def line_circle_hit(p1, p2, center, radius):
    if p1 is None or p2 is None:
        return False

    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return False

    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    dist = math.hypot(closest_x - cx, closest_y - cy)
    return dist <= radius

# ===============================
# PARTICLE CLASS
# ===============================
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-4, 4)
        self.vy = random.uniform(-4, 4)
        self.life = PARTICLE_LIFE
        self.color = color

    def move(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, frame):
        if self.life > 0:
            r = max(1, int(4 * self.life / PARTICLE_LIFE))
            cv.circle(frame, (int(self.x), int(self.y)), r, self.color, -1)

# ===============================
# BUBBLE CLASS
# ===============================
class Bubble:
    def __init__(self, level):
        self.is_bomb = random.random() < 0.15
        self.radius = random.randint(MIN_RADIUS, MAX_RADIUS)

        self.x = float(random.randint(self.radius, WIDTH - self.radius))
        self.y = float(HEIGHT + random.randint(0, HEIGHT))

        base_speed = random.randint(2, 4)
        self.vy = -(base_speed + level * 1.5)
        self.vx = random.uniform(-1.2, 1.2)

        if self.is_bomb:
            self.color = (0, 0, 0)
        else:
            self.color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        cv.circle(frame, (cx, cy), self.radius, self.color, -1)
        cv.circle(frame, (cx, cy), self.radius, (40, 40, 40), 2)

        if self.is_bomb:
            offset = int(self.radius * 0.6)
            cv.line(frame, (cx - offset, cy - offset),
                    (cx + offset, cy + offset), (0, 0, 255), 4)
            cv.line(frame, (cx + offset, cy - offset),
                    (cx - offset, cy + offset), (0, 0, 255), 4)
        else:
            cv.circle(
                frame,
                (int(cx - self.radius / 3), int(cy - self.radius / 3)),
                self.radius // 4,
                (255, 255, 255),
                -1
            )

# ===============================
# GAME RESET
# ===============================
def reset_game():
    return {
        "bubbles": [],
        "particles": [],
        "score": 0,
        "lives": START_LIVES,
        "level": 1,
        "last_spawn": time.time(),
        "finger_trail": deque(maxlen=TRAIL_LENGTH),
        "smooth_pos": None,
        "prev_slice": None,
        "game_over": False,
        "flash_color": None,
        "flash_timer": 0
    }

# ===============================
# INITIAL SETUP
# ===============================
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

state = reset_game()

# ===============================
# MAIN LOOP
# ===============================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    raw_pos = None

    if result.multi_hand_landmarks and not state["game_over"]:
        lm = result.multi_hand_landmarks[0].landmark[8]
        raw_pos = (int(lm.x * WIDTH), int(lm.y * HEIGHT))

    if raw_pos:
        if state["smooth_pos"] is None:
            state["smooth_pos"] = raw_pos
        else:
            sx, sy = state["smooth_pos"]
            rx, ry = raw_pos
            sx = int(SMOOTHING_ALPHA * rx + (1 - SMOOTHING_ALPHA) * sx)
            sy = int(SMOOTHING_ALPHA * ry + (1 - SMOOTHING_ALPHA) * sy)
            state["smooth_pos"] = (sx, sy)

        state["finger_trail"].appendleft(state["smooth_pos"])

    for i in range(1, len(state["finger_trail"])):
        thickness = max(3, int(TRAIL_BASE_THICKNESS * (1 - i / len(state["finger_trail"]))))
        cv.line(frame, state["finger_trail"][i - 1],
                state["finger_trail"][i], (0, 200, 255), thickness)

    finger_pos = state["smooth_pos"]

    if not state["game_over"]:
        now = time.time()
        spawn_interval = max(0.6, SPAWN_INTERVAL - state["level"] * 0.05)

        if now - state["last_spawn"] > spawn_interval:
            if len(state["bubbles"]) < MAX_BUBBLES:
                state["bubbles"].append(Bubble(state["level"]))
                state["last_spawn"] = now

        for bubble in state["bubbles"][:]:
            bubble.move()
            bubble.draw(frame)

            if finger_pos and state["prev_slice"]:
                dist = math.hypot(
                    finger_pos[0] - state["prev_slice"][0],
                    finger_pos[1] - state["prev_slice"][1]
                )

                if dist > MIN_SLICE_DISTANCE and line_circle_hit(
                    state["prev_slice"], finger_pos,
                    (bubble.x, bubble.y), bubble.radius
                ):
                    state["bubbles"].remove(bubble)

                    if bubble.is_bomb:
                        bomb_sound.play()
                        state["game_over"] = True
                        state["flash_color"] = (0, 0, 255)
                        state["flash_timer"] = FLASH_DURATION
                    else:
                        bubble_sound.play()
                        state["score"] += 1
                        state["flash_color"] = bubble.color
                        state["flash_timer"] = FLASH_DURATION
                        for _ in range(PARTICLE_COUNT):
                            state["particles"].append(
                                Particle(bubble.x, bubble.y, bubble.color)
                            )

            if bubble.y < -bubble.radius:
                state["bubbles"].remove(bubble)
                if not bubble.is_bomb:
                    state["lives"] -= 1

        state["level"] = min(MAX_LEVEL, state["score"] // 10 + 1)
        if state["lives"] <= 0:
            state["game_over"] = True

        state["prev_slice"] = finger_pos

    for p in state["particles"][:]:
        p.move()
        p.draw(frame)
        if p.life <= 0:
            state["particles"].remove(p)

    if state["flash_timer"] > 0:
        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (WIDTH, HEIGHT),
                     state["flash_color"], -1)
        alpha = state["flash_timer"] / FLASH_DURATION * 0.35
        frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        state["flash_timer"] -= 1

    cv.putText(frame, f"Score: {state['score']}", (30, 50),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv.putText(frame, f"Lives: {state['lives']}", (30, 100),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv.putText(frame, f"Level: {state['level']}", (30, 150),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if state["game_over"]:
        cv.putText(frame, "GAME OVER", (400, 320),
                   cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
        cv.putText(frame, "Press R to Restart | Q to Quit", (360, 380),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv.imshow("Bubble Popper", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r') and state["game_over"]:
        state = reset_game()

cap.release()
cv.destroyAllWindows()
