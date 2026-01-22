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
MID_X = WIDTH // 2

MAX_BUBBLES = 5
MIN_RADIUS = 22
MAX_RADIUS = 45

MAX_LEVEL = 5
START_LIVES = 10
SPAWN_INTERVAL = 1.2

TRAIL_LENGTH = 18
TRAIL_BASE_THICKNESS = 10
SMOOTHING_ALPHA = 0.35
MIN_SLICE_DISTANCE = 25

PARTICLE_COUNT = 18
PARTICLE_LIFE = 15
FLASH_DURATION = 6

# ===============================
# AUDIO
# ===============================
pygame.mixer.init()
bubble_sound = pygame.mixer.Sound("bubblepop.mp3")
bomb_sound = pygame.mixer.Sound("bomb.mp3")

# ===============================
# MEDIAPIPE
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# GEOMETRY
# ===============================
def line_circle_hit(p1, p2, center, radius):
    if p1 is None or p2 is None:
        return False
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return False
    t = ((cx-x1)*dx + (cy-y1)*dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    px, py = x1 + t*dx, y1 + t*dy
    return math.hypot(px-cx, py-cy) <= radius

# ===============================
# PARTICLES
# ===============================
class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
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
# BUBBLE
# ===============================
class Bubble:
    def __init__(self, level, xmin, xmax):
        self.is_bomb = random.random() < 0.15
        self.radius = random.randint(MIN_RADIUS, MAX_RADIUS)
        self.x = float(random.randint(xmin + self.radius, xmax - self.radius))
        self.y = float(HEIGHT + random.randint(0, HEIGHT))
        self.vy = -(random.randint(2, 4) + level * 1.5)
        self.vx = random.uniform(-1.2, 1.2)
        self.color = (0, 0, 0) if self.is_bomb else (
            random.randint(100,255),
            random.randint(100,255),
            random.randint(100,255)
        )

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        cv.circle(frame, (cx, cy), self.radius, self.color, -1)
        cv.circle(frame, (cx, cy), self.radius, (40,40,40), 2)
        if self.is_bomb:
            o = int(self.radius*0.6)
            cv.line(frame,(cx-o,cy-o),(cx+o,cy+o),(0,0,255),4)
            cv.line(frame,(cx+o,cy-o),(cx-o,cy+o),(0,0,255),4)
        else:
            cv.circle(frame,(cx-self.radius//3,cy-self.radius//3),
                      self.radius//4,(255,255,255),-1)

# ===============================
# GAME STATE
# ===============================
def new_state(xmin, xmax):
    return {
        "bubbles": [],
        "particles": [],
        "score": 0,
        "lives": START_LIVES,
        "level": 1,
        "last_spawn": time.time(),
        "trail": deque(maxlen=TRAIL_LENGTH),
        "smooth": None,
        "prev": None,
        "over": False,
        "flash_color": None,
        "flash_timer": 0,
        "xmin": xmin,
        "xmax": xmax
    }

# ===============================
# CAMERA
# ===============================
cap = cv.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

mode = "MENU"
p1 = new_state(0, MID_X)
p2 = new_state(MID_X, WIDTH)

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = hands.process(rgb)

    fingers = []
    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            lm = h.landmark[8]
            fingers.append((int(lm.x*WIDTH), int(lm.y*HEIGHT)))

    # ================= MENU =================
    if mode == "MENU":
        cv.putText(frame,"SLICE TO CHOOSE",(380,150),
                   cv.FONT_HERSHEY_SIMPLEX,1.8,(255,255,255),4)
        cv.rectangle(frame,(200,300),(500,450),(0,200,255),-1)
        cv.rectangle(frame,(780,300),(1080,450),(0,200,255),-1)
        cv.putText(frame,"1 PLAYER",(245,380),
                   cv.FONT_HERSHEY_SIMPLEX,1.4,(0,0,0),4)
        cv.putText(frame,"2 PLAYER",(815,380),
                   cv.FONT_HERSHEY_SIMPLEX,1.4,(0,0,0),4)

        if fingers:
            fx, fy = fingers[0]
            if 200 < fx < 500 and 300 < fy < 450:
                mode = "SINGLE"
                p1 = new_state(0, WIDTH)
            if 780 < fx < 1080 and 300 < fy < 450:
                mode = "DOUBLE"
                p1 = new_state(0, MID_X)
                p2 = new_state(MID_X, WIDTH)

    # ================= GAME MODES =================
    states = []
    if mode == "SINGLE":
        states = [p1]
    elif mode == "DOUBLE":
        states = [p1, p2]
        cv.line(frame,(MID_X,0),(MID_X,HEIGHT),(255,255,255),3)

    for idx, state in enumerate(states):
        finger = None
        for f in fingers:
            if state["xmin"] < f[0] < state["xmax"]:
                finger = f

        if finger:
            if state["smooth"] is None:
                state["smooth"] = finger
            else:
                sx, sy = state["smooth"]
                rx, ry = finger
                state["smooth"] = (
                    int(SMOOTHING_ALPHA*rx + (1-SMOOTHING_ALPHA)*sx),
                    int(SMOOTHING_ALPHA*ry + (1-SMOOTHING_ALPHA)*sy)
                )
            state["trail"].appendleft(state["smooth"])

        for i in range(1,len(state["trail"])):
            t = max(3,int(TRAIL_BASE_THICKNESS*(1-i/len(state["trail"]))))
            cv.line(frame,state["trail"][i-1],state["trail"][i],
                    (0,200,255),t)

        if not state["over"]:
            now = time.time()
            if now - state["last_spawn"] > SPAWN_INTERVAL:
                if len(state["bubbles"]) < MAX_BUBBLES:
                    state["bubbles"].append(
                        Bubble(state["level"],state["xmin"],state["xmax"])
                    )
                    state["last_spawn"] = now

            for b in state["bubbles"][:]:
                b.move()
                b.draw(frame)

                if finger and state["prev"]:
                    if math.hypot(finger[0]-state["prev"][0],
                                  finger[1]-state["prev"][1]) > MIN_SLICE_DISTANCE:
                        if line_circle_hit(state["prev"],finger,
                                           (b.x,b.y),b.radius):
                            state["bubbles"].remove(b)
                            state["flash_color"] = (0,0,255) if b.is_bomb else b.color
                            state["flash_timer"] = FLASH_DURATION

                            if b.is_bomb:
                                bomb_sound.play()
                                state["over"] = True
                            else:
                                bubble_sound.play()
                                state["score"] += 1
                                for _ in range(PARTICLE_COUNT):
                                    state["particles"].append(
                                        Particle(b.x,b.y,b.color)
                                    )

                if b.y < -b.radius:
                    state["bubbles"].remove(b)
                    if not b.is_bomb:
                        state["lives"] -= 1

            state["level"] = min(MAX_LEVEL, state["score"] // 10 + 1)
            if state["lives"] <= 0:
                state["over"] = True
            state["prev"] = finger

        for p in state["particles"][:]:
            p.move()
            p.draw(frame)
            if p.life <= 0:
                state["particles"].remove(p)

        # FLASH (PER PLAYER AREA)
        if state["flash_timer"] > 0:
            overlay = frame.copy()
            cv.rectangle(
                overlay,
                (state["xmin"], 0),
                (state["xmax"], HEIGHT),
                state["flash_color"],
                -1
            )
            alpha = state["flash_timer"] / FLASH_DURATION * 0.35
            frame = cv.addWeighted(overlay, alpha, frame, 1-alpha, 0)
            state["flash_timer"] -= 1

        ox = 30 + idx * MID_X
        cv.putText(frame,f"Score: {state['score']}",(ox,40),
                   cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        cv.putText(frame,f"Lives: {state['lives']}",(ox,80),
                   cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        cv.putText(frame,f"Level: {state['level']}",(ox,120),
                   cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    # ================= WINNER =================
    if mode == "DOUBLE" and p1["over"] and p2["over"]:
        winner = "PLAYER 1 WINS" if p1["score"] > p2["score"] else "PLAYER 2 WINS"
        cv.putText(frame,winner,(380,320),
                   cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),6)
        cv.putText(frame,"R-RESTART  M-MENU  Q-QUIT",(300,380),
                   cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
    # ================= SINGLE PLAYER POST GAME =================
    if mode == "SINGLE" and p1["over"]:
        cv.putText(frame, "GAME OVER", (420, 320),
                   cv.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6)

        cv.putText(frame, "R - Restart   M - Menu   Q - Quit",
                   (300, 380),
                   cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 255), 3)

    cv.imshow("Bubble Popper", frame)

    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('m'):
        mode = "MENU"
    if k == ord('r'):
        if mode == "SINGLE":
            p1 = new_state(0, WIDTH)
        if mode == "DOUBLE":
            p1 = new_state(0, MID_X)
            p2 = new_state(MID_X, WIDTH)

cap.release()
cv.destroyAllWindows()
