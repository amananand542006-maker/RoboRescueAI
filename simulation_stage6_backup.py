import pygame
import math
import sys

# ==========================================
# RoboRescueAI - Stage 6
# 2D Rescue Robot Simulation
# ==========================================

pygame.init()

WIDTH, HEIGHT = 1100, 720
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RoboRescueAI - Rescue Robot Simulation")

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 22)
small_font = pygame.font.SysFont("Arial", 16)
title_font = pygame.font.SysFont("Arial", 30, bold=True)

# Colors
BACKGROUND = (20, 28, 38)
GRID = (37, 48, 61)
WHITE = (240, 245, 250)
GREEN = (60, 210, 130)
RED = (235, 80, 80)
BLUE = (65, 160, 255)
YELLOW = (255, 205, 70)
GREY = (110, 125, 140)
DARK = (12, 18, 26)

# Map
MAP = pygame.Rect(20, 100, 760, 580)
PANEL = pygame.Rect(800, 100, 280, 580)

# Robot
robot_x = 90.0
robot_y = 150.0
robot_radius = 17
robot_speed = 3.0
robot_trail = []

# Buildings and obstacles
obstacles = [
    pygame.Rect(180, 145, 125, 100),
    pygame.Rect(390, 110, 115, 145),
    pygame.Rect(590, 190, 120, 100),
    pygame.Rect(130, 370, 145, 115),
    pygame.Rect(365, 390, 120, 145),
    pygame.Rect(570, 450, 145, 100),
]

# Simulated people
people = [
    {"x": 335, "y": 170, "rescued": False},
    {"x": 540, "y": 335, "rescued": False},
    {"x": 320, "y": 590, "rescued": False},
    {"x": 735, "y": 390, "rescued": False},
    {"x": 85, "y": 560, "rescued": False},
]

collisions = 0
mission_complete = False


def draw_text(text, x, y, color=WHITE, font_obj=font):
    image = font_obj.render(str(text), True, color)
    screen.blit(image, (x, y))


def hits_obstacle(x, y):
    robot_rect = pygame.Rect(
        int(x - robot_radius),
        int(y - robot_radius),
        robot_radius * 2,
        robot_radius * 2,
    )

    if not MAP.contains(robot_rect):
        return True

    return any(
        robot_rect.colliderect(obstacle)
        for obstacle in obstacles
    )


def move_robot(dx, dy):
    global robot_x, robot_y, collisions

    next_x = robot_x + dx

    if not hits_obstacle(next_x, robot_y):
        robot_x = next_x
    elif dx:
        collisions += 1

    next_y = robot_y + dy

    if not hits_obstacle(robot_x, next_y):
        robot_y = next_y
    elif dy:
        collisions += 1


def update_rescues():
    for person in people:
        if person["rescued"]:
            continue

        distance = math.hypot(
            robot_x - person["x"],
            robot_y - person["y"],
        )

        if distance < 30:
            person["rescued"] = True


def draw_grid():
    for x in range(MAP.left, MAP.right, 25):
        pygame.draw.line(
            screen, GRID, (x, MAP.top), (x, MAP.bottom)
        )

    for y in range(MAP.top, MAP.bottom, 25):
        pygame.draw.line(
            screen, GRID, (MAP.left, y), (MAP.right, y)
        )


def draw_obstacles():
    for obstacle in obstacles:
        pygame.draw.rect(
            screen, GREY, obstacle, border_radius=5
        )
        pygame.draw.rect(
            screen, WHITE, obstacle, 2, border_radius=5
        )

        label = small_font.render("BUILDING", True, DARK)
        screen.blit(label, label.get_rect(center=obstacle.center))


def draw_people():
    for person in people:
        x, y = int(person["x"]), int(person["y"])

        color = GREEN if person["rescued"] else RED
        pygame.draw.circle(screen, color, (x, y), 12)
        pygame.draw.circle(screen, WHITE, (x, y), 12, 2)

        label = "RESCUED" if person["rescued"] else "PERSON"
        label_color = GREEN if person["rescued"] else WHITE

        draw_text(
            label, x - 30, y - 29,
            label_color, small_font
        )


def draw_robot():
    if len(robot_trail) > 1:
        pygame.draw.lines(
            screen, BLUE, False, robot_trail, 3
        )

    pygame.draw.circle(
        screen, BLUE,
        (int(robot_x), int(robot_y)),
        robot_radius,
    )

    pygame.draw.circle(
        screen, WHITE,
        (int(robot_x), int(robot_y)),
        robot_radius, 2,
    )

    pygame.draw.circle(
        screen, YELLOW,
        (int(robot_x + 7), int(robot_y - 5)),
        4,
    )


def draw_dashboard():
    pygame.draw.rect(
        screen, DARK, PANEL, border_radius=12
    )

    pygame.draw.rect(
        screen, GREY, PANEL, 2, border_radius=12
    )

    rescued = sum(person["rescued"] for person in people)
    remaining = len(people) - rescued

    draw_text(
        "MISSION DASHBOARD",
        PANEL.x + 18, PANEL.y + 20,
        YELLOW, small_font,
    )

    draw_text("Robot status", PANEL.x + 18, PANEL.y + 75, WHITE, small_font)
    draw_text("ACTIVE", PANEL.x + 18, PANEL.y + 100, GREEN)

    draw_text("People detected", PANEL.x + 18, PANEL.y + 160, WHITE, small_font)
    draw_text(len(people), PANEL.x + 18, PANEL.y + 185, YELLOW)

    draw_text("People rescued", PANEL.x + 18, PANEL.y + 235, WHITE, small_font)
    draw_text(rescued, PANEL.x + 18, PANEL.y + 260, GREEN)

    draw_text("Awaiting rescue", PANEL.x + 18, PANEL.y + 310, WHITE, small_font)
    draw_text(remaining, PANEL.x + 18, PANEL.y + 335, RED if remaining else GREEN)

    draw_text("Collision events", PANEL.x + 18, PANEL.y + 385, WHITE, small_font)
    draw_text(collisions, PANEL.x + 18, PANEL.y + 410, YELLOW)

    draw_text("CONTROLS", PANEL.x + 18, PANEL.y + 465, YELLOW, small_font)
    draw_text("WASD / Arrow keys", PANEL.x + 18, PANEL.y + 490, WHITE, small_font)
    draw_text("R - Reset mission", PANEL.x + 18, PANEL.y + 515, WHITE, small_font)
    draw_text("ESC - Exit", PANEL.x + 18, PANEL.y + 540, WHITE, small_font)

    if mission_complete:
        draw_text(
            "MISSION COMPLETE!",
            PANEL.x + 18, PANEL.y + 565,
            GREEN, small_font,
        )


def reset_mission():
    global robot_x, robot_y, robot_trail
    global collisions, mission_complete

    robot_x, robot_y = 90.0, 150.0
    robot_trail = []
    collisions = 0
    mission_complete = False

    for person in people:
        person["rescued"] = False


# Main loop
running = True

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                reset_mission()

    keys = pygame.key.get_pressed()

    if not mission_complete:
        dx = 0
        dy = 0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= robot_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += robot_speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= robot_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += robot_speed

        if dx and dy:
            dx *= 0.7071
            dy *= 0.7071

        move_robot(dx, dy)

        if dx or dy:
            robot_trail.append((int(robot_x), int(robot_y)))

            if len(robot_trail) > 350:
                robot_trail.pop(0)

        update_rescues()

        if all(person["rescued"] for person in people):
            mission_complete = True

    screen.fill(BACKGROUND)

    draw_text(
        "RoboRescueAI - Stage 6",
        25, 25, WHITE, title_font,
    )

    draw_text(
        "2D Rescue Robot Simulation",
        28, 65, GREY, small_font,
    )

    pygame.draw.rect(screen, BACKGROUND, MAP)

    draw_grid()
    draw_obstacles()
    draw_people()
    draw_robot()
    draw_dashboard()

    pygame.display.flip()

pygame.quit()
sys.exit()