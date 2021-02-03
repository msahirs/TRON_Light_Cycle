import pygame as py
import sys

# import random
# import math
# import time
# import numpy as np

screen_res = (650, 800)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


class grid:  # Include initiation for grid layout, colours, traps, etc.
    def __init__(self, screen, color):
        divisions = 5
        self.h_lines = []
        self.v_lines = []
        self.color = color
        for x in range(divisions):
            self.h_lines.append([screen, self.color, [0, (py.display.get_window_size()[1] / divisions) * x],
                                 [py.display.get_window_size()[0], (py.display.get_window_size()[1] / divisions) * x],
                                 4])
        for y in range(divisions):
            self.v_lines.append([screen, self.color, [(py.display.get_window_size()[0] / divisions) * y, 0],
                                 [(py.display.get_window_size()[0] / divisions) * y, py.display.get_window_size()[1]],
                                 4])

    def SetColor(self, color):
        self.color = color


class cycles:  # Player objects with position, speed, colour, state
    def __init__(self, color, name, start_pos):
        self.cycle = py.Rect(start_pos, (30, 30))
        self.name = name
        self.color = color
        self.speed = 5
        self.orientation = [0, -1]  # Velocity unit vector
        self.tracer = [[py.Rect(self.cycle.center, (5,5)), False]]

    def move(self):
        self.cycle = py.Rect.move(self.cycle, self.speed * self.orientation[0], self.speed * self.orientation[1])
        if self.cycle[0] >= py.display.get_window_size()[0] - 15:
            self.cycle[0] = py.display.get_window_size()[0] - 15
        if self.cycle[0] <= 0:
            self.cycle[0] = 0
        if self.cycle[1] >= py.display.get_window_size()[1] - 15:
            self.cycle[1] = py.display.get_window_size()[1] - 15
        if self.cycle[1] <= 0:
            self.cycle[1] = 0

        self.tracer.append([py.Rect(self.genCoords(), (5, 5)), False])

    def genCoords(self):
        return self.cycle.center


def color_limit_check(x):
    if x > 255:
        x = 255
    elif x < 0:
        x = 0
    return x


# Globals' setup
screen = py.display.set_mode(screen_res)
fpsClock = py.time.Clock()
FPS = 60
dt = 1 / FPS
col_iter = 0
game_iter=0
grid_color = [255, 0, 0]
Main_grid = grid(screen, grid_color)
color_ramp_speed = 10

# Player Instantiation
Player1 = cycles(WHITE, "Boi", (py.display.get_window_size()[0] // 2, py.display.get_window_size()[1] - py.display.get_window_size()[1] / 10))

# AI Setup

# Main Loop
while True:

    # Debug Stuff
    game_iter += 1

    # Colour Ramping
    if grid_color[(col_iter + 1) % 3] < 255:
        grid_color[(col_iter + 1) % 3] += color_ramp_speed
        if grid_color[(col_iter + 1) % 3] > 255:
            grid_color[(col_iter + 1) % 3] = 255
    elif 0 < grid_color[col_iter % 3] <= 255:
        grid_color[col_iter % 3] -= color_ramp_speed
        if grid_color[col_iter % 3] < 0:
            grid_color[col_iter % 3] = 0
        if grid_color[col_iter % 3] == 0:
            col_iter += 1

    # Background & Grid setup
    screen.fill(BLACK)
    Main_grid.SetColor(grid_color)
    for x in Main_grid.h_lines:
        py.draw.line(x[0], x[1], x[2], x[3], x[4])
    for y in Main_grid.v_lines:
        py.draw.line(y[0], y[1], y[2], y[3], y[4])

    # Event Queue and Management
    for events in py.event.get():
        if events.type == py.QUIT:
            sys.exit()
        if events.type == py.KEYDOWN:
            if events.key == py.K_ESCAPE:
                py.quit()
                sys.exit()
            if events.key == py.K_w:
                Player1.orientation = [0, -1]
                Player1.orientation_changed = True
            if events.key == py.K_a:
                Player1.orientation = [-1, 0]
                Player1.orientation_changed = True
            if events.key == py.K_s:
                Player1.orientation = [0, 1]
                Player1.orientation_changed = True
            if events.key == py.K_d:
                Player1.orientation = [1, 0]
                Player1.orientation_changed = True

    # Update Objects
    Player1.move()
    for trace in Player1.tracer:
        py.draw.rect(screen, RED, trace[0])
    print([Player1.tracer[t][0] for t in range(len(Player1.tracer))])
    if Player1.cycle.collidelist([Player1.tracer[t][0] for t in range(len(Player1.tracer))]):
        Player1.color = BLUE
    else:
        Player1.color = RED

    py.draw.rect(screen, Player1.color, Player1.cycle)
    py.display.update()
    fpsClock.tick(FPS)
