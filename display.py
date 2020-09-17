import pygame
import os
import random

class UI:
    manual_click = False
    click_coordinates = []
    tile_size = 40
    colors = {
        "agent": (255, 57, 127, 100),
        "obstacle": (66, 75, 84),
        "target": (31, 243, 180),
        "historic_path": (170,170,170),
        "background": (215, 208, 200),
        "repetition": (255, 170, 170),
    }
    random_grading = []
    is_rendering = False
    manual_stop = False

    def __init__(self, grid_shape):
        self.width = self.tile_size * grid_shape[1]
        self.height = self.tile_size * grid_shape[0]
        for i in range(grid_shape[0]):
            self.random_grading.append([random.randint(-10, 10) for j in range(grid_shape[1])])

    def handle_input_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYUP and pygame.key.get_focused():
                if event.key == pygame.K_r:
                    self.is_rendering = True
                if event.key == pygame.K_ESCAPE:
                    self.manual_stop = True

    def render_grid(self, e, fps=4):
        t = self.tile_size
        pygame.init()
        dis=pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        while True:
            dis.fill(self.colors["background"])
            world = yield None
            if self.is_rendering:
                e = world["env"]
                a = world["agent"]
                if world["done"]: self.is_rendering = False
                for i, row in enumerate(e.grid):
                    for j, col in enumerate(row):
                        color_grade = self.random_grading[i][j]
                        if col == 0:
                            color = (35+int(color_grade/2), 103+int(color_grade/2), int(146+color_grade/2))
                            pygame.draw.rect(dis, color, [j * t, i * t, t, t], 0, 1)
                        else:
                            color = (36+int(color_grade/3), 39+int(color_grade/3), int(56+color_grade/3))
                            pygame.draw.rect(dis, color, [j * t, i * t, t, t], 0, 1) 
                for target in e.targets:
                    pygame.draw.rect(dis, self.colors["target"], [target[1] * t, target[0] * t, t, t], 0, 4)
                    pygame.draw.rect(dis, (36, 198, 151), [target[1] * t, target[0] * t, t, t], 1, 4)
                g = 0
                for pp in e.past_positions:
                    pygame.draw.rect(dis, (84+g, 43+g, 72+g), [pp[1] * t, pp[0] * t, t, t], 0, 0)
                    g += 1
                pygame.draw.rect(dis, self.colors["agent"], [int(a.position[1]*t), int(a.position[0]*t), t, t], 0, 4)
                pygame.display.update()
            clock.tick(fps)