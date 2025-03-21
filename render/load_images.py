import pygame
from pygame import Surface, SurfaceType


def load_images(cell_size) -> dict[str, Surface | SurfaceType]:
    try:
        images = {
            "base": pygame.transform.scale(pygame.image.load("data/images/base.png"), (cell_size, cell_size)),
            "agent": pygame.transform.scale(pygame.image.load("data/images/agent.png"), (cell_size, cell_size)),
            "fire": pygame.transform.scale(pygame.image.load("data/images/fire.png"), (cell_size, cell_size)),
            "obstacle": pygame.transform.scale(pygame.image.load("data/images/obstacle.png"), (cell_size, cell_size)),
            "wind": pygame.transform.scale(pygame.image.load("data/images/wind.png"), (cell_size, cell_size)),
            "houses": pygame.transform.scale(pygame.image.load("data/images/houses.png"), (cell_size, cell_size)),
            "tree": pygame.transform.scale(pygame.image.load("data/images/tree.png"), (cell_size, cell_size)),
        }
        return images
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load image: {e}")