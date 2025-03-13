import pygame
from pygame import Surface, SurfaceType


def load_images(cell_size) -> dict[str, Surface | SurfaceType]:
    try:
        images = {
            "base": pygame.transform.scale(pygame.image.load("data/images/base.jpg"), (cell_size, cell_size)),
            "agent": pygame.transform.scale(pygame.image.load("data/images/agent.jpg"), (cell_size, cell_size)),
            "fire": pygame.transform.scale(pygame.image.load("data/images/fire.jpg"), (cell_size, cell_size)),
            "obstacle": pygame.transform.scale(pygame.image.load("data/images/tree.jpg"), (cell_size, cell_size)),
        }
        return images
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load image: {e}")