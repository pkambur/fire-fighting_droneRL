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
            "houses": pygame.transform.scale(pygame.image.load("data/images/houses.jpg"),
                                             (cell_size * 2, cell_size * 2)),
            "burned": pygame.transform.scale(pygame.image.load("data/images/burned.png"), (cell_size, cell_size)),
            "aircraft": pygame.transform.scale(pygame.image.load("data/images/aircraft.png"),
                                             (cell_size * 3, cell_size * 3)),
        }
        return images
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load image: {e}")
