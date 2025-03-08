from typing import Dict

import pygame
from pygame import Surface, SurfaceType

import constants.colors as colors
import tkinter as tk
from tkinter import messagebox

# Константы для Pygame
FONT_SIZE = 36

# Глобальная переменная для отслеживания состояния Pygame
pygame_initialized = False


def init_pygame(size, caption):
    """Инициализирует Pygame и возвращает экран и шрифт, если Pygame ещё не инициализирован."""
    global pygame_initialized
    if not pygame_initialized:
        pygame.init()
        pygame_initialized = True
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(caption)
    font = pygame.font.Font(None, FONT_SIZE)
    return screen, font


def draw_text(screen, text, font, color, x, y, center=False):
    """Универсальная функция для отрисовки текста."""
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(x, y)) if center else (x, y)
    screen.blit(surface, rect)


def show_input_window():
    """Окно ввода количества очагов и препятствий перед игрой с использованием tkinter."""
    root = tk.Tk()
    root.title("Настройки игры")
    root.geometry("300x300")

    tk.Label(root, text="Поле 10x10 клеток", font=("Arial", 12)).pack(pady=10)
    tk.Label(root, text="Введите:", font=("Arial", 10)).pack(pady=5)

    fire_var = tk.StringVar()
    obstacle_var = tk.StringVar()

    tk.Label(root, text="Очаги").pack()
    tk.Entry(root, textvariable=fire_var).pack()
    tk.Label(root, text="Препятствия").pack()
    tk.Entry(root, textvariable=obstacle_var).pack()

    result = [None, None]  # [fire, obstacles]

    def submit():
        try:
            fire = int(fire_var.get()) if fire_var.get() else 0
            obstacles = int(obstacle_var.get()) if obstacle_var.get() else 0
            if fire + obstacles > 50:
                messagebox.showerror("Ошибка", "Сумма > 50")
            else:
                result[0] = fire
                result[1] = obstacles
                root.quit()
        except ValueError:
            messagebox.showerror("Ошибка", "Только числа")

    tk.Button(root, text="Подтвердить", command=submit).pack(pady=10)

    root.mainloop()
    root.destroy()

    if result[0] is None or result[1] is None:
        return None, None
    return result[0], result[1]


def show_summary_window(fire_count, obstacle_count, iteration_count, total_reward):
    """Отображает итоговое окно с информацией о завершенной игре."""
    global pygame_initialized
    screen, font = init_pygame((400, 300), "Итоги игры")

    screen.fill(colors.WHITE)
    lines = [
        f"Итераций: {iteration_count}",
        f"Очагов: {fire_count}",
        f"Препятствий: {obstacle_count}",
        f"Награда: {total_reward}"
    ]
    for i, line in enumerate(lines):
        draw_text(screen, line, font, colors.BLACK, 50, 50 + i * 40)

    pygame.display.flip()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    # Не вызываем pygame.quit() здесь, чтобы избежать проблем с повторной инициализацией


def show_test_prompt_window():
    """Окно с запросом, хочет ли пользователь протестировать модель."""
    global pygame_initialized
    screen, font = init_pygame((400, 200), "Тестирование модели")

    screen.fill(colors.WHITE)
    draw_text(screen, "Тестировать модель?", font, colors.BLACK, 200, 40, center=True)
    yes_button = pygame.Rect(50, 100, 100, 50)
    no_button = pygame.Rect(250, 100, 100, 50)

    pygame.draw.rect(screen, colors.GREEN, yes_button)
    pygame.draw.rect(screen, colors.RED, no_button)
    draw_text(screen, "Да", font, colors.BLACK, yes_button.centerx, yes_button.centery, center=True)
    draw_text(screen, "Нет", font, colors.BLACK, no_button.centerx, no_button.centery, center=True)

    pygame.display.flip()
    choice = None
    while choice is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                choice = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_button.collidepoint(event.pos):
                    choice = True
                elif no_button.collidepoint(event.pos):
                    choice = False

    return choice  # Убираем pygame.quit() здесь


# Функция для явного завершения Pygame в конце программы
def quit_pygame():
    """Завершает Pygame, если он был инициализирован."""
    global pygame_initialized
    if pygame_initialized:
        pygame.quit()
        pygame_initialized = False


def _load_images(cell_size) -> dict[str, Surface | SurfaceType]:
    """Загружает и масштабирует изображения для рендеринга."""
    try:
        images = {
            "base": pygame.transform.scale(pygame.image.load("data/images/base.jpg"),
                                           (cell_size, cell_size)),
            "agent": pygame.transform.scale(pygame.image.load("data/images/agent.jpg"),
                                            (cell_size, cell_size)),
            "fire": pygame.transform.scale(pygame.image.load("data/images/fire.jpg"),
                                           (cell_size, cell_size)),
            "obstacle": pygame.transform.scale(pygame.image.load("data/images/tree.jpg"),
                                               (cell_size, cell_size)),
        }
        return images
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {e}")
