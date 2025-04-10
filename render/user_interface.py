import pygame
import tkinter as tk
from tkinter import messagebox

from pygame import Surface

from envs import GRID_SIZE
from render import WEIGHT, HEIGHT, FONT_SIZE
import constants.colors as colors

pygame_initialized = False


def init_pygame(size, caption):
    global pygame_initialized
    if not pygame_initialized:
        pygame.init()
        pygame_initialized = True
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(caption)
    font = pygame.font.Font(None, FONT_SIZE)
    return screen, font


def draw_text(screen: Surface, text: str, font: pygame.font,
              color: tuple[int, int, int], x: int, y: int, center=False):
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(x, y)) if center else (x, y)
    screen.blit(surface, rect)


def show_input_window():
    root = tk.Tk()
    root.title("Настройки игры")
    root.geometry("350x350")
    tk.Label(root, text=f"Поле {GRID_SIZE}x{GRID_SIZE}", font=("Arial", 12)).pack(pady=10)
    tk.Label(root, text="Введите:", font=("Arial", 10)).pack(pady=5)

    scenario_var = tk.StringVar()
    tk.Label(root, text="Сценарий 1 - 2").pack()
    tk.Entry(root, textvariable=scenario_var).pack()

    fire_var = tk.StringVar()
    tk.Label(root, text="Очаги").pack()
    tk.Entry(root, textvariable=fire_var).pack()

    obstacle_var = tk.StringVar()
    tk.Label(root, text="Препятствия").pack()
    tk.Entry(root, textvariable=obstacle_var).pack()
    result = [None, None, None]

    def submit():
        try:
            scenario = int(scenario_var.get()) if scenario_var.get() else 1
            fire = int(fire_var.get()) if fire_var.get() else 0
            obstacles = int(obstacle_var.get()) if obstacle_var.get() else 0
            if fire + obstacles > GRID_SIZE ** 2 // 2:
                messagebox.showerror("Ошибка", "Сумма > 50")
            else:
                result[0] = scenario
                result[1] = fire
                result[2] = obstacles
                root.quit()
        except ValueError:
            messagebox.showerror("Ошибка", "Вводить только цифры!")

    tk.Button(root, text="Начать", command=submit).pack(pady=10)
    root.mainloop()
    root.destroy()
    if result[0] is None or result[1] is None or result[2] is None:
        return None, None, None
    return result[0], result[1], result[2]


def show_summary_window(fire_count, fire_done, obstacle_count, iteration_count, total_reward):
    global pygame_initialized
    screen, font = init_pygame((WEIGHT, HEIGHT), "Game Summary")
    screen.fill(colors.WHITE)
    lines = [
        f"Количество итераций: {iteration_count}",
        f"Количество очагов: {fire_count}",
        f"Потушено: {fire_done}",
        f"Препятствий: {obstacle_count}",
        f"Суммарная награда: {round(total_reward, 2)}"
    ]
    for i, line in enumerate(lines):
        draw_text(screen, line, font, colors.BLACK, 50, 50 + i * 40)
    pygame.display.flip()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


def choice_window(caption: str, question: str, choices: list[str], button_width=150, button_height=50):
    global pygame_initialized
    # Окно размером 700x300
    screen, font = init_pygame((700, 300), caption)
    screen.fill(colors.WHITE)
    draw_text(screen, question, font, colors.BLACK, 700 // 2, 40, center=True)
    
    buttons = []
    for i, choice in enumerate(choices):
        if i < 3:  # Первый ряд: "Обучение", "Тест", "Optuna"
            button = pygame.Rect(50 + i * (button_width + 20), 100, button_width, button_height)
        else:  # Второй ряд: "Обучение с MongoDB", "Тест с MongoDB"
            # Увеличиваем ширину кнопок второго ряда до 290 и смещаем вторую кнопку
            button_width_extended = 290
            if i == 3:  # "Обучение с MongoDB"
                button = pygame.Rect(50, 170, button_width_extended, button_height)
            else:  # "Тест с MongoDB"
                button = pygame.Rect(50 + button_width_extended + 20, 170, button_width_extended, button_height)
        buttons.append(button)
        # Задаем цвета в зависимости от индекса кнопки
        if i == 0:  # "Обучение"
            color = colors.GREEN
        elif i == 1:  # "Тест"
            color = colors.RED
        elif i == 2:  # "Optuna"
            color = colors.GRAY
        else:  # "Обучение с MongoDB" и "Тест с MongoDB"
            color = colors.BLUE
        pygame.draw.rect(screen, color, button)
        draw_text(screen, choice, font, colors.BLACK, button.centerx, button.centery, center=True)
    
    pygame.display.flip()
    choice = None
    while choice is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                choice = -1  # Выход
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, button in enumerate(buttons):
                    if button.collidepoint(event.pos):
                        choice = i
                        break
    quit_pygame()
    return choice


def show_test_prompt_window():
    caption = "Test Model"
    question = "Запустить тестирование модели?"
    choices = ["Да", "Нет"]
    choice = choice_window(caption, question, choices, button_width=120)
    return choice == 0  # 0 - Да, 1 - Нет


def show_start_window():
    caption = "Choose Action"
    question = "Сделайте выбор"
    choices = ["Обучение", "Тест", "Optuna", "Обучение с MongoDB", "Тест с MongoDB"]
    choice = choice_window(caption, question, choices, button_width=150, button_height=50)
    return choice  # Возвращаем индекс выбранного варианта (0-4, или -1 при выходе)


def quit_pygame():
    global pygame_initialized
    if pygame_initialized:
        pygame.quit()
        pygame_initialized = False




# import pygame
# import tkinter as tk
# from tkinter import messagebox

# from pygame import Surface

# from envs import GRID_SIZE
# from render import WEIGHT, HEIGHT, FONT_SIZE
# import constants.colors as colors

# pygame_initialized = False


# def init_pygame(size, caption):
#     global pygame_initialized
#     if not pygame_initialized:
#         pygame.init()
#         pygame_initialized = True
#     screen = pygame.display.set_mode(size)
#     pygame.display.set_caption(caption)
#     font = pygame.font.Font(None, FONT_SIZE)
#     return screen, font


# def draw_text(screen: Surface, text: str, font: pygame.font,
#               color: tuple[int, int, int], x: int, y: int, center=False):
#     surface = font.render(text, True, color)
#     rect = surface.get_rect(center=(x, y)) if center else (x, y)
#     screen.blit(surface, rect)


# def show_input_window():
#     root = tk.Tk()
#     root.title("Настройки игры")
#     root.geometry("350x350")
#     tk.Label(root, text=f"Поле {GRID_SIZE}x{GRID_SIZE}", font=("Arial", 12)).pack(pady=10)
#     tk.Label(root, text="Введите:", font=("Arial", 10)).pack(pady=5)

#     scenario_var = tk.StringVar()
#     tk.Label(root, text="Сценарий 1 - 2").pack()
#     tk.Entry(root, textvariable=scenario_var).pack()

#     fire_var = tk.StringVar()
#     tk.Label(root, text="Очаги").pack()
#     tk.Entry(root, textvariable=fire_var).pack()

#     obstacle_var = tk.StringVar()
#     tk.Label(root, text="Препятствия").pack()
#     tk.Entry(root, textvariable=obstacle_var).pack()
#     result = [None, None, None]

#     def submit():
#         try:
#             scenario = int(scenario_var.get()) if scenario_var.get() else 1
#             fire = int(fire_var.get()) if fire_var.get() else 0
#             obstacles = int(obstacle_var.get()) if obstacle_var.get() else 0
#             if fire + obstacles > GRID_SIZE ** 2 // 2:
#                 messagebox.showerror("Ошибка", "Сумма > 50")
#             else:
#                 result[0] = scenario
#                 result[1] = fire
#                 result[2] = obstacles
#                 root.quit()
#         except ValueError:
#             messagebox.showerror("Ошибка", "Вводить только цифры!")

#     tk.Button(root, text="Начать", command=submit).pack(pady=10)
#     root.mainloop()
#     root.destroy()
#     if result[0] is None or result[1] is None or result[2] is None:
#         return None, None, None
#     return result[0], result[1], result[2]


# def show_summary_window(fire_count, fire_done, obstacle_count, iteration_count, total_reward):
#     global pygame_initialized
#     screen, font = init_pygame((WEIGHT, HEIGHT), "Game Summary")
#     screen.fill(colors.WHITE)
#     lines = [
#         f"Количество итераций: {iteration_count}",
#         f"Количество очагов: {fire_count}",
#         f"Потушено: {fire_done}",
#         f"Препятствий: {obstacle_count}",
#         f"Суммарная награда: {round(total_reward, 2)}"
#     ]
#     for i, line in enumerate(lines):
#         draw_text(screen, line, font, colors.BLACK, 50, 50 + i * 40)
#     pygame.display.flip()
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False


# def choice_window(caption: str, question: str, choices: list[str], button_width=100):
#     global pygame_initialized
#     screen, font = init_pygame((WEIGHT, WEIGHT // 2), caption)
#     screen.fill(colors.WHITE)
#     draw_text(screen, question,
#               font, colors.BLACK, WEIGHT // 2, 40, center=True)
#     yes_button = pygame.Rect(50, 100, button_width, 50)
#     no_button = pygame.Rect(300, 100, button_width, 50)
#     pygame.draw.rect(screen, colors.GREEN, yes_button)
#     pygame.draw.rect(screen, colors.RED, no_button)
#     draw_text(screen, choices[0], font, colors.BLACK, yes_button.centerx, yes_button.centery, center=True)
#     draw_text(screen, choices[1], font, colors.BLACK, no_button.centerx, no_button.centery, center=True)
#     pygame.display.flip()
#     choice = None
#     while choice is None:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 choice = False
#             elif event.type == pygame.MOUSEBUTTONDOWN:
#                 if yes_button.collidepoint(event.pos):
#                     choice = True
#                 elif no_button.collidepoint(event.pos):
#                     choice = False
#     quit_pygame()
#     return choice


# def show_test_prompt_window():
#     caption = "Test Model"
#     question = "Запустить тестирование модели?"
#     choices = ["Да", "Нет"]
#     choice = choice_window(caption, question, choices)
#     return choice


# def show_start_window():
#     caption = "Choose Action"
#     question = "Сделайте выбор"
#     choices = ["Обучение", "Тест"]
#     choice = choice_window(caption, question, choices, button_width=120)
#     return choice


# def quit_pygame():
#     global pygame_initialized
#     if pygame_initialized:
#         pygame.quit()
#         pygame_initialized = False
