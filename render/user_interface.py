import pygame
import config.colors as colors

# Константы
WINDOW_WIDTH, WINDOW_HEIGHT = 500, 400
SUMMARY_SIZE = (400, 300)
TEST_SIZE = (400, 200)
FONT_SIZE = 36
INPUT_COLOR = (200, 200, 200)
ACTIVE_COLOR = (150, 150, 150)

def draw_text(screen, text, font, color, x, y, center=False):
    """Универсальная функция для отрисовки текста."""
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(x, y)) if center else (x, y)
    screen.blit(surface, rect)

def show_input_window():
    """Окно ввода количества очагов и препятствий перед игрой."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Настройки игры")
    font = pygame.font.Font(None, FONT_SIZE)

    input_boxes = [
        {"rect": pygame.Rect(300, 180, 100, 40), "text": "", "active": False},  # Очаги
        {"rect": pygame.Rect(300, 240, 100, 40), "text": "", "active": False}   # Препятствия
    ]
    hints = ["Очаги", "Препятствия"]
    error = ""
    running = True

    while running:
        screen.fill(colors.WHITE)
        draw_text(screen, "Поле 10x10 клеток", font, colors.BLACK, 50, 50)
        draw_text(screen, "Введите:", font, colors.BLACK, 50, 90)

        for idx, box in enumerate(input_boxes):
            color = ACTIVE_COLOR if box["active"] else INPUT_COLOR
            pygame.draw.rect(screen, color, box["rect"])
            draw_text(screen, box["text"], font, colors.BLACK, box["rect"].x + 10, box["rect"].y + 10)
            draw_text(screen, hints[idx], font, colors.BLACK, box["rect"].x - 150, box["rect"].y + 10)

        if error:
            draw_text(screen, error, font, colors.RED, 50, 300)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for box in input_boxes:
                    box["active"] = box["rect"].collidepoint(event.pos)
            elif event.type == pygame.KEYDOWN:
                for box in input_boxes:
                    if box["active"]:
                        if event.key == pygame.K_RETURN:
                            try:
                                fire = int(input_boxes[0]["text"]) if input_boxes[0]["text"] else 0
                                obstacles = int(input_boxes[1]["text"]) if input_boxes[1]["text"] else 0
                                if fire + obstacles > 50:
                                    error = "Сумма > 50"
                                else:
                                    running = False
                            except ValueError:
                                error = "Только числа"
                        elif event.key == pygame.K_BACKSPACE:
                            box["text"] = box["text"][:-1]
                        elif event.unicode.isdigit() and len(box["text"]) < 3:
                            box["text"] += event.unicode

    pygame.quit()
    return fire, obstacles

def show_summary_window(fire_count, obstacle_count, iteration_count, total_reward):
    """Отображает итоговое окно с информацией о завершенной игре."""
    pygame.init()
    screen = pygame.display.set_mode(SUMMARY_SIZE)
    pygame.display.set_caption("Итоги игры")
    font = pygame.font.Font(None, FONT_SIZE)

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
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

def show_test_prompt_window():
    """Окно с запросом, хочет ли пользователь протестировать модель."""
    pygame.init()
    screen = pygame.display.set_mode(TEST_SIZE)
    pygame.display.set_caption("Тестирование модели")
    font = pygame.font.Font(None, FONT_SIZE)

    screen.fill(colors.WHITE)
    draw_text(screen, "Тестировать модель?", font, colors.BLACK, TEST_SIZE[0] // 2, 40, center=True)
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
                pygame.quit()
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if yes_button.collidepoint(event.pos):
                    choice = True
                elif no_button.collidepoint(event.pos):
                    choice = False

    pygame.quit()
    return choice