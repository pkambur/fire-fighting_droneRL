import pygame
import config.colors as colors


def show_input_window():
    """ Окно ввода количества очагов и препятствий перед игрой. """
    if not pygame.get_init():
        pygame.init()

    screen = pygame.display.set_mode((500, 400))
    pygame.display.set_caption("Настройки игры")
    font = pygame.font.Font(None, 36)

    input_boxes = [
        {"rect": pygame.Rect(300, 180, 100, 40), "text": "", "active": False},  # Поле для очагов
        {"rect": pygame.Rect(300, 240, 100, 40), "text": "", "active": False}  # Поле для препятствий
    ]

    instructions = [
        "Имеется поле 10x10 клеток.",
        "Введите:"
    ]

    error_message = ""
    running = True
    while running:
        screen.fill(colors.WHITE)

        y = 50
        for line in instructions:
            text_surface = font.render(line, True, colors.BLACK)
            screen.blit(text_surface, (50, y))
            y += 40

        for idx, box in enumerate(input_boxes):
            pygame.draw.rect(screen, (200, 200, 200), box["rect"])
            text_surface = font.render(box["text"], True, colors.BLACK)
            screen.blit(text_surface, (box["rect"].x + 10, box["rect"].y + 10))
            hint = "Количество очагов" if idx == 0 else "Количество препятствий"
            hint_surface = font.render(hint, True, colors.BLACK)
            screen.blit(hint_surface, (box["rect"].x - 300, box["rect"].y + 10))

        if error_message:
            error_surface = font.render(error_message, True, colors.RED)
            screen.blit(error_surface, (50, 300))

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
                                fire_count = int(input_boxes[0]["text"]) if input_boxes[0]["text"] else 0
                                obstacle_count = int(input_boxes[1]["text"]) if input_boxes[1]["text"] else 0

                                if fire_count + obstacle_count > 50:
                                    error_message = "Ошибка: Сумма очагов и препятствий не должна превышать 50."
                                else:
                                    running = False
                            except ValueError:
                                error_message = "Ошибка: Введите числовые значения."
                        elif event.key == pygame.K_BACKSPACE:
                            box["text"] = box["text"][:-1]
                        elif event.unicode.isdigit():
                            box["text"] += event.unicode

    pygame.quit()
    return fire_count, obstacle_count

def show_summary_window(fire_count, obstacle_count, iteration_count, total_reward):
    """ Отображает итоговое окно с информацией о завершенной игре. """
    if not pygame.get_init():
        pygame.init()

    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Итоги игры")
    font = pygame.font.Font(None, 36)

    screen.fill(colors.WHITE)

    text1 = font.render(f"Количество итераций: {iteration_count}", True, colors.BLACK)
    text2 = font.render(f"Количество очагов: {fire_count}", True, colors.BLACK)
    text3 = font.render(f"Количество препятствий: {obstacle_count}", True, colors.BLACK)
    text4 = font.render(f"Суммарная награда: {total_reward}", True, colors.BLACK)

    screen.blit(text1, (50, 50))
    screen.blit(text2, (50, 90))
    screen.blit(text3, (50, 130))
    screen.blit(text4, (50, 170))

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

    pygame.quit()

def show_test_prompt_window():
    """ Окно с запросом, хочет ли пользователь протестировать модель. """
    if not pygame.get_init():
        pygame.init()

    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Тестирование модели")
    font = pygame.font.Font(None, 36)

    screen.fill(colors.WHITE)

    text = font.render("Хотите протестировать модель? (Да/Нет)", True, colors.BLACK)
    yes_button = pygame.Rect(50, 100, 100, 50)
    no_button = pygame.Rect(250, 100, 100, 50)

    pygame.draw.rect(screen, colors.GREEN, yes_button)
    pygame.draw.rect(screen, colors.RED, no_button)

    yes_text = font.render("Да", True, colors.BLACK)
    no_text = font.render("Нет", True, colors.BLACK)

    screen.blit(text, (50, 20))
    screen.blit(yes_text, (yes_button.x + 30, yes_button.y + 10))
    screen.blit(no_text, (no_button.x + 30, no_button.y + 10))

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