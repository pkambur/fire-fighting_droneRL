from constants.grid import GRID_SIZE


def get_data_from_user():
    """Получает входные данные от пользователя с валидацией"""

    print("\n----- Настройки сценария ------")
    scenario_description = """
    Доступные сценарии:
    1 - Сценарий поиска и тушения пожаров
    2 - Сценарий дотушивания
    """
    print(scenario_description)

    scenario = get_int_input("Выберите сценарий: ", min_val=1, max_val=3)

    if scenario == 1:
        fire_count = get_int_input(
            "Введите количество очагов (минимум 3): ",
            min_val=3,
            max_val=GRID_SIZE ** 2 * 0.5 - 1
        )
    else:
        fire_count = 4

    obstacle_count = get_int_input(
        "Введите количество препятствий: ",
        min_val=0,
        max_val=GRID_SIZE ** 2 * 0.5 - fire_count
    )

    print(f"\n-----Выбраны параметры-----:")
    print(f"- Сценарий: {scenario}")
    print(f"- Очаги: {fire_count}")
    print(f"- Препятствия: {obstacle_count}")
    return scenario, fire_count, obstacle_count


def get_int_input(prompt, min_val=None, max_val=None, default=None):
    """Функция для ввода целых чисел с валидацией"""
    while True:
        try:
            value = input(prompt)
            if default and value == "":
                return default
            value = int(value)
            if min_val is not None and value < min_val:
                print(f"Значение должно быть >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Значение должно быть <= {max_val}")
                continue
            return value
        except ValueError:
            print("Ошибка: введите целое число")
