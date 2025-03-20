from tkinter import messagebox
from stable_baselines3 import PPO

from models.train_model import train_and_evaluate
from models.test_model import test_model
from render.user_interface import show_input_window, show_test_prompt_window, show_start_window
from utils.logging_files import model_name
from utils.logger import setup_logger

logger = setup_logger()
render_mode = True


def run():
    if render_mode:
        if show_start_window():
            scenario, fire_count, obstacle_count = show_input_window()
            logger.info("Starting model training...")
            model = train_and_evaluate(scenario, fire_count, obstacle_count)
            logger.info("Training completed!")

            if show_test_prompt_window():
                test_model(scenario, model, fire_count, obstacle_count)
        else:
            try:
                scenario, fire_count, obstacle_count = show_input_window()
                model = PPO.load(model_name + str(scenario))
                test_model(scenario, model, fire_count, obstacle_count)
            except ValueError:
                messagebox.showerror("Ошибка",
                                     "Введенные данные не соответствуют модели!")
    else:
        print("Выберите режим работы\n"
              "1 - обучение модели\n"
              "2 - тестирование модели")
        mode = int(input())
        if mode == 1:
            try:
                scenario, fire_count, obstacle_count = get_data_from_user()
                logger.info("Starting model training...")
                model = train_and_evaluate(scenario, fire_count, obstacle_count)
                logger.info("Training completed!")
                print("Для проведения тестирования нажмите 1")
                if int(input()) == 1:
                    test_model(scenario, model, fire_count, obstacle_count, render=False)
                else:
                    print("Недопустимая операция")
            except ValueError:
                print("Нужно ввести целые числа.")
        elif mode == 2:
            try:
                scenario, fire_count, obstacle_count = get_data_from_user()
                model = PPO.load(model_name + str(scenario))
                test_model(scenario, model, fire_count, obstacle_count, render=False)
            except ValueError:
                print("Введенные данные не соответствуют модели!")
        else:
            print("Недопустимая операция")


def get_data_from_user():
    print('Выберите сценарий 1 or 2:')
    scenario = int(input())
    print('Количество очагов > 2:')
    fire_count = int(input())
    while fire_count < 3:
        print('Количество очагов  > 2:')
        fire_count = int(input())
    print('Количество препятствий :')
    obstacle_count = int(input())
    return scenario, fire_count, obstacle_count
