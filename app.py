from tkinter import messagebox
from stable_baselines3 import PPO

from models.train_model import train_and_evaluate
from models.test_model import test_model
from render.user_interface import show_input_window, show_test_prompt_window, show_start_window
from utils.logging_files import model_name
from utils.logger import setup_logger

logger = setup_logger()


def run():
    if show_start_window():
        fire_count, obstacle_count = show_input_window()
        logger.info("Starting model training...")
        model = train_and_evaluate(fire_count, obstacle_count)
        logger.info("Training completed!")

        if show_test_prompt_window():
            test_model(model, fire_count, obstacle_count)
    else:
        try:
            model = PPO.load(model_name)
            fire_count, obstacle_count = show_input_window()
            test_model(model, fire_count, obstacle_count)
        except ValueError:
            messagebox.showerror("Ошибка",
                                 "Введенные данные не соответсвуют модели!")
