import logging
from tkinter import messagebox

from stable_baselines3 import PPO

from models.ppo import train_and_evaluate
from models.test_model import test_model
from render.user_interface import show_input_window, show_test_prompt_window, show_start_window
from utils.logging_files import model_name


def run():
    if show_start_window():
        fire_count, obstacle_count = show_input_window()
        logging.info("Starting model training...")
        model = train_and_evaluate(fire_count, obstacle_count)
        logging.info("Training completed!")

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
