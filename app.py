from pymongo.errors import ServerSelectionTimeoutError
from stable_baselines3 import PPO

from models.optuna_train import optimize_hyperparameters
from models.train_model import train_and_evaluate
from models.test_model import test_model
from mongo.mongo_integration_for_training import train_and_evaluate_with_mongo
from mongo.mongo_test_integration import test_model_with_mongo
from render.user_interface import show_input_window, show_test_prompt_window, show_start_window
from utils.get_console_data import get_data_from_user, get_int_input
from utils.logging_files import model_name, log_dir
from utils.logger import setup_logger

logger = setup_logger()


def run(render_mode=True):
    start_msg = "Starting model training..."
    final_msg = "Training completed!"
    try:
        if render_mode:
            choice = show_start_window()
            if choice == -1:  # Выход при закрытии окна
                return
            
            scenario, fire_count, obstacle_count = show_input_window()
            if scenario is None:  # Пользователь закрыл окно ввода
                return

            match choice:
                case 0:  # Обучение
                    logger.info(start_msg)
                    model = train_and_evaluate(scenario, fire_count, obstacle_count)
                    logger.info(final_msg)
                    if show_test_prompt_window():
                        test_model(scenario, model, fire_count, obstacle_count)
                case 1:  # Тест
                    model = PPO.load(log_dir + "/" + model_name + str(scenario))
                    test_model(scenario, model, fire_count, obstacle_count)
                case 2:  # Optuna
                    logger.info(start_msg)
                    optimize_hyperparameters(scenario, fire_count, obstacle_count)
                    logger.info(final_msg)
                case 3:  # Обучение с MongoDB
                    logger.info(start_msg)
                    experiment_name = input("Введите название эксперимента или пропустите шаг: ") or None
                    model, experiment_id = train_and_evaluate_with_mongo(scenario, fire_count, obstacle_count, experiment_name)
                    logger.info(final_msg + f" Experiment ID: {experiment_id}")
                    if show_test_prompt_window():
                        test_model_with_mongo(scenario, model, fire_count, obstacle_count, experiment_id)
                case 4:  # Тест с MongoDB
                    experiment_id = input("Введите ID эксперимента (оставьте пустым, если не требуется): ").strip() or None
                    model = PPO.load(log_dir + "/" + model_name + str(scenario))
                    test_model_with_mongo(scenario, model, fire_count, obstacle_count, experiment_id)

        else:
            promt = ("\n----- Выберите режим работы ------\n"
                     "1 - обучение модели\n"
                     "2 - тестирование модели\n"
                     "3 - подбор optuna\n"
                     "4 - обучение с записью в MongoDB\n"
                     "5 - тестирование с записью в MongoDB\n"
                    )

            mode = get_int_input(promt, min_val=1, max_val=5)
            print('_______________________________\n')
            scenario, fire_count, obstacle_count = get_data_from_user()
            match mode:
                case 1:
                    logger.info(start_msg)
                    model = train_and_evaluate(scenario, fire_count, obstacle_count)
                    logger.info(final_msg)
                    promt = ("Для проведения тестирования нажмите - 1\n"
                             "Для выхода - 0\n")
                    test = get_int_input(promt, min_val=0, max_val=1)
                    if test == 1:
                        test_model(scenario, model, fire_count, obstacle_count, render=False)
                case 2:
                    model = PPO.load(log_dir + "/" + model_name + str(scenario))
                    test_model(scenario, model, fire_count, obstacle_count, render=False)
                case 3:
                    logger.info(start_msg)
                    optimize_hyperparameters(scenario, fire_count, obstacle_count)
                    logger.info(final_msg)
                case 4:
                    logger.info(start_msg)
                    print("Введите название эксперимента или пропустите шаг")
                    experiment_name = input() or None
                    model, experiment_id = train_and_evaluate_with_mongo(scenario,
                                                                         fire_count,
                                                                         obstacle_count,
                                                                         experiment_name)
                    logger.info(final_msg + f" Experiment ID: {experiment_id}")
                    promt = ("Для проведения тестирования с MongoDB нажмите - 1\n"
                             "Для выхода - 0\n")
                    test = get_int_input(promt, min_val=0, max_val=1)
                    if test == 1:
                        test_model_with_mongo(
                            scenario,
                            model,
                            fire_count,
                            obstacle_count,
                            experiment_id,
                            render=False)
                case 5:
                    print("Введите ID эксперимента (оставьте пустым, если не требуется):")
                    experiment_id = input().strip()
                    experiment_id = experiment_id if experiment_id else None
                    model = PPO.load(log_dir + "/" + model_name + str(scenario))
                    test_model_with_mongo(scenario,
                                          model,
                                          fire_count,
                                          obstacle_count,
                                          experiment_id,
                                          render=False)

    except ValueError:
        print("Неизвестный сбой в работе")
    except ServerSelectionTimeoutError:
        print("Работа с БД возможна только на сервере")
    except FileNotFoundError:
        print("Модель не найдена.\n"
              "Проверьте сохранилась ли она")







# from pymongo.errors import ServerSelectionTimeoutError
# from stable_baselines3 import PPO

# from models.optuna_train import optimize_hyperparameters
# from models.train_model import train_and_evaluate
# from models.test_model import test_model
# from mongo.mongo_integration_for_training import train_and_evaluate_with_mongo
# from mongo.mongo_test_integration import test_model_with_mongo
# from render.user_interface import show_input_window, show_test_prompt_window, show_start_window
# from utils.get_console_data import get_data_from_user, get_int_input
# from utils.logging_files import model_name, log_dir
# from utils.logger import setup_logger

# logger = setup_logger()


# def run(render_mode=False):
#     start_msg = "Starting model training..."
#     final_msg = "Training completed!"
#     try:
#         if render_mode:  # Нельзя ввести не подходящие данные
#             if show_start_window():
#                 scenario, fire_count, obstacle_count = show_input_window()
#                 logger.info(start_msg)
#                 model = train_and_evaluate(scenario, fire_count, obstacle_count)
#                 logger.info(final_msg)
#                 if show_test_prompt_window():
#                     test_model(scenario, model, fire_count, obstacle_count)
#             else:
#                 scenario, fire_count, obstacle_count = show_input_window()
#                 model = PPO.load(log_dir + "/" + model_name + str(scenario))
#                 test_model(scenario, model, fire_count, obstacle_count)
#         else:
#             promt = ("\n----- Выберите режим работы ------\n"
#                      "1 - обучение модели\n"
#                      "2 - тестирование модели\n"
#                      "3 - подбор optuna\n"
#                      "4 - обучение с записью в MongoDB\n"
#                      "5 - тестирование с записью в MongoDB\n"
#                     )

#             mode = get_int_input(promt, min_val=1, max_val=6)
#             print('_______________________________\n')
#             scenario, fire_count, obstacle_count = get_data_from_user()
#             match mode:
#                 case 1:
#                     logger.info(start_msg)
#                     model = train_and_evaluate(scenario, fire_count, obstacle_count)
#                     logger.info(final_msg)
#                     promt = ("Для проведения тестирования нажмите - 1\n"
#                              "Для выхода - 0\n")
#                     test = get_int_input(promt, min_val=0, max_val=1)
#                     if test == 1:
#                         test_model(scenario, model, fire_count, obstacle_count, render=False)
#                 case 2:
#                     model = PPO.load(log_dir + "/" + model_name + str(scenario))
#                     test_model(scenario, model, fire_count, obstacle_count, render=False)
#                 case 3:
#                     logger.info(start_msg)
#                     optimize_hyperparameters(scenario, fire_count, obstacle_count)
#                     logger.info(final_msg)
#                 case 4:
#                     logger.info(start_msg)
#                     print("Введите название эксперимента или пропустите шаг")
#                     experiment_name = input() or None
#                     model, experiment_id = train_and_evaluate_with_mongo(scenario,
#                                                                          fire_count,
#                                                                          obstacle_count,
#                                                                          experiment_name)
#                     logger.info(final_msg + "Experiment ID: {experiment_id}")
#                     promt = ("Для проведения тестирования с MongoDB нажмите - 1\n"
#                              "Для выхода - 0\n")
#                     test = get_int_input(promt, min_val=0, max_val=1)
#                     if test == 1:
#                         test_model_with_mongo(
#                             scenario,
#                             model,
#                             fire_count,
#                             obstacle_count,
#                             experiment_id,
#                             render=False)
#                 case 5:
#                     print("Введите ID эксперимента (оставьте пустым, если не требуется):")
#                     experiment_id = input().strip()
#                     experiment_id = experiment_id if experiment_id else None
#                     model = PPO.load(model_name + str(scenario))
#                     test_model_with_mongo(scenario,
#                                           model,
#                                           fire_count,
#                                           obstacle_count,
#                                           experiment_id,
#                                           render=False)
                

#     except ValueError:
#         print("Неизвестный сбой в работе")
#     except ServerSelectionTimeoutError:
#         print("Работа с БД возможна только на сервере")
#     except FileNotFoundError:
#         print("Модель не найдена.\n"
#               "Проверьте сохранилась ли она")
