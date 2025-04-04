penalties = {"step": -0.02,  # штраф за шаг
             "obstacle": -0.2,  # штраф за препятствие
             "out_of_bounds": -0.1,  # штраф за выход за границы
             "crash": -0.3,
             "wind": -0.15,
             "stagnation": -0.1,
             "repeat_step": -0.05,
             "not_done": -5,
             "fire_spread": -0.5,
             "too_close": -0.02}

rewards = {
    "fire": 1,  # награда за тушение
    "near_fire": 0.05,  # бонус за приближение
    "final": 5,  # награда за тушение всех пожаров
    "new_step": 0.05,
    "wind_avoid_bonus": 0.01
}

rewards_sys_sc3 = {
    "step_penalty": -0.1,  # штраф за шаг
    "fire_reward": 50,  # 1 награда за тушение
    "final_reward": 50,  # награда за тушение всех пожаров
    "obstacle_penalty": -40,  # -0.5  #  штраф за препятствие
    "out_of_bounds_penalty": -30,  # -0.5  #  штраф за выход за границы
    "base_return_reward": 150,  # награда за возврат на базу после тушения очередного поджара
    "distance_coef": 0.1}  # 0.1 #Приближение к пожару
