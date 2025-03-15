def calculate_wind_cells(start, direction, strength, grid):
    x_start, y_start = start
    dx, dy = direction
    wind_cells = []

    for i in range(strength + 1):
        x = x_start + i * dx
        y = y_start + i * dy
        if 0 <= x < grid and 0 <= y < grid:
            wind_cells.append((x, y))
        else:
            break
    return wind_cells

