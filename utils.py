import numpy as np
import math

def get_heatmap(coordinate_list, target_shape, sigma=10.0):
    heatmap = np.zeros(target_shape[:2], dtype='float64')

    for x,y in coordinate_list:
        update_heatmap(heatmap, (x,y), sigma)

    return heatmap

def update_heatmap(heatmap, center, sigma):
    center_x, center_y = center
    height, width = heatmap.shape

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[y][x] = max(heatmap[y][x], math.exp(-exp))
            heatmap[y][x] = min(heatmap[y][x], 1.0)

