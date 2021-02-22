import numpy as np


def move_nearest_point(self, tag_corners):
    distance_short = 1000
    point_short = np.array([0, 0])
    center = [int(tag_corners[:, 0].mean()), int(tag_corners[:, 1].mean())]
    for point in self.path:
        distance = np.sqrt(np.sum(np.square(point - center)))
        if distance < distance_short:
            distance_short = distance
            point_short = point

    
    return point_short, distance_short


a = np.array([1, 0])
point_short = np.array([2, 2])
point_distance = point_short - a
print(point_distance)

tag_corners = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])

tag_corners += point_distance
print(tag_corners)
