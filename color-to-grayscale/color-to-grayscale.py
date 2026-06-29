import numpy as np
def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    result = []
    for i in range(0, len(image)):
        row_i = []
        for j in range(0, len(image[0])):
            y = 0.299*image[i][j][0] + 0.587*image[i][j][1] + 0.114*image[i][j][2]
            row_i.append(y)
        result.append(row_i)

    return result