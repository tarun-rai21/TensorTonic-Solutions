def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    result = []
    for i in range(0, len(image)):
        row_i = []
        for j in range(0, len(image[0])):
            pixel = image[i][j]

            R = pixel[0]
            G = pixel[1]
            B = pixel[2]
            
            y = 0.299*R + 0.587*G + 0.114*B
            row_i.append(y)
        result.append(row_i)

    return result