import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

color_pattern = {
    "red": (90, 90, 170),
    "green": (78, 220, 110),
    "blue": (228, 168, 110),
    "orange": (52, 108, 252),
    "yellow": (145, 231, 185),
    "white": (213, 220, 200)
}


def classify_tile_color(color):
    # Find the closest color in the standard pattern
    min_dist = float('inf')
    closest_color = None
    for color_name, color_value in color_pattern.items():
        dist = np.linalg.norm(np.array(color_value) - np.array(color))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    return closest_color

def get_average_color(tile):
    # Calculate the average color of a tile using K-means clustering
    pixels = tile.reshape((-1, 3))

    # Apply K-means clustering to cluster pixels
    kmeans = KMeans(n_clusters=1, n_init="auto")
    kmeans.fit(pixels)

    # Get the dominant color (average color)
    dominant_color = kmeans.cluster_centers_[0]

    # Convert the dominant color to BGR format and integer type
    dominant_color_bgr = tuple(int(c) for c in dominant_color)

    return dominant_color_bgr

folder_path = "cube image"

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    image = cv2.imread(filepath)
    if (image is not None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.fastNlMeansDenoising(gray, 10, 15, 7, 21)
        blurred = cv2.GaussianBlur(gray, (15, 15), 2)
        canny = cv2.Canny(blurred, 20, 40)
        kernel = np.ones((5, 5), np.uint8)

        dilated = cv2.dilate(canny, kernel, iterations=25)
        thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        # Create a copy of the original image to draw rectangles
        image_copy = image.copy()

        contour_area = [] 

        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            if (len(approx) == 4):
                contour_area.append(w * h)

        average_area = sum(contour_area) / len(contour_area)

        images = []

        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            if (len(approx) == 4):
                if (w * h < average_area/2):
                    cropped_image = image[y:y+h, x:x+w]
                    average_color = get_average_color(cropped_image)
                    closest_color = classify_tile_color(average_color)
                    cv2.imshow("Cropped", cropped_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(closest_color)
        break
