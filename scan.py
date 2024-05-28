import cv2
import os
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

class Cube_Scanner:
    def __init__(self):
        self.color_pattern = {
            "red": (90, 90, 170),
            "green": (78, 220, 110),
            "blue": (228, 168, 110),
            "orange": (52, 108, 252),
            "yellow": (145, 231, 185),
            "white": (213, 220, 200)
        }
        self.tiles = []

    def classify_tile_color(self, color):
        # Find the closest color in the standard pattern
        min_dist = float('inf')
        closest_color = None
        for color_name, color_value in self.color_pattern.items():
            dist = np.linalg.norm(np.array(color_value) - np.array(color))
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
        return closest_color

    def get_average_color(self, tile):
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
    
    def cube_preprocess(self, image):
        if (image is not None):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            gray = cv2.fastNlMeansDenoising(gray, 10, 15, 7, 21)
            blurred = cv2.GaussianBlur(gray, (15, 15), 2)
            canny = cv2.Canny(blurred, 20, 40)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(canny, kernel, iterations=25)
            thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)[1]
            return thresh
    
    def find_stickers(self, image):
        # Preprocess the image
        new_image = self.cube_preprocess(image)
        contours = cv2.findContours(new_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour_areas = [w * h for cnt in contours for epsilon in [0.1 * cv2.arcLength(cnt, True)] for approx in [cv2.approxPolyDP(cnt, epsilon, True)] if len(approx) == 4 for (x, y, w, h) in [cv2.boundingRect(approx)]]
        average_area = sum(contour_areas) / len(contour_areas)
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            (x, y, w, h) = cv2.boundingRect(approx)
            if (len(approx) == 4 and w*h <= average_area/2):
                self.tiles.append((x, y, w, h))
    
    def scan(self, image_path):
        for filename in os.listdir(image_path):
            self.tiles.clear()
            filepath = os.path.join(image_path, filename)
            image = cv2.imread(filepath)
            if image is not None:
                self.find_stickers(image)
                self.sort_tiles()
                self.draw_tiles(image)
    
    def sort_tiles(self):
        self.tiles.sort(key=lambda tile: tile[0])
        grouped_tiles = [self.tiles[i:i+3] for i in range(0, len(self.tiles), 3)]
        sorted_tile = [sorted(group, key=lambda x: x[1], reverse=True) for group in grouped_tiles]
        self.tiles = [tile for group in sorted_tile for tile in group]

    def draw_tiles(self,image):
        for tile in self.tiles:
            cv2.rectangle(image,(tile[0:2]),(tile[0]+tile[2],tile[1]+tile[3]),(0,255,0),30)
            cv2.imshow("Image",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
if __name__ == "__main__":
    cube_scanner = Cube_Scanner()
    cube_scanner.scan("cube image")
