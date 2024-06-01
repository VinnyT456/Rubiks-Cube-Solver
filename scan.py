import cv2
import numpy as np
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
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def classify_tile_color(self, color):
        min_dist = float('inf')
        closest_color = None
        for color_name, color_value in self.color_pattern.items():
            dist = np.linalg.norm(np.array(color_value) - np.array(color))
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
        return closest_color

    def get_average_color(self, tile):
        pixels = tile.reshape((-1, 3))
        kmeans = KMeans(n_clusters=1, n_init="auto")
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]
        dominant_color_bgr = tuple(int(c) for c in dominant_color)
        return dominant_color_bgr
    
    def cube_preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 2)
        canny = cv2.Canny(blurred, 20, 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(canny, kernel)
        thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)[1]
        return thresh
    
    def find_stickers(self, image):
        new_image = self.cube_preprocess(image)
        contours, _ = cv2.findContours(new_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        if not contour_areas:
            return

        average_area = sum(contour_areas) / len(contour_areas)
        min_area = average_area * 0.5
        max_area = average_area * 1.5

        self.tiles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 0.8 <= aspect_ratio <= 1.2:
                self.tiles.append((x, y, w, h))
        
        if (len(self.tiles) == 9):
            self.draw_tiles(image)

    def scan(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            self.find_stickers(frame)
            cv2.imshow("Image", frame)
            self.center_window("Image")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def sort_tiles(self):
        self.tiles.sort(key=lambda tile: tile[0])
        grouped_tiles = [self.tiles[i:i+3] for i in range(0, len(self.tiles), 3)]
        sorted_tile = [sorted(group, key=lambda x: x[1], reverse=True) for group in grouped_tiles]
        self.tiles = [tile for group in sorted_tile for tile in group]

    def draw_tiles(self, image):
        for tile in self.tiles:
            cv2.rectangle(image, (tile[0], tile[1]), (tile[0]+tile[2], tile[1]+tile[3]), (0, 255, 0), 2)

    def center_window(self, window_name):
        screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        if screen_width > 0 and screen_height > 0:
            cv2.moveWindow(window_name, int(screen_width/2 - self.width/2), int(screen_height/2 - self.height/2))

if __name__ == "__main__":
    cube_scanner = Cube_Scanner()
    cube_scanner.scan()
