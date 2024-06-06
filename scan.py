import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

class Cube_Scanner:
    def __init__(self):
        self.color_pattern = {
            'red'   : (0, 0, 255),
            'orange': (0, 165, 255),
            'blue'  : (255, 0, 0),
            'green' : (0, 255, 0),
            'white' : (255, 255, 255),
            'yellow': (0, 255, 255)
        }
        self.color_names = list(self.color_pattern.keys())
        self.color_values = list(self.color_pattern.values())
        self.kdtree = cKDTree(self.color_values)
        self.tiles = []
        self.frame = None
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scramble = []

    def classify_tile_color(self, bgr_color) -> str:
        dist, idx = self.kdtree.query(bgr_color)
        return self.color_names[idx]

    def get_average_color(self, tile, min_brightness=80, max_brightness=230, k=1) -> tuple:
        x, y, w, h = tile
        tile_roi = self.frame[y:y+h, x:x+w]

        # Convert ROI to grayscale to check brightness
        tile_roi_gray = cv2.cvtColor(tile_roi, cv2.COLOR_BGR2GRAY)

        # Mask to exclude too dark or too bright pixels
        mask = cv2.inRange(tile_roi_gray, min_brightness, max_brightness)

        # Apply the mask to filter out pixels
        filtered_pixels = tile_roi[mask > 0]

        pixels = filtered_pixels.reshape((-1, 3))

        if pixels.size == 0:
            print("Warning: No pixels found for tile:", tile)
            return (0, 0, 0)  # Return a default color or handle accordingly

        # Apply K-means clustering to cluster pixels
        kmeans = KMeans(n_clusters=k, n_init="auto")
        kmeans.fit(pixels)

        # Find the most dominant cluster
        dominant_cluster_index = np.argmax(np.bincount(kmeans.labels_))
        dominant_color = kmeans.cluster_centers_[dominant_cluster_index]

        # Convert the dominant color to BGR format and integer type
        dominant_color = tuple(int(c) for c in dominant_color)

        return dominant_color
        
    
    def cube_preprocess(self) -> None:
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 2)
        canny = cv2.Canny(blurred, 20, 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(canny, kernel)
        thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)[1]
        return thresh
    
    def find_stickers(self) -> None:
        new_image = self.cube_preprocess()
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

        
        if (len(self.tiles) == 9 and self.check_valid_tiles_1()):
            dominant_colors = [self.get_average_color(tile) for tile in self.tiles]
            colors = [self.classify_tile_color(color) for color in dominant_colors]
            self.draw_tiles()
            print(colors)
        '''if (len(self.tiles) == 8):
            self.special_sort()                           
            print(self.tiles)
            self.draw_tiles()'''
    
    def check_valid_tiles_1(self) -> bool:
        self.sort_tiles()
        grouped_tiles = [self.tiles[i:i+3] for i in range(0, len(self.tiles), 3)]
        for group in grouped_tiles:
            avg = np.mean(group, axis=0)
            for x in group:
                if (x[0] < avg[0] - 5) or (x[0] > avg[0] + 5):
                    return False
        return True
    
    def scan(self) -> None:
        while True:
            ret, self.frame = self.cam.read()
            if not ret:
                break
            self.find_stickers()
            cv2.imshow("Image", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def sort_tiles(self) -> None:
        self.tiles.sort(key=lambda tile: tile[0])
        grouped_tiles = [self.tiles[i:i+3] for i in range(0, len(self.tiles), 3)]
        sorted_tile = [sorted(group, key=lambda x: x[1], reverse=True) for group in grouped_tiles]
        self.tiles = [tile for group in sorted_tile for tile in group]

    def special_sort(self) -> None:
        self.tiles.sort(key=lambda tile: tile[0])
        grouped_tiles = [self.tiles[:3],self.tiles[3:5],self.tiles[5:]]
        sorted_tile = [sorted(group, key=lambda x: x[1], reverse=True) for group in grouped_tiles]
        self.tiles = [tile for group in sorted_tile for tile in group]

    def draw_tiles(self) -> None:
        for tile in self.tiles:
            cv2.rectangle(self.frame, (tile[0], tile[1]), (tile[0]+tile[2], tile[1]+tile[3]), (0, 255, 0),10)
            '''cv2.imshow("Image", self.frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
if __name__ == "__main__":
    cube_scanner = Cube_Scanner()
    cube_scanner.scan()
