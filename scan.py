import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

class Cube_Scanner:
    def __init__(self):
        self.color_pattern = {
            "White": (),
            "Yellow": (),
            "Orange": (),
            "Red": (),
            "Blue": (),
            "Green": ()
        }
        self.sticker_colors = {
            "White": (255, 255, 255),
            "Yellow": (0,255,255),
            "Orange": (0,165,255),
            "Red": (0,0,255),
            "Blue": (255,0,0),
            "Green": (0,255,0)
        }
        self.color_names = list(self.color_pattern.keys())
        self.tiles = []
        self.scramble = []
        self.centers = []
        self.current_face = []
        self.bgr = []
        self.frame = None
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_color_index = 0
        self.predict_color_state = False
        self.extract_color_state = False
        self.draw_color_state = False
        self.draw_preview_face_state = False

    def classify_tile_color(self, bgr_color) -> str:
        color_values = list(self.color_pattern.values())
        kdtree = cKDTree(color_values)
        dist, idx = kdtree.query(bgr_color)
        return self.color_names[idx]
    def get_average_color(self, min_brightness=80, max_brightness=250) -> list[tuple]:
        average_colors = []
        for tile in self.tiles:
            x, y, w, h = tile
            tile_roi = self.frame[y:y+h, x:x+w]

            # Convert ROI to grayscale to check brightness
            tile_roi_gray = cv2.cvtColor(tile_roi, cv2.COLOR_BGR2GRAY)

            # Mask to exclude too dark or too bright pixels
            mask = cv2.inRange(tile_roi_gray, min_brightness, max_brightness)

            # Apply the mask to filter out pixels
            filtered_pixels = tile_roi[mask > 0]

            # Reshape filtered pixels to a 2D array where each row is a pixel (B, G, R)
            pixels = tile_roi.reshape((-1, 3))

            if pixels.size == 0:
                return (0, 0, 0)  # Return a default color or handle accordingly

            # Apply K-means clustering to cluster pixels
            kmeans = KMeans(n_clusters=1, n_init='auto')
            kmeans.fit(pixels)

            # Find the most dominant cluster
            dominant_cluster_index = np.argmax(np.bincount(kmeans.labels_))
            dominant_color = kmeans.cluster_centers_[dominant_cluster_index]

            # Convert the dominant color to BGR format and integer type
            dominant_color = tuple(int(c) for c in dominant_color)

            dominant_color_rgb = (dominant_color[2], dominant_color[1], dominant_color[0])

            average_colors.append(dominant_color_rgb)
        
        return average_colors
        
    
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
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.tiles.append((x, y, w, h))

    def predict_color(self):
        self.find_stickers()
        if (len(self.tiles) == 9 and self.check_valid_tiles_1() and self.check_center_piece()):
            dominant_colors = self.get_average_color()
            colors = [self.classify_tile_color(color) for color in dominant_colors]
            self.draw_tiles()
            print(colors if len(colors) == 9 or 8 else "")
            if (colors[4] not in self.centers):
                self.current_face.append(colors) 
            self.centers.append(colors[4])
        elif (len(self.tiles) == 8 and self.check_valid_tiles_2() and self.check_center_piece() != True):
            dominant_colors = self.get_average_color()
            dominant_colors.insert(4, self.color_pattern["White"])
            colors = [self.classify_tile_color(color) for color in dominant_colors]
            self.draw_tiles()
            print(colors if len(colors) == 9 else "")
            if (colors[4] not in self.centers):
                self.current_face.append(colors)
            self.centers.append(colors[4])

    def check_center_piece(self) -> bool:
        # Determine bounding box
        left = min(tile[0] for tile in self.tiles)
        right = max(tile[0] + tile[2] for tile in self.tiles)
        top = min(tile[1] for tile in self.tiles)
        bottom = max(tile[1] + tile[3] for tile in self.tiles)

        # Create a coverage matrix
        width = right - left
        height = bottom - top
        coverage = np.zeros((height, width), dtype=bool)

        # Mark the coverage of each tile in the matrix
        for x, y, w, h in self.tiles:
            coverage[(y-top):(y-top+h), (x-left):(x-left+w)] = True

        # Calculate the center region within the bounding box
        center_x = width // 2
        center_y = height // 2

        # Define the center piece region (assuming center piece is 1x1)
        # Check if the center of the bounding box is covered
        has_center_piece = coverage[center_y, center_x]

        return has_center_piece
    
    def check_valid_tiles_1(self) -> bool:
        self.sort_tiles()
        grouped_tiles = [self.tiles[i:i+3] for i in range(0, len(self.tiles), 3)]
        for group in grouped_tiles:
            avg = np.mean(group, axis=0)
            for x in group:
                if (x[0] < avg[0] - 5) or (x[0] > avg[0] + 5):
                    return False
        return True
    
    def check_valid_tiles_2(self) -> bool:
        self.special_sort()
        tile_1 = self.tiles[0]
        tile_2 = self.tiles[-1]
        for x in self.tiles:
            if (tile_1[0]-5 >= x[0] >= tile_2[0]+5) and (tile_1[1]-5 >= x[1] >= tile_2[1]+5):
                return False
        return True
    
    def extract_color(self) -> None:
        self.find_stickers()
        if (len(self.tiles) == 9 and self.check_valid_tiles_1() and self.check_center_piece):
            dominant_colors = self.get_average_color()
            self.bgr.append(dominant_colors)
            rgb_values = [y for x in self.bgr for y in x]
            average_rgb = tuple(np.mean(rgb_values, axis=0).astype(int))
            self.bgr.clear()
            self.color_pattern[self.color_names[self.current_color_index]] = average_rgb
        elif (len(self.tiles) == 8 and self.check_valid_tiles_2() and self.check_center_piece() != True): 
            dominant_colors = self.get_average_color()
            self.bgr.append(dominant_colors)
            rgb_values = [y for x in self.bgr for y in x]
            average_rgb = tuple(np.mean(rgb_values,axis=0).astype(int))
            self.bgr.clear()
            self.color_pattern[self.color_names[self.current_color_index]] = average_rgb

    def draw_extracted_color(self):
        if (self.draw_color_state):
            overlay = self.frame.copy()
            # Draw rectangles
            for position, (color_name, color) in enumerate(self.color_pattern.items()):
                if color != ():
                    cv2.rectangle(overlay, (10, 10 + (50 * position)), (50, 50 + (50 * position)), (int(color[2]),int(color[1]),int(color[0])),-1)
                else:
                    cv2.rectangle(overlay, (10, 10 + (50 * position)), (50, 50 + (50 * position)), (0, 0, 0),2)
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            for position, color_name in enumerate(self.color_pattern.keys()):
                cv2.putText(overlay, color_name, (60, 35 + position * 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            # Blend overlay with the frame
            alpha = 1.0  # Transparency factor
            self.frame = cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0)

    def draw_preview_face(self):
        if (self.draw_preview_face_state):
            overlay = self.frame.copy()
            # Draw rectangles
            for i in range(3):
                for j in range(3):
                    top_left = (10 + j * 70, 10 + i * 70)
                    bottom_right = (10 + (j + 1) * 70, 10 + (i + 1) * 70)
                    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), 2)
            alpha = 1.0  # Transparency factor
            self.frame = cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0)

                
    def scan(self) -> None:
        while True:
            ret, self.frame = self.cam.read()

            if not ret:
                break
            key = cv2.waitKey(1) & 0xFF

            self.draw_extracted_color()

            self.draw_preview_face()

            cv2.imshow("Image", self.frame)

            if (self.extract_color_state and self.current_color_index < len(self.color_names)):
                self.extract_color()
                if (self.color_pattern[self.color_names[self.current_color_index]] != ()):
                    self.current_color_index += 1
                    self.extract_color_state = False

            if (self.predict_color_state and self.extract_color_state != True):
                self.predict_color()

            if (key == ord('q')):
                exit()
            elif (key == ord('e')):
                self.extract_color_state = not self.extract_color_state
                self.draw_color_state = True
            elif (key == ord('s') and all(value != () for value in self.color_pattern.values())):
                self.predict_color_state = not self.predict_color_state
                self.draw_preview_face_state = not self.draw_preview_face_state
                self.draw_color_state = False

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
            cv2.rectangle(self.frame, (tile[0], tile[1]), (tile[0]+tile[2], tile[1]+tile[3]), (0, 255, 0),5)
            cv2.imshow("Image",self.frame)

if __name__ == "__main__":
    cube_scanner = Cube_Scanner()
    cube_scanner.scan()
