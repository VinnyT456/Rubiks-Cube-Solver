import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from collections import Counter
import matplotlib.pyplot as plt

class Cube_Scanner:
    def __init__(self):
        self.color_pattern = {
            "red": (),
            "orange": (),
            "blue": (),
            "green": (),
            "white": (),
            "yellow": ()
        }
        self.color_names = list(self.color_pattern.keys())
        self.color_values = list(self.color_pattern.values())
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

    def classify_tile_color(self, bgr_color) -> str:
        kdtree = cKDTree(self.color_values)
        dist, idx = kdtree.query(bgr_color)
        return self.color_names[idx]
    
    def visualize_clusters(self, tile_roi, mask, kmeans, pixels):
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Create a blank image to visualize clusters
        cluster_img = np.zeros((50, 500, 3), dtype=np.uint8)

        # Calculate the number of pixels in each cluster
        counts = np.bincount(labels)

        # Sort clusters by the number of pixels
        sorted_indices = np.argsort(counts)[::-1]

        # Calculate the width of each segment
        total_pixels = pixels.shape[0]
        start = 0

        for i in sorted_indices:
            cluster_color = cluster_centers[i].astype(int)
            width = int(counts[i] / total_pixels * cluster_img.shape[1])
            end = start + width
            cluster_img[:, start:end, :] = cluster_color
            start = end

        # Plot the original ROI, mask, and color clusters
        plt.figure(figsize=(15, 5))

        # Original ROI
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(tile_roi, cv2.COLOR_BGR2RGB))
        plt.title('Original ROI')
        plt.axis('off')

        # Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Brightness Mask')
        plt.axis('off')

        # Color Clusters
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(cluster_img, cv2.COLOR_BGR2RGB))
        plt.title('Color Clusters')
        plt.axis('off')

        plt.show()

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

            #self.visualize_clusters(tile_roi, mask, kmeans, pixels)

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
                self.tiles.append((x, y, w, h))

    def predict_color(self):
        self.find_stickers()
        if (len(self.tiles) == 9 and self.check_valid_tiles_1()):
            dominant_colors = self.get_average_color()
            colors = [self.classify_tile_color(color) for color in dominant_colors]
            self.draw_tiles()
            print(colors if len(colors) == 9 or 8 else "")
            if (colors[4] not in self.centers):
                self.current_face.append(colors) 
            if (len(self.current_face) == 100):
                self.find_final_face()
                exit
        '''
        if (len(self.tiles) == 8 and self.check_valid_tiles_2()):
            dominant_colors = [self.get_average_color(tile) for tile in self.tiles]
            colors = [self.classify_tile_color(color) for color in dominant_colors]
            colors.insert(4, "white")
            self.draw_tiles()
            print(colors if len(colors) == 9 else "")
            if (colors[4] not in self.centers):
                self.current_face.append(colors) '''
    def find_final_face(self) -> None:
        tuple_list = [tuple(sublist) for sublist in self.current_face]
        face = Counter(tuple_list)
        print(face)

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
            print(*self.tiles,sep="\n")
            self.draw_tiles()
            self.bgr.append(dominant_colors)
            if (len(self.bgr) == 25):
                rgb_values = [y for x in self.bgr for y in x]
                average_rgb = tuple(np.mean(rgb_values, axis=0).astype(int))
                self.bgr.clear()
                self.color_pattern[self.color_names[self.current_color_index]] = average_rgb
                self.current_color_index += 1
                self.extract_color_state = False
                print("Done")
        elif (len(self.tiles) == 8 and self.check_valid_tiles_2() and self.check_center_piece() != True): 
            dominant_colors = self.get_average_color()
            print(*self.tiles,sep="\n")
            self.draw_tiles()
            self.bgr.append(dominant_colors)
            if (len(self.bgr) == 25):
                rgb_values = [y for x in self.bgr for y in x]
                average_rgb = tuple(np.mean(rgb_values,axis=0).astype(int))
                self.bgr.clear()
                self.color_pattern[self.color_names[self.current_color_index]] = average_rgb
                self.current_color_index += 1
                self.extract_color_state = False
                print("Done")

    def draw_extracted_color(self):
        cv2.rectangle(self.frame, (10,10), (50,50), (0,0,0), 3)
        cv2.rectangle(self.frame, (10,60), (50,100), (0,0,0), 3)
        cv2.rectangle(self.frame, (10,110), (50,150), (0,0,0), 3)
        cv2.rectangle(self.frame, (10,160), (50,200), (0,0,0), 3)
        cv2.rectangle(self.frame, (10,210), (50,250), (0,0,0), 3)
        cv2.rectangle(self.frame, (10,260), (50,300), (0,0,0), 3)
                
    def scan(self) -> None:
        while True:
            ret, self.frame = self.cam.read()
            if not ret:
                break
            key = cv2.waitKey(1) & 0xFF

            self.draw_extracted_color()

            cv2.imshow("Image", self.frame)

            if (self.extract_color_state and self.current_color_index < len(self.color_names)):
                self.extract_color()

            if (key == ord('q')):
                break
            elif (key == ord('e')):
                self.extract_color_state = True
        print(self.color_pattern)

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
