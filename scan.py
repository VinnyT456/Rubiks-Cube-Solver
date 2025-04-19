import cv2
import numpy as np
import os
import torch
import copy
from solver import *
from torchvision import transforms, models
from PIL import Image
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMenu
)
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject

class VideoThread(QThread):
    frame_processed = pyqtSignal(np.ndarray)
    colors_detected = pyqtSignal(list)

    def __init__(self, scanner):
        super().__init__()
        self.scanner = scanner
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.scanner.cam.read()
            if ret:

                self.process_frame(frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_processed.emit(rgb_frame)

    def process_frame(self, frame):
        if self.scanner.predict_color_state:
            self.scanner.frame = frame.copy()
            if not self.scanner.current_face:
                self.scanner.predict_color()
                if self.scanner.current_face:
                    self.colors_detected.emit(self.scanner.current_face)

class CubeGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout()
        self.grid.setSpacing(-1)
        self.grid.setContentsMargins(-1, -1, -1, -1)
        self.setLayout(self.grid)
        self.cells = {}

        self.current_face_scanned = []
        self.verified_face = []
        
        self.sticker_colors = {
            "White": (255, 255, 255),
            "Yellow": (0, 255, 255),
            "Orange": (0, 165, 255),
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0)
        }

        self.background_color = "#333333"
        self.border_color = "#404040"
        self.hover_border_color = "#606060"

        for row in range(3):
            for col in range(3):
                cell = QLabel()
                cell.setFixedSize(105, 105)
                cell.setStyleSheet(f"""
                    QLabel {{
                        background-color: {self.background_color};
                        border: 2px solid {self.border_color};
                    }}
                    QLabel:hover {{
                        border: 2px solid {self.hover_border_color};
                    }}
                """)
                cell.mousePressEvent = lambda event, r=row, c=col: self.show_color_menu(r, c)
                self.grid.addWidget(cell, row, col)
                self.cells[(row, col)] = cell

    def create_color_icon(self, rgb):
        pixmap = QPixmap(24, 24)
        pixmap.fill(QColor(*rgb))
        return QIcon(pixmap)

    def show_color_menu(self, row, col):
        if (len(self.current_face_scanned) >= 9):
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background-color: #404040;
                    color: white;
                    padding: 4px;
                }
                QMenu::item:selected {
                    background-color: #505050;
                }
            """)
            for name, bgr in self.sticker_colors.items():
                action = QAction(self.create_color_icon(bgr), name, self)
                action.triggered.connect(lambda _, r=row, c=col, rgb=bgr, n=name: 
                                    self.update_cell_and_storage(r, c, rgb, n))
                menu.addAction(action)
            menu.exec(self.cells[(row, col)].mapToGlobal(self.cells[(row, col)].rect().center()))

    def update_cell_and_storage(self, row, col, rgb, color_name):
        self.set_cell_color(row, col, rgb)
        index = row * 3 + col
        if index < len(self.current_face_scanned):
            self.current_face_scanned[index] = color_name
        else:
            self.current_face_scanned += [None] * (index - len(self.current_face_scanned) + 1)
            self.current_face_scanned[index] = color_name

    def set_cell_color(self, row, col, rgb):
        self.cells[(row, col)].setStyleSheet(f"""
            QLabel {{
                background-color: rgb({rgb[2]}, {rgb[1]}, {rgb[0]});
                border: 2px solid #404040;
            }}
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube Solver")
        self.setFixedSize(1040, 640)
        
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(700, 640)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #2A2A2A;")
        
        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(8)
        
        self.cube_grid = CubeGrid()
        control_layout.addWidget(self.cube_grid)

        # Add mode toggle button
        self.mode_toggle_btn = QPushButton("Sticker Mode")
        self.mode_toggle_btn.setFixedHeight(50)
        self.mode_toggle_btn.setCheckable(True)
        
        self.scan_btn = QPushButton("Start Scan")
        self.scan_btn.setFixedHeight(50)
        self.scan_btn.setCheckable(True)
        
        self.verify_btn = QPushButton("Verify Face")
        self.verify_btn.setFixedHeight(50)
        self.verify_btn.setCheckable(True)

        self.solver_btn = QPushButton("Full Cube Solver")
        self.solver_btn.setFixedHeight(50)
        self.solver_btn.setCheckable(True)
        
        control_layout.addWidget(self.scan_btn)
        control_layout.addWidget(self.verify_btn)
        control_layout.addWidget(self.mode_toggle_btn)
        control_layout.addWidget(self.solver_btn)
        
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(360)
        control_panel.setFixedHeight(600)
        
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(control_panel)
        
        self.set_style()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def set_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2A2A2A;
            }
            QPushButton {
                background-color: #444444;
                color: white;
                border: none;
                font: bold 16px;
                padding: 4px;
                        
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:checked {
                background-color: #666666;
            }
        """)

    def closeEvent(self, event):
        self.scanner.close_application()
        event.accept()

class CubeScanner(QObject):
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")
        self.window = MainWindow()
        self.window.scanner = self
        
        self.sticker_colors = {
            "Blue": (255,0,0),
            "Green": (0,255,0),
            "Orange": (0,165,255),
            "Red": (0,0,255),
            "White": (255,255,255),
            "Yellow": (0,255,255)
        }

        self.color_names = list(self.sticker_colors.keys())
        self.tiles = []
        self.scramble = []
        self.current_face = []
        self.centers = []
        self.frame = None

        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.predict_color_state = False
        self.draw_preview_face_state = False
        self.stickerless_mode = False

        self.solvers = ['Full Cube Solver','Cross Solver','OLL Solver','PLL Solver']
        self.solvers_index = 0

        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model = self.load_model("best_model.pth")
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.init_thread()
        self.connect_ui()

    def init_thread(self):
        self.video_thread = VideoThread(self)
        self.video_thread.frame_processed.connect(self.update_video_display)
        self.video_thread.colors_detected.connect(self.update_gui_grid)

    def connect_ui(self):
        self.window.scan_btn.clicked.connect(self.toggle_scan)
        self.window.verify_btn.clicked.connect(self.toggle_verify)
        self.window.mode_toggle_btn.clicked.connect(self.toggle_mode)
        self.window.solver_btn.clicked.connect(self.toggle_solver)

    @pyqtSlot(np.ndarray)
    def update_video_display(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.window.video_label.setPixmap(pixmap)

    @pyqtSlot(list)
    def update_gui_grid(self, colors):
        cell_position = [(2,0),(1,0),(0,0),(2,1),(1,1),(0,1),(2,2),(1,2),(0,2)]
        for position, color_name in zip(cell_position, colors):
            row, col = position
            self.window.cube_grid.set_cell_color(row, col, self.sticker_colors[color_name])
            self.window.cube_grid.current_face_scanned.append(color_name)

    def toggle_scan(self):
        self.predict_color_state = not self.predict_color_state
        self.window.scan_btn.setText("Stop Scan" if self.predict_color_state else "Start Scan")

    def toggle_verify(self):
        if (self.window.cube_grid.current_face_scanned != [] or self.current_face.clear() != []):
            self.scramble.append(self.window.cube_grid.current_face_scanned.copy())
            self.window.cube_grid.current_face_scanned.clear()
            self.current_face.clear()
            cell_position = [(2,0),(1,0),(0,0),(2,1),(1,1),(0,1),(2,2),(1,2),(0,2)]
            for position in cell_position:
                row, col = position
                self.window.cube_grid.set_cell_color(row, col, (51,51,51))
            if len(self.scramble) == 6:
                for _ in range(2): 
                    self.scramble[_] = self.scramble[_].copy()[::-1]

    def toggle_mode(self):
        self.stickerless_mode = not self.stickerless_mode
        self.window.mode_toggle_btn.setText("Stickerless Mode" if self.stickerless_mode else "Sticker Mode")

    def toggle_solver(self):
        self.solvers_index = (self.solvers_index + 1) % len(self.solvers)
        self.window.solver_btn.setText(self.solvers[self.solvers_index])

    def load_model(self, model_path, num_classes=6):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        model.eval()
        model.to(self.device)
        return model

    def classify_tile_color(self) -> str:
        colors = []
        class_names = ['Blue', 'Green', 'Orange', 'Red', 'White', 'Yellow']
        for tile in self.tiles:
            x, y, w, h = tile
            tile_roi = self.frame[y:y+h, x:x+w]

            # Convert numpy array (OpenCV image) to PIL Image
            tile_pil = Image.fromarray(cv2.cvtColor(tile_roi, cv2.COLOR_BGR2RGB))

            # Apply the transformation to the PIL image
            image_tensor = self.test_transform(tile_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)

            predicted_class = class_names[predicted.item()]
            colors.append(predicted_class)

        return colors
    
    def cube_preprocess(self) -> None:
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 2)
        canny = cv2.Canny(blurred, 20, 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(canny, kernel)
        thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)[1]
        return thresh

    def find_stickers(self) -> None:
        if (self.stickerless_mode):
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
                if (0.8 <= aspect_ratio <= 1.2):
                    self.tiles.append((x, y, w, h))

        else:
            pass

    def predict_color(self):
        self.find_stickers()
        if (len(self.tiles) == 8 and self.check_valid_tiles_1 and self.check_center_piece != True):
            self.find_center_piece()
        if (len(self.tiles) == 9 and self.check_valid_tiles_1() and self.check_center_piece()):
            colors = self.classify_tile_color() 
            if (colors[4] not in self.centers):   
                self.centers.append(colors[4])
                self.current_face = colors

    def position_windows(self):
        # Get screen dimensions
        screen = self.app.primaryScreen().availableGeometry()
        
        # OpenCV window position (left side)
        cv2.namedWindow("Cube Scanner", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Cube Scanner", screen.left(), screen.center().y()-480)
        
        # Qt window position (right side)
        self.window.move(screen.left() + 640, screen.center().y()-480)
    
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
    
    def find_center_piece(self):
        # Select the relevant tiles
        self.tiles_copy = [self.tiles[1], self.tiles[3], self.tiles[4], self.tiles[6]]
        
        # Determine bounding box
        left = min(tile[0] for tile in self.tiles_copy)
        right = max(tile[0] + tile[2] for tile in self.tiles_copy)
        top = min(tile[1] for tile in self.tiles_copy)
        bottom = max(tile[1] + tile[3] for tile in self.tiles_copy)

        # Calculate center of the bounding box
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        # Calculate average width and height of the tiles
        average_width = sum(tile[2] for tile in self.tiles_copy) // len(self.tiles_copy)
        average_height = sum(tile[3] for tile in self.tiles_copy) // len(self.tiles_copy)

        # Calculate top-left corner of the center piece
        center_piece_x = center_x - average_width // 2
        center_piece_y = center_y - average_height // 2

        # The center piece dimensions
        center_piece_w = average_width
        center_piece_h = average_height

        center_piece = (center_piece_x, center_piece_y, center_piece_w, center_piece_h)
        self.tiles.insert(4,center_piece)
    
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

    def run(self):
        self.window.show()
        self.video_thread.start()
        self.app.exec()

if __name__ == "__main__":
    scanner = CubeScanner()
    scanner.run()