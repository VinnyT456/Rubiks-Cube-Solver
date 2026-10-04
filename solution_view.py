import copy
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QListWidgetItem, QSizePolicy
)
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QBrush, QPen, QFont, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QTimer, QRectF, QSize
from solver import Cube
from solution import algorithm_word_conversion

# Same sticker gradients as the scanner grid
STICKER_GRADIENTS = {
    "y": ("#fff176", "#fdd835"),
    "w": ("#ffffff", "#e0e0e0"),
    "g": ("#66bb6a", "#388e3c"),
    "r": ("#ef5350", "#c62828"),
    "b": ("#42a5f5", "#1565c0"),
    "o": ("#ffa726", "#f57c00"),
}

def move_to_words(move):
    # PLL algorithms use spellings like "M2'" and "U2'", which turn the same as "M2" and "U2"
    normalized = move.replace("2'", "2")
    return algorithm_word_conversion.get(normalized, move)


class CubeNet(QWidget):
    """Unfolded cube: U on top, L F R B across the middle, D on the bottom."""
    layout_positions = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}

    def __init__(self):
        super().__init__()
        self.state = None
        self.setMinimumSize(440, 330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_state(self, state):
        self.state = state
        self.update()

    def paintEvent(self, event):
        if self.state is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gap = 10
        face = min((self.width() - 3 * gap) / 4, (self.height() - 2 * gap) / 3)
        x0 = (self.width() - (4 * face + 3 * gap)) / 2
        y0 = (self.height() - (3 * face + 2 * gap)) / 2
        pad = face * 0.04
        sticker_gap = face * 0.035
        sticker = (face - 2 * pad - 2 * sticker_gap) / 3

        for name, (col, row) in self.layout_positions.items():
            fx = x0 + col * (face + gap)
            fy = y0 + row * (face + gap)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1c1b24"))
            painter.drawRoundedRect(QRectF(fx, fy, face, face), face * 0.09, face * 0.09)

            for r in range(3):
                for c in range(3):
                    sx = fx + pad + c * (sticker + sticker_gap)
                    sy = fy + pad + r * (sticker + sticker_gap)
                    start, end = STICKER_GRADIENTS[str(self.state[name][r][c])]
                    gradient = QLinearGradient(sx, sy, sx + sticker, sy + sticker)
                    gradient.setColorAt(0, QColor(start))
                    gradient.setColorAt(1, QColor(end))
                    painter.setBrush(QBrush(gradient))
                    painter.drawRoundedRect(QRectF(sx, sy, sticker, sticker), sticker * 0.16, sticker * 0.16)


class StagePill(QWidget):
    """Stage name and move count, filled left to right as its moves are played."""
    def __init__(self, name, count):
        super().__init__()
        self.name = name
        self.count = count
        self.progress = 0.0
        self.setFixedHeight(44)
        self.setMinimumWidth(150)

    def set_progress(self, progress):
        self.progress = progress
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        radius = 12

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#a593e6") if self.progress > 0 else QColor("#efeaff"))
        painter.drawRoundedRect(rect, radius, radius)

        if self.progress > 0:
            gradient = QLinearGradient(0, 0, rect.width(), rect.height())
            gradient.setColorAt(0, QColor("#673ab7"))
            gradient.setColorAt(1, QColor("#9c27b0"))
            painter.save()
            painter.setClipRect(QRectF(0, 0, rect.width() * self.progress, rect.height()))
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(rect, radius, radius)
            painter.restore()

        painter.setPen(QColor("white") if self.progress > 0 else QColor("#4b3d79"))
        font = QFont()
        font.setPointSize(13)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        text_rect = rect.adjusted(14, 0, -14, 0)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.name)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, str(self.count))


class SolutionWindow(QWidget):
    """Steps through a solve move by move: the cube net, stage progress, and each move in words."""
    def __init__(self, start_state, stages, interval_ms=600):
        super().__init__()
        self.setWindowTitle("Cube Solution")
        self.setFixedSize(1140, 640)

        # Drop empty stages' "" entries and keep every stage, even an empty one, as a pill
        self.stages = [(name, [move for move in moves if move]) for name, moves in stages]
        self.moves = [(name, move) for name, moves in self.stages for move in moves]

        # Precompute the cube after every move so stepping backwards is free
        cube = Cube(copy.deepcopy(start_state))
        self.states = [cube.get_state()]
        for _, move in self.moves:
            cube.apply_move(move)
            self.states.append(cube.get_state())

        self.index = 0
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.step_forward)

        self.build_ui()
        self.set_style()
        self.show_index(0)

        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self.step_forward)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self.step_back)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.toggle_play)

    def build_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(24)

        self.net = CubeNet()
        main_layout.addWidget(self.net, 1)

        card = QWidget()
        card.setObjectName("card")
        card.setFixedWidth(560)
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(14)

        pill_grid = QGridLayout()
        pill_grid.setSpacing(10)
        self.pills = []
        for i, (name, moves) in enumerate(self.stages):
            pill = StagePill(name, len(moves))
            pill_grid.addWidget(pill, i // 3, i % 3)
            self.pills.append(pill)
        card_layout.addLayout(pill_grid)

        self.current_move = QLabel()
        self.current_move.setObjectName("currentMove")
        self.current_move.setWordWrap(True)
        self.current_move.setMinimumHeight(64)
        card_layout.addWidget(self.current_move)

        self.move_list = QListWidget()
        self.move_list.setObjectName("moveList")
        self.move_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for i, (_, move) in enumerate(self.moves):
            item = QListWidgetItem(f"{i + 1}.  {move}  —  {move_to_words(move)}")
            item.setSizeHint(QSize(0, 30))
            self.move_list.addItem(item)
        self.move_list.itemClicked.connect(lambda item: self.jump_to(self.move_list.row(item) + 1))
        card_layout.addWidget(self.move_list, 1)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.restart_btn = QPushButton("Restart")
        self.prev_btn = QPushButton("Prev")
        self.play_btn = QPushButton("Play")
        self.next_btn = QPushButton("Next")
        for button, name in [(self.restart_btn, "restartButton"), (self.prev_btn, "prevButton"),
                             (self.play_btn, "playButton"), (self.next_btn, "nextButton")]:
            button.setObjectName(name)
            button.setFixedHeight(44)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            controls.addWidget(button)
        self.restart_btn.clicked.connect(lambda: self.jump_to(0))
        self.prev_btn.clicked.connect(self.step_back)
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn.clicked.connect(self.step_forward)
        card_layout.addLayout(controls)

        self.counter = QLabel()
        self.counter.setObjectName("counter")
        card_layout.addWidget(self.counter)

        card.setLayout(card_layout)
        main_layout.addWidget(card)
        self.setLayout(main_layout)

    def set_style(self):
        self.setStyleSheet("""
            SolutionWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7f9cf5, stop:0.5 #b299f8, stop:1 #a15ee0);
            }
            QWidget#card {
                background: rgba(255, 255, 255, 240);
                border-radius: 24px;
            }
            QLabel#currentMove {
                color: #2a2340;
                font-size: 18px;
                font-weight: 600;
            }
            QLabel#counter {
                color: #6b5fa0;
                font-size: 14px;
                font-weight: 600;
            }
            QListWidget#moveList {
                background: transparent;
                border: none;
                color: #9a8fbf;
                font-size: 14px;
            }
            QListWidget#moveList::item:selected {
                background: #efeaff;
                color: #2a2340;
                border-radius: 8px;
            }
            QPushButton {
                color: white;
                border: none;
                border-radius: 14px;
                font: bold 15px;
            }
            QPushButton:disabled { color: rgba(255, 255, 255, 120); }
            QPushButton#restartButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3f51b5, stop:1 #2196f3); }
            QPushButton#prevButton, QPushButton#nextButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #667eea, stop:1 #764ba2); }
            QPushButton#playButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #9c27b0, stop:1 #e91e63); }
        """)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def show_index(self, index):
        self.index = index
        self.net.set_state(self.states[index])

        done = 0
        for pill, (_, moves) in zip(self.pills, self.stages):
            if len(moves) == 0:
                pill.set_progress(1.0 if index >= done else 0.0)
            else:
                pill.set_progress(min(max((index - done) / len(moves), 0.0), 1.0))
            done += len(moves)

        total = len(self.moves)
        if total == 0:
            self.current_move.setText("Already solved.")
        elif index == 0:
            self.current_move.setText("Hold the cube with yellow on top and green in front, then press Next or Play.")
        else:
            stage, move = self.moves[index - 1]
            self.current_move.setText(f"{stage} · {move}\n{move_to_words(move)}")
        if index == total and total > 0:
            self.current_move.setText(self.current_move.text() + "\nSolved!")

        if index > 0:
            self.move_list.setCurrentRow(index - 1)
            self.move_list.scrollToItem(self.move_list.item(index - 1), QListWidget.ScrollHint.PositionAtCenter)
        else:
            self.move_list.clearSelection()
            self.move_list.scrollToTop()

        self.counter.setText(f"Move {index} / {total}")
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < total)

    def step_forward(self):
        if self.index < len(self.moves):
            self.show_index(self.index + 1)
        if self.index >= len(self.moves):
            self.stop()

    def step_back(self):
        self.stop()
        if self.index > 0:
            self.show_index(self.index - 1)

    def jump_to(self, index):
        self.stop()
        self.show_index(index)

    def toggle_play(self):
        if self.timer.isActive():
            self.stop()
        else:
            if self.index >= len(self.moves):
                self.show_index(0)
            self.timer.start()
            self.play_btn.setText("Pause")

    def stop(self):
        self.timer.stop()
        self.play_btn.setText("Play")
