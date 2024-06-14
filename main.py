import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QSlider, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Лабораторная_4')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.open_button = QPushButton('Открыть изображение', self)
        self.open_button.clicked.connect(self.open_image)

        self.effect_combo = QComboBox(self)
        self.effect_combo.addItem("Резкость")
        self.effect_combo.addItem("Размытие")
        self.effect_combo.addItem("Тиснение")
        self.effect_combo.addItem("Медианный фильтр")
        self.effect_combo.addItem("Canny детектор")
        self.effect_combo.addItem("Roberts детектор")
        self.effect_combo.currentIndexChanged.connect(self.apply_effect)
        self.effect_combo.setEnabled(False)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.apply_effect)
        self.slider.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.open_button)
        layout.addWidget(self.effect_combo)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image = None
        self.original_image = None

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.original_image = self.image.copy()
            self.display_image(self.image)
            self.slider.setEnabled(True)
            self.effect_combo.setEnabled(True)

    def display_image(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(out_image).scaledToHeight(500))
        self.image_label.setScaledContents(True)

    def apply_effect(self):
        if self.original_image is not None:
            effect = self.effect_combo.currentText()
            if effect == "Резкость":
                self.apply_sharpen()
            elif effect == "Размытие":
                self.apply_motion_blur()
            elif effect == "Тиснение":
                self.apply_emboss()
            elif effect == "Медианный фильтр":
                self.apply_median_filter()
            elif effect == "Canny детектор":
                self.apply_canny()
            elif effect == "Roberts детектор":
                self.apply_roberts()

    def apply_sharpen(self):
        if self.original_image is not None:
            alpha = self.slider.value() / 10.0  # Normalize alpha to a reasonable range
            blurred_image = cv2.GaussianBlur(self.original_image, (7, 7), 0)
            sharpened_image = cv2.addWeighted(self.original_image, 1.0 + alpha, blurred_image, -alpha, 0)
            self.display_image(sharpened_image)

    def apply_motion_blur(self):
        if self.original_image is not None:
            size = self.slider.value()
            if size == 0:
                self.display_image(self.original_image)
                return
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            motion_blurred = cv2.filter2D(self.original_image, -1, kernel_motion_blur)
            self.display_image(motion_blurred)

    def apply_emboss(self):
        if self.original_image is not None:
            kernel_emboss = np.array([[ -2, -1, 0],
                                      [ -1,  1, 1],
                                      [  0,  1, 2]])
            embossed_image = cv2.filter2D(self.original_image, -1, kernel_emboss)
            embossed_image = cv2.cvtColor(embossed_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for better effect
            self.display_image(embossed_image)

    def apply_median_filter(self):
        if self.original_image is not None:
            ksize = self.slider.value() // 2 * 2 + 1  # Ensure ksize is odd
            if ksize == 1:
                self.display_image(self.original_image)
                return
            median_filtered = cv2.medianBlur(self.original_image, ksize)
            self.display_image(median_filtered)

    def apply_canny(self):
        if self.original_image is not None:
            threshold1 = self.slider.value()
            threshold2 = threshold1 * 2
            canny_edges = cv2.Canny(self.original_image, threshold1, threshold2)
            canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for display
            self.display_image(canny_edges)

    def apply_roberts(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            roberts_x = cv2.filter2D(gray_image, -1, kernel_roberts_x)
            roberts_y = cv2.filter2D(gray_image, -1, kernel_roberts_y)
            roberts_edges = cv2.addWeighted(np.abs(roberts_x), 0.5, np.abs(roberts_y), 0.5, 0)
            roberts_edges = cv2.cvtColor(roberts_edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for display
            self.display_image(roberts_edges)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
