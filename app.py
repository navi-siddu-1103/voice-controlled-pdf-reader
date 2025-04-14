import sys
import os
import threading
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QSlider,
                             QProgressBar, QMessageBox, QScrollArea, QFrame, )
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPalette, QColor, QImage, QPainter, QPen, QPainterPath
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QRect, QTimer, QPoint

from PyQt5.QtWidgets import QGraphicsOpacityEffect
from PyQt5 import QtCore
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QRadialGradient
from PyQt5.QtCore import QEasingCurve
from PyQt5.QtGui import QLinearGradient
#from PyQt6 import sip
import speech_recognition as sr
import torch
import torchaudio
from torch.nn import functional as F
import fitz  # PyMuPDF
import resources_rc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class RoundButton(QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(40, 40)
        self.setMaximumSize(40, 40)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def paintEvent(self, event):
        # Use a custom paint event to make the button round
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create gradient
        if self.isDown():
            gradient = QRadialGradient(self.rect().center(), self.width() / 2)
            gradient.setColorAt(0, QColor("#1E88E5"))
            gradient.setColorAt(1, QColor("#0D47A1"))
        else:
            gradient = QRadialGradient(self.rect().center(), self.width() / 2)
            gradient.setColorAt(0, QColor("#2196F3"))
            gradient.setColorAt(1, QColor("#1976D2"))

        # Draw circle button
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.rect())

        # Draw icon or text
        painter.setPen(QColor("white"))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())


class SlidingStackedWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.current_widget = None

    def setCurrentWidget(self, widget):
        # Remove old widget if exists
        if self.current_widget:
            self.layout.removeWidget(self.current_widget)
            self.current_widget.hide()

        # Add new widget with animation
        self.current_widget = widget
        self.layout.addWidget(widget)
        widget.show()

        # Add animation
        self.anim = QPropertyAnimation(widget, b"geometry")
        self.anim.setDuration(300)
        self.anim.setStartValue(QRect(self.width(), 0, widget.width(), widget.height()))
        self.anim.setEndValue(QRect(0, 0, widget.width(), widget.height()))
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()


# Voice command model and audio processor classes remain the same
class VoiceCommandModel(torch.nn.Module):
    def __init__(self, n_input=40, n_output=4):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_input, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(0.3)
        self.lstm = torch.nn.LSTM(64, 64, batch_first=True)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)  # (batch, channels, time) -> (batch, time, channels)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Create dummy model
model = VoiceCommandModel()

# Initialize with random weights
for param in model.parameters():
    param.data.normal_(0, 0.01)

# Save model weights
torch.save(model.state_dict(), "models/voice_commands.pth")
print("Dummy model saved to models/voice_commands.pth")

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=40):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )

    def preprocess(self, waveform):
        # Convert to tensor if it's not already
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # If 2D, take first channel
        if waveform.dim() == 2:
            waveform = waveform[0]

        # Normalize waveform
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # Compute mel spectrogram
        mel = self.mel_spectrogram(waveform)

        # Convert to dB scale
        mel = torchaudio.transforms.AmplitudeToDB()(mel)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        return mel


class VoiceControlThread(QThread):
    command_detected = pyqtSignal(str)
    status_update = pyqtSignal(str)
    listening_state = pyqtSignal(bool)

    def __init__(self, model_path=None):
        super().__init__()
        self.running = True
        self.paused = False
        self.processor = AudioProcessor()

        # Initialize model
        self.model = VoiceCommandModel()

        # Load model weights if available, otherwise start from scratch
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.status_update.emit("Model loaded successfully!")
            except Exception as e:
                self.status_update.emit(f"Error loading model: {str(e)}")
                # Initialize dummy model for demonstration
                self._initialize_dummy_model()
        else:
            self.status_update.emit("No model found - initializing with dummy weights")
            self._initialize_dummy_model()

        self.model.eval()

        # Command mapping
        self.commands = {0: "up", 1: "down", 2: "background", 3: "unknown"}

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 2000
        self.recognizer.dynamic_energy_threshold = True

    def _initialize_dummy_model(self):
        """Initialize model with dummy weights for demonstration"""
        # This is just for demo purposes when no trained model is available
        for param in self.model.parameters():
            param.data.normal_(0.0, 0.01)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.wait()

    def process_audio(self, audio_data, sample_rate):
        # Resample if needed
        if sample_rate != self.processor.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.processor.sample_rate
            )
            audio_tensor = torch.tensor(np.frombuffer(audio_data, dtype=np.int16), dtype=torch.float32)
            audio_tensor = resampler(audio_tensor)
        else:
            audio_tensor = torch.tensor(np.frombuffer(audio_data, dtype=np.int16), dtype=torch.float32)

        # Normalize
        audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-8)

        # Process audio
        mel = self.processor.preprocess(audio_tensor)

        # Make prediction
        with torch.no_grad():
            # Add batch dimension
            mel = mel.unsqueeze(0)
            output = self.model(mel)
            probabilities = F.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()

            # Return command only if confidence is high enough
            if confidence > 0.6 and predicted_idx < 2:  # Only up/down commands
                return self.commands[predicted_idx]
            elif predicted_idx == 2:  # Background noise
                return None
            else:
                return None  # Unknown or low confidence

    def run(self):
        self.status_update.emit("Voice control started")

        # For simple command recognition without ML model
        keywords = {
            "up": ["up", "scroll up", "move up", "go up", "upward", "top"],
            "down": ["down", "scroll down", "move down", " go down", "downward", "bottom"],
            "next": ["next", "go", "next page", "forward"],
            "previous": ["previous", "back", "previous page", "backward"]
        }

        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                try:
                    self.status_update.emit("Listening...")
                    self.listening_state.emit(True)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    self.listening_state.emit(False)
                    # Two approaches: ML model and speech-to-text
                    # Try ML model first (faster response)
                    command = self.process_audio(audio.get_raw_data(), audio.sample_rate)
                    if command:
                        self.command_detected.emit(command)
                        self.status_update.emit(f"Command detected: {command}")
                        time.sleep(0.5)  # Prevent multiple rapid commands
                        continue

                    # If ML model doesn't detect command, try speech-to-text
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        self.status_update.emit(f"Heard: {text}")

                        for direction, phrases in keywords.items():
                            if any(phrase in text for phrase in phrases):
                                self.command_detected.emit(direction)
                                self.status_update.emit(f"Command detected: {direction}")
                                break

                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        self.status_update.emit("Speech service unavailable")

                except sr.WaitTimeoutError:
                    pass
                except Exception as e:
                    self.status_update.emit(f"Error: {str(e)}")

                time.sleep(0.1)

        self.status_update.emit("Voice control stopped")


class MicrophoneIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active = False
        self.setMinimumSize(30, 30)
        self.setMaximumSize(30, 30)

        # Create pulse animation
        self.animation = QPropertyAnimation(self, b"minimumWidth")
        self.animation.setDuration(800)
        self.animation.setStartValue(30)
        self.animation.setEndValue(40)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Make it loop
        self.animation.finished.connect(self.toggleAnimation)

    def setActive(self, active):
        self.active = active
        if active and not self.animation.state():
            self.animation.start()
        elif not active:
            self.animation.stop()
            self.setMinimumSize(30, 30)
            self.setMaximumSize(30, 30)
        self.update()

    def toggleAnimation(self):
        if self.active:
            # Reverse the animation
            self.animation.setDirection(
                QPropertyAnimation.Backward
                if self.animation.direction() == QPropertyAnimation.Forward
                else QPropertyAnimation.Forward
            )
            self.animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.active:
            color = QColor("#F44336")  # Red when active
        else:
            color = QColor("#9E9E9E")  # Gray when inactive

        # Draw microphone
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)

        # Circle background
        circle_rect = self.rect().adjusted(5, 5, -5, -5)
        painter.drawEllipse(circle_rect)

        # Microphone icon
        painter.setPen(QColor("white"))
        painter.setBrush(QColor("white"))

        # Simplified microphone shape
        mic_width = circle_rect.width() * 0.4
        mic_height = circle_rect.height() * 0.6
        mic_x = circle_rect.center().x() - mic_width / 2
        mic_y = circle_rect.center().y() - mic_height / 2

        painter.drawRoundedRect(int(mic_x), int(mic_y), int(mic_width), int(mic_height), 2, 2)
        painter.drawRect(circle_rect.center().x() - 1,
                         mic_y + mic_height,
                         2,
                         mic_height * 0.3)

        # Base
        base_width = mic_width * 1.5
        painter.drawRoundedRect(
            circle_rect.center().x() - base_width / 2,
            mic_y + mic_height + mic_height * 0.3 - 1,
            base_width,
            3,
            1, 1
        )


class PDFViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc = None
        self.current_page = 0
        self.zoom_factor = 1.0
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #ffffff; border-radius: 6px;")

    def load_document(self, filepath):
        try:
            self.doc = fitz.open(filepath)
            self.current_page = 0
            self.render_page()
            return self.doc.page_count
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            return 0

    def render_page(self):
        if not self.doc:
            return

        if 0 <= self.current_page < self.doc.page_count:
            page = self.doc[self.current_page]
            zoom_matrix = fitz.Matrix(2.0 * self.zoom_factor, 2.0 * self.zoom_factor)
            pixmap = page.get_pixmap(matrix=zoom_matrix)

            # Convert pixmap to QImage and then to QPixmap
            img = QImage(pixmap.samples, pixmap.width, pixmap.height,
                         pixmap.stride, QImage.Format_RGB888)
            qpixmap = QPixmap.fromImage(img)

            self.setPixmap(qpixmap)
            self.setMinimumSize(qpixmap.width(), qpixmap.height())
            return True
        return False

    def next_page(self):
        if self.doc and self.current_page < self.doc.page_count - 1:
            self.current_page += 1
            self.render_page()
            return self.current_page
        return -1

    def prev_page(self):
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.render_page()
            return self.current_page
        return -1

    def go_to_page(self, page_num):
        if self.doc and 0 <= page_num < self.doc.page_count:
            self.current_page = page_num
            self.render_page()
            return self.current_page
        return -1

    def zoom_in(self):
        self.zoom_factor *= 1.25
        self.render_page()

    def zoom_out(self):
        self.zoom_factor /= 1.25
        self.render_page()


class StatusBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMaximumHeight(40)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 0)

        # Gradient background
        self.setAutoFillBackground(True)
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#263238"))
        gradient.setColorAt(1, QColor("#37474F"))
        palette.setBrush(QPalette.Window, gradient)
        self.setPalette(palette)

        # Page info
        self.page_label = QLabel("Page: 0/0")
        self.page_label.setStyleSheet("color: white; font-weight: bold;")

        # Page slider
        self.page_slider = QSlider(Qt.Horizontal)
        self.page_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4A4A4A;
                height: 8px;
                background: #222222;
                margin: 2px 0;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #1976D2;
                width: 18px;
                margin: -8px 0;
                border-radius: 9px;
            }
        """)

        # Status message
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #BBDEFB;")

        # Microphone indicator
        self.mic_indicator = MicrophoneIndicator()

        # Add everything to layout
        self.layout.addWidget(self.page_label)
        self.layout.addWidget(self.page_slider, 1)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.mic_indicator)


class ButtonToolbar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMaximumHeight(80)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)
        self.setStyleSheet("QFrame { border-bottom: 1px solid rgba(255, 255, 255, 50); }")

        def resizeEvent(self, event):
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor("#1976D2"))
            gradient.setColorAt(1, QColor("#0D47A1"))
            palette = self.palette()
            palette.setBrush(QPalette.Window, gradient)
            self.setPalette(palette)
            super().resizeEvent(event)

        # Create buttons with consistent style
        button_style = """
            QPushButton {
                background-color: rgba(255, 255, 255, 30);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 12px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 60);
            }
            QPushButton:pressed {
                background-color: rgba(0, 0, 0, 30);
            }
            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 10);
                color: rgba(255, 255, 255, 120);
            }
        """

        # Document control buttons
        self.open_btn = QPushButton("Open PDF")
        self.open_btn.setIcon(QIcon("icons/open.jpg"))
        self.open_btn.setStyleSheet(button_style)

        # Voice control button
        self.voice_control_btn = QPushButton("Start Voice Control")
        self.voice_control_btn.setIcon(QIcon("icons/mic.jpg"))
        self.voice_control_btn.setStyleSheet(button_style)

        # Auto-scroll button
        self.auto_scroll_btn = QPushButton("Auto-Scroll")
        self.auto_scroll_btn.setIcon(QIcon("icons/scroll.jpg"))
        self.auto_scroll_btn.setStyleSheet(button_style)

        # Navigation buttons in a container frame
        nav_frame = QFrame()
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(5)

        self.prev_btn = QPushButton()
        self.prev_btn.setIcon(QIcon("icons/previous.jpg"))
        self.prev_btn.setIconSize(QSize(24, 24))
        self.prev_btn.setFixedSize(36, 36)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 20);
                border-radius: 18px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 40);
            }
            QPushButton:pressed {
                background-color: rgba(0, 0, 0, 20);
            }
        """)

        self.next_btn = QPushButton()
        self.next_btn.setIcon(QIcon("icons/next.jpg"))
        self.next_btn.setIconSize(QSize(24, 24))
        self.next_btn.setFixedSize(36, 36)
        self.next_btn.setStyleSheet(self.prev_btn.styleSheet())

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        # Zoom buttons
        zoom_frame = QFrame()
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(5)

        self.zoom_out_btn = QPushButton()
        self.zoom_out_btn.setIcon(QIcon("icons/zoom_out.jpg"))
        self.zoom_out_btn.setIconSize(QSize(24, 24))
        self.zoom_out_btn.setFixedSize(36, 36)
        self.zoom_out_btn.setStyleSheet(self.prev_btn.styleSheet())

        self.zoom_in_btn = QPushButton()
        self.zoom_in_btn.setIcon(QIcon("icons/zoom_in.jpg"))
        self.zoom_in_btn.setIconSize(QSize(24, 24))
        self.zoom_in_btn.setFixedSize(36, 36)
        self.zoom_in_btn.setStyleSheet(self.prev_btn.styleSheet())

        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_in_btn)

        # Add everything to layout
        self.layout.addWidget(self.open_btn)
        self.layout.addWidget(self.voice_control_btn)
        self.layout.addWidget(self.auto_scroll_btn)
        self.layout.addStretch(1)
        self.layout.addWidget(nav_frame)
        self.layout.addStretch(1)
        self.layout.addWidget(zoom_frame)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw rounded rectangle background
        rect = self.rect()
        painter.setBrush(QColor("#1565C0"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 15, 15)

        # Draw app title
        painter.setPen(QColor("white"))
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, "VoicePDF")

        # Draw tagline
        painter.setPen(QColor("#BBDEFB"))
        font = QFont("Arial", 12)
        painter.setFont(font)
        rect_tagline = rect.adjusted(0, 40, 0, 0)
        painter.drawText(rect_tagline, Qt.AlignHCenter, "Voice Controlled PDF Reader")

        # Draw loading message at bottom
        font = QFont("Arial", 10)
        painter.setFont(font)
        rect_message = rect.adjusted(0, 0, 0, -20)
        # Make sure message() method exists or provide a default
        if hasattr(self, 'message') and callable(self.message):
            painter.drawText(rect_message, Qt.AlignHCenter | Qt.AlignBottom, self.message())
        else:
            painter.drawText(rect_message, Qt.AlignHCenter | Qt.AlignBottom, "Loading...")


class AutoScrollThread(QThread):
    scroll_update = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.paused = True
        self.scroll_speed = 1  # pixels per update (can be adjusted)
        self.scroll_interval = 0.05  # seconds between updates
        self.current_position = 0
        self.max_position = 0

    def set_max_position(self, max_pos):
        self.max_position = max_pos

    def set_speed(self, speed):
        self.scroll_speed = speed

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        while self.running:
            if not self.paused and self.max_position > 0:
                self.current_position += self.scroll_speed
                if self.current_position > self.max_position:
                    self.current_position = self.max_position
                    self.paused = True  # Stop when we reach the end

                self.scroll_update.emit(self.current_position)

            time.sleep(self.scroll_interval)


# New component for visual command feedback
class CommandVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self.setStyleSheet("background-color: transparent;")
        self.command = None
        self.opacity = 0.0
        self.animation = QPropertyAnimation(self, b"geometry")
        self.timer = QTimer(self)  # Create timer as instance variable

    def show_command(self, command):
        self.command = command
        # Start animation
        self.opacity = 1.0
        self.update()
        # Fade effect
        self.animation.setDuration(800)
        self.animation.setStartValue(QRect(self.x(), self.y(), self.width(), self.height()))
        self.animation.setEndValue(QRect(self.x(), self.y() - 50, self.width(), self.height()))
        self.animation.start()
        # Reset after animation
        self.timer.singleShot(800, self.reset)  # Use the instance timer

    def reset(self):
        self.opacity = 0.0
        self.update()

    def paintEvent(self, event):
        if not self.command or self.opacity <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(self.opacity)

        # Draw arrow based on command
        if self.command == "up":
            self.draw_arrow(painter, Qt.UpArrow)
        elif self.command == "down":
            self.draw_arrow(painter, Qt.DownArrow)
        elif self.command == "next":
            self.draw_arrow(painter, Qt.RightArrow)
        elif self.command == "previous":
            self.draw_arrow(painter, Qt.LeftArrow)

    def draw_arrow(self, painter, direction):
        # Draw circle background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(41, 128, 185, 180))
        painter.drawEllipse(10, 10, 100, 100)

        # Draw arrow
        painter.setPen(QPen(QColor(255, 255, 255), 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QColor(255, 255, 255))

        # Calculate arrow points based on direction
        if direction == Qt.UpArrow:
            points = [QPoint(60, 30), QPoint(30, 70), QPoint(90, 70)]
        elif direction == Qt.DownArrow:
            points = [QPoint(60, 90), QPoint(30, 50), QPoint(90, 50)]
        elif direction == Qt.LeftArrow:
            points = [QPoint(30, 60), QPoint(70, 30), QPoint(70, 90)]
        elif direction == Qt.RightArrow:
            points = [QPoint(90, 60), QPoint(50, 30), QPoint(50, 90)]

        # Draw the arrow
        if points:
            path = QPainterPath()
            path.moveTo(points[0])
            path.lineTo(points[1])
            path.lineTo(points[2])
            path.lineTo(points[0])
            painter.drawPath(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoicePDF - Voice Controlled PDF Reader")
        self.setMinimumSize(800, 600)
        #label = QLabel("Main Window Loaded", self)
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create toolbar
        self.toolbar = ButtonToolbar()
        main_layout.addWidget(self.toolbar)

        # Create content area with PDF viewer
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Create scroll area for PDF content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("background-color: #f0f0f0; border-radius: 8px;")

        # Create PDF viewer
        self.pdf_viewer = PDFViewer()
        self.scroll_area.setWidget(self.pdf_viewer)
        content_layout.addWidget(self.scroll_area)

        main_layout.addWidget(content_widget, 1)  # Give content area stretch factor of 1

        # Create status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)

        # Create command visualizer (overlay for visual feedback)
        self.command_vis = CommandVisualizer(self)

        # Position it in the center right of the window
        self.command_vis.move(self.width() - 150, self.height() // 2)

        # Set up voice control thread
        self.voice_thread = VoiceControlThread("models/voice_commands.pth")  # Path to model
        self.voice_thread.command_detected.connect(self.handle_voice_command)
        self.voice_thread.status_update.connect(self.update_status)
        self.voice_thread.listening_state.connect(self.status_bar.mic_indicator.setActive)

        # Set up auto-scroll thread
        self.auto_scroll = AutoScrollThread()
        self.auto_scroll.scroll_update.connect(self.scroll_area.verticalScrollBar().setValue)

        # Connect UI signals
        self.toolbar.open_btn.clicked.connect(self.open_pdf)
        self.toolbar.voice_control_btn.clicked.connect(self.toggle_voice_control)
        self.toolbar.auto_scroll_btn.clicked.connect(self.toggle_auto_scroll)
        self.toolbar.prev_btn.clicked.connect(self.prev_page)
        self.toolbar.next_btn.clicked.connect(self.next_page)
        self.toolbar.zoom_in_btn.clicked.connect(self.pdf_viewer.zoom_in)
        self.toolbar.zoom_out_btn.clicked.connect(self.pdf_viewer.zoom_out)

        self.status_bar.page_slider.valueChanged.connect(self.go_to_page)

        # Initialize state
        self.current_file = None
        self.update_status("Ready")

        # Set central widget
        self.setCentralWidget(central_widget)

    # Start threads
    def initialize_threads(self):
        # Start auto_scroll if not already running
        if self.auto_scroll and not self.auto_scroll.isRunning():
            self.auto_scroll.start()

        # Safely attempt to start voice_thread
        try:
            if self.voice_thread and not self.voice_thread.isRunning():
                self.voice_thread.start()
        except RuntimeError:
            # Thread was deleted, recreate and start
            self.voice_thread = VoiceControlThread(parent=self)
            self.voice_thread.start()

        self.update_status("Voice control and auto-scroll initialized")

    def update_status(self, message):
        self.status_bar.status_label.setText(message)

    def update_page_info(self):
        if self.pdf_viewer.doc:
            # Convert from 0-based to 1-based for display
            current_page = self.pdf_viewer.current_page + 1
            total_pages = self.pdf_viewer.doc.page_count
            self.status_bar.page_label.setText(f"Page: {current_page}/{total_pages}")

            # Update slider without triggering signals
            self.status_bar.page_slider.blockSignals(True)
            self.status_bar.page_slider.setMaximum(max(0, self.pdf_viewer.doc.page_count - 1))
            self.status_bar.page_slider.setValue(self.pdf_viewer.current_page)
            self.status_bar.page_slider.blockSignals(False)

    def open_pdf(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open PDF File", "", "PDF Files (*.pdf)"
        )

        if filepath:
            page_count = self.pdf_viewer.load_document(filepath)
            if page_count > 0:
                self.current_file = filepath
                self.update_page_info()
                self.auto_scroll.set_max_position(self.pdf_viewer.height())
                self.update_status(f"Loaded: {os.path.basename(filepath)}")

                # Enable buttons that require an open document
                self.toolbar.prev_btn.setEnabled(True)
                self.toolbar.next_btn.setEnabled(True)
                self.toolbar.zoom_in_btn.setEnabled(True)
                self.toolbar.zoom_out_btn.setEnabled(True)
                self.toolbar.auto_scroll_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Failed to load PDF document.")

    def next_page(self):
        if self.pdf_viewer.next_page() >= 0:
            self.update_page_info()
            # Reset scroll position for new page
            self.scroll_area.verticalScrollBar().setValue(0)

    def prev_page(self):
        if self.pdf_viewer.prev_page() >= 0:
            self.update_page_info()
            # Reset scroll position for new page
            self.scroll_area.verticalScrollBar().setValue(0)

    def handle_voice_command(self, command):
        # Show visual feedback
        self.command_vis.show_command(command)

        if command == "up":
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - 100
            )
        elif command == "down":
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() + 100
            )
        elif command == "next":
            self.next_page()

        elif command == "previous":
            self.prev_page()

    def go_to_page(self, page_num):
        if self.pdf_viewer.go_to_page(page_num) >= 0:
            self.update_page_info()
            # Reset scroll position for new page
            self.scroll_area.verticalScrollBar().setValue(0)

    def toggle_voice_control(self):
        if self.voice_thread.paused:
            self.voice_thread.resume()
            self.toolbar.voice_control_btn.setText("Stop Voice Control")
            self.toolbar.voice_control_btn.setIcon(QIcon("icons/mic-off.jpg"))
            self.update_status("Voice control active")
        else:
            self.voice_thread.pause()
            self.toolbar.voice_control_btn.setText("Start Voice Control")
            self.toolbar.voice_control_btn.setIcon(QIcon("icons/mic-on.jpg"))
            self.update_status("Voice control paused")

    def toggle_auto_scroll(self):
        if self.auto_scroll.paused:
            # Set the maximum position to the document height
            self.auto_scroll.set_max_position(self.pdf_viewer.height())
            # Start from current position
            self.auto_scroll.current_position = self.scroll_area.verticalScrollBar().value()
            self.auto_scroll.resume()
            self.toolbar.auto_scroll_btn.setText("Stop Auto-Scroll")
            self.toolbar.auto_scroll_btn.setIcon(QIcon("icons/scroll-off.jpg"))
            self.update_status("Auto-scroll active")
        else:
            self.auto_scroll.pause()
            self.toolbar.auto_scroll_btn.setText("Auto-Scroll")
            self.toolbar.auto_scroll_btn.setIcon(QIcon("icons/scroll.jpg"))
            self.update_status("Auto-scroll paused")

    def resizeEvent(self, event):
        # Reposition command visualizer when window is resized
        self.command_vis.move(
            self.width() - self.command_vis.width() - 20,
            (self.height() - self.command_vis.height()) // 2
        )
        super().resizeEvent(event)

    def closeEvent(self, event):
        # Clean up threads before closing
        self.voice_thread.stop()
        self.auto_scroll.stop()
        event.accept()


class SplashScreen(QWidget):
    finished = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoicePDF")
        self.setFixedSize(500, 300)
        self.setWindowFlag(Qt.FramelessWindowHint)

        # Center on screen
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #E0E0E0;
                border-radius: 10px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 10px;
            }
        """)

        # Add to layout with some spacing
        layout.addStretch(2)
        layout.addWidget(self.progress)
        layout.addStretch(1)

        # Status message
        self.message_label = QLabel("Loading resources...")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #BBDEFB; font-size: 12px;")
        layout.addWidget(self.message_label)

        # Set background color
        self.setStyleSheet("background-color: #1565C0;")

        # For animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0

    def start_loading(self):
        self.timer.start(30)  # Update every 30ms

    def update_progress(self):
        self.progress_value += 1
        self.progress.setValue(self.progress_value)

        # Update message based on progress
        if self.progress_value < 30:
            self.message_label.setText("Loading resources...")
        elif self.progress_value < 60:
            self.message_label.setText("Initializing voice recognition...")
        elif self.progress_value < 90:
            self.message_label.setText("Setting up user interface...")
        else:
            self.message_label.setText("Ready to launch...")

        # When complete, stop the timer and emit finished signal
        if self.progress_value >= 100:
            self.timer.stop()
            print("Splash screen finished loading")
            self.finished.emit()
            QTimer.singleShot(200, self.close_splash)
            # Use QTimer.singleShot to give UI a chance to update before closing

    def close_splash(self):
        print("Closing splash screen")
        self.close()

    def paintEvent(self, event):
        # Custom paint event to create an attractive splash screen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create rounded rectangle for the window
        rect = self.rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#1565C0"))
        painter.drawRoundedRect(rect, 15, 15)

        # Draw app logo/icon
        # For demo purposes, we'll just draw a simplified icon
        icon_rect = QRect(rect.width() // 2 - 50, 40, 100, 100)
        gradient = QRadialGradient(icon_rect.center(), 50)
        gradient.setColorAt(0, QColor("#42A5F5"))
        gradient.setColorAt(1, QColor("#1E88E5"))
        painter.setBrush(gradient)
        painter.drawEllipse(icon_rect)

        # Draw microphone symbol
        painter.setPen(QPen(QColor("white"), 4))
        painter.setBrush(QColor("white"))
        mic_width = 30
        mic_height = 50
        mic_x = icon_rect.center().x() - mic_width // 2
        mic_y = icon_rect.center().y() - mic_height // 2
        painter.drawRoundedRect(mic_x, mic_y, mic_width, mic_height, 5, 5)
        painter.drawRect(icon_rect.center().x() - 2, mic_y + mic_height, 4, 15)
        painter.drawRoundedRect(icon_rect.center().x() - 15, mic_y + mic_height + 15, 30, 4, 2, 2)

        # Draw app name
        painter.setPen(QColor("white"))
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        painter.drawText(rect.adjusted(0, icon_rect.bottom() + 10, 0, 0),
                         Qt.AlignHCenter, "VoicePDF")

        # Draw tagline
        painter.setPen(QColor("#BBDEFB"))
        font = QFont("Arial", 12)
        painter.setFont(font)
        painter.drawText(rect.adjusted(0, icon_rect.bottom() + 50, 0, 0),
                         Qt.AlignHCenter, "Voice Controlled PDF Reader")


def main():
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Set dark palette for application
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    # Create the main window but don't show it yet
    splash = SplashScreen()
    main_window = MainWindow()

    # Define the function before using it
    def launch_main():
        print("Splash closed, launching main window...")
        splash.close()  # Safe call
        main_window.show()
        main_window.initialize_threads()

    #splash = SplashScreen()
    #splash.destroyed.connect(on_splash_closed)
    splash.show()
    splash.start_loading()

    # Trigger launch after 2.5 seconds (or however long your splash runs)
    QTimer.singleShot(2500, launch_main)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()