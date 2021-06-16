import sys
from PyQt5.QtWidgets import QApplication
from predict_music_genre_window import predict_music_genre_window

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = predict_music_genre_window()
    window.show()
    sys.exit(app.exec_())