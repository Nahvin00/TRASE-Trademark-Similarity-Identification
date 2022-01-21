from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QPlainTextEdit, QPushButton, QLabel
from PyQt5 import QtGui
from PyQt5 import uic
import sys
from test import proc


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        uic.loadUi("trase_ui.ui", self)

        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.textedit1 = self.findChild(QPlainTextEdit, "TextEdit_1")
        self.textedit2 = self.findChild(QPlainTextEdit, "TextEdit_2")
        self.button_check = self.findChild(QPushButton, "but_check")
        self.button_clear = self.findChild(QPushButton, "but_clear")
        self.logo = self.findChild(QLabel, "logo")
        pixmap_logo = QPixmap("TRASE LOGO.png")
        self.logo.setPixmap(pixmap_logo)
        self.label_res = self.findChild(QLabel, "label_res")
        self.label_res.setVisible(0)
        self.res_img = self.findChild(QLabel, "res_img")
        self.res_img.setVisible(0)
        self.res_text = self.findChild(QLabel, "res_text")
        self.res_text.setVisible(0)

        self.button_check.clicked.connect(self.clickedBtn_check)
        self.button_clear.clicked.connect(self.clickedBtn_clear)

        self.show()

    def clickedBtn_check(self):
        self.label_res.setVisible(0)
        self.res_img.setVisible(0)
        self.res_text.setVisible(0)
        inp1 = self.textedit1.toPlainText()
        inp2 = self.textedit2.toPlainText()
        a, b = proc(inp1, inp2)
        oup = "Prediction: {}\n\nConfidence: {}".format(a, b)
        self.res_text.setText(oup)
        pixmap = QPixmap("res.png")
        self.res_img.setPixmap(pixmap)
        self.label_res.setVisible(1)
        self.res_img.setVisible(1)
        self.res_text.setVisible(1)

    def clickedBtn_clear(self):
        self.label_res.setVisible(0)
        self.res_img.setVisible(0)
        self.res_text.setVisible(0)
        self.textedit1.clear()
        self.textedit2.clear()


app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
