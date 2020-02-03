## GUI MAIN RUN

import sys
from PyQt5.QtWidgets import QApplication
from MNIST_Window import MNIST_Window

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MNIST_Window()
    mymnist.show()
    app.exec_()
 