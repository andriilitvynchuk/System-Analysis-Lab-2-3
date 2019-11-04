import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import sys

sys.path.append('..')
from backend import AdditiveModel


def my_int(number):
    try:
        return int(number)
    except:
        return 2


class Equation(QWidget):

    def __init__(self):
        super().__init__()
        self.dataset_size = 40
        self.x_path = '../data/x.tsv'
        self.y_path = '../data/y.tsv'
        self.x_size = [2, 2, 2]
        self.y_size = 1
        self.b_type = 'norm'
        self.polynom_type = 'chebyshev'
        self.polynom_degrees = [2, 2, 2]
        self.polynom_search = True
        self.lambda_type = 'separately'
        self.output_file = '../data/output.txt'
        self.norm_graph = True
        self.y_number = 1
        self.initUI()

    def initUI(self):
        fontBold = QFont()
        fontBold.setBold(True)

        topleft = QFrame(self)
        topleft.setFrameShape(QFrame.StyledPanel)
        topleft.resize(400, 400)
        labelData = QLabel("Вихідні дані", topleft)
        labelData.setFont(fontBold)
        labelData.move(2, 1)
        labelAmount = QLabel("Розмір вибірки", topleft)
        labelAmount.move(2, 20)

        # self.choiceAmount = "1"

        size = QComboBox(topleft)
        size.addItems([str(i) for i in range(0, 100)])
        size.move(100, 15)
        size.activated[str].connect(self.sizeData)

        labelOpen = QLabel("Файл вихідних даних", topleft)
        labelOpen.move(2, 50)
        self.file_open = QLineEdit(topleft)
        self.file_open.setFixedWidth(40)
        self.file_open.move(140, 50)

        open_file = QPushButton('...', topleft)
        open_file.setCheckable(True)
        open_file.move(180, 45)
        open_file.clicked[bool].connect(self.openFileNameDialog)

        labelResult = QLabel("Файл результатів", topleft)
        labelResult.move(2, 80)
        self.file_res = QLineEdit(topleft)
        self.file_res.setFixedWidth(40)
        self.file_res.move(140, 80)

        save_file = QPushButton('...', topleft)
        save_file.setCheckable(True)
        save_file.move(180, 75)
        save_file.clicked[bool].connect(self.saveFileDialog)

        labelOpenY = QLabel("Файл Y", topleft)
        labelOpenY.move(2, 110)
        self.file_y = QLineEdit(topleft)
        self.file_y.setFixedWidth(40)
        self.file_y.move(140, 110)

        save_y = QPushButton('...', topleft)
        save_y.setCheckable(True)
        save_y.move(180, 105)
        save_y.clicked[bool].connect(self.openFileNameDialogY)

        labelVector = QLabel("Вектори", topleft)
        labelVector.setFont(fontBold)
        labelVector.move(2, 180)

        labelx1 = QLabel("Розмірність x1", topleft)
        labelx1.move(2, 200)
        self.x1Array = QLineEdit(topleft)
        self.x1Array.setFixedWidth(35)
        self.x1Array.move(150, 200)

        labelx2 = QLabel("Розмірність x2", topleft)
        labelx2.move(2, 230)
        self.x2Array = QLineEdit(topleft)
        self.x2Array.setFixedWidth(35)
        self.x2Array.move(150, 230)

        labelx3 = QLabel("Розмірність x3", topleft)
        labelx3.move(2, 260)
        self.x3Array = QLineEdit(topleft)
        self.x3Array.setFixedWidth(35)
        self.x3Array.move(150, 260)

        labely = QLabel("Розмірність Y", topleft)
        labely.move(2, 290)
        self.yArray = QLineEdit(topleft)
        self.yArray.setFixedWidth(35)
        self.yArray.move(150, 290)

        central = QFrame(self)
        central.setFrameShape(QFrame.StyledPanel)
        central.resize(400, 400)

        labelPolynom = QLabel("Вигляд поліномів", central)
        labelPolynom.setFont(fontBold)
        labelPolynom.move(2, 1)

        polynoms = QComboBox(central)
        polynoms.addItems(["Поліноми Чебишева", "Поліноми Лежандра",
                           "Поліноми Лаггера", "Поліноми Ерміта"])
        polynoms.move(2, 15)
        polynoms.activated[str].connect(self.polynomType)

        labelx1_power = QLabel("Для х1", central)
        labelx1_power.move(2, 50)
        self.x1_power = QLineEdit(central)
        self.x1_power.setFixedWidth(35)
        self.x1_power.move(100, 45)

        labelx2_power = QLabel("Для x2", central)
        labelx2_power.move(2, 80)
        self.x2_power = QLineEdit(central)
        self.x2_power.setFixedWidth(35)
        self.x2_power.move(100, 75)

        labelx3_power = QLabel("Для x3", central)
        labelx3_power.move(2, 110)
        self.x3_power = QLineEdit(central)
        self.x3_power.setFixedWidth(35)
        self.x3_power.move(100, 105)

        auto_polynom_search = QCheckBox("Автоматичний пошук степені полінома", central)
        auto_polynom_search.move(2, 130)
        auto_polynom_search.toggle()
        auto_polynom_search.stateChanged.connect(self.polynomicalSearch)

        topright = QFrame(self)
        topright.setFrameShape(QFrame.StyledPanel)
        topright.resize(400, 400)
        labelAdditional = QLabel("Додатково", topright)
        labelAdditional.setFont(fontBold)
        labelAdditional.move(2, 1)

        labelVectorB = QLabel("Спосіб підрахунку вектора b", topright)
        labelVectorB.move(2, 20)
        vector_b = QComboBox(topright)
        vector_b.addItems(["Норма цільової змінної", "Середнє арифметичне"])
        vector_b.move(190, 13)
        vector_b.activated[str].connect(self.typeB)

        lambda_cb = QCheckBox("Визначити lambda з трьох систем рівнянь", topright)
        lambda_cb.move(2, 45)
        lambda_cb.toggle()
        lambda_cb.stateChanged.connect(self.findLambda)

        button_execute = QPushButton('Виконати', topright)
        button_execute.move(150, 100)
        button_execute.clicked.connect(self.execute)
        button_graph = QPushButton('Графік', topright)
        button_graph.move(250, 100)
        button_graph.clicked.connect(self.graphic)
        lambda_cb = QCheckBox("Графік у нормованому вигляді", topright)
        lambda_cb.move(155, 130)
        lambda_cb.toggle()
        lambda_cb.stateChanged.connect(self.graphNorm)

        labelNumbY = QLabel("Цільова змінна: ", topright)
        labelNumbY.move(155, 70)
        numbY = QComboBox(topright)
        numbY.addItems([str(i) for i in range(1, 101)])
        numbY.move(250, 65)
        numbY.activated[str].connect(self.numb_y)

        bottom = QFrame(self)
        bottom.setFrameShape(QFrame.StyledPanel)

        layout = QHBoxLayout()
        for frame in [topleft, central, topright]:
            layout.addWidget(frame)

        verticalLayout = QVBoxLayout(self)
        verticalLayout.addLayout(layout)
        verticalLayout.addWidget(bottom)
        self.setLayout(verticalLayout)

        self.setGeometry(300, 300, 1200, 1200)
        self.setWindowTitle('Відтворення функціональних залежностей в адитивній формі')
        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.txt)", options=options)
        if fileName:
            self.file_open.setText(fileName)
            self.x_path = fileName

    def openFileNameDialogY(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.txt)", options=options)
        if fileName:
            self.file_y.setText(fileName)
            self.y_path = fileName

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            self.file_res.setText(fileName)
        self.output_file = fileName

    def sizeData(self, size):
        self.dataset_size = int(size)

    def polynomType(self, type_):
        if 'Чебишева' in type_:
            self.polynom_type = 'chebyshev'
        elif 'Лежандра' in type_:
            self.polynom_type = 'legendre'
        elif 'Лаггера' in type_:
            self.polynom_type = 'laguerre'
        elif 'Ерміта' in type_:
            self.polynom_type = 'hermit'
        else:
            self.polynom_type = 'chebyshev'

    def polynomicalSearch(self, state):
        if state == Qt.Checked:
            self.polynom_search = True
        else:
            self.polynom_search = False

    def findLambda(self, state):
        if state == Qt.Checked:
            self.lambda_type = 'separately'
        else:
            self.lambda_type = 'all'

    def graphNorm(self, state):
        if state == Qt.Checked:
            self.norm_graph = True
        else:
            self.norm_graph = False

    def typeB(self, type_):
        if 'Середнє' in type_:
            self.b_type = 'mean'
        else:
            self.b_type = 'norm'

    def numb_y(self, numb):
        self.y_number = numb

    def execute(self):
        self.x_size = [my_int(self.x1Array.text()), my_int(self.x2Array.text()), my_int(self.x3Array.text())]
        self.y_size = my_int(self.yArray.text())
        self.polynom_degrees = [my_int(self.x1_power.text()), my_int(self.x2_power.text()),
                                my_int(self.x3_power.text())]
        if self.polynom_type == '':
            self.polynom_type = 'chebyshev'
        if self.lambda_type == '':
            self.lambda_type = 'separately'
        if self.b_type == '':
            self.b_type = 'norm'
        if self.dataset_size == 0:
            self.dataset_size = 40

        attr = {'dataset_size': int(self.dataset_size),
                'x_path': self.x_path, 'y_path': self.y_path,
                'x_size': self.x_size, 'y_size': self.y_size,
                'b_type': self.b_type, 'polynom_type': self.polynom_type,
                'polynom_degrees': self.polynom_degrees, 'polynom_search': self.polynom_search,
                'lambda_type': self.lambda_type, 'output_file': self.output_file}
        # print("Attributes for execution:")
        print(attr)
        self.additive_model = AdditiveModel(**attr)
        self.additive_model.find_additive_model()
        self.content = self.additive_model.write_in_file()
        print(self.content)

    def graphic(self):
        if self.y_number == '':
            self.y_number = '1'
        attr = {'norm': self.norm_graph, 'y_number': int(self.y_number)}
        self.additive_model.get_plot(**attr)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Equation()
    sys.exit(app.exec_())
