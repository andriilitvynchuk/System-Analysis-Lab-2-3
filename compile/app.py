import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import eval_chebyt, eval_hermite, eval_legendre, eval_laguerre
from scipy.optimize import fmin_cg
from copy import deepcopy
from sklearn.linear_model import Ridge as LinearRegression
from tqdm import tqdm
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *


def get_coef(X, y):
    reg = LinearRegression(fit_intercept=False, alpha=0.0001).fit(X, y)
    coef = reg.coef_
    return coef


# def get_coef(A, b):
#     coef = fmin_cg(lambda x: np.sum((b - np.dot(A, x)) ** 2), np.ones((A.shape[1])), maxiter=2000, disp=0)
#     return coef


class AdditiveModel:

    def __init__(self, dataset_size, x_path, y_path, x_size, y_size, b_type, polynom_type, polynom_degrees,
                 polynom_search, lambda_type, output_file):
        self.dataset_size = dataset_size
        self.x = np.loadtxt(x_path)
        self.y = np.loadtxt(y_path)
        self.y = self.y.reshape(self.y.shape[0], -1)
        self.x_size = x_size
        self.y_size = y_size
        self.y = self.y[:, :self.y_size]

        if polynom_type == 'chebyshev':
            self.polynom_function = eval_chebyt
        elif polynom_type == 'hermit':
            self.polynom_function = eval_hermite
        elif polynom_type == 'legendre':
            self.polynom_function = eval_legendre
        elif polynom_type == 'laguerre':
            self.polynom_function = eval_laguerre
        self.polynom_type = polynom_type
        self.b_type = b_type
        self.polynom_search = polynom_search
        if not self.polynom_search:
            self.polynom_degrees = polynom_degrees
        self.lambda_type = lambda_type
        self.output_file = output_file

        self.cache_y = None
        self.cache_min_max = None
        self.b = None
        self.coef_lambda = None
        self.coef_a = None
        self.coef_c = None
        self.X_coef_lambda = None
        self.X_coef_a = None
        self.X_coef_c = None

    def norm(self):
        self.cache_min_max = dict()
        for index in range(self.x.shape[1]):
            key = 'x' + str(index)
            self.cache_min_max[key] = [self.x[:, index].min(), self.x[:, index].max()]
            self.x[:, index] = (self.x[:, index] - self.cache_min_max[key][0]) / (
                    self.cache_min_max[key][1] - self.cache_min_max[key][0])

        self.cache_y = deepcopy(self.y)
        for index in range(self.y.shape[1]):
            key = 'y' + str(index)
            self.cache_min_max[key] = [self.y[:, index].min(), self.y[:, index].max()]
            self.y[:, index] = (self.y[:, index] - self.cache_min_max[key][0]) / (
                    self.cache_min_max[key][1] - self.cache_min_max[key][0])

    def set_b(self):
        if self.b_type == 'norm':
            self.b = deepcopy(self.y)
        elif self.b_type == 'mean':
            self.b = np.zeros((self.y.shape[0], 1))
            for i in range(self.y.shape[0]):
                self.b[i, :] = (np.max(self.y[i, :self.y_size]) + np.min(self.y[i, :self.y_size])) / 2

    def evaluate_degrees(self, degrees):
        b_mean = np.mean(self.b, axis=1)
        new_shape_X = (self.x.shape[0], np.sum((np.array(degrees) + 1) * np.array(self.x_size)))
        X = np.zeros(new_shape_X)
        pointer_X = 0
        pointer_x = 0
        for i in range(3):
            for j in range(self.x_size[i]):
                for d in range(degrees[i] + 1):
                    X[:, pointer_X] = self.polynom_function(d, self.x[:, pointer_x])
                    pointer_X += 1
                pointer_x += 1
        score = np.abs((b_mean - np.dot(X, get_coef(X, b_mean)))).mean()
        return score

    def find_polynom_degrees(self):
        degrees = [1, 1, 1]
        for index in tqdm(range(3)):
            best_score = np.inf
            best_degree = 1
            for degree in range(1, 11):
                degrees[index] = degree
                score = self.evaluate_degrees(degrees)
                if score < best_score:
                    best_score = score
                    best_degree = degree
            degrees[index] = best_degree
        print(degrees, best_score)
        self.polynom_degrees = degrees

    def find_coef_lambda_all(self):
        b_mean = np.mean(self.b, axis=1)
        new_shape_X = (self.x.shape[0], np.sum((np.array(self.polynom_degrees) + 1) * np.array(self.x_size)))
        X = np.zeros(new_shape_X)
        pointer_X = 0
        pointer_x = 0
        for i in range(3):
            for j in range(self.x_size[i]):
                for d in range(self.polynom_degrees[i] + 1):
                    X[:, pointer_X] = self.polynom_function(d, self.x[:, pointer_x])
                    pointer_X += 1
                pointer_x += 1
        self.coef_lambda = get_coef(X, b_mean)
        self.X_coef_lambda = X

    def find_coef_lambda_separately(self):
        b_mean = np.mean(self.b, axis=1)
        coef_lambda = []
        X_coef_lambda = []
        pointer_x = 0
        for i in range(3):
            tmp_X = np.zeros((self.x.shape[0], (self.polynom_degrees[i] + 1) * self.x_size[i]))
            pointer_X = 0
            for j in range(self.x_size[i]):
                for d in range(self.polynom_degrees[i] + 1):
                    tmp_X[:, pointer_X] = self.polynom_function(d, self.x[:, pointer_x])
                    pointer_X += 1
                pointer_x += 1
            tmp_coef_lambda = get_coef(tmp_X, b_mean)
            coef_lambda.append(tmp_coef_lambda)
            X_coef_lambda.append(tmp_X)

        self.X_coef_lambda = np.concatenate(X_coef_lambda, axis=1)
        self.coef_lambda = np.concatenate(coef_lambda, axis=0)

    def find_coef_a(self):
        coef_a = dict()
        X_coef_a = dict()
        for index in range(self.y_size):
            tmp_y = self.y[:, index]
            coef_a[index] = []
            X_coef_a[index] = []
            pointer = 0
            for i in range(3):
                tmp_X = np.zeros((self.X_coef_lambda.shape[0], self.x_size[i]))
                for j in range(self.x_size[i]):
                    polynom_subset = self.X_coef_lambda[:, pointer:pointer + self.polynom_degrees[i] + 1]
                    lambda_coef_subset = self.coef_lambda[pointer:pointer + self.polynom_degrees[i] + 1]
                    tmp_X[:, j] = np.dot(polynom_subset, lambda_coef_subset)
                    pointer += self.polynom_degrees[i] + 1
                tmp_coef_a = get_coef(tmp_X, tmp_y)

                X_coef_a[index].append(tmp_X)
                coef_a[index].append(tmp_coef_a)

        self.coef_a = coef_a
        self.X_coef_a = X_coef_a

    def find_coef_c(self):
        coef_c = dict()
        X_coef_c = dict()
        for index in range(self.y_size):
            tmp_y = self.y[:, index]
            tmp_X = np.zeros((self.X_coef_lambda.shape[0], 3))
            for i in range(3):
                tmp_X[:, i] = np.dot(self.X_coef_a[index][i], self.coef_a[index][i])
            tmp_coef_c = get_coef(tmp_X, tmp_y)
            X_coef_c[index] = tmp_X
            coef_c[index] = tmp_coef_c

        self.X_coef_c = X_coef_c
        self.coef_c = coef_c

    def find_additive_model(self):
        self.norm()
        self.set_b()

        if self.polynom_search:
            self.find_polynom_degrees()

        if self.lambda_type == 'all':
            self.find_coef_lambda_all()
        elif self.lambda_type == 'separately':
            self.find_coef_lambda_separately()

        self.find_coef_a()
        self.find_coef_c()

    def get_coef_lambda(self):
        string = f'Коефіцієнти \u03BB \n'
        pointer = 0
        for i in range(3):
            for j in range(self.x_size[i]):
                for d in range(self.polynom_degrees[i] + 1):
                    coef = self.coef_lambda[pointer]
                    string += f'\u03BB{i + 1}{j + 1}{d}={coef:.4f}  '
                    pointer += 1
                string += '\n'
        return string

    def get_coef_a(self):
        string = 'Коефіцієнти а \n'
        for index in range(self.y_size):
            string += f'i = {index + 1} \n'
            for i, coef in enumerate(self.coef_a[index]):
                for j in range(len(coef)):
                    string += f'a{i + 1}{j + 1}={coef[j]:.4f} '
                string += '\n'
        return string

    def get_coef_c(self):
        string = 'Коефіцієнти с \n'
        for index in range(self.y_size):
            string += f'i = {index + 1} \n'
            for j in range(len(self.coef_c[index])):
                string += f'c{j + 1}={self.coef_c[index][j]:.4f} '
            string += '\n'
        return string

    def get_function_theta(self):
        string = 'Функції \u03A8 \n'
        pointer = 0
        for i in range(3):
            for j in range(self.x_size[i]):
                string += f'\u03A8{i + 1}{j + 1}(x{i + 1}{j + 1}) = '
                for d in range(self.polynom_degrees[i] + 1):
                    coef = self.coef_lambda[pointer]
                    string += f'{coef:.4f}*T{d}(x{i + 1}{j + 1})'
                    pointer += 1
                    if d != self.polynom_degrees[i]:
                        string += '+  '
                string += '\n'
        return string

    def get_function_f_i(self):
        string = 'Функції Ф_ij \n'
        for index in range(self.y_size):
            for i, coef in enumerate(self.coef_a[index]):
                string += f'Ф{index + 1}{i + 1}(x{i + 1})= '
                for j in range(len(coef)):
                    string += f'{coef[j]:.4f}*\u03A8{i + 1}{j + 1}(x{i + 1}{j + 1})'
                    if j != len(coef) - 1:
                        string += '+  '
                string += '\n'
        return string

    def get_final_approximation_f(self):
        string = 'Одержані функції через Ф \n'
        for index in range(self.y_size):
            string += f'Ф{index + 1}(x1, x2, x3) = '
            for j in range(len(self.coef_c[index])):
                string += f'{self.coef_c[index][j]:.4f}*Ф{index + 1}{j + 1}(x{j + 1})'
                if j != len(self.coef_c[index]) - 1:
                    string += '+  '
            string += '\n'
        return string

    def get_final_approximation_t(self):
        string = 'Одержані функції через поліноми \n'
        for index in range(self.y_size):
            string += f'Ф{index + 1}(x1, x2, x3) = '
            pointer = 0
            for i in range(3):
                for j in range(self.x_size[i]):
                    for d in range(self.polynom_degrees[i] + 1):
                        coef = self.coef_lambda[pointer] * self.coef_a[index][i][j] * self.coef_c[index][i]
                        string += f'{coef:.4f}*T{d}(x{i + 1}{j + 1})+  '
                        pointer += 1
            string = string[:-3] + '\n'
        return string

    def get_final_approximation_polynoms(self):
        string = 'Одержані функції у вигляді многочленів(у нормованому вигляді) \n'
        mapping = dict(chebyshev=0.4310,
                       hermit=0.3154,
                       legendre=0.6421,
                       laguerre=0.2587)
        for index in range(self.y_size):
            string += f'Ф{index + 1}(x1, x2, x3) = '
            pointer = 0
            bias = 0
            for i in range(3):
                for j in range(self.x_size[i]):
                    for d in range(self.polynom_degrees[i] + 1):
                        if d != 0:
                            coef = self.coef_lambda[pointer] * self.coef_a[index][i][j] * self.coef_c[index][i]
                            coef *= 1 / mapping[self.polynom_type]
                            string += f'{coef:.4f}*x{i + 1}{j + 1}^{d} +  '
                        else:
                            bias += self.coef_lambda[pointer] * self.coef_a[index][i][j] * self.coef_c[index][i]
                        pointer += 1
            string += f'{bias:.4f} \n'
        return string

    def get_final_approximation_polynoms_denorm(self):
        string = 'Одержані функції у вигляді многочленів(у відтвореному вигляді) \n'
        mapping = dict(chebyshev=0.4310,
                       hermit=0.3154,
                       legendre=0.6421,
                       laguerre=0.2587)
        for index in range(self.y_size):
            y_min, y_max = self.cache_min_max[f'y{index}']
            string += f'Ф{index + 1}(x1, x2, x3) = '
            pointer = 0
            bias = 0
            for i in range(3):
                for j in range(self.x_size[i]):
                    for d in range(self.polynom_degrees[i] + 1):
                        if d != 0:
                            coef = self.coef_lambda[pointer] * self.coef_a[index][i][j] * self.coef_c[index][i]
                            coef *= 1 / mapping[self.polynom_type]
                            coef *= y_max - y_min
                            string += f'{coef:.4f}*x{i + 1}{j + 1}^{d} +  '
                        else:
                            bias += self.coef_lambda[pointer] * self.coef_a[index][i][j] * self.coef_c[index][i]
                        pointer += 1
            bias += y_min
            string += f'{bias:.4f} \n'
        return string

    def write_in_file(self):
        with open(self.output_file, 'w') as file:
            if self.polynom_search:
                file.write(f'Найкращі степені поліномів : '
                           f'{self.polynom_degrees[0]} {self.polynom_degrees[1]} {self.polynom_degrees[2]} \n\n')
            file.write(self.get_coef_lambda() + '\n')
            file.write(self.get_coef_a() + '\n')
            file.write(self.get_coef_c() + '\n')
            file.write(self.get_function_theta() + '\n')
            file.write(self.get_function_f_i() + '\n')
            file.write(self.get_final_approximation_f() + '\n')
            file.write(self.get_final_approximation_t() + '\n')
            file.write(self.get_final_approximation_polynoms() + '\n')
            file.write(self.get_final_approximation_polynoms_denorm())

        with open(self.output_file, 'r') as file:
            content_to_print = file.read()

        return content_to_print

    def get_plot(self, y_number=1, norm=True):
        ground_truth = self.y[:, y_number - 1]
        predict = np.dot(self.X_coef_c[y_number - 1], self.coef_c[y_number - 1])
        if not norm:
            y_min, y_max = self.cache_min_max[f'y{y_number - 1}']
            ground_truth = ground_truth * (y_max - y_min) + y_min
            predict = predict * (y_max - y_min) + y_min
        error = np.mean(np.abs(predict - ground_truth))
        plt.title(f'Відновлена функціональна залежність з похибкою {error:.6f}')
        plt.plot(np.arange(1, self.dataset_size + 1),
                 ground_truth,
                 label=f'Y{y_number}')
        plt.plot(np.arange(1, self.dataset_size + 1),
                 predict,
                 label='Ф11 + Ф12 + Ф13',
                 linestyle='--')
        plt.legend()
        plt.show()


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
        self.output = QTextEdit(bottom)
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QTextEdit.NoWrap)
        self.output.setFixedWidth(1140)
        self.output.setMinimumHeight(300)
        #self.output.setMaximumHeight(1000)
        self.output.move(5,5)

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
        self.output.setText(self.content)

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