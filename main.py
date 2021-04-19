import numpy as np
import math
import matplotlib.pyplot as plt
import time
import numba

speed = 250
rad = 1000  # для движения по окружности
angle = 0

period = 2
t = (rad * np.pi / 1000) / speed * 3600
N = math.ceil(t / period)  # число наблюдений
#N = 300  # число наблюдений
print(N)
dNoise = 20  # дисперсия шума
dSignal = 5  # дисперсия сигнала
r = 1  # коэффициент корреляции в модели движения
en = 1  # дисперсия СВ в модели движения

M = 2  # размерность вектора координат положения объекта
R = np.array([[r, 0], [0, r]])
Vksi = np.eye(M) * en  # диагональная матрица с дисперсиями en по главной диагонали
V = np.eye(M) * dNoise  # диагональная матрица с дисперсиями ошибок наблюдений

coord = np.zeros(N * M).reshape(N, M)  # истинные координаты перемещения (пока просто нули)
# coord[:][0] = np.random.normal(0, dSignal, M)  # формирование первой координаты
coord[:][0] = 0

"""for i in range(1, N):  # формирование координат
    if i <N/3:
        coord[i][0] = coord[i if i == 0 else i - 1][0] - 4
        coord[i][1] = coord[i if i == 0 else i - 1][1]
    elif N/3<=i<2*N/3:
        coord[i][0] = coord[i if i == 0 else i - 1][0] + rad * (10 / (np.pi * N)) * np.sin(angle * np.pi - np.pi / 2)
        coord[i][1] = coord[i if i == 0 else i - 1][1] + rad * (10 / (np.pi * N)) * np.cos(angle * np.pi + np.pi / 2)
    else:
        coord[i][0] = coord[i if i == 0 else i - 1][0] + 4
        coord[i][1] = coord[i if i == 0 else i - 1][1]
    angle -= 1/N"""

for i in range(1, N):  # формирование координат
    #coord[i][0] = coord[i if i == 0 else i - 1][0] + rad * (10 / (np.pi * N)) * np.sin(angle * np.pi - np.pi / 2)
    #coord[i][1] = coord[i if i == 0 else i - 1][1] + rad * (10 / (np.pi * N)) * np.cos(angle * np.pi + np.pi / 2)
    coord[i][0] = coord[i if i == 0 else i - 1][0] + speed
    coord[i][1] = coord[i if i == 0 else i - 1][1]
    #print(math.sqrt((rad * (10 / (np.pi * N)) * np.sin(angle * np.pi - np.pi / 2))**2+(rad * (10 / (np.pi * N)) * np.cos(angle * np.pi + np.pi / 2))**2))
    angle -= 1 / N

coordWithNoise = coord + np.random.normal(0, dNoise, size=(N, M))  # формирование наблюдений
start_time1 = time.time()

# По Калману
###########################################################################################################################################################
# фильтрация сигнала с помощью фильтра Калмана
coordKalman = np.zeros(N * M).reshape(N, M)  # вектор для хранения оценок перемещений
P = np.zeros(M * M).reshape(M, M)  # вектор для хранения дисперсий ошибок оценивания
coordKalman[:][0] = coordWithNoise[:][0]  # первая оценка
P = V  # дисперсия первой оценки

Vinv = np.linalg.inv(V)  # вычисление обратной матрицы дисперсий ошибок наблюдений

# рекуррентное вычисление оценок по фильтру Калмана
for i in range(1, N):
    Pe = np.dot(np.dot(R, P), R.T) + Vksi
    P = np.dot(Pe, V) * np.linalg.inv(Pe + V)
    xe = np.dot(R, coordKalman[:][i - 1])
    coordKalman[:][i] = xe + np.dot(np.dot(P, Vinv), (coordWithNoise[:][i] - xe))


###########################################################################################################################################################
print("--- %s seconds ---" % (time.time() - start_time1))
start_time2 = time.time()

# По курсу и скорости
###########################################################################################################################################################
def speedCalculate(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def courseCalculate(point11, point12, point21=(0, 0), point22=(1, 0)):
    A1 = point11[1] - point12[1]
    A2 = point21[1] - point22[1]
    B1 = point12[0] - point11[0]
    B2 = point22[0] - point21[0]
    return math.acos((A1 * A2 + B1 * B2) / (np.sqrt(A1 ** 2 + B1 ** 2) * np.sqrt(A2 ** 2 + B2 ** 2))) / np.pi


def courseAndSpeed(point1, point2, point3):
    V = speedCalculate(point1, point2)
    Q = courseCalculate(point1, point2)
    dQ = Q - courseCalculate(point2, point3)
    dV = V - speedCalculate(point2, point3)
    l = (V + dV)
    x = point3[0]
    y = point3[1]
    Xextr = x + l * np.sin(Q + dQ)
    Yextr = y + l * np.cos(Q + dQ)
    speedExtr = dV + speedCalculate((Xextr, Yextr), point3)
    courseExtr = dQ + courseCalculate((Xextr, Yextr), point3)
    return np.array([Xextr, Yextr])


coordCourseAndSpeed = np.zeros(N * M).reshape(N, M)
for i in range(3, N):
    coordCourseAndSpeed[i] = courseAndSpeed(coordWithNoise[i - 3], coordWithNoise[i - 2], coordWithNoise[i - 1])

print("--- %s seconds ---" % (time.time() - start_time2))
###########################################################################################################################################################
start_time3 = time.time()

def multiplePointPrediction(points):
    x = 0
    y = 0
    for i in range(len(points)):
        x += ((6 * (i + 1) - 2 * len(points) - 2) / (len(points) * (len(points) + 1))) * points[i][0]
        y += ((6 * (i + 1) - 2 * len(points) - 2) / (len(points) * (len(points) + 1))) * points[i][1]
    return (x, y)


countsOfPoints = 3  # количество точек для Предсказания по нескольким отметкам
coordMultiplePoints = np.zeros(N * M).reshape(N, M)

for i in range(countsOfPoints, N):
    coordMultiplePoints[i] = multiplePointPrediction(coordWithNoise[i: i + countsOfPoints])
    # print(coordMultiplePoints[i])
print("--- %s seconds ---" % (time.time() - start_time3))
# отображение результатов
fig, a = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
"""
res = xx.reshape(M * N)
resX = x.reshape(M * N)
resZ = z.reshape(M * N)

axX.plot(resX[0:N * M:M], color="green");
axX.plot(resZ[0:N * M:M], color="red");
axX.plot(res[0:N * M:M], color="blue")
axY.plot(resX[1:N * M:M], color="green");
axY.plot(resZ[1:N * M:M], color="red");
axY.plot(res[1:N * M:M], color="blue")

axX.set_ylabel('Ось X')
axY.set_ylabel('Ось Y')

axX.grid(True)
axY.grid(True)"""

a.plot([i[0] for i in coord], [i[1] for i in coord], color="green")
a.plot([i[0] for i in coordWithNoise], [i[1] for i in coordWithNoise], color="red")
a.plot([i[0] for i in coordKalman], [i[1] for i in coordKalman], color="blue")
a.plot([i[0] for i in coordCourseAndSpeed], [i[1] for i in coordCourseAndSpeed], color="orange")
a.plot([i[0] for i in coordMultiplePoints], [i[1] for i in coordMultiplePoints], color="purple")

"""c = np.correlate([i[0] for i in coord], [i[0] for i in coord], 'full')
c1 = np.correlate([i[0] for i in coord], [i[0] for i in coordKalman], 'full')
c2 = np.correlate([i[0] for i in coord], [i[0] for i in coordCourseAndSpeed], 'full')
c3 = np.correlate([i[0] for i in coord], [i[0] for i in coordMultiplePoints], 'full')


b.plot(c, color="green")
b.plot(c1, color="blue")
b.plot(c2, color="orange")
b.plot(c3, color="purple")"""

plt.show()
def get_R_numba_mp1(p, q):
    R = np.empty((p.shape[0], q.shape[1]))

    for i in range(p.shape[0]):
        for j in range(q.shape[1]):
            rx = p[i, 0] - q[0, j]
            ry = p[i, 1] - q[1, j]
            rz = p[i, 2] - q[2, j]

            R[i, j] = 1 / (1 + math.sqrt(rx * rx + ry * ry + rz * rz))

    return R


"""from numba import float64, jit

@jit(float64[:, :](float64[:, :], float64[:, :]), nopython=True, parallel=True)
def get_R_numba_mp(p, q):
    R = np.empty((p.shape[0], q.shape[1]))

    for i in range(p.shape[0]):
        for j in range(q.shape[1]):
            rx = p[i, 0] - q[0, j]
            ry = p[i, 1] - q[1, j]
            rz = p[i, 2] - q[2, j]

            R[i, j] = 1 / (1 + math.sqrt(rx * rx + ry * ry + rz * rz))

    return R

xd1 = np.zeros(100 * 3).reshape(100, 3)  # истинные координаты перемещения (пока просто нули)
xd1[:][0] = np.random.normal(0, dSignal, 3)  # формирование первой координаты
xdd1 = xd1 + np.random.normal(0, dNoise, size=(100, 3))  # формирование наблюдений
xd2 = np.zeros(100 * 3).reshape(100, 3)  # истинные координаты перемещения (пока просто нули)
xd2[:][0] = np.random.normal(0, dSignal, 3)  # формирование первой координаты
xdd2 = xd2 + np.random.normal(0, dNoise, size=(100, 3))  # формирование наблюдений

start_time4 = time.time()
get_R_numba_mp(xdd1, xdd2)
print("--- %s seconds ---" % (time.time() - start_time4))

start_time5 = time.time()
get_R_numba_mp1(xdd1, xdd2)
print("--- %s seconds ---" % (time.time() - start_time5))"""