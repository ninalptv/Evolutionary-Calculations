import matplotlib.pylab as pl
import numpy as np


def levi(x):
    a = (np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (
            1 + (np.sin(3 * np.pi * x[1])) ** 2)
         + (x[1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[1]) ** 2)))
    return a


def rozen(x):
    result = 100.0 * (x[1] - x[0] ** 2.0) ** 2.0 + (1 - x[0]) ** 2.0
    return result


def rastrigin(x):
    return 10 * len(x) + (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1]))


def huka_djivs(func, delta=0.25, alpha=1, beta=0.5, epsilon=0.15, xk=np.array([2, 0])):
    """Введен начальный шаг поиска по обнаружению delta, коэф ускорения alpha(alpha>=1), коэфф снижения beta(0<beta<1),
    допустимая погрешность epsilon(epsilon>0), начальная точка xk
    delta, alpha, beta, epsilon, xk = 0.5, 1, 0.5, 0.2, np.array([2, 0])"""
    yk = xk.copy()
    dim = len(xk)  # Размер
    k = 1  # Инициализация итераций
    points = []

    while delta > epsilon:
        # Вывод основной информации из поиска
        print(f"Итерация №{k}:")
        print('\tБазисная точка:', xk)
        points.append(xk)
        print('\tЗначение функции в базисной точке:', func(xk))
        print('\tНачальная точка для обнаружения:', yk)
        print('\tЗначение функции в начальной точке:', func(yk))
        print('\tПробный шаг поиска delta:', delta)

        # Обнаружение движения
        for i in range(dim):
            # Генерация координатного направления движения
            e = np.zeros([1, dim])[0]
            e[i] = 1
            t1, t2 = func(yk + delta * e), func(yk)
            # print(f"Точка 1 = {t1}, Точка 2 = {t2} ")
            if t1 < t2:
                yk = yk + delta * e
            else:
                t1, t2 = func(yk - delta * e), func(yk)
                # print(f"Точка 1 = {t1}, Точка 2 = {t2} ")
                if t1 < t2:
                    yk = yk - delta * e
            print(i + 1, 'точка, полученная при вычислении: ', yk)
            print('Значение функции в этой точке:', func(yk))

        # Определяется новая базовая точка и вычисляется новая начальная точка обнаружения
        t1, t2 = func(yk), func(xk)
        if t1 < t2:
            xk, yk = yk, yk + alpha * (yk - xk)
        else:
            delta, yk = delta * beta, xk
        k += 1

        print("\n")

    print("Минимальная точка:", yk)
    print("Минимальное значение:", func(yk))

    f = lambda x, y: func([x, y])
    X = np.linspace(-5, 5, 200)
    Y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    xs = []
    ys = []
    pl.contour(X, Y, Z, 8, alpha=.75, cmap='jet')

    for i in range(len(points)):
        xs.append(points[i][0])
        ys.append(points[i][1])

    pl.plot(xs, ys, marker='o', linestyle='--', color='r', label='Square')
    pl.scatter(yk[0], yk[1], c="magenta", s=100)
    pl.show()


if __name__ == '__main__':
    while True:
        inp = input('\nКакой функцией проверим:\nРастрыгина(введите 1),\nРозенброка(введите 2),'
                    '\nЛеви(введите 3)\nдля выхода нажмите "q" :_ ')
        if inp == "1":
            print("Тест метода Хука-Дживса функцией Растригина:\n")
            huka_djivs(rastrigin, delta=1, alpha=1, beta=0.5, epsilon=0.15, xk=np.array([-2, 2]))
        if inp == "2":
            print("\n\nТест метода Хука-Дживса функцией Розенброка:\n")
            huka_djivs(rozen2)
        if inp == "3":
            print("\n\nТест метода Хука-Дживса функцией Леви:\n")
            huka_djivs(levi, delta=1)
        if inp == "q":
            break
