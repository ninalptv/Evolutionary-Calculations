import matplotlib.pyplot as pl
import numpy as np


def levi(point):
    x, y = point
    a = (np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (
            1 + (np.sin(3 * np.pi * y)) ** 2)
         + (y - 1) ** 2 * (1 + (np.sin(2 * np.pi * y) ** 2)))
    return a


def rozen(point):
    x, y = point
    result = 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0
    return result


def rastrigin(point):
    x, y = point
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))


class Point(object):
    """Вспомогательный класс Vector и перегружаем операторы для возможности производить с векторами базовые операции"""

    def __init__(self, x, y):
        """ Создается вектор"""
        self.x = x
        self.y = y

    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)

    def __rmul__(self, other):
        x = self.x * other
        y = self.y * other
        return Point(x, y)

    def __truediv__(self, other):
        x = self.x / other
        y = self.y / other
        return Point(x, y)

    def vec(self):
        """Возвращает точку"""
        return [self.x, self.y]


def nelder_mead(func, p1=Point(0, 0), p2=Point(1, 0), p3=Point(0, 1),
                alpha=1, beta=0.5, gamma=2, max_iter=10, epsilon=0.01):
    """
    func: Функция для оптимизации
    p1: первая точка для симплекса
    p2: вторая точка для симплекса
    p3: третья точка для симплекса
    alpha: коэффициент при отражении точки
    beta: коэффициент сжатия
    gamma: коэффициент увеличения расстояния
    maxiter:
    return:
    """
    points_plot = []
    for i in range(max_iter):
        print(f"Итерация {i + 1}:")
        adict = {p1: func(p1.vec()), p2: func(p2.vec()), p3: func(p3.vec())}
        print(f"Значение функции в точках {adict}")
        points = sorted(adict.items(), key=lambda x: x[1])  # сортируем точки, чтобы найти лучшую
        best_point = points[0][0]
        points_plot.append(best_point.vec())
        good_point = points[1][0]
        points_plot.append(good_point.vec())
        worst_point = points[2][0]
        points_plot.append(worst_point.vec())
        points_plot.append(best_point.vec())
        print(f"Худшая точка: {worst_point}, отражаем относительно нее.")
        mid = (good_point + best_point) / 2  # находим середину отрезка
        xr = mid + alpha * (mid - worst_point)  # отражение точки worst_point относительно отрезка
        print(f"Новая точка:{xr}, значение функции в ней: {func(xr.vec())}.")
        if func(xr.vec()) < func(good_point.vec()):  # если ситуация немного улучшилась, то xr новая худшая точка
            worst_point = xr
        else:
            if func(xr.vec()) < func(worst_point.vec()):
                worst_point = xr
            c = (worst_point + mid) / 2
            if func(c.vec()) < func(worst_point.vec()):
                worst_point = c
        if func(xr.vec()) < func(best_point.vec()):
            print("Значение минимальное из всех-пробуем растянуть.")
            xe = mid + gamma * (xr - mid)  # Растяжение
            if func(xe.vec()) < func(xr.vec()):
                worst_point = xe
            else:
                worst_point = xr
            print(f"Новая точка:{worst_point}, значение функции в ней: {func(worst_point.vec())}.")

        if func(xr.vec()) > func(good_point.vec()):
            print("Значение не минимальное из всех-Сжимаем.")
            xc = mid + beta * (worst_point - mid)  # Сжатие
            if func(xc.vec()) < func(worst_point.vec()):
                worst_point = xc
            print(f"Новая точка:{worst_point}, значение функции в ней: {func(worst_point.vec())}.")

        # Обновленные точки
        p1 = worst_point
        p2 = good_point
        p3 = best_point

        print(f"Новые точки: {p1} ,{p2}, {p3}\n")
    print(f"\tМинимальная точка: {best_point}\n\tМинимальное значение функции в ней: {func(best_point.vec())}")
    f = lambda x, y: func([x, y])
    X = np.linspace(-5, 5, 200)
    Y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)
    pl.contour(X, Y, Z, 8, alpha=.75, cmap='jet')
    points1 = np.array(points_plot)
    # print(points1)
    xs = []
    ys = []
    for i in range(len(points1)):
        xs.append(points1[i][0])
        ys.append(points1[i][1])

    pl.plot(xs, ys, marker='o', color='r')
    pl.scatter(best_point.x, best_point.y, c="magenta", s=100)

    pl.show()


if __name__ == '__main__':
    while True:
        inp = input('Какой функцией проверим:\nРастрыгина(введите 1),\nРозенброка(введите 2),'
                    '\nЛеви(введите 3)\nдля выхода нажмите "q" :_ ')
        if inp == "1":
            print("Тест метода Нелдера-Мида функцией Растригина:\n")
            nelder_mead(rastrigin)
        if inp == "2":
            print("\n\nТест метода Нелдера-Мида функцией Розенброка:\n")
            nelder_mead(rozen)
        if inp == "3":
            print("\n\nТест метода Нелдера-Мида функцией Леви:\n")
            nelder_mead(levi)
        if inp == "q":
            break
