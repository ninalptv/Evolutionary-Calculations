import random as rnd
import time

import matplotlib.pyplot as plt
import numpy as np


def levi(x, y):
    a = (np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (
            1 + (np.sin(3 * np.pi * y)) ** 2)
         + (y - 1) ** 2 * (1 + (np.sin(2 * np.pi * y) ** 2)))
    return a


def rozen(x, y):
    result = 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0
    return result


def rastrigin(x, y):
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))


class Unit:
    """класс для одной частицы"""

    def __init__(self, start, end, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, function):
        # область поиска
        self.start = start
        self.end = end
        # коэффициенты для изменения скорости(коэффициенты приоритетов смещения частиц к разным точкам)
        self.currentVelocityRatio = currentVelocityRatio
        self.localVelocityRatio = localVelocityRatio
        self.globalVelocityRatio = globalVelocityRatio
        # функция
        self.function = function
        # лучшая локальная позиция
        self.localBestPos = [rnd.uniform(self.start, self.end), rnd.uniform(self.start, self.end)]
        self.localBestScore = self.function(*self.localBestPos)
        # текущая позиция
        self.currentPos = self.localBestPos[:]
        self.score = self.function(*self.localBestPos)
        # значение глобальной позиции
        self.globalBestPos = []

        # скорость
        minval = -(self.end - self.start)
        maxval = self.end - self.start
        self.velocity = [rnd.uniform(minval, maxval), rnd.uniform(minval, maxval)]

    def nextIteration(self):
        """ Метод для нахождения новой позиции частицы"""
        # случайные данные для изменения скорости
        rndCurrentBestPosition = [rnd.random(), rnd.random()]
        rndGlobalBestPosition = [rnd.random(), rnd.random()]
        # делаем перерасчет скорости частицы исходя из всех введенных параметров
        velocityRatio = self.localVelocityRatio + self.globalVelocityRatio
        commonVelocityRatio = 2 * self.currentVelocityRatio / abs(
            2 - velocityRatio - np.sqrt(velocityRatio ** 2 - 4 * velocityRatio))
        multLocal = list(map(lambda x: x * commonVelocityRatio * self.localVelocityRatio, rndCurrentBestPosition))
        betweenLocalAndCurPos = [self.localBestPos[0] - self.currentPos[0], self.localBestPos[1] - self.currentPos[1]]
        betweenGlobalAndCurPos = [self.globalBestPos[0] - self.currentPos[0],
                                  self.globalBestPos[1] - self.currentPos[1]]
        multGlobal = list(map(lambda x: x * commonVelocityRatio * self.globalVelocityRatio, rndGlobalBestPosition))
        newVelocity1 = list(map(lambda coord: coord * commonVelocityRatio, self.velocity))
        newVelocity2 = [coord1 * coord2 for coord1, coord2 in zip(multLocal, betweenLocalAndCurPos)]
        newVelocity3 = [coord1 * coord2 for coord1, coord2 in zip(multGlobal, betweenGlobalAndCurPos)]
        self.velocity = [coord1 + coord2 + coord3 for coord1, coord2, coord3 in
                         zip(newVelocity1, newVelocity2, newVelocity3)]
        # передвигаем частицу и смотрим, какое значение целевой фунции получается
        self.currentPos = [coord1 + coord2 for coord1, coord2 in zip(self.currentPos, self.velocity)]
        newScore = self.function(*self.currentPos)
        if newScore < self.localBestScore:
            self.localBestPos = self.currentPos[:]
            self.localBestScore = newScore
        return newScore


class Swarm:
    """класс роя, который на вход будет принимать размер роя,
    коэффициенты приоритетов смещения частиц к разным точкам,
    количество итераций алгоритма, целевую функцию и область поиска экстремума"""

    def __init__(self, sizeSwarm,
                 currentVelocityRatio,
                 localVelocityRatio,
                 globalVelocityRatio,
                 numbersOfLife,
                 function,
                 start,
                 end):
        # размер популяции частиц
        self.sizeSwarm = sizeSwarm
        # коэффициенты изменения скорости
        self.currentVelocityRatio = currentVelocityRatio
        self.localVelocityRatio = localVelocityRatio
        self.globalVelocityRatio = globalVelocityRatio
        # количество итераций алгоритма
        self.numbersOfLife = numbersOfLife
        # функция для поиска экстремума
        self.function = function
        # область поиска
        self.start = start
        self.end = end
        # рой частиц
        self.swarm = []
        # данные о лучшей позиции
        self.globalBestPos = []
        self.globalBestScore = float('inf')
        self.bestscore_plot = []

    def startSwarm(self):
        """ Метод для запуска алгоритма"""
        # создаем рой
        pack = [self.start, self.end, self.currentVelocityRatio, self.localVelocityRatio, self.globalVelocityRatio,
                self.function]
        self.swarm = [Unit(*pack) for _ in range(self.sizeSwarm)]
        # Рисуем график
        f = lambda x, y: self.function(x, y)
        X = np.linspace(-5, 5, 200)
        Y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(X, Y)
        Z = f(X, Y)
        plt.ion()
        fig, ax = plt.subplots()
        # пересчитываем лучшее значение для только что созданного роя
        for unit in self.swarm:
            if unit.localBestScore < self.globalBestScore:
                self.globalBestScore = unit.localBestScore
                self.globalBestPos = unit.localBestPos
        for _ in range(self.numbersOfLife):
            # ptMins = [точки минимума целевой функции]
            ax.clear()
            ax.contour(X, Y, Z, 8, alpha=.75, cmap='jet')
            # ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=1)
            xs = []
            ys = []
            for unit in self.swarm:
                xs.append(unit.currentPos[0])
                ys.append(unit.currentPos[1])
                unit.globalBestPos = self.globalBestPos
                score = unit.nextIteration()
                if score < self.globalBestScore:
                    self.globalBestScore = score
                    self.globalBestPos = unit.localBestPos
            ax.scatter(xs, ys, color='green', s=2, zorder=0)

            plt.draw()
            plt.gcf().canvas.flush_events()
            time.sleep(0.1)
            self.bestscore_plot.append(self.globalBestScore)


        print("МИНИМУМ ФУНКЦИИ:", self.globalBestScore, "В ТОЧКЕ:", self.globalBestPos)
        plt.ioff()
        plt.show()
        plt.plot(range(1, self.numbersOfLife + 1), self.bestscore_plot, color='red')
        plt.xlabel('Поколение')
        plt.ylabel('Оптимизированное значение функции')
        plt.title('Простой алгоритм роя частиц')
        plt.show()

if __name__ == '__main__':
    a = Swarm(50, 0.1, 1, 5, 50, levi, -5, 5)
    a.startSwarm()
