import copy
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


def clusterization(array, k):
    n = len(array)
    dim = len(array[0])

    cluster = [[0 for i in range(dim)] for q in range(k)]

    for i in range(dim):
        for q in range(k):
            cluster[q][i] = rnd.randint(-5, 5)

    cluster_content = [[] for i in range(k)]

    for i in range(n):
        min_distance = float('inf')
        situable_cluster = -1
        for j in range(k):
            distance = 0
            for q in range(dim):
                distance += (array[i][q] - cluster[j][q]) ** 2

            distance = distance ** (1 / 2)
            if distance < min_distance:
                min_distance = distance
                situable_cluster = j

        cluster_content[situable_cluster].append(array[i])

    privious_cluster = copy.deepcopy(cluster)
    while 1:
        k = len(cluster)
        for i in range(k):  # по i кластерам
            for q in range(dim):  # по q параметрам
                updated_parameter = 0
                for j in range(len(cluster_content[i])):
                    updated_parameter += cluster_content[i][j][q]
                if len(cluster_content[i]) != 0:
                    updated_parameter = updated_parameter / len(cluster_content[i])
                cluster[i][q] = updated_parameter
        cluster_content = [[] for i in range(k)]

        for i in range(n):
            min_distance = float('inf')
            situable_cluster = -1
            for j in range(k):
                distance = 0
                for q in range(dim):
                    distance += (array[i][q] - cluster[j][q]) ** 2

                distance = distance ** (1 / 2)
                if distance < min_distance:
                    min_distance = distance
                    situable_cluster = j

            cluster_content[situable_cluster].append(array[i])
        if cluster == privious_cluster:
            break
        privious_cluster = copy.deepcopy(cluster)
    return cluster_content


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
        self.globalBestPos1 = []
        self.globalBestPos2 = []
        # скорость
        minval = -(self.end - self.start)
        maxval = self.end - self.start
        self.velocity = [rnd.uniform(minval, maxval), rnd.uniform(minval, maxval)]

    def nextIteration1(self):
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
        betweenGlobalAndCurPos1 = [self.globalBestPos1[0] - self.currentPos[0],
                                   self.globalBestPos1[1] - self.currentPos[1]]
        betweenGlobalAndCurPos2 = [self.globalBestPos2[0] - self.currentPos[0],
                                   self.globalBestPos2[1] - self.currentPos[1]]
        multGlobal = list(map(lambda x: x * commonVelocityRatio * self.globalVelocityRatio, rndGlobalBestPosition))
        newVelocity1 = list(map(lambda coord: coord * commonVelocityRatio, self.velocity))
        newVelocity2 = [coord1 * coord2 for coord1, coord2 in zip(multLocal, betweenLocalAndCurPos)]
        newVelocity3 = [coord1 * coord2 for coord1, coord2 in zip(multGlobal, betweenGlobalAndCurPos1)]
        newVelocity4 = [coord1 * coord2 for coord1, coord2 in zip(multGlobal, betweenGlobalAndCurPos2)]
        self.velocity = [coord1 + coord2 + coord3 + coord4 for coord1, coord2, coord3, coord4 in
                         zip(newVelocity1, newVelocity2, newVelocity3, newVelocity4)]
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
        self.swarm1 = []
        self.swarm2 = []
        # данные о лучшей позиции
        self.globalBestPos = []
        self.globalBestPos1 = []
        self.globalBestPos2 = []
        self.globalBestScore = float('inf')
        self.globalBestScore1 = float('inf')
        self.globalBestScore2 = float('inf')
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

        cluster_swarm = []
        for unit in self.swarm:
            cluster_swarm.append(unit.currentPos)
        clus = clusterization(cluster_swarm, 2)
        clus1 = clus[0]
        clus2 = clus[1]

        for unit in self.swarm:
            for i in clus1:
                if unit.currentPos == i:
                    self.swarm1.append(unit)
            for i in clus2:
                if unit.currentPos == i:
                    self.swarm2.append(unit)

        for unit in self.swarm1:
            if unit.localBestScore < self.globalBestScore1:
                self.globalBestScore1 = unit.localBestScore
                self.globalBestPos1 = unit.localBestPos
        for unit in self.swarm2:
            if unit.localBestScore < self.globalBestScore2:
                self.globalBestScore2 = unit.localBestScore
                self.globalBestPos2 = unit.localBestPos
        for _ in range(self.numbersOfLife):
            # ptMins = [точки минимума целевой функции]
            ax.clear()
            ax.contour(X, Y, Z, 8, alpha=.75, cmap='jet')
            # ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=1)
            xs = []
            ys = []
            xs1 = []
            ys1 = []
            for unit in self.swarm1:
                xs.append(unit.currentPos[0])
                ys.append(unit.currentPos[1])
                unit.globalBestPos1 = self.globalBestPos1
                unit.globalBestPos2 = self.globalBestPos2
                score = unit.nextIteration1()
                if score < self.globalBestScore1:
                    self.globalBestScore1 = score
                    self.globalBestPos1 = unit.localBestPos
            for unit in self.swarm2:
                xs1.append(unit.currentPos[0])
                ys1.append(unit.currentPos[1])
                unit.globalBestPos1 = self.globalBestPos1
                unit.globalBestPos2 = self.globalBestPos2
                score = unit.nextIteration1()
                if score < self.globalBestScore2:
                    self.globalBestScore2 = score
                    self.globalBestPos2 = unit.localBestPos
            ax.scatter(xs, ys, color='green', s=2, zorder=0)
            ax.scatter(xs1, ys1, color='orange', s=2, zorder=0)

            plt.draw()
            plt.gcf().canvas.flush_events()
            time.sleep(0.1)
            self.bestscore_plot.append((self.globalBestScore2 + self.globalBestScore1) / 2)

        print("МИНИМУМ ФУНКЦИИ 1 КЛАСТЕРА ЧАСТИЦ:", self.globalBestScore1, "В ТОЧКЕ:", self.globalBestPos1)
        print("МИНИМУМ ФУНКЦИИ 2 КЛАСТЕРА ЧАСТИЦ:", self.globalBestScore2, "В ТОЧКЕ:", self.globalBestPos2)
        print("УСРЕДНЕННОЕ ЗНАЧЕНИЕ: МИНИМУМ:", (self.globalBestScore2 + self.globalBestScore1) / 2, "В ТОЧКЕ:",
              [(self.globalBestPos1[0] + self.globalBestPos2[0]) / 2,
               (self.globalBestPos1[1] + self.globalBestPos2[1]) / 2])
        plt.ioff()
        plt.show()
        plt.plot(range(1, self.numbersOfLife + 1), self.bestscore_plot, color='red')
        plt.xlabel('Поколение')
        plt.ylabel('Оптимизированное значение функции')
        plt.title('Алгоритм роя частиц с использованием кластеризации')
        plt.show()


if __name__ == '__main__':
    a = Swarm(100, 0.1, 1, 5, 50, levi, -5, 5)
    a.startSwarm()
