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


class Individ():
    """ Класс одного индивида в популяции"""

    def __init__(self, start, end, mutationSteps, function):
        # пределы поиска минимума
        self.start = start
        self.end = end
        # позиция индивида по Х (первый раз определяется случайно)
        self.x = rnd.triangular(self.start, self.end)
        # позиция индивида по Y (первый раз определяется случайно)
        self.y = rnd.triangular(self.start, self.end)
        # значение функции, которую реализует индивид
        self.score = 0
        # передаем функцию для оптимизации
        self.function = function
        # количество шагов мутации
        self.mutationSteps = mutationSteps
        # считаем сразу значение функции
        self.calculateFunction()

    def calculateFunction(self):
        """ Функция для пересчета значения значение в индивиде"""
        self.score = self.function(self.x, self.y)

    def mutate(self):
        """ Функция для мутации индивида"""
        # задаем отклонение по Х
        delta = 0
        for i in range(1, self.mutationSteps + 1):
            if rnd.random() < 1 / self.mutationSteps:
                delta += 1 / (2 ** i)
        if rnd.randint(0, 1):
            delta = self.end * delta
        else:
            delta = self.start * delta
        self.x += delta
        # ограничим наших индивидом по Х
        if self.x < 0:
            self.x = max(self.x, self.start)
        else:
            self.x = min(self.x, self.end)
        # отклонение по У
        delta = 0
        for i in range(1, self.mutationSteps + 1):
            if rnd.random() < 1 / self.mutationSteps:
                delta += 1 / (2 ** i)
        if rnd.randint(0, 1):
            delta = self.end * delta
        else:
            delta = self.start * delta
        self.y += delta
        # ограничим наших индивидом по У
        if self.y < 0:
            self.y = max(self.y, self.start)
        else:
            self.y = min(self.y, self.end)


class Genetic:
    """ Класс, отвечающий за реализацию генетического алгоритма"""

    def __init__(self,
                 numberOfIndividums,
                 crossoverRate,
                 mutationSteps,
                 chanceMutations,
                 numberLives,
                 function,
                 start,
                 end):
        # размер популяции
        self.numberOfIndividums = numberOfIndividums
        # какая часть популяции должна производить потомство (в % соотношении)
        self.crossoverRate = crossoverRate
        # количество шагов мутации
        self.mutationSteps = mutationSteps
        # шанс мутации особи
        self.chanceMutations = chanceMutations
        # сколько раз будет появляться новое поколение (сколько раз будет выполняться алгоритм)
        self.numberLives = numberLives
        # функция для поиска минимума
        self.function = function

        # самое минимальное значение, которое было в нашей популяции
        self.bestScore = float('inf')
        # точка Х, У, где нашли минимальное значение
        self.xy = [float('inf'), float('inf')]
        self.bestscore_plot=[]
        # область поиска
        self.start = start
        self.end = end

    def crossover(self, parent1: Individ, parent2: Individ):
        """ Функция для скрещивания двух родителей

        :return: 2 потомка, полученных путем скрещивания
        """
        # создаем 2х новых детей
        child1 = Individ(self.start, self.end, self.mutationSteps, self.function)
        child2 = Individ(self.start, self.end, self.mutationSteps, self.function)
        # создаем новые координаты для детей
        alpha = rnd.uniform(0.01, 1)
        child1.x = parent1.x + alpha * (parent2.x - parent1.x)

        alpha = rnd.uniform(0.01, 1)
        child1.y = parent1.y + alpha * (parent2.y - parent1.y)

        alpha = rnd.uniform(0.01, 1)
        child2.x = parent1.x + alpha * (parent1.x - parent2.x)

        alpha = rnd.uniform(0.01, 1)
        child2.y = parent1.y + alpha * (parent1.y - parent2.y)
        return child1, child2

    def startGenetic(self):

        # создаем стартовую популяцию
        pack = [self.start, self.end, self.mutationSteps, self.function]
        population = [Individ(*pack) for _ in range(self.numberOfIndividums)]
        # Рисуем график
        f = lambda x, y: self.function(x, y)
        X = np.linspace(-5, 5, 200)
        Y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(X, Y)
        Z = f(X, Y)
        plt.ion()
        fig, ax = plt.subplots()
        cluster_pop = []
        for i in population:
            cluster_pop.append([i.x, i.y, i.function(i.x, i.y)])
        cluster = clusterization(cluster_pop, 2)
        clus1 = cluster[0]
        clus2 = cluster[1]
        population1 = []
        population2 = []
        for i in population:
            for j in clus1:
                if j[0] == i.x and j[1] == i.y:
                    population1.append(i)
            for j in clus2:
                if j[0] == i.x and j[1] == i.y:
                    population2.append(i)
        numberOfIndividumsPop1 = len(population1)
        numberOfIndividumsPop2 = len(population2)

        cros_value = "1"
        # запускаем алгоритм
        for _ in range(self.numberLives):
            # ptMins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]
            ax.clear()
            ax.contour(X, Y, Z, 8, alpha=.75, cmap='jet')
            # ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=1)
            xs = []
            ys = []
            xs1 = []
            ys1 = []
            for i in range(numberOfIndividumsPop1):
                xs.append(population1[i].x)
                ys.append(population1[i].y)
            for i in range(numberOfIndividumsPop2):
                xs1.append(population2[i].x)
                ys1.append(population2[i].y)
            ax.scatter(xs, ys, color='green', s=2, zorder=0)
            ax.scatter(xs1, ys1, color='red', s=2, zorder=0)

            plt.draw()
            plt.gcf().canvas.flush_events()
            time.sleep(0.1)
            # сортируем популяцию по значению score
            population1 = sorted(population1, key=lambda item: item.score)
            population2 = sorted(population2, key=lambda item: item.score)
            # берем ту часть лучших индивидов, которых будем скрещивать между собой
            bestPopulation1 = population1[:int(numberOfIndividumsPop1 * self.crossoverRate)]
            bestPopulation2 = population2[:int(numberOfIndividumsPop2 * self.crossoverRate)]
            # теперь проводим скрещивание столько раз, сколько было задано по коэффициенту кроссовера
            childs = []
            if cros_value == "1":
                for individ1 in bestPopulation1:
                    # находим случайную пару для каждого индивида и скрещиваем
                    individ2 = rnd.choice(bestPopulation1)
                    while individ1 == individ2:
                        individ2 = rnd.choice(bestPopulation1)
                    child1, child2 = self.crossover(individ1, individ2)
                    childs.append(child1)
                    childs.append(child2)
                population1.extend(childs)
                for individ in population1:
                    # проводим мутации для каждого индивида
                    individ.mutate()
                    # пересчитываем значение функции для каждого индивида
                    individ.calculateFunction()
                # отбираем лучших индивидов
                population1 = sorted(population1, key=lambda item: item.score)
                population1 = population1[:numberOfIndividumsPop1]
                # теперь проверим значение функции лучшего индивида на наилучшее значение экстремума
                if population1[0].score < self.bestScore:
                    self.bestScore = population1[0].score
                    self.xy = [population1[0].x, population1[0].y]


            if cros_value == "2":
                for individ1 in bestPopulation2:
                    # находим случайную пару для каждого индивида и скрещиваем
                    individ2 = rnd.choice(bestPopulation2)
                    while individ1 == individ2:
                        individ2 = rnd.choice(bestPopulation2)
                    child1, child2 = self.crossover(individ1, individ2)
                    childs.append(child1)
                    childs.append(child2)
                population2.extend(childs)
                for individ in population2:
                    # проводим мутации для каждого индивида
                    individ.mutate()
                    # пересчитываем значение функции для каждого индивида
                    individ.calculateFunction()
                # отбираем лучших индивидов
                population2 = sorted(population2, key=lambda item: item.score)
                population2 = population2[:numberOfIndividumsPop2]
                # теперь проверим значение функции лучшего индивида на наилучшее значение экстремума
                if population2[0].score < self.bestScore:
                    self.bestScore = population2[0].score
                    self.xy = [population2[0].x, population2[0].y]

            if cros_value == "3":
                for individ1 in bestPopulation1:
                    # находим случайную пару для каждого индивида и скрещиваем
                    individ2 = rnd.choice(bestPopulation2)
                    child1, child2 = self.crossover(individ1, individ2)
                    childs.append(child1)
                    childs.append(child2)
                population1.extend(childs)
                for individ in population1:
                    # проводим мутации для каждого индивида
                    individ.mutate()
                    # пересчитываем значение функции для каждого индивида
                    individ.calculateFunction()
                # отбираем лучших индивидов
                population1 = sorted(population1, key=lambda item: item.score)
                population1 = population1[:numberOfIndividumsPop1]
                # теперь проверим значение функции лучшего индивида на наилучшее значение экстремума
                if population1[0].score < self.bestScore:
                    self.bestScore = population1[0].score
                    self.xy = [population1[0].x, population1[0].y]

            self.bestscore_plot.append(self.bestScore)
            cros_value = rnd.choice(["1", "2", "3"])

        print("ОПТИМИЗИРОВАННОЕ ЗНАЧЕНИЕ ФУНКЦИИ:", self.xy, "В ТОЧКЕ:", self.bestScore)
        print(len(self.bestscore_plot))
        plt.ioff()
        plt.show()
        plt.plot(range(1,self.numberLives+1), self.bestscore_plot, color='red')
        plt.xlabel('Поколение')
        plt.ylabel('Оптимизированное значение функции')
        plt.title('Генетический алгоритм с использованием кластеризации')
        plt.show()


if __name__ == '__main__':
    # В качестве параметров в класс будем передавать:
    # размер популяции;
    # процент популяции, который сможет воспроизводить потомство;
    # количество шагов для мутации;
    # шанс мутации особи;
    # количество исполнений алгоритма (количество раз для появления нового потомства),
    # целевую функцию, а также область поиска экстремума.
    a = Genetic(numberOfIndividums=200, crossoverRate=0.5, mutationSteps=15, chanceMutations=0.4,
                numberLives=50, function=levi, start=-5, end=5)
    a.startGenetic()
