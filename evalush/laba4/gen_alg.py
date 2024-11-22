import random as rnd
import matplotlib.pyplot as plt
import time
import numpy as np


def levi(x,y):
    a = (np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (
            1 + (np.sin(3 * np.pi * y)) ** 2)
         + (y - 1) ** 2 * (1 + (np.sin(2 * np.pi * y) ** 2)))
    return a


def rozen(x, y):
    result = 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0
    return result


def rastrigin(x,y):
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))


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
        self.population=[]



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
        self.population = [Individ(*pack) for _ in range(self.numberOfIndividums)]
        # Рисуем график
        f = lambda x, y: self.function(x, y)
        X = np.linspace(-5, 5, 200)
        Y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(X, Y)
        Z = f(X, Y)
        plt.ion()
        fig, ax = plt.subplots()
        # запускаем алгоритм
        for _ in range(self.numberLives):
            # ptMins = [points min]
            ax.clear()
            ax.contour(X, Y, Z, 8, alpha=.75, cmap='jet')
            # ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=1)
            xs = []
            ys = []
            for i in range(self.numberOfIndividums):
                xs.append(self.population[i].x)
                ys.append(self.population[i].y)
            ax.scatter(xs, ys, color='green', s=2, zorder=0)

            plt.draw()
            plt.gcf().canvas.flush_events()
            time.sleep(0.1)
            # сортируем популяцию по значению score
            self.population = sorted(self.population, key=lambda item: item.score)
            # берем ту часть лучших индивидов, которых будем скрещивать между собой
            bestPopulation = self.population[:int(self.numberOfIndividums * self.crossoverRate)]
            # теперь проводим скрещивание столько раз, сколько было задано по коэффициенту кроссовера
            childs = []
            for individ1 in bestPopulation:
                # находим случайную пару для каждого индивида и скрещиваем
                individ2 = rnd.choice(bestPopulation)
                while individ1 == individ2:
                    individ2 = rnd.choice(bestPopulation)
                child1, child2 = self.crossover(individ1, individ2)
                childs.append(child1)
                childs.append(child2)
            # добавляем всех новых потомков в нашу популяцию
            self.population.extend(childs)

            for individ in self.population:
                # проводим мутации для каждого индивида
                individ.mutate()
                # пересчитываем значение функции для каждого индивида
                individ.calculateFunction()
            # отбираем лучших индивидов
            self.population = sorted(self.population, key=lambda item: item.score)
            self.population = self.population[:self.numberOfIndividums]
            # теперь проверим значение функции лучшего индивида на наилучшее значение экстремума
            if self.population[0].score < self.bestScore:
                self.bestScore = self.population[0].score
                self.xy = [self.population[0].x, self.population[0].y]
            self.bestscore_plot.append(self.bestScore)


        print("ОПТИМИЗИРОВАННОЕ ЗНАЧЕНИЕ ФУНКЦИИ:", self.xy, "В ТОЧКЕ:", self.bestScore)
        plt.ioff()
        plt.show()
        plt.plot(range(1, self.numberLives + 1), self.bestscore_plot, color='red')
        plt.xlabel('Поколение')
        plt.ylabel('Оптимизированное значение функции')
        plt.title('Простой генетический алгоритм')
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


