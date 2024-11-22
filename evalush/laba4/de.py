import random as rnd
import matplotlib.pyplot as plt
import numpy as np


def rozen(x, y, z):
    result = 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0 + 100.0 * (z - y ** 2.0) ** 2.0 + (1 - y) ** 2.0
    return result


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
        # позиция индивида по Z (первый раз определяется случайно)
        self.z = rnd.triangular(self.start, self.end)
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
        self.score = self.function(self.x, self.y, self.z)

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
        # ограничим наших индивидов по Х
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
        # ограничим наших индивидов по У
        if self.y < 0:
            self.y = max(self.y, self.start)
        else:
            self.y = min(self.y, self.end)
        # отклонение по Z
        delta = 0
        for i in range(1, self.mutationSteps + 1):
            if rnd.random() < 1 / self.mutationSteps:
                delta += 1 / (2 ** i)
        if rnd.randint(0, 1):
            delta = self.end * delta
        else:
            delta = self.start * delta
        self.z += delta
        # ограничим наших индивидов по Z
        if self.z < 0:
            self.z = max(self.z, self.start)
        else:
            self.z = min(self.z, self.end)


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
        # точка Х, У,Z, где нашли минимальное значение
        self.xyz = [float('inf'), float('inf'), float('inf')]
        self.bestscore_plot = []
        # область поиска
        self.start = start
        self.end = end
        self.population = []

    def dif_ev(self, individ1: Individ, individ2: Individ, individ3: Individ):
        """ Функция дифференциальной эволюции
                :return: 1 индивид
                """
        individ_v = Individ(self.start, self.end, self.mutationSteps, self.function)
        alpha = rnd.uniform(0.01, 1)
        betta = alpha / 10.0
        individ_v.x = individ1.x + alpha * (individ2.x - individ3.x) + betta * (individ2.x + individ3.x)
        individ_v.y = individ1.y + alpha * (individ2.y - individ3.y) + betta * (individ2.y + individ3.y)
        individ_v.z = individ1.z + alpha * (individ2.z - individ3.z) + betta * (individ2.z + individ3.z)
        return individ_v

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
        child1.z = parent1.z + alpha * (parent2.z - parent1.z)

        alpha = rnd.uniform(0.01, 1)
        child2.x = parent1.x + alpha * (parent1.x - parent2.x)

        alpha = rnd.uniform(0.01, 1)
        child2.y = parent1.y + alpha * (parent1.y - parent2.y)

        alpha = rnd.uniform(0.01, 1)
        child2.z = parent1.z + alpha * (parent1.z - parent2.z)
        return child1, child2

    def startGenetic(self):

        # создаем стартовую популяцию
        pack = [self.start, self.end, self.mutationSteps, self.function]
        self.population = [Individ(*pack) for _ in range(self.numberOfIndividums)]
        # Рисуем график
        f = lambda x, y, z: self.function(x, y, z)
        X = np.arange(-5, 5, 0.2)
        Y = np.arange(-5, 5, 0.2)
        Z = np.arange(-5, 5, 0.2)
        X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")
        F = f(X, Y, Z)
        plt.ion()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(15, 200)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        mask = f(X, Y, Z) > 0.01
        idx = np.arange(int(np.prod(
            F.shape)))  # np.prod-вычисляет произведение всех элементов в заданном массиве вдоль указанной оси или по всем осям.
        x, y, z = np.unravel_index(idx, F.shape)

        # запускаем алгоритм
        for _ in range(self.numberLives):
            ax.clear()
            # plt.xlim(-5, 10)
            # plt.ylim(-5, 10)
            ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
            ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
            ax.set_zticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

            ax.axes.set_xlim3d(left=0.2, right=9.8)
            ax.axes.set_ylim3d(bottom=0.2, top=9.8)
            ax.axes.set_zlim3d(bottom=0.2, top=9.8)
            ax.scatter(x, y, z, c=F.flatten(), s=30.0 * mask, edgecolor="face", alpha=0.2, cmap='jet', marker="o",
                       linewidth=0)
            xs = []
            ys = []
            zs = []
            for i in range(len(self.population)):
                xs.append(self.population[i].x)
                ys.append(self.population[i].y)
                zs.append(self.population[i].z)

            ax.scatter(xs, ys, zs, color='black', s=2, marker="o", linewidth=0)

            plt.draw()
            plt.gcf().canvas.flush_events()
            # self.population = sorted(self.population, key=lambda item: item.score)
            copy_poulation = self.population[:int(self.numberOfIndividums * self.crossoverRate)]
            childs = []
            for n in range(len(copy_poulation)):
                i1 = i2 = i3 = i4 = 0
                while i1 == i2 or i1 == i3 or i2 == i3 or i1 == i4 or i2 == i4 or i3 == i4:
                    i1, i2, i3 = rnd.randint(0, len(copy_poulation) - 1), rnd.randint(0,
                                                                                      len(copy_poulation) - 1), rnd.randint(
                        0, len(copy_poulation) - 1)
                ind_x = copy_poulation[i1]
                ind_v = self.dif_ev(copy_poulation[i2], copy_poulation[i3], copy_poulation[i4])
                child1, child2 = self.crossover(ind_x, ind_v)
                childs.append(child1)
                childs.append(child2)
                self.population.extend(childs)
                if ind_x.score > ind_v.score:
                    self.population.append(ind_v)
                    for i, ind in enumerate(self.population):
                        if ind == ind_x:
                            self.population.remove(self.population[i])

            # # сортируем популяцию по значению score
            # self.population = sorted(self.population, key=lambda item: item.score)
            # # берем ту часть лучших индивидов, которых будем скрещивать между собой
            # bestPopulation = self.population[:int(self.numberOfIndividums * self.crossoverRate)]
            # # теперь проводим скрещивание столько раз, сколько было задано по коэффициенту кроссовера
            # childs = []
            # for individ1 in bestPopulation:
            #     # находим случайную пару для каждого индивида и скрещиваем
            #     individ2 = rnd.choice(bestPopulation)
            #     while individ1 == individ2:
            #         individ2 = rnd.choice(bestPopulation)
            #     child1, child2 = self.crossover(individ1, individ2)
            #     childs.append(child1)
            #     childs.append(child2)
            # # добавляем всех новых потомков в нашу популяцию
            # self.population.extend(childs)

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
                self.xyz = [self.population[0].x, self.population[0].y, self.population[0].z]
            self.bestscore_plot.append(self.bestScore)

        print("ОПТИМИЗИРОВАННОЕ ЗНАЧЕНИЕ ФУНКЦИИ:", self.bestScore, "В ТОЧКЕ:", self.xyz)
        plt.ioff()
        plt.show()
        plt.plot(range(1, self.numberLives + 1), self.bestscore_plot, color='red')
        plt.xlabel('Поколение')
        plt.ylabel('Оптимизированное значение функции')
        plt.title('Дифференциальная эволюция')
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
                numberLives=50, function=rozen, start=-5, end=5)
    a.startGenetic()
