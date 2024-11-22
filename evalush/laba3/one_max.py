# Решение задачи onemax генетическим алгоритмом с одноточечным кроссовером
import random
import matplotlib.pyplot as plt

# константы задачи
ONE_MAX_LENGTH = 200  # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 2000  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений

# счетчик псевдослучайных чисел
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class FitnessMax():
    def __init__(self):
        self.values = [0]  # начальная приспособленность особей будет принимать нулевое значение


class Individual(list):  # наследуется от базового класса list(список),тк индмвмд представлен в виде списка(хромосомы)
    """Представление каждого индивидуума в популяции"""

    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()  # связано с вычислением приспособленности данного индивидуума


def oneMaxFitness(individual):
    """функция принадлежности для определения приспособленности отдельной особи"""
    return sum(individual),  # кортеж


def individualCreator():
    """создает отдельного индивидуума"""
    return Individual([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])


def populationCreator(n=0):
    """создает всю популяцию"""
    return list([individualCreator() for i in range(n)])


population = populationCreator(n=POPULATION_SIZE)  # создаем популяцию
generationCounter = 0  # счетчик числа поколений

fitnessValues = list(map(oneMaxFitness, population))  # вычислим приспособленность каждой сформированний особи

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue  # свойству values(кот находится в классе FitnessMax()) присваеваем вычисленное значение функции приспособленности

maxFitnessValues = []  # будут хранить статистику алгоритма, мах приспособленность особи в тек популяции
meanFitnessValues = []  # будут хранить статистику алгоритма, средняя приспособленность всех особей в тек популяци


def clone(value):
    """необходимая после отбора проклонировать каждого индивидуума,
    тк в процессе отбора может быть отобран один и тот же индивидуум дважды,
    и в популяции будет две ссылки на один и тот же список"""
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind


def selTournament(population, p_len):  # параметрами передается популяция и ее зазмер
    """Функция турнирного отбора"""
    offspring = []  # будет формироваться новый список из отобранных особей
    for n in range(p_len):  # делаем цикл по всей популяции
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:  # случайным образом отбираем трёх особей, что бы индексы быси различными
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[
            0]))  # среди этих особей отбираем ту, у которойц максимальная приспособленность

    return offspring


def cxOnePoint(child1, child2):
    """Функция одноточечного кроссовера"""
    s = random.randint(2,
                       len(child1) - 3)  # точка разреза хромосомы(определяем случайным образом, но чтобы границы не попадали)
    child1[s:], child2[s:] = child2[s:], child1[s:]  # меняем части хромосом




def mutFlipBit(mutant, indpb=0.01):  # indpb-вероятность мутации отдельного гена
    """функция мутации"""
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1


fitnessValues = [individual.fitness.values[0] for individual in
                 population]  # коллекция, состоящее из значений приспособленностей особей в данной популяции

# Основной цикл работы генетического алгоритма
while max(
        fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:  # цикл пока либо не найде лучшее решение либо не пройдем 50 поколений
    generationCounter += 1
    offspring = selTournament(population, len(population))  # отбираем особей с помощью турнирного отбора
    offspring = list(map(clone, offspring))  # клонируем, чтобы не было повторений

    for child1, child2 in zip(offspring[::2], offspring[
                                              1::2]):  # выполняем скрещивание, выбираем из списка четный элемент и нечетный(четная особь и нечетная особь из популяции), то образом перебираются неповторяющие пары родителей
        if random.random() < P_CROSSOVER:  # с вероятностью кроссовера, если условие выполнилось то родители превращаются в потомков, если не выполнилось то родители остаются в популяции
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant, indpb=1.0 / ONE_MAX_LENGTH)

    freshFitnessValues = list(map(oneMaxFitness, offspring))  # обновляем приспособленность особи новой популяции
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue  # записываем их в свойство values для каждого индивидуума

    population[:] = offspring  # обновляем список популяции

    fitnessValues = [ind.fitness.values[0] for ind in
                     population]  # обновляем список, значение приспособленности каждой особи в популяции

    maxFitness = max(fitnessValues)  # особь с максимальной приспособленностью
    meanFitness = sum(fitnessValues) / len(population)  # средняя приспособленность всех особей в популяции
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()

# То есть, реализация генетического алгоритма при имитации эвалюции позволило получить наиболее приспособленную особь состоящую из всех едениц
# Генетический алгоритм нашел решение поставленной задачи