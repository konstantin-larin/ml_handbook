from cProfile import label

import numpy as np
from numpy.ma.core import argmax
from sklearn.neighbors import NearestNeighbors # буду сверяться по нему
import time

# здесь я буду сверять алгоритмы метода поиска ближайших соседей на скорость и корректность.
# за сверяющий корректность алгоритм возьму NearestNeighbors от scikit-learn


# Обучающая выборка (3D-векторы + метки классов)
X_train = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [2.0, 3.0, 4.0],
    [6.0, 7.0, 8.0],
    [3.0, 5.0, 7.0]
])
# Метки классов (допустим, бинарная классификация)
Y_train = np.array([0, 1, 0, 1, 0, 1])
# Новый объект, для которого будем искать ближайших соседей
new_obj = np.array([5.0, 5.0, 5.0])

# функция, измеряющая время алгоритма
def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return end-start, result
# функция проверяющая алгоритм относительно sklearn.neighbors.NearestNeighbors
def check_alg(func, k, data, labels, new_vect):
    # алгоритм от scikit-learn (заведомо правильный)
    sk_alg = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='brute')
    def use_sk_alg():
        sk_alg.fit(data)
        # здесь на вход нужен двумерный массив, поэтому я оборачиваю
        return sk_alg.kneighbors(X=np.array([new_vect]))
    sk_time, sk_result =  measure_time(use_sk_alg)
    sk_distances, sk_indices = sk_result
    print("Ближайшие расстояния по sk_learn:", sk_distances)
    print("Ближайшие объекты по sk_learn:", data[sk_indices])
    print("Классы ближайших объектов по sk_learn:", labels[sk_indices])
    print("Время алгоритма sklearn:", sk_time)
    print("")


    print("В дело вступает:", func.__name__)
    my_time, my_result = measure_time(func,data, new_vect, k)
    my_distances, my_indices = my_result
    print("Ближайшие расстояния по", func.__name__, ":", my_distances)
    print("Ближайшие объекты по", func.__name__, ":", data[my_indices])
    print("Классы ближайших объектов по:", func.__name__, ":", labels[my_indices])
    print("Время алгоритма", func.__name__, ":", my_time)


# простой перебор
def brute(data, new_vect, k=3):
    k_nearest = list()
    max_dist = np.inf
    for i in range(len(data)):
        vec = data[i]
        distance = np.linalg.norm(new_vect - vec)
        if distance < max_dist:
            if len(k_nearest) == k:
                k_nearest_distances = [n[0] for n in k_nearest]
                k_nearest_max_distance, n_index = np.max(k_nearest_distances), np.argmax(k_nearest_distances)
                if k_nearest_max_distance > distance:
                    del k_nearest[n_index]
                    k_nearest.append((distance, i))
                    max_dist = k_nearest_max_distance
                else:
                    max_dist = distance
            else:
                k_nearest.append((distance, i))
    k_nearest = np.array(k_nearest)
    k_nearest_sorted_indices = np.argsort(k_nearest[:, 0])
    k_nearest = k_nearest[k_nearest_sorted_indices]

    return k_nearest[:, 0], k_nearest[:, 1].astype(int)

check_alg(brute, 4, X_train, Y_train, new_obj)
# k-d дерево
# сначала хочу вникнуть в бинарный поиск
def explain_binary():
    class BinarySearchTree:
        def __init__(self, value, label=None):
            self.value = value #значение узла - объект
            self.label = label #метка класса
            self.left = None #левый потомок
            self.right = None #правый потомок
        def insert(self, value, label):
            if value < self.value:
                if self.left is None:
                    self.left = BinarySearchTree(value, label) #инициализируем потомка
                else:
                    self.left.insert(value, label) # вызываем insert у потомка
            elif value > self.value:
                if self.right is None:
                    self.right = BinarySearchTree(value, label) #инициализируем потомка
                else:
                    self.right.insert(value, label) # вызываем insert у потомка

        def find_k_nearest(self, target, k, nearest=None):
            if nearest is None:
                nearest = []
            # вычисляем расстояние от текущего узла до target
            distance = abs(self.value - target)
            # добавляем текущий узел в список ближайших соседей
            nearest.append((distance, self.value, self.label))
            #     сортируем список по distance
            nearest.sort(key=lambda x: x[0])
            if len(nearest) > k:
                nearest.pop()

            if target < self.value and self.left:
                self.left.find_k_nearest(target, k, nearest)
            elif target > self.value and self.right:
                self.right.find_k_nearest(target, k, nearest)
            return nearest


        def print_tree(self, level=0, prefix="Root: "):
            """Метод для красивой печати дерева"""
            print(" " * (level * 4) + prefix + str(self.value))  # Отступы для визуализации
            if self.left:  # Если есть левый потомок, печатаем его
                self.left.print_tree(level + 1, "L--- ")
            if self.right:  # Если есть правый потомок, печатаем его
                self.right.print_tree(level + 1, "R--- ")

    # check_alg(func=brute, k=4, data=X_train, labels=Y_train, new_vect=new_obj)
    # Создаём бинарное дерево поиска из данных
    data = [1, 100, 3, 30, 5, 6, 7, 8, 10, 14, 20, 21]
    labels = [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]

    root = BinarySearchTree(data[0], labels[0])
    for i in range(1, len(data)):
        root.insert(data[i], labels[i])
    # Новый объект для поиска ближайших соседей
    new_obj_1 = 9
    k = 3
    #поиск
    nearest = root.find_k_nearest(new_obj_1, k)
    print(nearest)
# explain_binary()

#когда не впадлу будет - реализую KD-tree https://ru.hexlet.io/courses/algorithms-trees/lessons/kdtrees/theory_unit
