import numpy as np
from collections import deque

class TransportSolver:
    """
    Класс для решения транспортной задачи методом потенциалов.
    Решает задачу линейного программирования: найти оптимальный план перевозок
    с минимальными транспортными затратами.
    """
    
    def __init__(self, costs, supply, demand):
        """
        Инициализация решателя транспортной задачи.
        
        Parameters:
        -----------
        costs : list[list[float]] или np.ndarray
            Матрица стоимостей перевозок размера m×n, где
            costs[i][j] - стоимость перевозки единицы груза от i-го поставщика к j-му потребителю
        
        supply : list[float] или np.ndarray
            Вектор запасов поставщиков длиной m
        
        demand : list[float] или np.ndarray
            Вектор потребностей потребителей длиной n
        """
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        
        # Приведение задачи к закрытой модели (балансировка)
        self._balance()
        
        # Определение размерности задачи
        self.m, self.n = self.costs.shape
        
    def _balance(self):
        """
        Балансировка транспортной задачи.
        
        Если сумма запасов не равна сумме потребностей, задача приводится к закрытой модели:
        - При избытке запасов добавляется фиктивный потребитель
        - При недостатке запасов добавляется фиктивный поставщик
        Стоимость перевозок к фиктивному узлу устанавливается в 0.
        """
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        # Проверяем необходимость балансировки
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                # Добавляем фиктивного потребителя с недостающей потребностью
                deficit = total_supply - total_demand
                self.demand = np.append(self.demand, deficit)
                # Добавляем столбец нулевых стоимостей
                self.costs = np.column_stack([self.costs, np.zeros(self.costs.shape[0])])
            else:
                # Добавляем фиктивного поставщика с недостающим запасом
                deficit = total_demand - total_supply
                self.supply = np.append(self.supply, deficit)
                # Добавляем строку нулевых стоимостей
                self.costs = np.row_stack([self.costs, np.zeros(self.costs.shape[1])])
    
    def minimal_cost(self):
        """
        Построение начального опорного плана методом минимального элемента.
        
        Алгоритм:
        1. Среди всех доступных клеток (где есть остатки запасов и потребностей) выбираем клетку с минимальной стоимостью
        2. Назначаем в эту клетку максимально возможный объем перевозки (минимум из остатка запаса и потребности)
        3. Уменьшаем остатки запасов и потребностей на величину перевозки
        4. Если остаток запаса поставщика стал нулевым, исключаем его строку из рассмотрения
        5. Если остаток потребности потребителя стал нулевым, исключаем его столбец из рассмотрения
        6. Повторяем до полного распределения всех запасов
        
        Returns:
        --------
        np.ndarray
            Начальный опорный план перевозок (матрица m×n)
        """
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Создаем маску доступных клеток: True если есть и запасы и потребности
        available = np.ones((self.m, self.n), dtype=bool)
        
        # Продолжаем пока есть нераспределенные запасы и потребности
        while np.sum(supply) > 0 and np.sum(demand) > 0:
            # Создаем маску для доступных клеток
            mask = np.outer(supply > 0, demand > 0)
            
            # Устанавливаем стоимость в недоступных клетках как бесконечность
            masked_costs = np.where(mask, self.costs, np.inf)
            
            # Находим клетку с минимальной стоимостью среди доступных
            i, j = np.unravel_index(np.argmin(masked_costs), masked_costs.shape)
            
            # Определяем объем перевозки как минимум из остатков
            amount = min(supply[i], demand[j])
            plan[i, j] = amount
            
            # Обновляем остатки
            supply[i] -= amount
            demand[j] -= amount
        
        return plan
    
    def calculate_potentials(self, plan):
        """
        Вычисление потенциалов поставщиков (u) и потребителей (v).
        
        Потенциалы вычисляются из системы уравнений:
            u[i] + v[j] = costs[i][j] для всех базисных клеток (plan[i][j] > 0)
        
        Для определенности системы одному из потенциалов задается произвольное значение
        (обычно u[0] = 0).
        
        Parameters:
        -----------
        plan : np.ndarray
            Текущий опорный план перевозок
        
        Returns:
        --------
        tuple (np.ndarray, np.ndarray)
            Вектор потенциалов поставщиков u и вектор потенциалов потребителей v
        """
        u = np.full(self.m, np.nan)  # Потенциалы поставщиков (инициализированы как NaN)
        v = np.full(self.n, np.nan)  # Потенциалы потребителей (инициализированы как NaN)
        
        # Находим первую базисную клетку для инициализации
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 0:
                    u[i] = 0  # Задаем произвольное значение
                    v[j] = self.costs[i, j] - u[i]  # Вычисляем v[j] из уравнения
                    break
            if not np.isnan(u[i]):  # Если нашли начальную клетку, выходим
                break
        
        # Итеративное решение системы уравнений методом последовательных приближений
        changed = True
        while changed:
            changed = False
            
            # Проходим по всем клеткам
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] > 0:  # Только базисные клетки
                        # Если известен u[i] но неизвестен v[j]
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                        # Если известен v[j] но неизвестен u[i]
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_cycle(self, plan, i0, j0):
        """
        Поиск цикла перераспределения для свободной клетки.
        
        Цикл - это последовательность базисных клеток, начинающаяся и заканчивающаяся
        в клетке (i0, j0), с чередованием горизонтальных и вертикальных шагов.
        
        Parameters:
        -----------
        plan : np.ndarray
            Текущий опорный план
        i0, j0 : int
            Координаты свободной клетки, для которой ищем цикл
        
        Returns:
        --------
        list of tuples или None
            Список координат клеток цикла или None если цикл не найден
        """
        # Создаем список базисных клеток (ненулевые в плане)
        basic_cells = []
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 0:
                    basic_cells.append((i, j))
        
        # Временный список с добавленной клеткой (i0, j0)
        temp_cells = basic_cells + [(i0, j0)]
        
        # Поиск в ширину (BFS) для нахождения цикла
        queue = deque()
        # Каждый элемент очереди: (текущая строка, текущий столбец, путь, направление движения)
        # direction = True: двигались по строке, False: двигались по столбцу
        queue.append((i0, j0, [(i0, j0)], True))
        
        visited = set()  # Множество посещенных состояний
        
        while queue:
            i, j, path, row_move = queue.popleft()
            
            # Если вернулись в начальную клетку и путь содержит более одной точки
            if len(path) > 1 and (i, j) == (i0, j0):
                return path
            
            # Проверяем, не посещали ли уже это состояние
            key = (i, j, row_move)
            if key in visited:
                continue
            visited.add(key)
            
            if row_move:
                # Двигались по строке → теперь ищем в столбце
                for next_i, next_j in temp_cells:
                    if next_j == j and next_i != i:  # Та же колонка, другая строка
                        queue.append((next_i, next_j, path + [(next_i, next_j)], False))
            else:
                # Двигались по столбцу → теперь ищем в строке
                for next_i, next_j in temp_cells:
                    if next_i == i and next_j != j:  # Та же строка, другая колонка
                        queue.append((next_i, next_j, path + [(next_i, next_j)], True))
        
        return None  # Цикл не найден
    
    def improve(self, plan):
        """
        Улучшение текущего опорного плана методом потенциалов.
        
        Алгоритм:
        1. Вычисляем потенциалы для текущего плана
        2. Для каждой свободной клетки вычисляем оценку Δ = u[i] + v[j] - costs[i][j]
        3. Если все Δ ≤ 0, план оптимален
        4. Иначе выбираем клетку с максимальным Δ > 0
        5. Для этой клетки находим цикл перераспределения
        6. Перераспределяем груз по циклу, улучшая стоимость
        
        Parameters:
        -----------
        plan : np.ndarray
            Текущий опорный план
        
        Returns:
        --------
        tuple (np.ndarray, bool)
            Новый улучшенный план и флаг: был ли план улучшен
        """
        # 1. Вычисление потенциалов
        u, v = self.calculate_potentials(plan)
        
        # 2. Поиск свободной клетки с максимальной положительной оценкой
        max_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] == 0:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta + 1e-10:  # С учетом погрешности вычислений
                        max_delta = delta
                        best_i, best_j = i, j
        
        # 3. Проверка критерия оптимальности
        if max_delta <= 1e-10:
            return plan, False  # План оптимален
        
        # 4. Поиск цикла для клетки с максимальной положительной оценкой
        cycle = self.find_cycle(plan, best_i, best_j)
        if cycle is None:
            return plan, False  # Цикл не найден
        
        # 5. Находим минимальный объем в минусовых клетках цикла
        min_amount = float('inf')
        # В цикле клетки с нечетными индексами (1, 3, 5...) - "минусовые"
        for idx in range(1, len(cycle), 2):
            i, j = cycle[idx]
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        # 6. Перераспределение груза по циклу
        new_plan = plan.copy()
        # В цикле клетки с четными индексами (0, 2, 4...) - "плюсовые"
        for idx, (i, j) in enumerate(cycle[:-1]):  # Последняя точка дублирует первую
            if idx % 2 == 0:  # Плюсовые клетки
                new_plan[i, j] += min_amount
            else:  # Минусовые клетки
                new_plan[i, j] -= min_amount
        
        # Очищаем очень маленькие значения (погрешности вычислений)
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def solve(self, method='minimal_cost'):
        """
        Полное решение транспортной задачи.
        
        Алгоритм:
        1. Построение начального опорного плана
        2. Добавление нулевых базисных клеток при необходимости (для невырожденности)
        3. Итеративное улучшение плана методом потенциалов
        
        Parameters:
        -----------
        method : str
            Метод построения начального плана:
            - 'minimal_cost': метод минимального элемента (по умолчанию)
            - 'north_west': метод северо-западного угла
        
        Returns:
        --------
        np.ndarray
            Оптимальный план перевозок
        """
        # 1. Построение начального опорного плана
        if method == 'north_west':
            # Метод северо-западного угла (не реализован в данном коде)
            plan = self.minimal_cost()  # Используем минимальный элемент как fallback
        else:
            plan = self.minimal_cost()
        
        # 2. Проверка на вырожденность и добавление нулевых базисных клеток
        # В транспортной задаче опорный план должен содержать ровно m + n - 1 базисных клеток
        basic_cells = np.sum(plan > 0)
        required = self.m + self.n - 1
        
        if basic_cells < required:
            # Ищем свободные клетки с минимальной стоимостью для добавления как нулевых базисных
            free_cells = []
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] == 0:
                        free_cells.append((self.costs[i, j], i, j))
            
            # Сортируем по стоимости (минимальная стоимость первая)
            free_cells.sort()
            
            # Добавляем нулевые базисные клетки
            for _, i, j in free_cells:
                if basic_cells < required:
                    plan[i, j] = 0  # Нулевая базисная клетка
                    basic_cells += 1
        
        # 3. Итеративное улучшение плана
        improved = True
        iterations = 0
        
        # Ограничиваем количество итераций для предотвращения бесконечного цикла
        while improved and iterations < 100:
            iterations += 1
            plan, improved = self.improve(plan)
        
        return plan
    
    def calculate_total_cost(self, plan):
        """
        Вычисление общей стоимости плана перевозок.
        
        Parameters:
        -----------
        plan : np.ndarray
            План перевозок
        
        Returns:
        --------
        float
            Общая стоимость всех перевозок по плану
        """
        return np.sum(plan * self.costs)


def test_variant(num, costs, supply, demand, expected):
    """
    Тестирование одного варианта транспортной задачи.
    
    Parameters:
    -----------
    num : int
        Номер варианта
    costs : list[list[float]]
        Матрица стоимостей
    supply : list[float]
        Вектор запасов
    demand : list[float]
        Вектор потребностей
    expected : float
        Ожидаемая оптимальная стоимость
    
    Returns:
    --------
    float
        Полученная стоимость решения
    """
    print(f"\nВариант {num}:")
    print(f"Ожидаемая стоимость: {expected}")
    
    # Создаем решатель и находим решение
    solver = TransportSolver(costs, supply, demand)
    plan = solver.solve()
    cost = solver.calculate_total_cost(plan)
    
    print(f"Полученная стоимость: {cost:.1f}")
    
    # Выводим ненулевые перевозки
    print("План перевозок:")
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            if plan[i, j] > 0:
                print(f"  A{i+1} → B{j+1}: {plan[i, j]:.1f} × {costs[i][j]}")
    
    # Проверяем соответствие ожидаемому результату
    if abs(cost - expected) < 0.1:
        print("✓ Решение верное")
    else:
        print("✗ Решение неверное")
    
    return cost


def solve_interactive():
    """
    Интерактивный режим решения транспортной задачи.
    Позволяет пользователю ввести данные своей задачи.
    """
    print("\n" + "=" * 60)
    print("РЕШЕНИЕ ПОЛЬЗОВАТЕЛЬСКОЙ ТРАНСПОРТНОЙ ЗАДАЧИ")
    print("=" * 60)
    
    try:
        # Ввод размерности задачи
        m = int(input("Введите количество поставщиков: ").strip())
        n = int(input("Введите количество потребителей: ").strip())
        
        # Ввод матрицы стоимостей
        print(f"\nВведите матрицу стоимостей ({m} строк × {n} столбцов):")
        costs = []
        for i in range(m):
            while True:
                row_input = input(f"  Строка {i+1} (числа через пробел): ").strip()
                values = row_input.split()
                if len(values) == n:
                    try:
                        row = [float(x) for x in values]
                        costs.append(row)
                        break
                    except ValueError:
                        print(f"Ошибка: в строке должны быть {n} чисел")
                else:
                    print(f"Ошибка: ожидалось {n} чисел, получено {len(values)}")
        
        # Ввод запасов поставщиков
        print(f"\nВведите запасы поставщиков ({m} чисел):")
        while True:
            supply_input = input("  Запасы: ").strip()
            values = supply_input.split()
            if len(values) == m:
                try:
                    supply = [float(x) for x in values]
                    break
                except ValueError:
                    print("Ошибка: введите числа через пробел")
            else:
                print(f"Ошибка: ожидалось {m} чисел, получено {len(values)}")
        
        # Ввод потребностей потребителей
        print(f"\nВведите потребности потребителей ({n} чисел):")
        while True:
            demand_input = input("  Потребности: ").strip()
            values = demand_input.split()
            if len(values) == n:
                try:
                    demand = [float(x) for x in values]
                    break
                except ValueError:
                    print("Ошибка: введите числа через пробел")
            else:
                print(f"Ошибка: ожидалось {n} чисел, получено {len(values)}")
        
        # Решение задачи
        print("\n" + "=" * 60)
        print("РЕШЕНИЕ ЗАДАЧИ:")
        print("=" * 60)
        
        solver = TransportSolver(costs, supply, demand)
        plan = solver.solve()
        cost = solver.calculate_total_cost(plan)
        
        print(f"Общая стоимость перевозок: {cost:.1f}")
        
        # Вывод подробного плана
        print("\nОптимальный план перевозок:")
        total_shipped = 0
        for i in range(plan.shape[0]):
            for j in range(plan.shape[1]):
                if plan[i, j] > 0:
                    cell_cost = plan[i, j] * costs[i][j]
                    total_shipped += plan[i, j]
                    print(f"  Из A{i+1} в B{j+1}: {plan[i, j]:.1f} ед. (стоимость: {cell_cost:.1f})")
        
        print(f"\nВсего перевезено: {total_shipped:.1f} единиц")
        
    except ValueError as e:
        print(f"\nОшибка ввода: {e}")
        print("Пожалуйста, проверьте правильность введенных данных.")
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}")


def main():
    """
    Главная функция программы.
    Выполняет тестирование двух вариантов и предоставляет интерактивный режим.
    """
    print("=" * 60)
    print("ПРОГРАММА РЕШЕНИЯ ТРАНСПОРТНОЙ ЗАДАЧИ")
    print("Метод потенциалов")
    print("=" * 60)
    
    # Автоматическое тестирование вариантов
    print("\nАВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ:")
    
    # Вариант 1
    costs1 = [[9, 5, 10, 7],
              [11, 8, 5, 6],
              [7, 6, 5, 4],
              [6, 4, 3, 2]]
    
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    expected1 = 1870.0
    
    cost1 = test_variant(1, costs1, supply1, demand1, expected1)
    
    # Вариант 2
    costs2 = [[5, 3, 4, 6, 4],
              [3, 4, 10, 5, 7],
              [4, 6, 9, 3, 4]]
    
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    expected2 = 340.0
    
    cost2 = test_variant(2, costs2, supply2, demand2, expected2)
    
    # Итоги тестирования
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"Вариант 1: {cost1:.1f} (ожидалось {expected1}) - "
          f"{'✓' if abs(cost1 - expected1) < 0.1 else '✗'}")
    print(f"Вариант 2: {cost2:.1f} (ожидалось {expected2}) - "
          f"{'✓' if abs(cost2 - expected2) < 0.1 else '✗'}")
    
    # Интерактивный режим
    while True:
        print("\n" + "=" * 60)
        print("ГЛАВНОЕ МЕНЮ:")
        print("1. Решить новую транспортную задачу")
        print("2. Выйти из программы")
        
        choice = input("\nВыберите действие (1 или 2): ").strip()
        
        if choice == '1':
            solve_interactive()
        elif choice == '2':
            print("\nВыход из программы. До свидания!")
            break
        else:
            print("Неверный выбор. Пожалуйста, введите 1 или 2.")


if __name__ == "__main__":
    main()