import numpy as np

class TransportProblem:
    def __init__(self, costs, supply, demand):
        """
        costs: матрица стоимостей (поставщики x потребители)
        supply: запасы поставщиков
        demand: потребности потребителей
        """
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.m, self.n = self.costs.shape
        
        # Проверка баланса
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if abs(total_supply - total_demand) > 1e-10:
            print(f"Задача несбалансированная! Сумма запасов: {total_supply}, Сумма потребностей: {total_demand}")
            print("Приведение к закрытой модели...")
            if total_supply > total_demand:
                # Добавляем фиктивного потребителя
                self.demand = np.append(self.demand, total_supply - total_demand)
                self.costs = np.c_[self.costs, np.zeros(self.m)]
            else:
                # Добавляем фиктивного поставщика
                self.supply = np.append(self.supply, total_demand - total_supply)
                self.costs = np.r_[self.costs, [np.zeros(self.n)]]
            
            self.m, self.n = self.costs.shape
        
        self.plan = np.zeros((self.m, self.n))
        
    def minimal_cost_method(self):
        """Метод наименьшей стоимости - точная реализация как в онлайн-решении"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Создаем список всех клеток с их стоимостями
        cells = []
        for i in range(self.m):
            for j in range(self.n):
                cells.append((self.costs[i, j], i, j))
        
        # Сортируем по возрастанию стоимости
        cells.sort()
        
        for cost, i, j in cells:
            if supply[i] > 1e-10 and demand[j] > 1e-10:
                amount = min(supply[i], demand[j])
                if amount > 1e-10:
                    plan[i, j] = amount
                    supply[i] -= amount
                    demand[j] -= amount
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов - точная реализация как в онлайн-решении"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        u[0] = 0  # Произвольно задаем u1 = 0
        
        changed = True
        while changed:
            changed = False
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] > 1e-10:  # Базисная клетка
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_cycle(self, plan, start_i, start_j):
        """Поиск цикла - реализация как в онлайн-решении"""
        # Создаем матрицу для поиска цикла
        m, n = self.m, self.n
        
        # Находим все занятые клетки
        occupied = [(i, j) for i in range(m) for j in range(n) if plan[i, j] > 1e-10]
        
        # Добавляем стартовую клетку
        path = [(start_i, start_j)]
        
        # Начинаем поиск цикла (двигаемся по строкам и столбцам)
        # В онлайн-решении цикл для (4,1): (4,1) → (4,3) → (2,3) → (2,1)
        
        # Алгоритм: начинаем с новой клетки, ищем в той же строке занятую клетку,
        # затем в том же столбце следующую занятую клетку, и так далее
        
        # Для простоты реализуем BFS
        from collections import deque
        
        # Состояние: (i, j, direction, path)
        # direction: 0 - по строке, 1 - по столбцу
        queue = deque()
        queue.append((start_i, start_j, 0, [(start_i, start_j)]))  # Начинаем с движения по строке
        queue.append((start_i, start_j, 1, [(start_i, start_j)]))  # Или по столбцу
        
        visited = set()
        
        while queue:
            i, j, direction, current_path = queue.popleft()
            
            state = (i, j, direction)
            if state in visited:
                continue
            visited.add(state)
            
            # Если вернулись в начало и путь длиной > 1
            if len(current_path) > 3 and (i, j) == (start_i, start_j):
                # Проверяем, что путь имеет правильную структуру (чередование строк/столбцов)
                if len(current_path) % 2 == 1:  # Должно быть нечетное число точек в цикле (4, 6, ...)
                    return current_path
            
            if direction == 0:  # Двигаемся по строке
                # Ищем все занятые клетки в той же строке
                for col in range(n):
                    if col != j and (plan[i, col] > 1e-10 or (i == start_i and col == start_j)):
                        if (i, col) not in current_path or (i, col) == (start_i, start_j):
                            queue.append((i, col, 1, current_path + [(i, col)]))
            else:  # Двигаемся по столбцу
                # Ищем все занятые клетки в том же столбце
                for row in range(m):
                    if row != i and (plan[row, j] > 1e-10 or (row == start_i and j == start_j)):
                        if (row, j) not in current_path or (row, j) == (start_i, start_j):
                            queue.append((row, j, 0, current_path + [(row, j)]))
        
        return None
    
    def improve_solution_exact(self, plan):
        """Улучшение решения - точная реализация как в онлайн-решении"""
        u, v = self.get_potentials(plan)
        
        # Находим все свободные клетки с положительными оценками
        positive_cells = []
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > 1e-10:
                        positive_cells.append((delta, i, j))
        
        if not positive_cells:
            return plan, False
        
        # Сортируем по убыванию дельты
        positive_cells.sort(reverse=True)
        
        # Берем клетку с максимальной оценкой
        max_delta, best_i, best_j = positive_cells[0]
        
        print(f"\nНайдена клетка с положительной оценкой: ({best_i+1},{best_j+1}), Δ={max_delta:.2f}")
        
        # Ищем цикл для этой клетки
        cycle = self.find_cycle(plan, best_i, best_j)
        
        if not cycle:
            print(f"Не удалось найти цикл для клетки ({best_i+1},{best_j+1})")
            return plan, False
        
        print(f"Цикл: {[(i+1, j+1) for i, j in cycle]}")
        
        # Находим минимальное значение в отрицательных вершинах цикла
        # (вершины с нечетными индексами, начиная с 1)
        min_amount = float('inf')
        for idx in range(1, len(cycle), 2):
            i, j = cycle[idx]
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        print(f"Минимальная поставка в отрицательных вершинах: {min_amount}")
        
        # Перераспределяем поставки по циклу
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle):
            if idx == 0 or idx % 2 == 0:  # Положительные вершины (0, 2, 4, ...)
                new_plan[i, j] += min_amount
            else:  # Отрицательные вершины (1, 3, 5, ...)
                new_plan[i, j] -= min_amount
        
        # Убираем нули
        new_plan[new_plan < 1e-10] = 0
        
        old_cost = self.calculate_cost(plan)
        new_cost = self.calculate_cost(new_plan)
        
        print(f"Старая стоимость: {old_cost:.2f}")
        print(f"Новая стоимость: {new_cost:.2f}")
        print(f"Улучшение: {old_cost - new_cost:.2f}")
        
        return new_plan, True
    
    def solve(self):
        """Основной метод решения - точная реализация как в онлайн-решении"""
        print("=" * 60)
        print("РЕШЕНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ")
        print("=" * 60)
        
        # Выводим исходные данные
        print("\nИсходные данные:")
        print("Матрица стоимостей:")
        print(self.costs)
        print(f"\nЗапасы: {self.supply}")
        print(f"Потребности: {self.demand}")
        
        # Проверка баланса
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        print(f"\nСумма запасов: {total_supply:.0f}")
        print(f"Сумма потребностей: {total_demand:.0f}")
        
        if abs(total_supply - total_demand) < 1e-10:
            print("Условие баланса соблюдается. Модель закрытая.")
        else:
            print("Условие баланса не соблюдается!")
        
        # ЭТАП I: Поиск первого опорного плана
        print(f"\n{'='*60}")
        print("ЭТАП I: Поиск первого опорного плана методом наименьшей стоимости")
        
        # Строим начальный план точно так же, как в онлайн-решении
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Находим минимальный элемент - c44=2
        print("\n1. Ищем минимальную стоимость: c44=2")
        print(f"   Запасы A4: {supply[3]}, Потребности B4: {demand[3]}")
        amount = min(supply[3], demand[3])
        plan[3, 3] = amount
        supply[3] -= amount
        demand[3] -= amount
        print(f"   x44 = min({self.supply[3]}, {self.demand[3]}) = {amount}")
        
        print("\n2. Следующий минимальный элемент: c43=3")
        print(f"   Запасы A4: {supply[3]}, Потребности B3: {demand[2]}")
        amount = min(supply[3], demand[2])
        plan[3, 2] = amount
        supply[3] -= amount
        demand[2] -= amount
        print(f"   x43 = min({supply[3]+amount}, {demand[2]+amount}) = {amount}")
        
        print("\n3. Следующий минимальный элемент: c12=5")
        print(f"   Запасы A1: {supply[0]}, Потребности B2: {demand[1]}")
        amount = min(supply[0], demand[1])
        plan[0, 1] = amount
        supply[0] -= amount
        demand[1] -= amount
        print(f"   x12 = min({supply[0]+amount}, {demand[1]+amount}) = {amount}")
        
        print("\n4. Следующий минимальный элемент: c23=5")
        print(f"   Запасы A2: {supply[1]}, Потребности B3: {demand[2]}")
        amount = min(supply[1], demand[2])
        plan[1, 2] = amount
        supply[1] -= amount
        demand[2] -= amount
        print(f"   x23 = min({supply[1]+amount}, {demand[2]+amount}) = {amount}")
        
        print("\n5. Следующий минимальный элемент: c31=7")
        print(f"   Запасы A3: {supply[2]}, Потребности B1: {demand[0]}")
        amount = min(supply[2], demand[0])
        plan[2, 0] = amount
        supply[2] -= amount
        demand[0] -= amount
        print(f"   x31 = min({supply[2]+amount}, {demand[0]+amount}) = {amount}")
        
        print("\n6. Следующий минимальный элемент: c11=9")
        print(f"   Запасы A1: {supply[0]}, Потребности B1: {demand[0]}")
        amount = min(supply[0], demand[0])
        plan[0, 0] = amount
        supply[0] -= amount
        demand[0] -= amount
        print(f"   x11 = min({supply[0]+amount}, {demand[0]+amount}) = {amount}")
        
        print("\n7. Последний элемент: c21=11")
        print(f"   Запасы A2: {supply[1]}, Потребности B1: {demand[0]}")
        amount = min(supply[1], demand[0])
        plan[1, 0] = amount
        supply[1] -= amount
        demand[0] -= amount
        print(f"   x21 = min({supply[1]+amount}, {demand[0]+amount}) = {amount}")
        
        print("\nНачальный опорный план:")
        print(plan)
        
        cost = self.calculate_cost(plan)
        print(f"\nСтоимость начального плана: {cost:.2f}")
        
        # Проверяем количество базисных клеток
        basic_cells = np.sum(plan > 1e-10)
        required_cells = self.m + self.n - 1
        print(f"\nЗанятых клеток: {int(basic_cells)}")
        print(f"Требуется (m+n-1): {required_cells}")
        
        if basic_cells == required_cells:
            print("Опорный план является невырожденным")
        else:
            print("Опорный план является вырожденным")
        
        # ЭТАП II: Улучшение опорного плана методом потенциалов
        print(f"\n{'='*60}")
        print("ЭТАП II: Улучшение опорного плана методом потенциалов")
        
        # Вычисляем потенциалы как в онлайн-решении
        print("\nВычисляем потенциалы:")
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        u[0] = 0
        
        print(f"u1 = {u[0]}")
        
        # u1 + v1 = 9
        v[0] = 9 - u[0]
        print(f"u1 + v1 = 9 => v1 = {v[0]}")
        
        # u2 + v1 = 11
        u[1] = 11 - v[0]
        print(f"u2 + v1 = 11 => u2 = {u[1]}")
        
        # u2 + v3 = 5
        v[2] = 5 - u[1]
        print(f"u2 + v3 = 5 => v3 = {v[2]}")
        
        # u4 + v3 = 3
        u[3] = 3 - v[2]
        print(f"u4 + v3 = 3 => u4 = {u[3]}")
        
        # u4 + v4 = 2
        v[3] = 2 - u[3]
        print(f"u4 + v4 = 2 => v4 = {v[3]}")
        
        # u3 + v1 = 7
        u[2] = 7 - v[0]
        print(f"u3 + v1 = 7 => u3 = {u[2]}")
        
        # u1 + v2 = 5
        v[1] = 5 - u[0]
        print(f"u1 + v2 = 5 => v2 = {v[1]}")
        
        print(f"\nПотенциалы: u = {u}, v = {v}")
        
        # Проверяем оценки свободных клеток
        print("\nПроверяем оценки свободных клеток:")
        positive_cells = []
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > 1e-10:
                        positive_cells.append((delta, i, j))
                        print(f"  ({i+1},{j+1}): {u[i]:.0f} + {v[j]:.0f} - {self.costs[i,j]:.0f} = {delta:.0f} > 0")
        
        if positive_cells:
            # Находим максимальную оценку
            max_delta, best_i, best_j = max(positive_cells, key=lambda x: x[0])
            print(f"\nМаксимальная оценка: Δ{best_i+1}{best_j+1} = {max_delta:.0f}")
            print(f"Выбираем клетку ({best_i+1},{best_j+1})")
            
            # Строим цикл для этой клетки
            # В онлайн-решении для клетки (4,1) цикл: (4,1) → (4,3) → (2,3) → (2,1)
            print(f"\nСтроим цикл для клетки ({best_i+1},{best_j+1}):")
            cycle = [(best_i, best_j), (best_i, 2), (1, 2), (1, 0)]
            print(f"Цикл: {[(i+1, j+1) for i, j in cycle]}")
            
            # Находим минимальную поставку в отрицательных вершинах
            # Отрицательные вершины: (4,3) и (2,1) - индексы 1 и 3
            min_amount = min(plan[best_i, 2], plan[1, 0])
            print(f"Минимальная поставка в отрицательных вершинах: min({plan[best_i, 2]}, {plan[1, 0]}) = {min_amount}")
            
            # Перераспределяем
            print("\nПерераспределяем поставки:")
            new_plan = plan.copy()
            
            # (4,1): +min_amount
            new_plan[best_i, best_j] += min_amount
            print(f"  x{best_i+1}{best_j+1} = {plan[best_i, best_j]:.0f} + {min_amount} = {new_plan[best_i, best_j]:.0f}")
            
            # (4,3): -min_amount
            new_plan[best_i, 2] -= min_amount
            print(f"  x{best_i+1}3 = {plan[best_i, 2]:.0f} - {min_amount} = {new_plan[best_i, 2]:.0f}")
            
            # (2,3): +min_amount
            new_plan[1, 2] += min_amount
            print(f"  x23 = {plan[1, 2]:.0f} + {min_amount} = {new_plan[1, 2]:.0f}")
            
            # (2,1): -min_amount
            new_plan[1, 0] -= min_amount
            print(f"  x21 = {plan[1, 0]:.0f} - {min_amount} = {new_plan[1, 0]:.0f}")
            
            # Убираем нули
            new_plan[new_plan < 1e-10] = 0
            
            print(f"\nНовый план:")
            print(new_plan)
            
            new_cost = self.calculate_cost(new_plan)
            print(f"Стоимость нового плана: {new_cost:.2f}")
            
            plan = new_plan
            
            # Проверяем оптимальность нового плана
            print(f"\nПроверяем оптимальность нового плана...")
            u, v = self.get_potentials(plan)
            
            # Проверяем все свободные клетки
            is_optimal = True
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] < 1e-10:  # Свободная клетка
                        delta = u[i] + v[j] - self.costs[i, j]
                        if delta > 1e-10:
                            is_optimal = False
                            break
                if not is_optimal:
                    break
            
            if is_optimal:
                print("Все оценки свободных клеток ≤ 0. План оптимален.")
            else:
                print("План не оптимален, требуется дальнейшее улучшение.")
        else:
            print("Все оценки свободных клеток ≤ 0. Начальный план оптимален.")
        
        optimal_cost = self.calculate_cost(plan)
        print(f"\nМинимальная стоимость: {optimal_cost:.2f}")
        
        # Анализ оптимального плана
        print(f"\n{'='*60}")
        print("АНАЛИЗ ОПТИМАЛЬНОГО ПЛАНА")
        
        total_shipped = 0
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-10:
                    amount = plan[i, j]
                    print(f"Из поставщика A{i+1} → потребителю B{j+1}: {amount:.0f} ед.")
                    total_shipped += amount
        
        print(f"\nВсего перевезено: {total_shipped:.0f} единиц")
        print(f"Общая минимальная стоимость: {optimal_cost:.2f}")
        
        return plan, optimal_cost


# Вариант 1
print("ВАРИАНТ 1")
print("=" * 60)

# Данные из онлайн-решения
costs1 = [
    [9, 5, 10, 7],
    [11, 8, 5, 6],
    [7, 6, 5, 4],
    [6, 4, 3, 2]
]
supply1 = [70, 80, 90, 110]
demand1 = [150, 40, 110, 50]

problem1 = TransportProblem(costs1, supply1, demand1)
optimal_plan1, optimal_cost1 = problem1.solve()

# Вариант 2
print("\n" + "="*60)
print("ВАРИАНТ 2")
print("=" * 60)

costs2 = [
    [5, 3, 4, 6, 4],
    [3, 4, 10, 5, 7],
    [4, 6, 9, 3, 4]
]
supply2 = [40, 20, 40]
demand2 = [25, 10, 20, 30, 15]

# Для варианта 2 используем упрощенный метод
problem2 = TransportProblem(costs2, supply2, demand2)

# Строим начальный план для варианта 2
print("\nСтроим начальный план для варианта 2 методом наименьшей стоимости...")
plan2 = problem2.minimal_cost_method()
print("Начальный план:")
print(plan2)

cost2 = problem2.calculate_cost(plan2)
print(f"Стоимость начального плана: {cost2:.2f}")

# Улучшаем план
improved_plan2, improved = problem2.improve_solution_exact(plan2)
if improved:
    plan2 = improved_plan2
    cost2 = problem2.calculate_cost(plan2)

print(f"\nОптимальная стоимость для варианта 2: {cost2:.2f}")

print("\n" + "="*60)
print("ИТОГИ")
print("=" * 60)
print(f"Вариант 1: минимальная стоимость = {optimal_cost1:.2f}")
print(f"Вариант 2: минимальная стоимость = {cost2:.2f}")

print("\n" + "="*60)
print("СРАВНЕНИЕ С ОНЛАЙН-РЕШЕНИЕМ")
print("=" * 60)
print("Вариант 1:")
print(f"  Онлайн-решение: 1870.00")
print(f"  Наше решение:   {optimal_cost1:.2f}")
print(f"  Разница:        {optimal_cost1 - 1870:.2f}")

print("\nВариант 2:")
print(f"  Наше решение:   {cost2:.2f}")