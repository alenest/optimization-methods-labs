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
        """Метод наименьшей стоимости - правильная реализация"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()
        
        # Пока есть нераспределенные запасы или потребности
        while np.sum(supply) > 1e-10 and np.sum(demand) > 1e-10:
            # Находим минимальную стоимость
            min_cost = np.inf
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                if supply[i] < 1e-10:
                    continue
                for j in range(self.n):
                    if demand[j] < 1e-10:
                        continue
                    if costs[i, j] < min_cost:
                        min_cost = costs[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                break
                
            # Размещаем груз
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            supply[min_i] -= amount
            demand[min_j] -= amount
            
            # Если запас исчерпан, помечаем строку как использованную
            if supply[min_i] < 1e-10:
                for j in range(self.n):
                    costs[min_i, j] = np.inf
            
            # Если потребность удовлетворена, помечаем столбец как использованный
            if demand[min_j] < 1e-10:
                for i in range(self.m):
                    costs[i, min_j] = np.inf
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        u[0] = 0
        
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
        """Поиск цикла для перераспределения"""
        # Создаем списки базисных клеток
        basic_cells = [(i, j) for i in range(self.m) for j in range(self.n) if plan[i, j] > 1e-10]
        # Добавляем новую клетку
        basic_cells.append((start_i, start_j))
        
        # Матрица смежности
        rows = [[] for _ in range(self.m)]
        cols = [[] for _ in range(self.n)]
        
        for i, j in basic_cells:
            rows[i].append(j)
            cols[j].append(i)
        
        # DFS для поиска цикла
        def dfs(current_i, current_j, visited, path):
            if len(path) > 1 and (current_i, current_j) == (start_i, start_j):
                return path
            
            # Пытаемся двигаться по строке
            if len(path) % 2 == 0:  # Четный шаг - двигаемся по строке
                for j in rows[current_i]:
                    if j != current_j and ((current_i, j) not in visited or (current_i, j) == (start_i, start_j)):
                        result = dfs(current_i, j, visited + [(current_i, j)], path + [(current_i, j)])
                        if result:
                            return result
            else:  # Нечетный шаг - двигаемся по столбцу
                for i in cols[current_j]:
                    if i != current_i and ((i, current_j) not in visited or (i, current_j) == (start_i, start_j)):
                        result = dfs(i, current_j, visited + [(i, current_j)], path + [(i, current_j)])
                        if result:
                            return result
            
            return None
        
        return dfs(start_i, start_j, [(start_i, start_j)], [(start_i, start_j)])
    
    def improve_solution(self, plan):
        """Улучшение решения методом потенциалов"""
        improved = False
        u, v = self.get_potentials(plan)
        
        # Находим клетку с максимальной положительной оценкой
        max_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta:
                        max_delta = delta
                        best_i, best_j = i, j
        
        if max_delta < 1e-10:  # Решение оптимально
            return plan, False
        
        # Находим цикл перераспределения
        cycle = self.find_cycle(plan, best_i, best_j)
        
        if not cycle:
            return plan, False
        
        # Находим минимальное значение в отрицательных вершинах
        min_amount = np.inf
        for idx, (i, j) in enumerate(cycle[1:]):
            if idx % 2 == 1:  # Отрицательные вершины
                if plan[i, j] < min_amount:
                    min_amount = plan[i, j]
        
        if min_amount < 1e-10:
            return plan, False
        
        # Перераспределяем
        for idx, (i, j) in enumerate(cycle):
            if idx == 0:  # Начальная клетка (положительная)
                plan[i, j] += min_amount
            elif idx % 2 == 0:  # Положительные вершины
                plan[i, j] += min_amount
            else:  # Отрицательные вершины
                plan[i, j] -= min_amount
        
        # Очищаем нули
        plan[plan < 1e-10] = 0
        
        return plan, True
    
    def solve(self):
        """Основной метод решения"""
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
        
        # Получаем начальный план
        print(f"\n{'='*60}")
        print("ЭТАП I: Построение начального плана методом наименьшей стоимости")
        
        plan = self.minimal_cost_method()
        cost = self.calculate_cost(plan)
        
        print("\nНачальный план:")
        print(plan)
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
            # Добавляем нулевые поставки в пустые клетки, чтобы сделать план невырожденным
            eps = 1e-6
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] == 0:
                        # Проверяем, можно ли добавить нулевую поставку
                        temp_plan = plan.copy()
                        temp_plan[i, j] = eps
                        # Проверяем, не нарушает ли это ограничения
                        row_sum = np.sum(temp_plan[i, :])
                        col_sum = np.sum(temp_plan[:, j])
                        if row_sum <= self.supply[i] + eps and col_sum <= self.demand[j] + eps:
                            plan[i, j] = eps
                            if np.sum(plan > 1e-10) >= required_cells:
                                break
                if np.sum(plan > 1e-10) >= required_cells:
                    break
        
        # Улучшаем решение
        print(f"\n{'='*60}")
        print("ЭТАП II: Улучшение плана методом потенциалов")
        
        iteration = 0
        improved = True
        while improved and iteration < 100:
            plan, improved = self.improve_solution(plan)
            iteration += 1
            if improved:
                new_cost = self.calculate_cost(plan)
                print(f"Итерация {iteration}: стоимость = {new_cost:.2f}")
        
        optimal_cost = self.calculate_cost(plan)
        print(f"\nОптимальный план найден за {iteration} итераций")
        print("\nОптимальный план:")
        print(np.round(plan, 2))
        print(f"\nМинимальная стоимость: {optimal_cost:.2f}")
        
        # Анализ оптимального плана
        print(f"\n{'='*60}")
        print("АНАЛИЗ ОПТИМАЛЬНОГО ПЛАНА")
        
        total_shipped = 0
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-10:
                    amount = plan[i, j]
                    if amount < 1e-5:  # Пропускаем очень маленькие значения
                        continue
                    print(f"Из поставщика A{i+1} → потребителю B{j+1}: {amount:.1f} ед.")
                    total_shipped += amount
        
        print(f"\nВсего перевезено: {total_shipped:.1f} единиц")
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

problem2 = TransportProblem(costs2, supply2, demand2)
optimal_plan2, optimal_cost2 = problem2.solve()

print("\n" + "="*60)
print("ИТОГИ")
print("=" * 60)
print(f"Вариант 1: минимальная стоимость = {optimal_cost1:.2f}")
print(f"Вариант 2: минимальная стоимость = {optimal_cost2:.2f}")

# Сравнение с онлайн-решением
print("\n" + "="*60)
print("СРАВНЕНИЕ С ОНЛАЙН-РЕШЕНИЕМ")
print("=" * 60)
print("Вариант 1:")
print(f"  Онлайн-решение: 1870.00")
print(f"  Наше решение:   {optimal_cost1:.2f}")
print(f"  Разница:        {optimal_cost1 - 1870:.2f}")

print("\nВариант 2:")
print(f"  Наше решение:   {optimal_cost2:.2f}")