import numpy as np

class TransportSolver:
    def __init__(self, costs, supply, demand, verbose=False):
        """Абстрактный решатель транспортной задачи"""
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.verbose = verbose
        
        # Приведение задачи к закрытой модели
        self._balance_problem()
        
    def _balance_problem(self):
        """Балансировка транспортной задачи"""
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if total_supply > total_demand:
            # Добавляем фиктивного потребителя
            self.demand = np.append(self.demand, total_supply - total_demand)
            self.costs = np.column_stack([self.costs, np.zeros(self.costs.shape[0])])
        elif total_demand > total_supply:
            # Добавляем фиктивного поставщика
            self.supply = np.append(self.supply, total_demand - total_supply)
            self.costs = np.row_stack([self.costs, np.zeros(self.costs.shape[1])])
        
        self.m = len(self.supply)
        self.n = len(self.demand)
        
        if self.verbose:
            print(f"Сбалансированная задача: {self.m}×{self.n}")
    
    def minimal_cost_method(self):
        """Метод минимального элемента для построения начального плана"""
        plan = np.zeros((self.m, self.n))
        supply_remaining = self.supply.copy()
        demand_remaining = self.demand.copy()
        
        if self.verbose:
            print("\n=== ПОСТРОЕНИЕ НАЧАЛЬНОГО ПЛАНА ===")
        
        while np.any(supply_remaining > 0) and np.any(demand_remaining > 0):
            # Находим клетку с минимальной стоимостью среди доступных
            mask = np.outer(supply_remaining > 0, demand_remaining > 0)
            available_costs = np.where(mask, self.costs, np.inf)
            
            min_idx = np.unravel_index(np.argmin(available_costs), available_costs.shape)
            i, j = min_idx
            
            # Определяем объем перевозки
            amount = min(supply_remaining[i], demand_remaining[j])
            plan[i, j] = amount
            
            # Обновляем остатки
            supply_remaining[i] -= amount
            demand_remaining[j] -= amount
            
            if self.verbose:
                print(f"Клетка ({i+1},{j+1}): стоимость={self.costs[i,j]}, количество={amount}")
        
        # Проверяем вырожденность и добавляем нулевые базисные клетки при необходимости
        self._ensure_non_degenerate(plan)
        
        return plan
    
    def _ensure_non_degenerate(self, plan):
        """Обеспечение невырожденности плана"""
        basic_cells = np.sum(plan > 0)
        required = self.m + self.n - 1
        
        while basic_cells < required:
            # Ищем свободную клетку с минимальной стоимостью
            free_cells = np.where(plan == 0)
            min_cost = np.inf
            best_i, best_j = -1, -1
            
            for i, j in zip(free_cells[0], free_cells[1]):
                if self.costs[i, j] < min_cost:
                    min_cost = self.costs[i, j]
                    best_i, best_j = i, j
            
            # Делаем эту клетку нулевой базисной
            if best_i != -1 and best_j != -1:
                plan[best_i, best_j] = 0
                basic_cells += 1
                
                if self.verbose:
                    print(f"Добавлена нулевая базисная клетка ({best_i+1},{best_j+1})")
    
    def calculate_potentials(self, plan):
        """Вычисление потенциалов методом северо-западного угла"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Начинаем с произвольного потенциала
        u[0] = 0
        
        # Итеративно вычисляем остальные потенциалы
        changed = True
        while changed:
            changed = False
            
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] >= 0:  # Базисная клетка (включая нулевые)
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_improvement_cycle(self, plan, start_i, start_j):
        """Поиск цикла перераспределения для свободной клетки"""
        # Временно добавляем клетку в базис
        temp_plan = plan.copy()
        temp_plan[start_i, start_j] = -1  # Маркер для новой клетки
        
        # Ищем цикл методом BFS
        from collections import deque
        
        # Структура для BFS: (i, j, path, direction)
        # direction: 'row' или 'col'
        queue = deque([(start_i, start_j, [(start_i, start_j)], 'row')])
        visited = set()
        
        while queue:
            i, j, path, direction = queue.popleft()
            
            # Проверяем, вернулись ли в начало (за исключением первой точки)
            if len(path) > 3 and i == start_i and j == start_j:
                return path
            
            key = (i, j, direction)
            if key in visited:
                continue
            visited.add(key)
            
            if direction == 'row':
                # Ищем базисные клетки в текущей строке
                for col in range(self.n):
                    if col != j and temp_plan[i, col] != 0:
                        # Проверяем, что это не исходная клетка (кроме начала)
                        if (i, col) == (start_i, start_j) and len(path) == 1:
                            continue
                        queue.append((i, col, path + [(i, col)], 'col'))
            else:  # direction == 'col'
                # Ищем базисные клетки в текущем столбце
                for row in range(self.m):
                    if row != i and temp_plan[row, j] != 0:
                        if (row, j) == (start_i, start_j) and len(path) == 1:
                            continue
                        queue.append((row, j, path + [(row, j)], 'row'))
        
        return None
    
    def improve_plan(self, plan):
        """Улучшение плана методом потенциалов"""
        u, v = self.calculate_potentials(plan)
        
        # Вычисляем оценки для свободных клеток
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] == 0:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > best_delta + 1e-10:
                        best_delta = delta
                        best_i, best_j = i, j
        
        # Если нет клеток с положительной оценкой - план оптимален
        if best_delta <= 1e-10:
            return plan, False
        
        # Находим цикл для улучшения
        cycle = self.find_improvement_cycle(plan, best_i, best_j)
        
        if not cycle:
            return plan, False
        
        # Находим минимальное значение в минусовых клетках
        min_amount = np.inf
        for idx in range(1, len(cycle), 2):
            i, j = cycle[idx]
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        # Перераспределяем груз
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle[:-1]):
            if idx % 2 == 0:  # Плюсовые клетки
                new_plan[i, j] += min_amount
            else:  # Минусовые клетки
                new_plan[i, j] -= min_amount
        
        # Очищаем отрицательные значения (должны быть только нули)
        new_plan[new_plan < 0] = 0
        
        return new_plan, True
    
    def solve(self):
        """Основной метод решения транспортной задачи"""
        # Шаг 1: Построение начального плана
        plan = self.minimal_cost_method()
        
        if self.verbose:
            print(f"\nНачальный план (стоимость={self.calculate_cost(plan):.1f}):")
            self.print_plan(plan)
        
        # Шаг 2: Итеративное улучшение плана
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            iterations += 1
            plan, improved = self.improve_plan(plan)
            
            if improved and self.verbose:
                print(f"\nИтерация {iterations} (стоимость={self.calculate_cost(plan):.1f}):")
                self.print_plan(plan)
        
        if self.verbose:
            print(f"\nОптимальный план найден за {iterations} итераций")
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def print_plan(self, plan):
        """Вывод плана в читаемом формате"""
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 0:
                    print(f"  A{i+1} → B{j+1}: {plan[i, j]:.1f} (стоимость {self.costs[i, j]})")


def test_solver():
    """Тестирование решателя на различных вариантах"""
    test_cases = [
        {
            'name': 'Вариант 1',
            'costs': [[9, 5, 10, 7], [11, 8, 5, 6], [7, 6, 5, 4], [6, 4, 3, 2]],
            'supply': [70, 80, 90, 110],
            'demand': [150, 40, 110, 50],
            'expected_cost': 1870
        },
        {
            'name': 'Вариант 2', 
            'costs': [[5, 3, 4, 6, 4], [3, 4, 10, 5, 7], [4, 6, 9, 3, 4]],
            'supply': [40, 20, 40],
            'demand': [25, 10, 20, 30, 15],
            'expected_cost': 340
        },
        {
            'name': 'Вариант 3',
            'costs': [[5, 4, 3, 2], [2, 3, 5, 6], [3, 2, 4, 3], [4, 1, 2, 4]],
            'supply': [120, 60, 80, 140],
            'demand': [100, 140, 100, 60],
            'expected_cost': 800
        }
    ]
    
    print("ТЕСТИРОВАНИЕ АБСТРАКТНОГО РЕШАТЕЛЯ ТРАНСПОРТНОЙ ЗАДАЧИ")
    print("=" * 70)
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"Матрица стоимостей: {test['costs']}")
        print(f"Запасы: {test['supply']}")
        print(f"Потребности: {test['demand']}")
        
        solver = TransportSolver(test['costs'], test['supply'], test['demand'], verbose=True)
        plan = solver.solve()
        actual_cost = solver.calculate_cost(plan)
        
        print(f"\nРезультат: {actual_cost:.1f} (ожидалось: {test['expected_cost']:.1f})")
        
        if abs(actual_cost - test['expected_cost']) < 0.1:
            print("✓ Решение верное!")
        else:
            print("✗ Решение неверное")
        
        print("-" * 50)


if __name__ == "__main__":
    test_solver()