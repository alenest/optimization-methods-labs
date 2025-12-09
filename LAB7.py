import numpy as np
from copy import deepcopy

class TransportSolver:
    def __init__(self, costs, supply, demand, verbose=False):
        """Решатель транспортной задачи"""
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.verbose = verbose
        
        # Балансировка задачи
        self._balance_problem()
        
        self.m, self.n = self.costs.shape
        
    def _balance_problem(self):
        """Балансировка транспортной задачи"""
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                # Добавляем фиктивного потребителя
                self.demand = np.append(self.demand, total_supply - total_demand)
                self.costs = np.column_stack([self.costs, np.zeros(self.costs.shape[0])])
            else:
                # Добавляем фиктивного поставщика
                self.supply = np.append(self.supply, total_demand - total_supply)
                self.costs = np.row_stack([self.costs, np.zeros(self.costs.shape[1])])
    
    def minimal_cost_method(self):
        """Метод минимального элемента для построения начального плана"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Создаем список всех клеток с их стоимостями
        cells = []
        for i in range(self.m):
            for j in range(self.n):
                cells.append((self.costs[i, j], i, j))
        
        # Сортируем по стоимости
        cells.sort(key=lambda x: x[0])
        
        for cost, i, j in cells:
            if supply[i] > 1e-10 and demand[j] > 1e-10:
                amount = min(supply[i], demand[j])
                plan[i, j] = amount
                supply[i] -= amount
                demand[j] -= amount
        
        # Добавляем нулевые базисные клетки при необходимости
        self._add_zero_basis_cells(plan)
        
        return plan
    
    def _add_zero_basis_cells(self, plan):
        """Добавление нулевых базисных клеток для невырожденности"""
        basic_cells = np.sum(plan > 1e-10)
        required = self.m + self.n - 1
        
        # Если недостаточно базисных клеток
        while basic_cells < required:
            # Ищем свободную клетку с минимальной стоимостью
            min_cost = float('inf')
            best_i, best_j = -1, -1
            
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] <= 1e-10:  # Свободная клетка
                        if self.costs[i, j] < min_cost:
                            min_cost = self.costs[i, j]
                            best_i, best_j = i, j
            
            # Делаем ее нулевой базисной
            if best_i != -1:
                plan[best_i, best_j] = 0
                basic_cells += 1
    
    def calculate_potentials(self, plan):
        """Вычисление потенциалов методом решения системы уравнений"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Начинаем с u1 = 0
        u[0] = 0
        
        # Итеративно вычисляем потенциалы
        changed = True
        iteration = 0
        while changed and iteration < 100:
            iteration += 1
            changed = False
            
            # Проходим по всем базисным клеткам
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] >= 0:  # Базисная клетка (включая нулевые)
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        # Проверяем, что все потенциалы вычислены
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # Если не все вычислены, устанавливаем недостающие в 0
            u[np.isnan(u)] = 0
            v[np.isnan(v)] = 0
        
        return u, v
    
    def find_cycle(self, plan, start_i, start_j):
        """Нахождение цикла для свободной клетки (улучшенный алгоритм)"""
        # Создаем граф базисных клеток
        rows = [[] for _ in range(self.m)]
        cols = [[] for _ in range(self.n)]
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] >= 0:  # Базисная клетка
                    rows[i].append(j)
                    cols[j].append(i)
        
        # Добавляем стартовую клетку временно
        rows[start_i].append(start_j)
        cols[start_j].append(start_i)
        
        # Поиск цикла методом BFS
        from collections import deque
        
        # Каждая вершина графа: (i, j, direction, path)
        # direction: 'row' или 'col'
        queue = deque()
        queue.append((start_i, start_j, 'row', [(start_i, start_j)]))
        
        while queue:
            i, j, direction, path = queue.popleft()
            
            if direction == 'row':
                # Идем по строке
                for next_j in rows[i]:
                    if next_j != j and (i, next_j) not in path[:-1]:
                        new_path = path + [(i, next_j)]
                        if next_j == start_j and i == start_i and len(new_path) > 3:
                            # Нашли цикл
                            rows[start_i].remove(start_j)
                            cols[start_j].remove(start_i)
                            return new_path
                        queue.append((i, next_j, 'col', new_path))
            else:
                # Идем по столбцу
                for next_i in cols[j]:
                    if next_i != i and (next_i, j) not in path[:-1]:
                        new_path = path + [(next_i, j)]
                        if next_i == start_i and j == start_j and len(new_path) > 3:
                            # Нашли цикл
                            rows[start_i].remove(start_j)
                            cols[start_j].remove(start_i)
                            return new_path
                        queue.append((next_i, j, 'row', new_path))
        
        # Убираем временную клетку
        if start_j in rows[start_i]:
            rows[start_i].remove(start_j)
        if start_i in cols[start_j]:
            cols[start_j].remove(start_i)
        
        return None
    
    def improve_plan(self, plan):
        """Улучшение плана методом потенциалов (исправленная версия)"""
        u, v = self.calculate_potentials(plan)
        
        # Находим клетку с максимальной положительной оценкой
        max_delta = -1e10
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] <= 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta + 1e-10:
                        max_delta = delta
                        best_i, best_j = i, j
        
        # Если нет клеток с положительной оценкой, план оптимален
        if max_delta <= 1e-10:
            return plan, False
        
        # Находим цикл
        cycle = self.find_cycle(plan, best_i, best_j)
        
        if not cycle:
            return plan, False
        
        # Находим минимальное значение в минусовых клетках
        min_amount = float('inf')
        for idx in range(1, len(cycle), 2):  # Минусовые клетки
            i, j = cycle[idx]
            if i == best_i and j == best_j:
                continue
            if plan[i, j] < min_amount - 1e-10:
                min_amount = plan[i, j]
        
        # Перераспределяем
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle[:-1]):
            if idx % 2 == 0:  # Плюсовые клетки
                new_plan[i, j] += min_amount
            else:  # Минусовые клетки
                new_plan[i, j] -= min_amount
        
        # Очищаем отрицательные значения
        new_plan[new_plan < 0] = 0
        
        # Добавляем нулевые базисные клетки при необходимости
        self._add_zero_basis_cells(new_plan)
        
        return new_plan, True
    
    def solve(self):
        """Решение транспортной задачи"""
        # Построение начального плана
        plan = self.minimal_cost_method()
        
        if self.verbose:
            print(f"Начальный план построен, стоимость: {self.calculate_cost(plan):.1f}")
        
        # Улучшение плана
        improved = True
        iteration = 0
        max_iterations = 50
        
        while improved and iteration < max_iterations:
            iteration += 1
            plan, improved = self.improve_plan(plan)
            
            if self.verbose and improved:
                print(f"Итерация {iteration}: стоимость = {self.calculate_cost(plan):.1f}")
        
        if self.verbose:
            print(f"Оптимальный план найден за {iteration} итераций")
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление стоимости плана"""
        # Игнорируем фиктивные строки/столбцы при расчете стоимости
        original_m = len(self.original_costs) if hasattr(self, 'original_costs') else self.m
        original_n = len(self.original_costs[0]) if hasattr(self, 'original_costs') else self.n
        
        total = 0
        for i in range(original_m):
            for j in range(original_n):
                total += plan[i, j] * self.costs[i, j]
        
        return total
    
    def print_solution(self, plan):
        """Вывод решения"""
        print("\nОПТИМАЛЬНОЕ РЕШЕНИЕ:")
        total_cost = 0
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-5:
                    cost = plan[i, j] * self.costs[i, j]
                    total_cost += cost
                    print(f"  Из A{i+1} в B{j+1}: {plan[i, j]:.1f} × {self.costs[i, j]:.1f} = {cost:.1f}")
        
        print(f"\nОбщая стоимость: {total_cost:.1f}")


def solve_problem(costs, supply, demand, verbose=False):
    """Функция для решения конкретной транспортной задачи"""
    solver = TransportSolver(costs, supply, demand, verbose)
    plan = solver.solve()
    cost = solver.calculate_cost(plan)
    
    if verbose:
        solver.print_solution(plan)
    
    return plan, cost


# Тестирование на предоставленных вариантах
if __name__ == "__main__":
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ РЕШАТЕЛЯ ТРАНСПОРТНОЙ ЗАДАЧИ")
    print("=" * 80)
    
    # Вариант 1
    print("\nВАРИАНТ 1 (ожидается 1870):")
    costs1 = [[9, 5, 10, 7], [11, 8, 5, 6], [7, 6, 5, 4], [6, 4, 3, 2]]
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    
    plan1, cost1 = solve_problem(costs1, supply1, demand1, verbose=True)
    print(f"Полученная стоимость: {cost1:.1f}")
    print(f"Соответствие ожидаемому: {'✓' if abs(cost1 - 1870) < 0.1 else '✗'}")
    
    # Вариант 2
    print("\n" + "=" * 80)
    print("ВАРИАНТ 2 (ожидается 340):")
    costs2 = [[5, 3, 4, 6, 4], [3, 4, 10, 5, 7], [4, 6, 9, 3, 4]]
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    
    plan2, cost2 = solve_problem(costs2, supply2, demand2, verbose=False)
    print(f"Полученная стоимость: {cost2:.1f}")
    print(f"Соответствие ожидаемому: {'✓' if abs(cost2 - 340) < 0.1 else '✗'}")
    
    # Вариант 3
    print("\n" + "=" * 80)
    print("ВАРИАНТ 3 (ожидается 800):")
    costs3 = [[5, 4, 3, 2], [2, 3, 5, 6], [3, 2, 4, 3], [4, 1, 2, 4]]
    supply3 = [120, 60, 80, 140]
    demand3 = [100, 140, 100, 60]
    
    plan3, cost3 = solve_problem(costs3, supply3, demand3, verbose=True)
    print(f"Полученная стоимость: {cost3:.1f}")
    print(f"Соответствие ожидаемому: {'✓' if abs(cost3 - 800) < 0.1 else '✗'}")
    
    # Итоги
    print("\n" + "=" * 80)
    print("ИТОГИ ТЕСТИРОВАНИЯ:")
    print("-" * 80)
    print(f"Вариант 1: {cost1:.1f} (ожидалось 1870.0) - {'ПРОЙДЕН' if abs(cost1 - 1870) < 0.1 else 'НЕ ПРОЙДЕН'}")
    print(f"Вариант 2: {cost2:.1f} (ожидалось 340.0) - {'ПРОЙДЕН' if abs(cost2 - 340) < 0.1 else 'НЕ ПРОЙДЕН'}")
    print(f"Вариант 3: {cost3:.1f} (ожидалось 800.0) - {'ПРОЙДЕН' if abs(cost3 - 800) < 0.1 else 'НЕ ПРОЙДЕН'}")