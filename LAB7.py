import numpy as np

class TransportSolver:
    def __init__(self, costs, supply, demand):
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        
        # Балансировка
        self._balance()
        
        self.m, self.n = self.costs.shape
        
    def _balance(self):
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                self.demand = np.append(self.demand, total_supply - total_demand)
                self.costs = np.column_stack([self.costs, np.zeros(self.costs.shape[0])])
            else:
                self.supply = np.append(self.supply, total_demand - total_supply)
                self.costs = np.row_stack([self.costs, np.zeros(self.costs.shape[1])])
    
    def north_west_corner(self):
        """Метод северо-западного угла"""
        plan = np.zeros((self.m, self.n))
        i, j = 0, 0
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        while i < self.m and j < self.n:
            amount = min(supply[i], demand[j])
            plan[i, j] = amount
            supply[i] -= amount
            demand[j] -= amount
            
            if supply[i] == 0:
                i += 1
            if demand[j] == 0:
                j += 1
        
        return plan
    
    def minimal_cost(self):
        """Метод минимального элемента"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        while np.sum(supply) > 0 and np.sum(demand) > 0:
            # Находим минимальную стоимость
            mask = np.outer(supply > 0, demand > 0)
            masked_costs = np.where(mask, self.costs, np.inf)
            i, j = np.unravel_index(np.argmin(masked_costs), masked_costs.shape)
            
            amount = min(supply[i], demand[j])
            plan[i, j] = amount
            supply[i] -= amount
            demand[j] -= amount
        
        return plan
    
    def calculate_potentials(self, plan):
        """Вычисление потенциалов"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Начинаем с первой базисной клетки
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 0:
                    u[i] = 0
                    v[j] = self.costs[i, j] - u[i]
                    break
            if not np.isnan(u[i]):
                break
        
        # Вычисляем остальные потенциалы
        changed = True
        while changed:
            changed = False
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] > 0:
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_cycle(self, plan, i0, j0):
        """Поиск цикла для свободной клетки"""
        # Создаем список базисных клеток
        basic_cells = []
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 0:
                    basic_cells.append((i, j))
        
        # Временный список с добавленной клеткой
        temp_cells = basic_cells + [(i0, j0)]
        
        # Поиск цикла методом BFS
        from collections import deque
        
        # Начинаем с (i0, j0)
        queue = deque()
        queue.append((i0, j0, [(i0, j0)], True))  # True - идем по строке
        
        visited = set()
        
        while queue:
            i, j, path, row_move = queue.popleft()
            
            # Если вернулись в начало и длина пути > 1
            if len(path) > 1 and (i, j) == (i0, j0):
                return path
            
            key = (i, j, row_move)
            if key in visited:
                continue
            visited.add(key)
            
            if row_move:
                # Ищем клетки в той же строке
                for next_i, next_j in temp_cells:
                    if next_i == i and next_j != j:
                        queue.append((next_i, next_j, path + [(next_i, next_j)], False))
            else:
                # Ищем клетки в том же столбце
                for next_i, next_j in temp_cells:
                    if next_j == j and next_i != i:
                        queue.append((next_i, next_j, path + [(next_i, next_j)], True))
        
        return None
    
    def improve(self, plan):
        """Улучшение плана методом потенциалов"""
        u, v = self.calculate_potentials(plan)
        
        # Ищем клетку с максимальной положительной оценкой
        max_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] == 0:
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta + 1e-10:
                        max_delta = delta
                        best_i, best_j = i, j
        
        # Если нет положительных оценок - план оптимален
        if max_delta <= 1e-10:
            return plan, False
        
        # Находим цикл
        cycle = self.find_cycle(plan, best_i, best_j)
        if cycle is None:
            return plan, False
        
        # Находим минимальное значение в минусовых клетках
        min_amount = float('inf')
        for idx in range(1, len(cycle), 2):
            i, j = cycle[idx]
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        # Перераспределяем
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle[:-1]):
            if idx % 2 == 0:
                new_plan[i, j] += min_amount
            else:
                new_plan[i, j] -= min_amount
        
        # Очищаем нули
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def solve(self, method='minimal_cost'):
        """Решение транспортной задачи"""
        if method == 'north_west':
            plan = self.north_west_corner()
        else:
            plan = self.minimal_cost()
        
        # Добавляем нулевые базисные клетки при необходимости
        basic_cells = np.sum(plan > 0)
        required = self.m + self.n - 1
        
        if basic_cells < required:
            # Ищем клетки с минимальной стоимостью для добавления
            free_cells = []
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] == 0:
                        free_cells.append((self.costs[i, j], i, j))
            
            free_cells.sort()
            for _, i, j in free_cells:
                if basic_cells < required:
                    plan[i, j] = 0
                    basic_cells += 1
        
        # Улучшение плана
        improved = True
        iterations = 0
        
        while improved and iterations < 100:
            iterations += 1
            plan, improved = self.improve(plan)
        
        return plan
    
    def calculate_total_cost(self, plan):
        """Вычисление общей стоимости"""
        return np.sum(plan * self.costs)


def test_variant(num, costs, supply, demand, expected):
    """Тестирование одного варианта"""
    print(f"\nВариант {num}:")
    print(f"Ожидаемая стоимость: {expected}")
    
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
    
    if abs(cost - expected) < 0.1:
        print("✓ Решение верное")
    else:
        print("✗ Решение неверное")
    
    return cost


def main():
    print("ТРАНСПОРТНАЯ ЗАДАЧА")
    print("=" * 60)
    
    # Вариант 1
    costs1 = [[9, 5, 10, 7], [11, 8, 5, 6], [7, 6, 5, 4], [6, 4, 3, 2]]
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    
    cost1 = test_variant(1, costs1, supply1, demand1, 1870)
    
    # Вариант 2
    costs2 = [[5, 3, 4, 6, 4], [3, 4, 10, 5, 7], [4, 6, 9, 3, 4]]
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    
    cost2 = test_variant(2, costs2, supply2, demand2, 340)
    
    # Вариант 3
    costs3 = [[5, 4, 3, 2], [2, 3, 5, 6], [3, 2, 4, 3], [4, 1, 2, 4]]
    supply3 = [120, 60, 80, 140]
    demand3 = [100, 140, 100, 60]
    
    cost3 = test_variant(3, costs3, supply3, demand3, 800)
    
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print(f"Вариант 1: {cost1:.1f} (ожидалось 1870.0) - {'✓' if abs(cost1-1870)<0.1 else '✗'}")
    print(f"Вариант 2: {cost2:.1f} (ожидалось 340.0) - {'✓' if abs(cost2-340)<0.1 else '✗'}")
    print(f"Вариант 3: {cost3:.1f} (ожидалось 800.0) - {'✓' if abs(cost3-800)<0.1 else '✗'}")


if __name__ == "__main__":
    main()