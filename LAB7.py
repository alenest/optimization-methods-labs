import numpy as np

class TransportProblem:
    def __init__(self, costs, supply, demand):
        """Абстрактный решатель транспортной задачи"""
        self.original_costs = np.array(costs, dtype=float)
        self.original_supply = np.array(supply, dtype=float)
        self.original_demand = np.array(demand, dtype=float)
        
        self._make_closed_model()
    
    def _make_closed_model(self):
        """Приведение задачи к закрытой модели"""
        self.costs = self.original_costs.copy()
        self.supply = self.original_supply.copy()
        self.demand = self.original_demand.copy()
        
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                self.demand = np.append(self.demand, total_supply - total_demand)
                self.costs = np.column_stack([self.costs, np.zeros(self.costs.shape[0])])
            else:
                self.supply = np.append(self.supply, total_demand - total_supply)
                self.costs = np.row_stack([self.costs, np.zeros(self.costs.shape[1])])
        
        self.m = len(self.supply)
        self.n = len(self.demand)
    
    def minimal_cost_method(self):
        """Метод минимального элемента"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        while True:
            # Находим минимальную стоимость среди доступных клеток
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                if supply[i] <= 1e-10:
                    continue
                for j in range(self.n):
                    if demand[j] <= 1e-10:
                        continue
                    if self.costs[i, j] < min_cost:
                        min_cost = self.costs[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                break
            
            # Распределяем
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            supply[min_i] -= amount
            demand[min_j] -= amount
        
        # Проверяем на вырожденность и добавляем нулевую базисную клетку если нужно
        basic_cells = np.sum(plan > 1e-10)
        if basic_cells < self.m + self.n - 1:
            # Добавляем нулевую базисную клетку
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] <= 1e-10:
                        plan[i, j] = 0
                        basic_cells += 1
                        if basic_cells >= self.m + self.n - 1:
                            break
                if basic_cells >= self.m + self.n - 1:
                    break
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов u и v"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Начинаем с u[0] = 0, как в онлайн-решателе
        u[0] = 0
        
        changed = True
        while changed:
            changed = False
            
            # Вычисляем v на основе известных u
            for i in range(self.m):
                if not np.isnan(u[i]):
                    for j in range(self.n):
                        if (plan[i, j] > 1e-10 or plan[i, j] == 0) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
            
            # Вычисляем u на основе известных v
            for j in range(self.n):
                if not np.isnan(v[j]):
                    for i in range(self.m):
                        if (plan[i, j] > 1e-10 or plan[i, j] == 0) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_cycle(self, plan, start_i, start_j):
        """Находит цикл для свободной клетки (алгоритм поиска в глубину)"""
        # Создаем список базисных клеток
        basic_cells = []
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-10 or plan[i, j] == 0:
                    basic_cells.append((i, j))
        
        # Добавляем начальную клетку как временно базисную
        basic_cells.append((start_i, start_j))
        
        # Ищем цикл с помощью DFS
        def dfs(current, path, visited_rows, visited_cols):
            i, j = current
            
            if len(path) > 1 and (i, j) == (start_i, start_j):
                return path
            
            # Пытаемся двигаться по строке
            if i not in visited_rows:
                visited_rows.add(i)
                for col in range(self.n):
                    if col != j and (i, col) in basic_cells:
                        new_path = dfs((i, col), path + [(i, col)], visited_rows.copy(), visited_cols.copy())
                        if new_path:
                            return new_path
                visited_rows.remove(i)
            
            # Пытаемся двигаться по столбцу
            if j not in visited_cols:
                visited_cols.add(j)
                for row in range(self.m):
                    if row != i and (row, j) in basic_cells:
                        new_path = dfs((row, j), path + [(row, j)], visited_rows.copy(), visited_cols.copy())
                        if new_path:
                            return new_path
                visited_cols.remove(j)
            
            return None
        
        return dfs((start_i, start_j), [(start_i, start_j)], set(), set())
    
    def improve_plan(self, plan):
        """Улучшение плана методом потенциалов"""
        u, v = self.get_potentials(plan)
        
        # Находим свободную клетку с максимальной положительной оценкой
        max_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] <= 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta + 1e-10:
                        max_delta = delta
                        best_i, best_j = i, j
        
        if max_delta <= 1e-10:
            return plan, False
        
        # Находим цикл
        cycle = self.find_cycle(plan, best_i, best_j)
        
        if not cycle or len(cycle) < 4:
            return plan, False
        
        # Находим минимальное значение в минусовых клетках
        min_amount = float('inf')
        for idx in range(1, len(cycle), 2):
            i, j = cycle[idx]
            if idx == 1:  # Пропускаем первую клетку (она свободная)
                continue
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        # Перераспределяем
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle[:-1]):  # Последняя равна первой
            if idx % 2 == 0:  # Плюсовые клетки
                new_plan[i, j] += min_amount
            else:  # Минусовые клетки
                new_plan[i, j] -= min_amount
        
        # Очищаем нули
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def solve(self, method='minimal_cost'):
        """Решение транспортной задачи"""
        # Построение начального плана
        plan = self.minimal_cost_method()
        initial_cost = self.calculate_cost(plan)
        
        # Улучшение плана
        improved = True
        iterations = 0
        
        while improved and iterations < 100:
            iterations += 1
            plan, improved = self.improve_plan(plan)
        
        final_cost = self.calculate_cost(plan)
        
        return {
            'plan': plan,
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'iterations': iterations
        }


def print_result(variant_num, costs, supply, demand, expected, method='minimal_cost'):
    """Выводит результат решения для одного варианта"""
    print(f"\n{'='*60}")
    print(f"ВАРИАНТ {variant_num}")
    print(f"Ожидаемый результат: {expected}")
    print('='*60)
    
    problem = TransportProblem(costs, supply, demand)
    result = problem.solve(method)
    
    print(f"Метод: {method}")
    print(f"Начальная стоимость: {result['initial_cost']:.2f}")
    print(f"Конечная стоимость:  {result['final_cost']:.2f}")
    print(f"Итераций улучшения:  {result['iterations']}")
    
    if abs(result['final_cost'] - expected) < 0.01:
        print("✓ РЕШЕНИЕ СОВПАДАЕТ С ОНЛАЙН-РЕШАТЕЛЕМ")
    else:
        print(f"✗ ОТЛИЧИЕ: {abs(result['final_cost'] - expected):.2f}")
    
    print("\nПлан распределения:")
    total = 0
    for i in range(result['plan'].shape[0]):
        for j in range(result['plan'].shape[1]):
            if result['plan'][i, j] > 1e-5:
                print(f"  A{i+1} → B{j+1}: {result['plan'][i, j]:.1f}")
                total += result['plan'][i, j]
    
    print(f"Всего перевезено: {total:.1f} единиц")
    
    return result['final_cost']


def main():
    print("ТРАНСПОРТНАЯ ЗАДАЧА")
    print("Абстрактный решатель - ФИНАЛЬНАЯ ВЕРСИЯ")
    print("="*60)
    
    test_cases = [
        {
            'num': 1,
            'costs': [[9, 5, 10, 7], [11, 8, 5, 6], [7, 6, 5, 4], [6, 4, 3, 2]],
            'supply': [70, 80, 90, 110],
            'demand': [150, 40, 110, 50],
            'expected': 1870
        },
        {
            'num': 2,
            'costs': [[5, 3, 4, 6, 4], [3, 4, 10, 5, 7], [4, 6, 9, 3, 4]],
            'supply': [40, 20, 40],
            'demand': [25, 10, 20, 30, 15],
            'expected': 340
        },
        {
            'num': 3,
            'costs': [[5, 4, 3, 2], [2, 3, 5, 6], [3, 2, 4, 3], [4, 1, 2, 4]],
            'supply': [120, 60, 80, 140],
            'demand': [100, 140, 100, 60],
            'expected': 800
        }
    ]
    
    for method in ['minimal_cost']:
        print(f"\n{'='*60}")
        print(f"МЕТОД: {method.upper()}")
        print('='*60)
        
        results = []
        for test in test_cases:
            cost = print_result(
                test['num'], 
                test['costs'], 
                test['supply'], 
                test['demand'], 
                test['expected'],
                method
            )
            results.append((test['num'], cost, test['expected']))
        
        print(f"\n{'='*60}")
        print(f"ИТОГИ ДЛЯ МЕТОДА {method.upper()}:")
        print('-'*60)
        
        success_count = 0
        for num, actual, expected in results:
            if abs(actual - expected) < 0.01:
                print(f"✓ Вариант {num}: {actual:.2f} (ожидалось {expected:.2f})")
                success_count += 1
            else:
                print(f"✗ Вариант {num}: {actual:.2f} (ожидалось {expected:.2f})")
        
        print(f"\nУспешно решено: {success_count} из {len(test_cases)}")
        
        if success_count == len(test_cases):
            print("\n✓ ВСЕ ЗАДАЧИ РЕШЕНЫ ПРАВИЛЬНО!")
        else:
            print("\n✗ ЕСТЬ ОШИБКИ В РЕШЕНИИ")


if __name__ == "__main__":
    main()