import numpy as np

class TransportProblem:
    def __init__(self, costs, supply, demand, verbose=False):
        """Абстрактный решатель транспортной задачи"""
        self.original_costs = np.array(costs, dtype=float)
        self.original_supply = np.array(supply, dtype=float)
        self.original_demand = np.array(demand, dtype=float)
        self.verbose = verbose
        
        self._make_closed_model()
    
    def _make_closed_model(self):
        """Приведение задачи к закрытой модели"""
        self.costs = self.original_costs.copy()
        self.supply = self.original_supply.copy()
        self.demand = self.original_demand.copy()
        
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if self.verbose:
            print(f"Сумма запасов: {total_supply}")
            print(f"Сумма потребностей: {total_demand}")
        
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                if self.verbose:
                    print(f"Добавляем фиктивного потребителя с потребностью {total_supply - total_demand}")
                self.demand = np.append(self.demand, total_supply - total_demand)
                self.costs = np.column_stack([self.costs, np.zeros(self.costs.shape[0])])
            else:
                if self.verbose:
                    print(f"Добавляем фиктивного поставщика с запасом {total_demand - total_supply}")
                self.supply = np.append(self.supply, total_demand - total_supply)
                self.costs = np.row_stack([self.costs, np.zeros(self.costs.shape[1])])
        else:
            if self.verbose:
                print("Модель сбалансирована")
        
        self.m = len(self.supply)
        self.n = len(self.demand)
        
        if self.verbose:
            print(f"Размер задачи: {self.m} поставщиков × {self.n} потребителей")
    
    def minimal_cost_method(self):
        """Метод минимального элемента"""
        if self.verbose:
            print("\n=== ПОСТРОЕНИЕ НАЧАЛЬНОГО ПЛАНА МЕТОДОМ МИНИМАЛЬНОГО ЭЛЕМЕНТА ===")
        
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        step = 1
        while np.sum(supply) > 1e-10 and np.sum(demand) > 1e-10:
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
            
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            supply[min_i] -= amount
            demand[min_j] -= amount
            
            if self.verbose:
                print(f"Шаг {step}: клетка ({min_i+1}, {min_j+1}) стоимостью {min_cost}, количество {amount}")
                print(f"  Остаток у поставщика {min_i+1}: {supply[min_i]:.1f}")
                print(f"  Остаток у потребителя {min_j+1}: {demand[min_j]:.1f}")
            
            step += 1
        
        # Добавляем нулевую базисную клетку, если нужно
        basic_cells = np.sum(plan > 1e-10)
        required = self.m + self.n - 1
        
        if basic_cells < required and self.verbose:
            print(f"\nДобавляем нулевую базисную клетку (базисных клеток: {basic_cells}, требуется: {required})")
        
        while basic_cells < required:
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] <= 1e-10:
                        plan[i, j] = 0
                        basic_cells += 1
                        if self.verbose:
                            print(f"  Добавлена нулевая базисная клетка ({i+1}, {j+1})")
                        if basic_cells >= required:
                            break
                if basic_cells >= required:
                    break
        
        if self.verbose:
            print("\nНачальный план построен:")
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] > 1e-10:
                        print(f"  x{i+1}{j+1} = {plan[i, j]:.1f}")
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов u и v"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Начинаем с u[0] = 0
        u[0] = 0
        
        if self.verbose:
            print("\n=== ВЫЧИСЛЕНИЕ ПОТЕНЦИАЛОВ ===")
            print(f"Полагаем u{1} = 0")
        
        changed = True
        iteration = 0
        while changed and iteration < 100:
            iteration += 1
            changed = False
            
            # Вычисляем v на основе известных u
            for i in range(self.m):
                if not np.isnan(u[i]):
                    for j in range(self.n):
                        if (plan[i, j] > 1e-10 or plan[i, j] == 0) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                            if self.verbose:
                                print(f"  u{i+1} = {u[i]}, c{i+1}{j+1} = {self.costs[i, j]} => v{j+1} = {v[j]}")
            
            # Вычисляем u на основе известных v
            for j in range(self.n):
                if not np.isnan(v[j]):
                    for i in range(self.m):
                        if (plan[i, j] > 1e-10 or plan[i, j] == 0) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
                            if self.verbose:
                                print(f"  v{j+1} = {v[j]}, c{i+1}{j+1} = {self.costs[i, j]} => u{i+1} = {u[i]}")
        
        if self.verbose:
            print("\nПолученные потенциалы:")
            print(f"  u = {u}")
            print(f"  v = {v}")
        
        return u, v
    
    def find_cycle(self, plan, start_i, start_j):
        """Находит цикл для свободной клетки"""
        # Создаем копию плана с добавленной клеткой
        temp_plan = plan.copy()
        temp_plan[start_i, start_j] = 1  # Помечаем как временно базисную
        
        # Ищем цикл с помощью DFS
        def dfs(i, j, path, direction, visited):
            # Если вернулись в начало
            if len(path) > 3 and i == start_i and j == start_j:
                return path
            
            key = (i, j, direction)
            if key in visited:
                return None
            
            visited.add(key)
            
            if direction == 'row':
                # Ищем в строке i другие базисные клетки
                for col in range(self.n):
                    if col != j and (temp_plan[i, col] > 1e-10 or temp_plan[i, col] == 0):
                        result = dfs(i, col, path + [(i, col)], 'col', visited.copy())
                        if result:
                            return result
            else:  # direction == 'col'
                # Ищем в столбце j другие базисные клетки
                for row in range(self.m):
                    if row != i and (temp_plan[row, j] > 1e-10 or temp_plan[row, j] == 0):
                        result = dfs(row, j, path + [(row, j)], 'row', visited.copy())
                        if result:
                            return result
            
            return None
        
        cycle = dfs(start_i, start_j, [(start_i, start_j)], 'row', set())
        
        if self.verbose and cycle:
            print(f"\nНайден цикл для клетки ({start_i+1}, {start_j+1}):")
            for idx, (i, j) in enumerate(cycle):
                sign = '+' if idx % 2 == 0 else '-'
                print(f"  {sign} ({i+1}, {j+1})")
        
        return cycle
    
    def improve_plan(self, plan, iteration):
        """Улучшение плана методом потенциалов"""
        if self.verbose:
            print(f"\n=== ИТЕРАЦИЯ УЛУЧШЕНИЯ {iteration} ===")
        
        u, v = self.get_potentials(plan)
        
        # Вычисляем оценки для свободных клеток
        if self.verbose:
            print("\nВЫЧИСЛЕНИЕ ОЦЕНОК СВОБОДНЫХ КЛЕТОК:")
        
        max_delta = -float('inf')
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] <= 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if self.verbose:
                        sign = '+' if delta > 1e-10 else ''
                        print(f"  Δ({i+1},{j+1}) = u{i+1} + v{j+1} - c{i+1}{j+1} = {u[i]:.1f} + {v[j]:.1f} - {self.costs[i, j]:.1f} = {sign}{delta:.2f}")
                    
                    if delta > max_delta:
                        max_delta = delta
                        best_i, best_j = i, j
        
        if self.verbose:
            print(f"\nМаксимальная положительная оценка: Δ({best_i+1},{best_j+1}) = {max_delta:.2f}")
        
        if max_delta <= 1e-10:
            if self.verbose:
                print("Все оценки неположительные - план оптимален!")
            return plan, False
        
        # Находим цикл для этой клетки
        cycle = self.find_cycle(plan, best_i, best_j)
        
        if not cycle:
            if self.verbose:
                print(f"Не удалось найти цикл для клетки ({best_i+1}, {best_j+1})")
            return plan, False
        
        # Находим минимальное значение в минусовых клетках
        min_amount = float('inf')
        min_cells = []
        
        for idx in range(1, len(cycle), 2):  # Минусовые клетки
            i, j = cycle[idx]
            if i == best_i and j == best_j:
                continue
            if plan[i, j] < min_amount - 1e-10:
                min_amount = plan[i, j]
                min_cells = [(i, j)]
            elif abs(plan[i, j] - min_amount) < 1e-10:
                min_cells.append((i, j))
        
        if self.verbose:
            print(f"Минимальное значение в минусовых клетках: {min_amount:.1f}")
        
        # Перераспределяем груз
        new_plan = plan.copy()
        
        for idx, (i, j) in enumerate(cycle[:-1]):  # Исключаем последнюю (дубликат первой)
            if idx % 2 == 0:  # Плюсовые клетки
                new_plan[i, j] += min_amount
                if self.verbose:
                    print(f"  x{i+1}{j+1}: {plan[i, j]:.1f} → {new_plan[i, j]:.1f} (+{min_amount:.1f})")
            else:  # Минусовые клетки
                new_plan[i, j] -= min_amount
                if self.verbose:
                    print(f"  x{i+1}{j+1}: {plan[i, j]:.1f} → {new_plan[i, j]:.1f} (-{min_amount:.1f})")
        
        # Очищаем почти нулевые значения
        new_plan[new_plan < 1e-10] = 0
        
        if self.verbose:
            new_cost = self.calculate_cost(new_plan)
            print(f"\nНовая стоимость: {new_cost:.1f}")
            print(f"Улучшение: {self.calculate_cost(plan) - new_cost:.1f}")
        
        return new_plan, True
    
    def solve(self, method='minimal_cost'):
        """Решение транспортной задачи"""
        if self.verbose:
            print("\n" + "="*60)
            print("РЕШЕНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ")
            print("="*60)
        
        # Построение начального плана
        plan = self.minimal_cost_method()
        initial_cost = self.calculate_cost(plan)
        
        if self.verbose:
            print(f"\nНачальная стоимость: {initial_cost:.1f}")
        
        # Улучшение плана
        improved = True
        iterations = 0
        
        while improved and iterations < 20:
            iterations += 1
            plan, improved = self.improve_plan(plan, iterations)
            
            if improved and self.verbose:
                cost = self.calculate_cost(plan)
                print(f"После итерации {iterations}: стоимость = {cost:.1f}")
        
        final_cost = self.calculate_cost(plan)
        
        if self.verbose:
            print(f"\nФинальная стоимость: {final_cost:.1f}")
            print(f"Всего итераций улучшения: {iterations}")
        
        return {
            'plan': plan,
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'iterations': iterations
        }


def solve_variant(costs, supply, demand, variant_num, expected, verbose=False):
    """Решает один вариант задачи с подробным выводом"""
    print(f"\n{'='*80}")
    print(f"ВАРИАНТ {variant_num}")
    print('='*80)
    print(f"Матрица стоимостей:")
    for i in range(len(costs)):
        print("  " + " ".join(f"{c:3}" for c in costs[i]))
    print(f"Запасы: {supply}")
    print(f"Потребности: {demand}")
    print(f"Ожидаемый результат: {expected}")
    
    problem = TransportProblem(costs, supply, demand, verbose)
    result = problem.solve()
    
    print(f"\nРезультат:")
    print(f"  Начальная стоимость: {result['initial_cost']:.2f}")
    print(f"  Конечная стоимость:  {result['final_cost']:.2f}")
    print(f"  Итераций улучшения:  {result['iterations']}")
    
    if abs(result['final_cost'] - expected) < 0.01:
        print("  ✓ РЕШЕНИЕ СОВПАДАЕТ С ОНЛАЙН-РЕШАТЕЛЕМ")
    else:
        print(f"  ✗ ОТЛИЧИЕ: {abs(result['final_cost'] - expected):.2f}")
    
    print("\nОптимальный план распределения:")
    total = 0
    for i in range(result['plan'].shape[0]):
        for j in range(result['plan'].shape[1]):
            if result['plan'][i, j] > 1e-5:
                print(f"  A{i+1} → B{j+1}: {result['plan'][i, j]:.1f} (стоимость: {costs[i][j] if i < len(costs) and j < len(costs[0]) else 0})")
                if i < len(costs) and j < len(costs[0]):
                    total += result['plan'][i, j] * costs[i][j]
    
    print(f"\nОбщая стоимость: {total:.1f}")
    
    return result['final_cost']


def main():
    print("ТРАНСПОРТНАЯ ЗАДАЧА")
    print("Абстрактный решатель с подробным выводом")
    
    test_cases = [
        {
            'num': 1,
            'costs': [[9, 5, 10, 7], [11, 8, 5, 6], [7, 6, 5, 4], [6, 4, 3, 2]],
            'supply': [70, 80, 90, 110],
            'demand': [150, 40, 110, 50],
            'expected': 1870,
            'verbose': False  # Меняйте на True для подробного вывода
        },
        {
            'num': 2,
            'costs': [[5, 3, 4, 6, 4], [3, 4, 10, 5, 7], [4, 6, 9, 3, 4]],
            'supply': [40, 20, 40],
            'demand': [25, 10, 20, 30, 15],
            'expected': 340,
            'verbose': False
        },
        {
            'num': 3,
            'costs': [[5, 4, 3, 2], [2, 3, 5, 6], [3, 2, 4, 3], [4, 1, 2, 4]],
            'supply': [120, 60, 80, 140],
            'demand': [100, 140, 100, 60],
            'expected': 800,
            'verbose': True  # Включаем подробный вывод для третьего варианта
        }
    ]
    
    results = []
    
    for test in test_cases:
        cost = solve_variant(
            test['costs'], 
            test['supply'], 
            test['demand'], 
            test['num'], 
            test['expected'],
            verbose=test['verbose']
        )
        results.append((test['num'], cost, test['expected']))
    
    print(f"\n{'='*80}")
    print("ИТОГИ:")
    print('-'*80)
    
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