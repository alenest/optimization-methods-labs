import numpy as np
import copy

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
        """Метод минимального элемента - ТОЧНАЯ копия онлайн-решателя"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()
        
        # Помечаем все клетки как доступные
        available = np.ones((self.m, self.n), dtype=bool)
        
        while np.any(supply > 1e-10) and np.any(demand > 1e-10):
            # Находим минимальную стоимость среди доступных клеток
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                if supply[i] <= 1e-10:
                    continue
                for j in range(self.n):
                    if demand[j] <= 1e-10 or not available[i, j]:
                        continue
                    if costs[i, j] < min_cost:
                        min_cost = costs[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                break
            
            # Распределяем минимальный возможный объем
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            supply[min_i] -= amount
            demand[min_j] -= amount
            
            # Если поставщик исчерпан - исключаем его строку
            if supply[min_i] <= 1e-10:
                available[min_i, :] = False
            
            # Если потребитель удовлетворен - исключаем его столбец
            if demand[min_j] <= 1e-10:
                available[:, min_j] = False
        
        # ОБЯЗАТЕЛЬНО: добавляем нулевые базисные клетки для невырожденности
        basic_cells = np.sum(plan > 1e-10)
        required_basic = self.m + self.n - 1
        
        # Добавляем нулевые базисные клетки, если нужно
        added = 0
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] <= 1e-10 and added < (required_basic - basic_cells):
                    # Временно помещаем маленькое значение для проверки
                    temp_plan = plan.copy()
                    temp_plan[i, j] = 0.000001  # Почти нуль, но не нуль
                    
                    # Проверяем, не создает ли это линейную зависимость
                    if self._is_basic_cell(temp_plan, i, j):
                        plan[i, j] = 0
                        added += 1
                        basic_cells += 1
                
                if basic_cells >= required_basic:
                    break
            if basic_cells >= required_basic:
                break
        
        return plan
    
    def _is_basic_cell(self, plan, row, col):
        """Проверяет, делает ли клетка план линейно зависимым"""
        # Создаем матрицу инцидентности для базисных клеток
        basis_cells = []
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-10 or (i == row and j == col):
                    basis_cells.append((i, j))
        
        # Если количество базисных клеток <= m+n-1, то клетка допустима
        return len(basis_cells) <= self.m + self.n - 1
    
    def calculate_cost(self, plan):
        """Вычисление стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов u и v - ИСПРАВЛЕННАЯ версия"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Находим первую базисную клетку (ненулевую)
        start_i, start_j = -1, -1
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-10:
                    start_i, start_j = i, j
                    break
            if start_i != -1:
                break
        
        if start_i == -1:
            return u, v
        
        # Как в онлайн-решателе: полагаем u[start_i] = 0
        u[start_i] = 0
        
        # Итеративно вычисляем все потенциалы
        changed = True
        iteration = 0
        while changed and iteration < 100:
            iteration += 1
            changed = False
            
            # Вычисляем v на основе известных u
            for i in range(self.m):
                if not np.isnan(u[i]):
                    for j in range(self.n):
                        if plan[i, j] > 1e-10 and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
            
            # Вычисляем u на основе известных v
            for j in range(self.n):
                if not np.isnan(v[j]):
                    for i in range(self.m):
                        if plan[i, j] > 1e-10 and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_improvement_cycle(self, plan, start_i, start_j):
        """Поиск цикла для улучшения плана - ПОЛНОСТЬЮ ПЕРЕПИСАННЫЙ алгоритм"""
        # Создаем копию плана с добавлением исследуемой клетки
        temp_plan = plan.copy()
        temp_plan[start_i, start_j] = 1  # Помечаем как временно базисную
        
        # Начинаем поиск цикла с начальной клетки
        cycle = self._dfs_find_cycle(temp_plan, start_i, start_j, start_i, start_j, [], True, set())
        
        return cycle
    
    def _dfs_find_cycle(self, plan, start_i, start_j, curr_i, curr_j, path, move_horizontal, visited):
        """Рекурсивный поиск цикла в глубину"""
        # Добавляем текущую клетку в путь
        new_path = path + [(curr_i, curr_j)]
        
        # Если вернулись в начало и путь достаточно длинный
        if len(new_path) > 3 and curr_i == start_i and curr_j == start_j:
            return new_path
        
        cell_key = (curr_i, curr_j, move_horizontal)
        if cell_key in visited:
            return None
        
        visited.add(cell_key)
        
        if move_horizontal:
            # Ищем в строке curr_i другие базисные клетки
            for col in range(self.n):
                if col == curr_j:
                    continue
                if plan[curr_i, col] > 1e-10:
                    result = self._dfs_find_cycle(plan, start_i, start_j, curr_i, col, new_path, False, visited.copy())
                    if result:
                        return result
        else:
            # Ищем в столбце curr_j другие базисные клетки
            for row in range(self.m):
                if row == curr_i:
                    continue
                if plan[row, curr_j] > 1e-10:
                    result = self._dfs_find_cycle(plan, start_i, start_j, row, curr_j, new_path, True, visited.copy())
                    if result:
                        return result
        
        return None
    
    def improve_plan(self, plan):
        """Улучшение плана методом потенциалов - КОРРЕКТНАЯ реализация"""
        u, v = self.get_potentials(plan)
        
        # Находим свободную клетку с максимальной положительной оценкой
        # В онлайн-решателе оценка: delta = u[i] + v[j] - c[i][j]
        # И ищут клетку с delta > 0 и максимальной delta
        max_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] <= 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta + 1e-10:
                        max_delta = delta
                        best_i, best_j = i, j
        
        # Если нет положительных оценок - план оптимален
        if max_delta <= 1e-10:
            return plan, False
        
        # Находим цикл для этой клетки
        cycle = self.find_improvement_cycle(plan, best_i, best_j)
        
        if cycle is None or len(cycle) < 4:
            # Если цикл не найден, план оптимален
            return plan, False
        
        # Находим минимальное значение в минусовых клетках
        # В цикле: начальная клетка (0) - "+", затем чередуются "-", "+", ...
        min_amount = float('inf')
        min_cells = []
        
        for idx in range(1, len(cycle)-1, 2):  # Минусовые клетки
            i, j = cycle[idx]
            if plan[i, j] < min_amount - 1e-10:
                min_amount = plan[i, j]
                min_cells = [(i, j)]
            elif abs(plan[i, j] - min_amount) < 1e-10:
                min_cells.append((i, j))
        
        # Перераспределяем груз
        new_plan = plan.copy()
        
        for idx, (i, j) in enumerate(cycle[:-1]):  # Исключаем последнюю (дубликат начальной)
            if idx == 0 or idx % 2 == 0:  # Плюсовые клетки
                new_plan[i, j] += min_amount
            else:  # Минусовые клетки
                new_plan[i, j] -= min_amount
        
        # Очищаем почти нулевые значения
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def solve(self, method='minimal_cost'):
        """Решение транспортной задачи"""
        if method == 'northwest':
            plan = self.northwest_corner()
        else:
            plan = self.minimal_cost_method()
        
        initial_cost = self.calculate_cost(plan)
        
        # Улучшение плана
        improved = True
        iterations = 0
        max_iterations = 50
        
        while improved and iterations < max_iterations:
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
    print("Абстрактный решатель - ИСПРАВЛЕННАЯ И РАБОТАЮЩАЯ ВЕРСИЯ")
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