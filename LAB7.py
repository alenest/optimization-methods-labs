import numpy as np

class TransportProblem:
    def __init__(self, costs, supply, demand):
        """
        costs: матрица стоимостей (поставщики x потребители)
        supply: запасы поставщиков
        demand: потребности потребителей
        """
        self.original_costs = np.array(costs, dtype=float)
        self.original_supply = np.array(supply, dtype=float)
        self.original_demand = np.array(demand, dtype=float)
        
        # Приведение к закрытой модели
        self._balance_problem()
        
    def _balance_problem(self):
        """Приведение задачи к закрытой модели"""
        self.costs = self.original_costs.copy()
        self.supply = self.original_supply.copy()
        self.demand = self.original_demand.copy()
        
        self.m, self.n = self.costs.shape
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                # Добавляем фиктивного потребителя
                self.demand = np.append(self.demand, total_supply - total_demand)
                self.costs = np.c_[self.costs, np.zeros(self.m)]
            else:
                # Добавляем фиктивного поставщика
                self.supply = np.append(self.supply, total_demand - total_supply)
                self.costs = np.r_[self.costs, [np.zeros(self.n)]]
            
            self.m, self.n = self.costs.shape
    
    def minimal_cost_method_exact(self):
        """
        Метод наименьшей стоимости - точная реализация как в онлайн-решателе
        """
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Копируем матрицу стоимостей для поиска минимумов
        temp_costs = self.costs.copy()
        
        # Маски для исключенных строк и столбцов
        excluded_rows = np.zeros(self.m, dtype=bool)
        excluded_cols = np.zeros(self.n, dtype=bool)
        
        while True:
            # Если все строки или столбцы исключены, выходим
            if np.all(excluded_rows) or np.all(excluded_cols):
                break
            
            # Находим минимальный элемент среди неисключенных
            min_cost = np.inf
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                if excluded_rows[i]:
                    continue
                for j in range(self.n):
                    if excluded_cols[j]:
                        continue
                    if temp_costs[i, j] < min_cost:
                        min_cost = temp_costs[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                break
            
            # Определяем объем поставки
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            
            # Уменьшаем запасы и потребности
            supply[min_i] -= amount
            demand[min_j] -= amount
            
            # Исключаем строку или столбец
            if abs(supply[min_i]) < 1e-10 and abs(demand[min_j]) < 1e-10:
                # Если и запас, и потребность обнулились, исключаем и строку, и столбец
                excluded_rows[min_i] = True
                excluded_cols[min_j] = True
            elif abs(supply[min_i]) < 1e-10:
                # Если запас обнулился, исключаем строку
                excluded_rows[min_i] = True
            elif abs(demand[min_j]) < 1e-10:
                # Если потребность обнулилась, исключаем столбец
                excluded_cols[min_j] = True
            
            # Для оставшихся клеток в исключенных строках/столбцах ставим очень большое число
            for i in range(self.m):
                if excluded_rows[i]:
                    temp_costs[i, :] = np.inf
            for j in range(self.n):
                if excluded_cols[j]:
                    temp_costs[:, j] = np.inf
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов - улучшенная реализация"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Начинаем с произвольной базисной клетки
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
        
        # Задаем начальное значение
        u[start_i] = 0
        
        # Итеративно вычисляем все потенциалы
        changed = True
        while changed:
            changed = False
            
            # Обновляем v на основе известных u
            for i in range(self.m):
                if not np.isnan(u[i]):
                    for j in range(self.n):
                        if plan[i, j] > 1e-10 and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
            
            # Обновляем u на основе известных v
            for j in range(self.n):
                if not np.isnan(v[j]):
                    for i in range(self.m):
                        if plan[i, j] > 1e-10 and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_cycle_improved(self, plan, start_i, start_j):
        """
        Улучшенный поиск цикла - использует алгоритм поиска в глубину
        """
        # Создаем списки смежности
        rows = [[] for _ in range(self.m)]
        cols = [[] for _ in range(self.n)]
        
        # Добавляем все базисные клетки
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-10:
                    rows[i].append(j)
                    cols[j].append(i)
        
        # Добавляем стартовую клетку
        rows[start_i].append(start_j)
        cols[start_j].append(start_i)
        
        # Используем стек для DFS
        stack = [(start_i, start_j, [(start_i, start_j)], 'row')]
        
        while stack:
            i, j, path, direction = stack.pop()
            
            # Если вернулись в начало и путь длиной > 1
            if len(path) > 3 and (i, j) == (start_i, start_j):
                return path
            
            if direction == 'row':
                # Ищем в строке i
                for next_j in rows[i]:
                    if next_j == j:
                        continue
                    # Проверяем, не создаем ли мы короткий цикл
                    if (i, next_j) not in path or (i, next_j) == (start_i, start_j):
                        stack.append((i, next_j, path + [(i, next_j)], 'col'))
            else:  # direction == 'col'
                # Ищем в столбце j
                for next_i in cols[j]:
                    if next_i == i:
                        continue
                    # Проверяем, не создаем ли мы короткий цикл
                    if (next_i, j) not in path or (next_i, j) == (start_i, start_j):
                        stack.append((next_i, j, path + [(next_i, j)], 'row'))
        
        return None
    
    def improve_plan_exact(self, plan):
        """Улучшение опорного плана - точная реализация"""
        u, v = self.get_potentials(plan)
        
        # Проверяем, все ли потенциалы вычислены
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return plan, False
        
        # Находим клетку с максимальной положительной оценкой
        max_delta = -np.inf
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta and delta > 1e-10:
                        max_delta = delta
                        best_i, best_j = i, j
        
        if best_i == -1:  # Нет клеток с положительной оценкой
            return plan, False
        
        # Находим цикл для этой клетки
        cycle = self.find_cycle_improved(plan, best_i, best_j)
        
        if cycle is None:
            return plan, False
        
        # Определяем, какие вершины положительные, какие отрицательные
        # В онлайн-решателе: начальная клетка "+", затем чередуются "-", "+", ...
        # У нас цикл может начинаться и заканчиваться в одной точке
        
        # Находим минимальное значение в отрицательных вершинах
        min_amount = np.inf
        for idx in range(1, len(cycle), 2):  # Отрицательные вершины (индексы 1, 3, 5, ...)
            i, j = cycle[idx]
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        if min_amount < 1e-10:  # Если минимальное значение слишком мало
            return plan, False
        
        # Перераспределяем поставки
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle):
            if idx == 0 or idx % 2 == 0:  # Положительные вершины
                new_plan[i, j] += min_amount
            else:  # Отрицательные вершины
                new_plan[i, j] -= min_amount
        
        # Убираем нули
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def check_optimality(self, plan):
        """Проверка оптимальности плана"""
        u, v = self.get_potentials(plan)
        
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return False, 0, -1, -1
        
        # Находим максимальную положительную оценку
        max_delta = -np.inf
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > max_delta:
                        max_delta = delta
                        if delta > 1e-10:
                            best_i, best_j = i, j
        
        is_optimal = (max_delta < 1e-10)
        return is_optimal, max_delta, best_i, best_j
    
    def solve(self, verbose=True):
        """Основной метод решения"""
        if verbose:
            print("=" * 60)
            print("РЕШЕНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ")
            print("=" * 60)
            
            print("\nИсходные данные:")
            print("Матрица стоимостей:")
            print(self.costs)
            print(f"\nЗапасы: {self.supply}")
            print(f"Потребности: {self.demand}")
            
            total_supply = np.sum(self.supply)
            total_demand = np.sum(self.demand)
            print(f"\nСумма запасов: {total_supply:.0f}")
            print(f"Сумма потребностей: {total_demand:.0f}")
            
            if abs(total_supply - total_demand) < 1e-10:
                print("Условие баланса соблюдается. Модель закрытая.")
            else:
                print("Условие баланса не соблюдается!")
        
        # Этап I: Построение начального плана
        if verbose:
            print(f"\n{'='*60}")
            print("ЭТАП I: Поиск первого опорного плана методом наименьшей стоимости")
        
        plan = self.minimal_cost_method_exact()
        
        if verbose:
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
        
        # Этап II: Улучшение плана
        if verbose:
            print(f"\n{'='*60}")
            print("ЭТАП II: Улучшение опорного плана методом потенциалов")
        
        iteration = 0
        improved = True
        
        while improved:
            iteration += 1
            
            # Проверяем оптимальность
            is_optimal, max_delta, best_i, best_j = self.check_optimality(plan)
            
            if is_optimal:
                if verbose:
                    print(f"\nИтерация {iteration}: все оценки свободных клеток ≤ 0")
                    print("План оптимален.")
                break
            
            if verbose:
                print(f"\nИтерация {iteration}:")
                print(f"Найдена клетка с положительной оценкой: ({best_i+1},{best_j+1}), Δ={max_delta:.2f}")
            
            # Улучшаем план
            new_plan, improved = self.improve_plan_exact(plan)
            
            if improved:
                old_cost = self.calculate_cost(plan)
                new_cost = self.calculate_cost(new_plan)
                
                if verbose:
                    print(f"Старая стоимость: {old_cost:.2f}")
                    print(f"Новая стоимость: {new_cost:.2f}")
                    print(f"Улучшение: {old_cost - new_cost:.2f}")
                    print(f"\nНовый план после итерации {iteration}:")
                    print(new_plan)
                
                plan = new_plan
            else:
                if verbose:
                    print("Не удалось улучшить план")
                break
        
        optimal_cost = self.calculate_cost(plan)
        
        if verbose:
            print(f"\nМинимальная стоимость: {optimal_cost:.2f}")
            
            print(f"\n{'='*60}")
            print("АНАЛИЗ ОПТИМАЛЬНОГО ПЛАНА")
            
            total_shipped = 0
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] > 1e-10:
                        amount = plan[i, j]
                        print(f"Из поставщика A{i+1} → потребителю B{j+1}: {amount:.1f} ед.")
                        total_shipped += amount
            
            print(f"\nВсего перевезено: {total_shipped:.1f} единиц")
            print(f"Общая минимальная стоимость: {optimal_cost:.2f}")
        
        return plan, optimal_cost


def test_variant(costs, supply, demand, variant_name, expected_cost=None):
    """Тестирование решения для конкретного варианта"""
    print(f"\n{'='*60}")
    print(f"ВАРИАНТ {variant_name}")
    print("=" * 60)
    
    problem = TransportProblem(costs, supply, demand)
    plan, cost = problem.solve(verbose=True)
    
    if expected_cost is not None:
        print(f"\nОжидаемая стоимость: {expected_cost:.2f}")
        print(f"Полученная стоимость: {cost:.2f}")
        print(f"Разница: {abs(cost - expected_cost):.2f}")
    
    return plan, cost


# Основная программа
if __name__ == "__main__":
    print("ТЕСТИРОВАНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ НА ТРЕХ ВАРИАНТАХ")
    
    # Вариант 1 (ожидаемая стоимость: 1870)
    costs1 = [
        [9, 5, 10, 7],
        [11, 8, 5, 6],
        [7, 6, 5, 4],
        [6, 4, 3, 2]
    ]
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    
    plan1, cost1 = test_variant(costs1, supply1, demand1, "1", 1870)
    
    # Вариант 2 (ожидаемая стоимость: 340)
    costs2 = [
        [5, 3, 4, 6, 4],
        [3, 4, 10, 5, 7],
        [4, 6, 9, 3, 4]
    ]
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    
    plan2, cost2 = test_variant(costs2, supply2, demand2, "2", 340)
    
    # Вариант 3 (ожидаемая стоимость: 800)
    costs3 = [
        [5, 4, 3, 2],
        [2, 3, 5, 6],
        [3, 2, 4, 3],
        [4, 1, 2, 4]
    ]
    supply3 = [120, 60, 80, 140]
    demand3 = [100, 140, 100, 60]
    
    plan3, cost3 = test_variant(costs3, supply3, demand3, "3", 800)
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ПО ВСЕМ ВАРИАНТАМ")
    print("=" * 60)
    print(f"Вариант 1: минимальная стоимость = {cost1:.2f} (ожидалось: 1870.00)")
    print(f"Вариант 2: минимальная стоимость = {cost2:.2f} (ожидалось: 340.00)")
    print(f"Вариант 3: минимальная стоимость = {cost3:.2f} (ожидалось: 800.00)")
    print("\n" + "="*60)