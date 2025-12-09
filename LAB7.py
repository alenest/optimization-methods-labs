import numpy as np
from collections import deque

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
        
        # Проверка баланса и приведение к закрытой модели
        self._balance_problem()
        
    def _balance_problem(self):
        """Приведение задачи к закрытой модели"""
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
    
    def minimal_cost_method(self):
        """Метод наименьшей стоимости - общая реализация"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Пока есть нераспределенные запасы или потребности
        while np.sum(supply) > 1e-10 and np.sum(demand) > 1e-10:
            # Находим минимальную стоимость среди всех доступных клеток
            min_cost = np.inf
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                if supply[i] < 1e-10:
                    continue
                for j in range(self.n):
                    if demand[j] < 1e-10:
                        continue
                    if self.costs[i, j] < min_cost:
                        min_cost = self.costs[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_j == -1:
                break
            
            # Определяем объем поставки
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            
            # Уменьшаем запасы и потребности
            supply[min_i] -= amount
            demand[min_j] -= amount
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов - общая реализация"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        u[0] = 0  # Произвольно задаем u1 = 0
        
        # Создаем список базисных клеток
        basic_cells = [(i, j) for i in range(self.m) for j in range(self.n) 
                      if plan[i, j] > 1e-10]
        
        # Итеративно вычисляем потенциалы
        changed = True
        while changed:
            changed = False
            for i, j in basic_cells:
                if not np.isnan(u[i]) and np.isnan(v[j]):
                    v[j] = self.costs[i, j] - u[i]
                    changed = True
                elif not np.isnan(v[j]) and np.isnan(u[i]):
                    u[i] = self.costs[i, j] - v[j]
                    changed = True
        
        return u, v
    
    def find_cycle(self, plan, start_i, start_j):
        """Поиск цикла для улучшения плана - общая реализация"""
        # Создаем граф из базисных клеток
        m, n = self.m, self.n
        
        # Находим все базисные клетки
        basic_cells = [(i, j) for i in range(m) for j in range(n) 
                      if plan[i, j] > 1e-10]
        
        # Добавляем стартовую клетку (она станет базисной после перераспределения)
        all_cells = basic_cells + [(start_i, start_j)]
        
        # Строим списки смежности по строкам и столбцам
        row_dict = {}
        col_dict = {}
        
        for i, j in all_cells:
            if i not in row_dict:
                row_dict[i] = []
            row_dict[i].append((i, j))
            
            if j not in col_dict:
                col_dict[j] = []
            col_dict[j].append((i, j))
        
        # Поиск в глубину для нахождения цикла
        def dfs(current, visited, path, direction):
            i, j = current
            
            if len(path) > 1 and (i, j) == (start_i, start_j):
                return path
            
            if direction == 'row':
                # Ищем в столбце j
                for cell in col_dict.get(j, []):
                    if cell == current:
                        continue
                    if cell in visited:
                        continue
                    new_visited = visited.copy()
                    new_visited.add(cell)
                    new_path = path + [cell]
                    result = dfs(cell, new_visited, new_path, 'col')
                    if result:
                        return result
            else:  # direction == 'col'
                # Ищем в строке i
                for cell in row_dict.get(i, []):
                    if cell == current:
                        continue
                    if cell in visited:
                        continue
                    new_visited = visited.copy()
                    new_visited.add(cell)
                    new_path = path + [cell]
                    result = dfs(cell, new_visited, new_path, 'row')
                    if result:
                        return result
            
            return None
        
        # Начинаем поиск с начальной клетки
        start_cell = (start_i, start_j)
        cycle = dfs(start_cell, {start_cell}, [start_cell], 'row')
        
        if cycle is None:
            # Пробуем начать с другого направления
            cycle = dfs(start_cell, {start_cell}, [start_cell], 'col')
        
        return cycle
    
    def improve_plan(self, plan):
        """Улучшение опорного плана методом потенциалов - общая реализация"""
        u, v = self.get_potentials(plan)
        
        # Находим свободные клетки с положительными оценками
        positive_cells = []
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > 1e-10:
                        positive_cells.append((delta, i, j))
        
        if not positive_cells:
            return plan, False  # План оптимален
        
        # Выбираем клетку с максимальной оценкой
        positive_cells.sort(reverse=True)
        max_delta, best_i, best_j = positive_cells[0]
        
        # Находим цикл для этой клетки
        cycle = self.find_cycle(plan, best_i, best_j)
        
        if cycle is None:
            return plan, False
        
        # Находим минимальное значение в отрицательных вершинах цикла
        # (вершины с нечетными индексами, начиная с 1)
        min_amount = float('inf')
        for idx in range(1, len(cycle), 2):
            i, j = cycle[idx]
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        # Перераспределяем поставки по циклу
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:  # Положительные вершины (0, 2, 4, ...)
                new_plan[i, j] += min_amount
            else:  # Отрицательные вершины (1, 3, 5, ...)
                new_plan[i, j] -= min_amount
        
        # Убираем нули
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def solve_transport_problem(self, verbose=True):
        """Основной метод решения транспортной задачи - общая реализация"""
        if verbose:
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
        if verbose:
            print(f"\n{'='*60}")
            print("ЭТАП I: Поиск первого опорного плана методом наименьшей стоимости")
        
        plan = self.minimal_cost_method()
        
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
        
        # ЭТАП II: Улучшение опорного плана методом потенциалов
        if verbose:
            print(f"\n{'='*60}")
            print("ЭТАП II: Улучшение опорного плана методом потенциалов")
        
        improved = True
        iteration = 0
        while improved:
            iteration += 1
            if verbose:
                print(f"\nИтерация {iteration}:")
            
            plan, improved = self.improve_plan(plan)
            
            if verbose and improved:
                cost = self.calculate_cost(plan)
                print(f"Стоимость после итерации {iteration}: {cost:.2f}")
        
        optimal_cost = self.calculate_cost(plan)
        
        if verbose:
            print(f"\nОптимальная стоимость: {optimal_cost:.2f}")
            
            # Анализ оптимального плана
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


def test_variant(costs, supply, demand, variant_name):
    """Тестирование решения для конкретного варианта"""
    print(f"\n{'='*60}")
    print(f"ВАРИАНТ {variant_name}")
    print("=" * 60)
    
    problem = TransportProblem(costs, supply, demand)
    plan, cost = problem.solve_transport_problem(verbose=True)
    
    return plan, cost


# Основная программа
if __name__ == "__main__":
    print("ТЕСТИРОВАНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ НА ТРЕХ ВАРИАНТАХ")
    
    # Вариант 1
    costs1 = [
        [9, 5, 10, 7],
        [11, 8, 5, 6],
        [7, 6, 5, 4],
        [6, 4, 3, 2]
    ]
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    
    plan1, cost1 = test_variant(costs1, supply1, demand1, "1")
    
    # Вариант 2
    costs2 = [
        [5, 3, 4, 6, 4],
        [3, 4, 10, 5, 7],
        [4, 6, 9, 3, 4]
    ]
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    
    plan2, cost2 = test_variant(costs2, supply2, demand2, "2")
    
    # Вариант 3
    costs3 = [
        [5, 4, 3, 2],
        [2, 3, 5, 6],
        [3, 2, 4, 3],
        [4, 1, 2, 4]
    ]
    supply3 = [120, 60, 80, 140]
    demand3 = [100, 140, 100, 60]
    
    plan3, cost3 = test_variant(costs3, supply3, demand3, "3")
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ПО ВСЕМ ВАРИАНТАМ")
    print("=" * 60)
    print(f"Вариант 1: минимальная стоимость = {cost1:.2f}")
    print(f"Вариант 2: минимальная стоимость = {cost2:.2f}")
    print(f"Вариант 3: минимальная стоимость = {cost3:.2f}")
    print("\n" + "="*60)