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
        self._balance_problem()
        
        # Для отслеживания истории выбора клеток
        self.selection_history = []
    
    def _balance_problem(self):
        """Приведение задачи к закрытой модели"""
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
    
    def minimal_cost_method_online(self):
        """
        Точная реализация метода наименьшей стоимости как в онлайн-решателе
        """
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Создаем копию стоимостей для вычеркивания
        temp_costs = self.costs.copy()
        
        # Маски для вычеркнутых строк/столбцов
        crossed_rows = np.zeros(self.m, dtype=bool)
        crossed_cols = np.zeros(self.n, dtype=bool)
        
        iteration = 0
        while np.any(supply > 1e-10) and np.any(demand > 1e-10):
            iteration += 1
            
            # Находим минимальную стоимость среди невычеркнутых
            min_val = np.inf
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                if crossed_rows[i]:
                    continue
                for j in range(self.n):
                    if crossed_cols[j]:
                        continue
                    if temp_costs[i, j] < min_val:
                        min_val = temp_costs[i, j]
                        min_i, min_j = i, j
            
            # Определяем объем поставки
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            
            # Обновляем запасы и потребности
            supply[min_i] -= amount
            demand[min_j] -= amount
            
            # Вычеркиваем строку/столбец как в онлайн-решателе
            if abs(supply[min_i]) < 1e-10 and abs(demand[min_j]) < 1e-10:
                # Вычеркиваем и строку, и столбец
                crossed_rows[min_i] = True
                crossed_cols[min_j] = True
            elif abs(supply[min_i]) < 1e-10:
                # Вычеркиваем только строку
                crossed_rows[min_i] = True
            elif abs(demand[min_j]) < 1e-10:
                # Вычеркиваем только столбец
                crossed_cols[min_j] = True
            
            # Запоминаем выбор
            self.selection_history.append({
                'iteration': iteration,
                'cell': (min_i, min_j),
                'cost': min_val,
                'amount': amount
            })
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials_online(self, plan):
        """
        Точная реализация вычисления потенциалов как в онлайн-решателе
        """
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Как в онлайн-решателе: полагаем u[0] = 0
        u[0] = 0.0
        
        # Итерационно вычисляем все потенциалы
        changed = True
        while changed:
            changed = False
            
            # Для каждой базисной клетки пытаемся вычислить потенциалы
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
    
    def find_cycle_online(self, plan, start_i, start_j):
        """
        Поиск цикла как в онлайн-решателе
        Возвращает список вершин цикла, начиная и заканчивая start_i, start_j
        """
        # Создаем списки базисных клеток по строкам и столбцам
        row_cells = [[] for _ in range(self.m)]
        col_cells = [[] for _ in range(self.n)]
        
        # Заполняем списки базисными клетками
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] > 1e-10:
                    row_cells[i].append(j)
                    col_cells[j].append(i)
        
        # Добавляем стартовую клетку (она станет базисной)
        row_cells[start_i].append(start_j)
        col_cells[start_j].append(start_i)
        
        # Используем DFS для поиска цикла
        def dfs(current_i, current_j, visited, path, came_from_row):
            # Если вернулись в начало и длина пути > 1
            if len(path) > 1 and (current_i, current_j) == (start_i, start_j):
                return path
            
            # В зависимости от того, откуда пришли, ищем следующий шаг
            if came_from_row:
                # Пришли по строке, теперь ищем в столбце
                for next_i in col_cells[current_j]:
                    if next_i == current_i:
                        continue
                    if (next_i, current_j) in visited and (next_i, current_j) != (start_i, start_j):
                        continue
                    new_visited = visited.copy()
                    new_visited.add((next_i, current_j))
                    result = dfs(next_i, current_j, new_visited, 
                                path + [(next_i, current_j)], False)
                    if result:
                        return result
            else:
                # Пришли по столбцу, теперь ищем в строке
                for next_j in row_cells[current_i]:
                    if next_j == current_j:
                        continue
                    if (current_i, next_j) in visited and (current_i, next_j) != (start_i, start_j):
                        continue
                    new_visited = visited.copy()
                    new_visited.add((current_i, next_j))
                    result = dfs(current_i, next_j, new_visited,
                                path + [(current_i, next_j)], True)
                    if result:
                        return result
            
            return None
        
        # Начинаем поиск с начальной клетки, двигаемся по строке
        cycle = dfs(start_i, start_j, {(start_i, start_j)}, [(start_i, start_j)], True)
        
        # Если не нашли, пробуем начать с движения по столбцу
        if cycle is None:
            cycle = dfs(start_i, start_j, {(start_i, start_j)}, [(start_i, start_j)], False)
        
        return cycle
    
    def improve_plan_online(self, plan):
        """
        Улучшение плана как в онлайн-решателе
        """
        # Вычисляем потенциалы
        u, v = self.get_potentials_online(plan)
        
        # Проверяем, все ли потенциалы вычислены
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return plan, False
        
        # Ищем свободную клетку с максимальной положительной оценкой
        max_delta = -np.inf
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > 1e-10 and delta > max_delta:
                        max_delta = delta
                        best_i, best_j = i, j
        
        # Если нет клеток с положительной оценкой
        if best_i == -1:
            return plan, False
        
        # Находим цикл для этой клетки
        cycle = self.find_cycle_online(plan, best_i, best_j)
        
        if cycle is None:
            return plan, False
        
        # Определяем минимальную поставку в отрицательных вершинах
        # В онлайн-решателе: отрицательные вершины - это вершины, из которых вычитаем
        min_amount = np.inf
        for idx in range(1, len(cycle), 2):  # Отрицательные вершины (1, 3, 5, ...)
            i, j = cycle[idx]
            # Если это не стартовая клетка (она пока нулевая)
            if not (i == best_i and j == best_j):
                if plan[i, j] < min_amount:
                    min_amount = plan[i, j]
        
        # Перераспределяем поставки
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle):
            if idx == 0 or idx % 2 == 0:  # Положительные вершины (0, 2, 4, ...)
                new_plan[i, j] += min_amount
            else:  # Отрицательные вершины (1, 3, 5, ...)
                new_plan[i, j] -= min_amount
        
        # Убираем нулевые поставки
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def check_degeneracy(self, plan):
        """Проверка на вырожденность и добавление фиктивной базисной клетки если нужно"""
        basic_cells = np.sum(plan > 1e-10)
        required = self.m + self.n - 1
        
        if basic_cells < required:
            # Находим свободную клетку с наименьшей стоимостью для добавления фиктивной поставки
            min_cost = np.inf
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] < 1e-10:  # Свободная клетка
                        if self.costs[i, j] < min_cost:
                            min_cost = self.costs[i, j]
                            min_i, min_j = i, j
            
            if min_i != -1:
                # Добавляем фиктивную поставку (0 или очень маленькое значение)
                plan[min_i, min_j] = 1e-10  # Почти ноль
        
        return plan
    
    def solve(self, verbose=True):
        """Основной метод решения как в онлайн-решателе"""
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
        
        # Сбрасываем историю выбора
        self.selection_history = []
        plan = self.minimal_cost_method_online()
        
        # Проверяем и исправляем вырожденность
        plan = self.check_degeneracy(plan)
        
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
            
            # Показываем историю выбора
            print("\nИстория выбора клеток методом наименьшей стоимости:")
            for step in self.selection_history:
                i, j = step['cell']
                print(f"  Шаг {step['iteration']}: клетка ({i+1},{j+1}), стоимость={step['cost']}, объем={step['amount']}")
        
        # Этап II: Улучшение плана
        if verbose:
            print(f"\n{'='*60}")
            print("ЭТАП II: Улучшение опорного плана методом потенциалов")
        
        iteration = 0
        improved = True
        
        while improved:
            iteration += 1
            
            # Вычисляем потенциалы
            u, v = self.get_potentials_online(plan)
            
            if verbose:
                print(f"\nИтерация {iteration}:")
                print(f"Потенциалы u: {u}")
                print(f"Потенциалы v: {v}")
            
            # Ищем клетку с максимальной положительной оценкой
            max_delta = -np.inf
            best_i, best_j = -1, -1
            positive_cells = []
            
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] < 1e-10:  # Свободная клетка
                        delta = u[i] + v[j] - self.costs[i, j]
                        if delta > 1e-10:
                            positive_cells.append((delta, i, j))
                            if delta > max_delta:
                                max_delta = delta
                                best_i, best_j = i, j
            
            if not positive_cells:
                if verbose:
                    print("Все оценки свободных клеток ≤ 0. План оптимален.")
                improved = False
                break
            
            if verbose:
                print(f"Найдены клетки с положительными оценками:")
                for delta, i, j in positive_cells:
                    print(f"  ({i+1},{j+1}): {u[i]:.1f} + {v[j]:.1f} - {self.costs[i,j]:.1f} = {delta:.2f}")
                print(f"Выбираем клетку ({best_i+1},{best_j+1}) с максимальной оценкой Δ={max_delta:.2f}")
            
            # Находим цикл для этой клетки
            cycle = self.find_cycle_online(plan, best_i, best_j)
            
            if cycle is None:
                if verbose:
                    print(f"Не удалось найти цикл для клетки ({best_i+1},{best_j+1})")
                improved = False
                break
            
            if verbose:
                cycle_str = " → ".join([f"({i+1},{j+1})" for i, j in cycle])
                print(f"Найден цикл: {cycle_str}")
            
            # Находим минимальную поставку в отрицательных вершинах
            min_amount = np.inf
            for idx in range(1, len(cycle), 2):  # Отрицательные вершины
                i, j = cycle[idx]
                # Пропускаем стартовую клетку (она пока нулевая)
                if not (i == best_i and j == best_j):
                    if plan[i, j] < min_amount:
                        min_amount = plan[i, j]
                        min_cell = (i, j)
            
            if verbose:
                print(f"Минимальная поставка в отрицательных вершинах: {min_amount} (клетка ({min_cell[0]+1},{min_cell[1]+1}))")
            
            # Перераспределяем
            new_plan = plan.copy()
            for idx, (i, j) in enumerate(cycle):
                if idx == 0 or idx % 2 == 0:  # Положительные вершины
                    new_plan[i, j] += min_amount
                else:  # Отрицательные вершины
                    new_plan[i, j] -= min_amount
            
            # Убираем нули
            new_plan[new_plan < 1e-10] = 0
            
            old_cost = self.calculate_cost(plan)
            new_cost = self.calculate_cost(new_plan)
            
            if verbose:
                print(f"Старая стоимость: {old_cost:.2f}")
                print(f"Новая стоимость: {new_cost:.2f}")
                print(f"Улучшение: {old_cost - new_cost:.2f}")
                print(f"\nНовый план:")
                print(new_plan)
            
            # Проверяем, что стоимость действительно уменьшилась
            if new_cost > old_cost + 1e-10:
                if verbose:
                    print("Внимание: стоимость увеличилась! Прерываем улучшение.")
                improved = False
            else:
                plan = new_plan
        
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
                        if amount > 1e-5:  # Игнорируем фиктивные очень маленькие значения
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
        print(f"\nСравнение с онлайн-решением:")
        print(f"Ожидаемая стоимость: {expected_cost:.2f}")
        print(f"Полученная стоимость: {cost:.2f}")
        print(f"Разница: {abs(cost - expected_cost):.2f}")
        
        if abs(cost - expected_cost) < 1e-5:
            print("✓ Решение совпадает с онлайн-решением!")
        else:
            print("✗ Решение не совпадает с онлайн-решением")
    
    return plan, cost


# Основная программа
if __name__ == "__main__":
    print("ТЕСТИРОВАНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ НА ТРЕХ ВАРИАНТАХ")
    
    # Вариант 1 (ожидаемая стоимость: 1870)
    print("\n" + "="*60)
    print("ВАРИАНТ 1 (ожидаемая стоимость: 1870)")
    print("=" * 60)
    
    costs1 = [
        [9, 5, 10, 7],
        [11, 8, 5, 6],
        [7, 6, 5, 4],
        [6, 4, 3, 2]
    ]
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    
    problem1 = TransportProblem(costs1, supply1, demand1)
    plan1, cost1 = problem1.solve(verbose=True)
    
    # Вариант 2 (ожидаемая стоимость: 340)
    print("\n" + "="*60)
    print("ВАРИАНТ 2 (ожидаемая стоимость: 340)")
    print("=" * 60)
    
    costs2 = [
        [5, 3, 4, 6, 4],
        [3, 4, 10, 5, 7],
        [4, 6, 9, 3, 4]
    ]
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    
    problem2 = TransportProblem(costs2, supply2, demand2)
    plan2, cost2 = problem2.solve(verbose=True)
    
    # Вариант 3 (ожидаемая стоимость: 800)
    print("\n" + "="*60)
    print("ВАРИАНТ 3 (ожидаемая стоимость: 800)")
    print("=" * 60)
    
    costs3 = [
        [5, 4, 3, 2],
        [2, 3, 5, 6],
        [3, 2, 4, 3],
        [4, 1, 2, 4]
    ]
    supply3 = [120, 60, 80, 140]
    demand3 = [100, 140, 100, 60]
    
    problem3 = TransportProblem(costs3, supply3, demand3)
    plan3, cost3 = problem3.solve(verbose=True)
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ПО ВСЕМ ВАРИАНТАМ")
    print("=" * 60)
    print(f"Вариант 1: минимальная стоимость = {cost1:.2f} (ожидалось: 1870.00)")
    print(f"Вариант 2: минимальная стоимость = {cost2:.2f} (ожидалось: 340.00)")
    print(f"Вариант 3: минимальная стоимость = {cost3:.2f} (ожидалось: 800.00)")
    
    # Подсчет успешных решений
    success_count = 0
    if abs(cost1 - 1870) < 1e-5:
        success_count += 1
    if abs(cost2 - 340) < 1e-5:
        success_count += 1
    if abs(cost3 - 800) < 1e-5:
        success_count += 1
    
    print(f"\nУспешно решено: {success_count} из 3 вариантов")
    print("\n" + "="*60)