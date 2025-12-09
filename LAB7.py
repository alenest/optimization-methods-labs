import numpy as np

class TransportProblem:
    def __init__(self, costs, supply, demand):
        """
        Абстрактный решатель транспортной задачи
        Реализует алгоритмы как в онлайн-решателе
        """
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.m, self.n = self.costs.shape
        
        # Проверка баланса
        self._check_balance()
    
    def _check_balance(self):
        """Проверка и приведение к закрытой модели"""
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)
        
        if abs(total_supply - total_demand) > 1e-10:
            if total_supply > total_demand:
                self.demand = np.append(self.demand, total_supply - total_demand)
                self.costs = np.column_stack([self.costs, np.zeros(self.m)])
            else:
                self.supply = np.append(self.supply, total_demand - total_supply)
                self.costs = np.row_stack([self.costs, np.zeros(self.n)])
            
            self.m, self.n = self.costs.shape
    
    def northwest_corner(self):
        """Метод северо-западного угла (для сравнения)"""
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        i, j = 0, 0
        while i < self.m and j < self.n:
            if supply[i] < 1e-10:
                i += 1
                continue
            if demand[j] < 1e-10:
                j += 1
                continue
            
            amount = min(supply[i], demand[j])
            plan[i, j] = amount
            supply[i] -= amount
            demand[j] -= amount
            
            if supply[i] < 1e-10:
                i += 1
            if demand[j] < 1e-10:
                j += 1
        
        return plan
    
    def minimal_cost_method(self):
        """
        Метод минимальной стоимости ТОЧНО как в онлайн-решателе
        Алгоритм из онлайн-решения:
        1. Найти минимальную стоимость в матрице
        2. Распределить min(запас, потребность)
        3. Исключить строку или столбец
        4. Повторять пока все не распределено
        """
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Создаем рабочую копию матрицы стоимостей
        work_costs = self.costs.copy()
        
        # Маски для исключенных строк и столбцов
        excluded_rows = np.zeros(self.m, dtype=bool)
        excluded_cols = np.zeros(self.n, dtype=bool)
        
        while True:
            # Находим минимальную стоимость среди неисключенных
            min_val = np.inf
            min_i, min_j = -1, -1
            
            for i in range(self.m):
                if excluded_rows[i]:
                    continue
                for j in range(self.n):
                    if excluded_cols[j]:
                        continue
                    if work_costs[i, j] < min_val:
                        min_val = work_costs[i, j]
                        min_i, min_j = i, j
            
            # Если не нашли или все распределено
            if min_i == -1 or (supply[min_i] < 1e-10 and demand[min_j] < 1e-10):
                break
            
            # Распределяем
            amount = min(supply[min_i], demand[min_j])
            plan[min_i, min_j] = amount
            supply[min_i] -= amount
            demand[min_j] -= amount
            
            # Исключаем строку/столбец как в онлайн-решении
            if abs(supply[min_i]) < 1e-10 and abs(demand[min_j]) < 1e-10:
                # Исключаем и строку, и столбец
                excluded_rows[min_i] = True
                excluded_cols[min_j] = True
            elif abs(supply[min_i]) < 1e-10:
                # Исключаем только строку
                excluded_rows[min_i] = True
            elif abs(demand[min_j]) < 1e-10:
                # Исключаем только столбец
                excluded_cols[min_j] = True
            
            # Помечаем исключенные клетки очень большим числом
            for i in range(self.m):
                if excluded_rows[i]:
                    work_costs[i, :] = 1e10
            for j in range(self.n):
                if excluded_cols[j]:
                    work_costs[:, j] = 1e10
        
        return plan
    
    def get_potentials(self, plan):
        """
        Вычисление потенциалов ТОЧНО как в онлайн-решателе
        u[i] + v[j] = c[i][j] для базисных клеток
        Начинаем с u[0] = 0
        """
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Находим первую базисную клетку
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
        
        # Как в онлайн-решателе: полагаем u[0] = 0
        # Но если start_i не 0, то все равно начинаем с u[start_i] = 0
        u[start_i] = 0
        
        # Итерационно вычисляем все потенциалы
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
        """
        Находит цикл улучшения для свободной клетки (start_i, start_j)
        Возвращает список клеток в порядке: [+, -, +, -, ...]
        """
        # Временная метка для стартовой клетки
        temp_plan = plan.copy()
        temp_plan[start_i, start_j] = 0.001  # Временно помечаем
        
        # Рекурсивный поиск цикла
        def dfs(i, j, path, from_row):
            # Если вернулись в начало
            if len(path) > 3 and i == start_i and j == start_j:
                return path
            
            if from_row:
                # Ищем в столбце
                for next_i in range(self.m):
                    if next_i == i:
                        continue
                    if temp_plan[next_i, j] > 1e-10:
                        if (next_i, j) not in path or (next_i == start_i and j == start_j):
                            result = dfs(next_i, j, path + [(next_i, j)], False)
                            if result:
                                return result
            else:
                # Ищем в строке
                for next_j in range(self.n):
                    if next_j == j:
                        continue
                    if temp_plan[i, next_j] > 1e-10:
                        if (i, next_j) not in path or (i == start_i and next_j == start_j):
                            result = dfs(i, next_j, path + [(i, next_j)], True)
                            if result:
                                return result
            
            return None
        
        # Начинаем поиск
        cycle = dfs(start_i, start_j, [(start_i, start_j)], True)
        return cycle
    
    def improve_with_potentials(self, plan):
        """
        Улучшение плана методом потенциалов
        Возвращает новый улучшенный план и флаг улучшения
        """
        # 1. Вычисляем потенциалы
        u, v = self.get_potentials(plan)
        
        # 2. Ищем клетку с максимальной положительной оценкой
        best_delta = -np.inf
        best_i, best_j = -1, -1
        
        for i in range(self.m):
            for j in range(self.n):
                if plan[i, j] < 1e-10:  # Свободная клетка
                    delta = u[i] + v[j] - self.costs[i, j]
                    if delta > best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
        
        # Если нет положительных оценок - план оптимален
        if best_delta < 1e-10:
            return plan, False
        
        # 3. Находим цикл для этой клетки
        cycle = self.find_improvement_cycle(plan, best_i, best_j)
        if cycle is None:
            return plan, False
        
        # 4. Находим минимальное значение в минусовых клетках
        min_amount = np.inf
        for idx in range(1, len(cycle), 2):  # Минусовые клетки (нечетные индексы)
            i, j = cycle[idx]
            if plan[i, j] < min_amount:
                min_amount = plan[i, j]
        
        if min_amount < 1e-10:
            return plan, False
        
        # 5. Перераспределяем
        new_plan = plan.copy()
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:  # Плюсовые клетки
                new_plan[i, j] += min_amount
            else:  # Минусовые клетки
                new_plan[i, j] -= min_amount
        
        # Очищаем очень маленькие значения
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def solve(self, method='minimal_cost'):
        """
        Решение транспортной задачи
        method: 'northwest' или 'minimal_cost'
        """
        # 1. Построение начального плана
        if method == 'northwest':
            plan = self.northwest_corner()
        else:
            plan = self.minimal_cost_method()
        
        initial_cost = np.sum(plan * self.costs)
        
        # 2. Улучшение плана методом потенциалов
        improved = True
        iterations = 0
        
        while improved and iterations < 50:
            iterations += 1
            plan, improved = self.improve_with_potentials(plan)
        
        final_cost = np.sum(plan * self.costs)
        
        return {
            'plan': plan,
            'initial_cost': initial_cost,
            'final_cost': final_cost,
            'iterations': iterations
        }


# Тестирование на трех вариантах
def test_all_variants():
    print("ТЕСТИРОВАНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ")
    print("=" * 60)
    
    # Вариант 1 (ожидаемый результат: 1870)
    print("\nВАРИАНТ 1 (ожидается: 1870)")
    print("-" * 40)
    
    costs1 = [
        [9, 5, 10, 7],
        [11, 8, 5, 6],
        [7, 6, 5, 4],
        [6, 4, 3, 2]
    ]
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    
    tp1 = TransportProblem(costs1, supply1, demand1)
    
    # Тест метода минимальной стоимости
    result1 = tp1.solve('minimal_cost')
    print(f"Метод минимальной стоимости:")
    print(f"  Начальная стоимость: {result1['initial_cost']:.2f}")
    print(f"  Конечная стоимость:  {result1['final_cost']:.2f}")
    print(f"  Итераций улучшения:  {result1['iterations']}")
    print(f"  Совпадение с онлайн: {'✓' if abs(result1['final_cost'] - 1870) < 0.01 else '✗'}")
    
    # Вариант 2 (ожидаемый результат: 340)
    print("\nВАРИАНТ 2 (ожидается: 340)")
    print("-" * 40)
    
    costs2 = [
        [5, 3, 4, 6, 4],
        [3, 4, 10, 5, 7],
        [4, 6, 9, 3, 4]
    ]
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    
    tp2 = TransportProblem(costs2, supply2, demand2)
    
    result2 = tp2.solve('minimal_cost')
    print(f"Метод минимальной стоимости:")
    print(f"  Начальная стоимость: {result2['initial_cost']:.2f}")
    print(f"  Конечная стоимость:  {result2['final_cost']:.2f}")
    print(f"  Итераций улучшения:  {result2['iterations']}")
    print(f"  Совпадение с онлайн: {'✓' if abs(result2['final_cost'] - 340) < 0.01 else '✗'}")
    
    # Вариант 3 (ожидаемый результат: 800)
    print("\nВАРИАНТ 3 (ожидается: 800)")
    print("-" * 40)
    
    costs3 = [
        [5, 4, 3, 2],
        [2, 3, 5, 6],
        [3, 2, 4, 3],
        [4, 1, 2, 4]
    ]
    supply3 = [120, 60, 80, 140]
    demand3 = [100, 140, 100, 60]
    
    tp3 = TransportProblem(costs3, supply3, demand3)
    
    result3 = tp3.solve('minimal_cost')
    print(f"Метод минимальной стоимости:")
    print(f"  Начальная стоимость: {result3['initial_cost']:.2f}")
    print(f"  Конечная стоимость:  {result3['final_cost']:.2f}")
    print(f"  Итераций улучшения:  {result3['iterations']}")
    print(f"  Совпадение с онлайн: {'✓' if abs(result3['final_cost'] - 800) < 0.01 else '✗'}")
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print("-" * 60)
    
    successes = 0
    if abs(result1['final_cost'] - 1870) < 0.01:
        successes += 1
    if abs(result2['final_cost'] - 340) < 0.01:
        successes += 1
    if abs(result3['final_cost'] - 800) < 0.01:
        successes += 1
    
    print(f"Успешно решено: {successes} из 3 вариантов")
    
    if successes == 3:
        print("✓ ВСЕ ЗАДАЧИ РЕШЕНЫ ПРАВИЛЬНО!")
    else:
        print("✗ ЕСТЬ ОШИБКИ В РЕШЕНИИ")
    
    return successes == 3


if __name__ == "__main__":
    success = test_all_variants()
    exit(0 if success else 1)