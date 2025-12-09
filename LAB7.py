import numpy as np

class TransportProblem:
    def __init__(self, costs, supply, demand):
        """
        Абстрактный решатель транспортной задачи
        """
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.m, self.n = self.costs.shape
        
        # Приведение к закрытой модели
        self._balance_problem()
    
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
    
    def northwest_corner_method(self):
        """Метод северо-западного угла - точная реализация"""
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
    
    def minimal_cost_method_online(self):
        """
        Метод минимальной стоимости как в онлайн-решателе
        Точная реализация алгоритма из онлайн-решения
        """
        plan = np.zeros((self.m, self.n))
        supply = self.supply.copy()
        demand = self.demand.copy()
        
        # Создаем список всех клеток с их стоимостями и координатами
        cells = []
        for i in range(self.m):
            for j in range(self.n):
                cells.append((self.costs[i, j], i, j))
        
        # Сортируем по стоимости (как в онлайн-решателе)
        cells.sort(key=lambda x: x[0])
        
        # Распределяем грузы
        for cost, i, j in cells:
            if supply[i] > 1e-10 and demand[j] > 1e-10:
                amount = min(supply[i], demand[j])
                plan[i, j] = amount
                supply[i] -= amount
                demand[j] -= amount
        
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """
        Вычисление потенциалов как в онлайн-решателе
        u[i] + v[j] = c[i][j] для базисных клеток
        """
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Как в онлайн-решателе: полагаем u[0] = 0
        u[0] = 0
        
        # Вычисляем потенциалы итеративно
        changed = True
        while changed:
            changed = False
            
            # Проходим по всем базисным клеткам
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i, j] > 1e-10:  # Базисная клетка
                        # Если известен u[i], вычисляем v[j]
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                        # Если известен v[j], вычисляем u[i]
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        
        return u, v
    
    def find_cycle_for_cell(self, plan, i, j):
        """
        Находит цикл для клетки (i, j) как в онлайн-решателе
        Возвращает список клеток цикла в порядке обхода
        """
        # Создаем копию плана с добавлением исследуемой клетки
        temp_plan = plan.copy()
        temp_plan[i, j] = 1  # Временно помечаем как базисную
        
        # Ищем цикл с помощью BFS
        from collections import deque
        
        # Очередь содержит (текущая_клетка, путь, пришли_по_строке)
        queue = deque()
        queue.append(((i, j), [(i, j)], True))
        
        while queue:
            (ci, cj), path, came_from_row = queue.popleft()
            
            # Если вернулись в начало и путь длинный
            if len(path) > 3 and (ci, cj) == (i, j):
                return path
            
            if came_from_row:
                # Ищем в том же столбце другие базисные клетки
                for ni in range(self.m):
                    if ni == ci:
                        continue
                    if temp_plan[ni, cj] > 1e-10 or (ni == i and cj == j):
                        if (ni, cj) not in path or (ni == i and cj == j):
                            queue.append(((ni, cj), path + [(ni, cj)], False))
            else:
                # Ищем в той же строке другие базисные клетки
                for nj in range(self.n):
                    if nj == cj:
                        continue
                    if temp_plan[ci, nj] > 1e-10 or (ci == i and nj == j):
                        if (ci, nj) not in path or (ci == i and nj == j):
                            queue.append(((ci, nj), path + [(ci, nj)], True))
        
        return None
    
    def improve_plan_with_potentials(self, plan):
        """
        Улучшение плана методом потенциалов как в онлайн-решателе
        """
        # Вычисляем потенциалы
        u, v = self.get_potentials(plan)
        
        # Находим свободную клетку с максимальной положительной оценкой
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
        
        # Находим цикл для этой клетки
        cycle = self.find_cycle_for_cell(plan, best_i, best_j)
        
        if cycle is None:
            return plan, False
        
        # Находим минимальное значение в минусовых клетках цикла
        min_amount = np.inf
        for idx in range(1, len(cycle), 2):  # Минусовые клетки - нечетные индексы
            ci, cj = cycle[idx]
            # Не учитываем стартовую клетку (она пока 0)
            if not (ci == best_i and cj == best_j):
                if plan[ci, cj] < min_amount:
                    min_amount = plan[ci, cj]
        
        # Перераспределяем поставки
        new_plan = plan.copy()
        for idx, (ci, cj) in enumerate(cycle):
            if idx % 2 == 0:  # Плюсовые клетки - четные индексы
                new_plan[ci, cj] += min_amount
            else:  # Минусовые клетки - нечетные индексы
                new_plan[ci, cj] -= min_amount
        
        # Убираем нули
        new_plan[new_plan < 1e-10] = 0
        
        return new_plan, True
    
    def solve_with_method(self, method='minimal_cost', max_iterations=50):
        """
        Решает транспортную задачу указанным методом
        """
        # Шаг 1: Строим начальный план
        if method == 'northwest':
            plan = self.northwest_corner_method()
        else:
            plan = self.minimal_cost_method_online()
        
        initial_cost = self.calculate_cost(plan)
        
        # Шаг 2: Улучшаем план методом потенциалов
        iteration = 0
        improved = True
        
        while improved and iteration < max_iterations:
            iteration += 1
            plan, improved = self.improve_plan_with_potentials(plan)
        
        final_cost = self.calculate_cost(plan)
        
        return plan, initial_cost, final_cost, iteration


# Функция для тестирования
def test_variant(costs, supply, demand, variant_num, expected_cost):
    """Тестирует решение для одного варианта"""
    print(f"\n{'='*60}")
    print(f"ВАРИАНТ {variant_num} (ожидаемая стоимость: {expected_cost})")
    print('='*60)
    
    problem = TransportProblem(costs, supply, demand)
    
    # Тестируем метод минимальной стоимости (как в онлайн-решателе)
    print("\nМЕТОД МИНИМАЛЬНОЙ СТОИМОСТИ (как в онлайн-решателе):")
    plan, init_cost, final_cost, iters = problem.solve_with_method('minimal_cost')
    
    print(f"Начальная стоимость: {init_cost:.2f}")
    print(f"Конечная стоимость:  {final_cost:.2f}")
    print(f"Количество итераций улучшения: {iters}")
    
    if abs(final_cost - expected_cost) < 0.01:
        print(f"✓ СОВПАДЕНИЕ С ОНЛАЙН-РЕШЕНИЕМ!")
    else:
        print(f"✗ ОТЛИЧИЕ ОТ ОНЛАЙН-РЕШЕНИЯ: {abs(final_cost - expected_cost):.2f}")
    
    # Выводим ненулевые поставки
    print("\nПлан распределения:")
    total = 0
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            if plan[i, j] > 1e-5:
                print(f"  A{i+1} → B{j+1}: {plan[i, j]:.1f} ед.")
                total += plan[i, j]
    
    print(f"Всего перевезено: {total:.1f} единиц")
    
    return final_cost


# Основная программа
def main():
    print("ТРАНСПОРТНАЯ ЗАДАЧА")
    print("Алгоритм соответствует онлайн-решателю")
    print("="*60)
    
    # Вариант 1 (из онлайн-решения)
    costs1 = [
        [9, 5, 10, 7],
        [11, 8, 5, 6],
        [7, 6, 5, 4],
        [6, 4, 3, 2]
    ]
    supply1 = [70, 80, 90, 110]
    demand1 = [150, 40, 110, 50]
    
    cost1 = test_variant(costs1, supply1, demand1, 1, 1870)
    
    # Вариант 2 (из онлайн-решения)
    costs2 = [
        [5, 3, 4, 6, 4],
        [3, 4, 10, 5, 7],
        [4, 6, 9, 3, 4]
    ]
    supply2 = [40, 20, 40]
    demand2 = [25, 10, 20, 30, 15]
    
    cost2 = test_variant(costs2, supply2, demand2, 2, 340)
    
    # Вариант 3 (из онлайн-решения)
    costs3 = [
        [5, 4, 3, 2],
        [2, 3, 5, 6],
        [3, 2, 4, 3],
        [4, 1, 2, 4]
    ]
    supply3 = [120, 60, 80, 140]
    demand3 = [100, 140, 100, 60]
    
    cost3 = test_variant(costs3, supply3, demand3, 3, 800)
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ")
    print("="*60)
    
    results = [
        ("Вариант 1", cost1, 1870),
        ("Вариант 2", cost2, 340),
        ("Вариант 3", cost3, 800)
    ]
    
    success_count = 0
    for name, actual, expected in results:
        match = abs(actual - expected) < 0.01
        status = "✓" if match else "✗"
        if match:
            success_count += 1
        print(f"{status} {name}: получено {actual:.2f}, ожидалось {expected:.2f}")
    
    print(f"\nУспешно решено: {success_count} из 3 вариантов")
    
    if success_count == 3:
        print("\n✓ ВСЕ ЗАДАЧИ РЕШЕНЫ ПРАВИЛЬНО!")
    else:
        print("\n✗ ЕСТЬ ОШИБКИ В РЕШЕНИИ")


if __name__ == "__main__":
    main()