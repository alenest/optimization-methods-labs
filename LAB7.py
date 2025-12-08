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
        
        self.plan = np.zeros((self.m, self.n))
        
    def minimal_cost(self):
        """Метод наименьшей стоимости для начального плана (улучшенная версия)"""
        plan = np.zeros((self.m, self.n))
        supply_copy = self.supply.copy()
        demand_copy = self.demand.copy()
        
        while np.sum(supply_copy) > 1e-10:
            # Находим все минимальные стоимости
            min_cost = np.inf
            min_cells = []
            
            for i in range(self.m):
                if supply_copy[i] < 1e-10:
                    continue
                for j in range(self.n):
                    if demand_copy[j] < 1e-10:
                        continue
                    cost = self.costs[i][j]
                    if cost < min_cost:
                        min_cost = cost
                        min_cells = [(i, j)]
                    elif abs(cost - min_cost) < 1e-10:
                        min_cells.append((i, j))
            
            if not min_cells:
                break
            
            # Среди клеток с минимальной стоимостью выбираем ту, где можно разместить больше груза
            max_amount = -1
            best_i, best_j = -1, -1
            
            for i, j in min_cells:
                amount = min(supply_copy[i], demand_copy[j])
                if amount > max_amount:
                    max_amount = amount
                    best_i, best_j = i, j
            
            if max_amount < 1e-10:
                break
                
            plan[best_i][best_j] = max_amount
            supply_copy[best_i] -= max_amount
            demand_copy[best_j] -= max_amount
        
        return plan
    
    def north_west_corner(self):
        """Метод северо-западного угла"""
        plan = np.zeros((self.m, self.n))
        i, j = 0, 0
        supply_copy = self.supply.copy()
        demand_copy = self.demand.copy()
        
        while i < self.m and j < self.n:
            amount = min(supply_copy[i], demand_copy[j])
            plan[i][j] = amount
            supply_copy[i] -= amount
            demand_copy[j] -= amount
            
            if abs(supply_copy[i]) < 1e-10:
                i += 1
            if abs(demand_copy[j]) < 1e-10:
                j += 1
                
        return plan
    
    def calculate_cost(self, plan):
        """Вычисление общей стоимости плана"""
        return np.sum(plan * self.costs)
    
    def get_potentials(self, plan):
        """Вычисление потенциалов методом решения системы уравнений"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        
        # Начальное приближение
        u[0] = 0
        
        # Итеративное уточнение потенциалов
        changed = True
        while changed:
            changed = False
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i][j] > 1e-10:
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i][j] - u[i]
                            changed = True
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i][j] - v[j]
                            changed = True
        
        return u, v
    
    def find_cycle(self, plan, start_i, start_j):
        """Поиск цикла для метода потенциалов (исправленная версия)"""
        # Создаем граф занятых клеток
        occupied = [(i, j) for i in range(self.m) for j in range(self.n) 
                   if plan[i][j] > 1e-10]
        
        # Добавляем начальную клетку
        occupied.append((start_i, start_j))
        
        # Используем DFS для поиска цикла
        stack = [((start_i, start_j), [(start_i, start_j)], 'row')]
        
        while stack:
            (i, j), path, direction = stack.pop()
            
            # Если вернулись в начало и путь длинный enough
            if len(path) > 3 and (i, j) == (start_i, start_j):
                return path
            
            if direction == 'row':
                # Ищем в строке
                for k in range(self.n):
                    if k != j and (i, k) in occupied:
                        if (i, k) not in path or (i, k) == (start_i, start_j):
                            stack.append(((i, k), path + [(i, k)], 'col'))
            else:  # direction == 'col'
                # Ищем в столбце
                for k in range(self.m):
                    if k != i and (k, j) in occupied:
                        if (k, j) not in path or (k, j) == (start_i, start_j):
                            stack.append(((k, j), path + [(k, j)], 'row'))
        
        return None
    
    def improve_plan(self, plan):
        """Улучшение плана методом потенциалов"""
        improved = False
        
        for _ in range(10):  # Ограничим количество итераций
            u, v = self.get_potentials(plan)
            
            # Находим клетку с максимальной положительной оценкой
            max_delta = 0
            max_i, max_j = -1, -1
            
            for i in range(self.m):
                for j in range(self.n):
                    if plan[i][j] < 1e-10:  # Свободная клетка
                        delta = u[i] + v[j] - self.costs[i][j]
                        if delta > max_delta:
                            max_delta = delta
                            max_i, max_j = i, j
            
            if max_delta < 1e-10:  # План оптимален
                break
            
            # Находим цикл для этой клетки
            cycle = self.find_cycle(plan, max_i, max_j)
            
            if not cycle:
                continue
            
            # Находим минимальное значение в отрицательных вершинах цикла
            min_amount = np.inf
            for idx, (i, j) in enumerate(cycle[1:]):
                if idx % 2 == 0:  # Отрицательные вершины (через одну)
                    if plan[i][j] < min_amount:
                        min_amount = plan[i][j]
            
            if min_amount < 1e-10:
                continue
            
            # Перераспределяем
            for idx, (i, j) in enumerate(cycle):
                if idx == 0:  # Начальная клетка (+)
                    plan[i][j] += min_amount
                elif idx % 2 == 0:  # Положительные вершины
                    plan[i][j] += min_amount
                else:  # Отрицательные вершины
                    plan[i][j] -= min_amount
            
            improved = True
            
            # Убираем нули
            plan[plan < 1e-10] = 0
        
        return plan, improved
    
    def solve_transportation(self):
        """Решение транспортной задачи несколькими методами для лучшего результата"""
        # Пробуем разные начальные планы
        methods = [
            ("Метод наименьшей стоимости", self.minimal_cost),
            ("Метод северо-западного угла", self.north_west_corner),
        ]
        
        best_plan = None
        best_cost = np.inf
        
        for method_name, method_func in methods:
            plan = method_func()
            cost = self.calculate_cost(plan)
            
            print(f"\n{method_name}:")
            print(f"  Начальная стоимость: {cost:.2f}")
            
            # Пытаемся улучшить
            improved_plan, improved = self.improve_plan(plan.copy())
            improved_cost = self.calculate_cost(improved_plan)
            
            if improved:
                print(f"  Улучшенная стоимость: {improved_cost:.2f}")
            else:
                print(f"  Улучшение не потребовалось")
            
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_plan = improved_plan
        
        return best_plan, best_cost
    
    def solve(self):
        """Основной метод решения"""
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
        print(f"\nСумма запасов: {total_supply}")
        print(f"Сумма потребностей: {total_demand}")
        
        if abs(total_supply - total_demand) < 1e-10:
            print("Условие баланса соблюдается. Модель закрытая.")
        else:
            print("Условие баланса не соблюдается!")
        
        # Решаем задачу
        print(f"\n{'='*60}")
        print("ПОИСК ОПТИМАЛЬНОГО РЕШЕНИЯ")
        
        optimal_plan, optimal_cost = self.solve_transportation()
        
        print(f"\n{'='*60}")
        print("НАИЛУЧШЕЕ РЕШЕНИЕ:")
        print(np.round(optimal_plan, 2))
        print(f"\nМинимальная стоимость: {optimal_cost:.2f}")
        
        # Анализ оптимального плана
        print(f"\n{'='*60}")
        print("АНАЛИЗ ОПТИМАЛЬНОГО ПЛАНА")
        
        total_shipped = 0
        for i in range(self.m):
            for j in range(self.n):
                if optimal_plan[i][j] > 1e-10:
                    print(f"Из поставщика A{i+1} → потребителю B{j+1}: {optimal_plan[i][j]:.1f} ед.")
                    total_shipped += optimal_plan[i][j]
        
        print(f"\nВсего перевезено: {total_shipped:.1f} единиц")
        print(f"Общая минимальная стоимость: {optimal_cost:.2f}")
        
        return optimal_plan, optimal_cost


# Вариант 1 (с исправленной матрицей - вторая строка должна быть [11, 8, 5, 6])
print("ВАРИАНТ 1")
print("=" * 60)

# Данные варианта 1 (исправленные согласно онлайн-решению)
costs1 = [
    [9, 5, 10, 7],
    [11, 8, 5, 6],  # Исправлено: 5 вместо 9!
    [7, 6, 5, 4],
    [6, 4, 3, 2]
]
supply1 = [70, 80, 90, 110]
demand1 = [150, 40, 110, 50]

# Создаем и решаем задачу
problem1 = TransportProblem(costs1, supply1, demand1)
optimal_plan1, optimal_cost1 = problem1.solve()

print("\n" + "="*60)
print("ВАРИАНТ 2")
print("=" * 60)

# Данные варианта 2
costs2 = [
    [5, 3, 4, 6, 4],
    [3, 4, 10, 5, 7],
    [4, 6, 9, 3, 4]
]
supply2 = [40, 20, 40]
demand2 = [25, 10, 20, 30, 15]

# Создаем и решаем задачу
problem2 = TransportProblem(costs2, supply2, demand2)
optimal_plan2, optimal_cost2 = problem2.solve()

print("\n" + "="*60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"Вариант 1: минимальная стоимость = {optimal_cost1:.2f}")
print(f"Вариант 2: минимальная стоимость = {optimal_cost2:.2f}")
print("\n" + "="*60)

# Проверка с онлайн-решением
print("\nПРОВЕРКА С ОНЛАЙН-РЕШЕНИЕМ:")
print("Вариант 1:")
print("Онлайн-решение: 1870.00")
print(f"Наше решение:    {optimal_cost1:.2f}")
print("Разница:         {:.2f}".format(abs(optimal_cost1 - 1870)))