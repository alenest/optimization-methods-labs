import numpy as np

class ArtificialBasisSolver:
    """
    Реализация метода искусственного базиса (метод Чарнса) для решения задач ЛП
    """
    
    def __init__(self, obj_coeffs, constraints, rhs_values, constraint_types, is_min=True, M=10000, tol=1e-9):
        """
        Инициализация решателя
        """
        # Проверка входных данных
        if not obj_coeffs:
            raise ValueError("Целевая функция не может быть пустой.")
        if not constraints:
            raise ValueError("Ограничения не могут быть пустыми.")
        if len(rhs_values) != len(constraints):
            raise ValueError("Количество свободных членов должно равняться количеству ограничений.")
        if len(constraint_types) != len(constraints):
            raise ValueError("Количество типов ограничений должно равняться количеству ограничений.")
        
        # Сохраняем исходные данные
        self.orig_obj = list(obj_coeffs)
        self.constraints = [list(row) for row in constraints]
        self.rhs = list(rhs_values)
        self.constraint_types = constraint_types
        self.is_min = is_min
        self.M = M
        self.tol = tol
        
        # Размеры задачи
        self.n_orig = len(obj_coeffs)
        self.m = len(constraints)
        
        # Списки для дополнительных переменных
        self.slack_vars = []
        self.artificial_vars = []
        
        # История итераций
        self.history = []
        
        # Подготовка задачи к решению
        self._prepare_problem()
    
    def _get_basis_vars_by_row(self):
        """Получение списка базисных переменных по строкам"""
        basis_vars = []
        for i in range(self.m):
            found = -1
            for j in range(self.n_total):
                if np.isclose(self.tableau[i, j], 1.0, atol=1e-8):
                    # Проверяем, что в других строках этого столбца 0
                    is_basis = True
                    for k in range(self.m):
                        if k != i and not np.isclose(self.tableau[k, j], 0.0, atol=1e-8):
                            is_basis = False
                            break
                    if is_basis:
                        found = j
                        break
            basis_vars.append(found)
        return basis_vars

    def _prepare_problem(self):
        """Подготовка задачи: приведение к канонической форме и добавление искусственных переменных"""
        new_constraints = [row[:] for row in self.constraints]
        new_rhs = self.rhs[:]
        new_obj = self.orig_obj[:]
        
        slack_count = 0
        artificial_count = 0
        
        # Проверяем существующие базисные переменные
        existing_basis = self._find_existing_basis_vars()
        
        # Обрабатываем каждое ограничение
        for i in range(self.m):
            constraint_type = self.constraint_types[i]
            
            if constraint_type == '<=':
                # Для ограничения типа <= добавляем slack-переменную с коэффициентом +1
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(1.0) # Единица в текущей строке
                    else:
                        new_constraints[j].append(0.0) # Нули в других строках
                new_obj.append(0.0) # В целевой функции коэффициент 0
                slack_count += 1
                self.slack_vars.append(self.n_orig + slack_count - 1)
                
            elif constraint_type == '>=':
                # Для ограничения типа >= добавляем slack-переменную с коэффициентом -1
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(-1.0)
                    else:
                        new_constraints[j].append(0.0)
                new_obj.append(0.0)
                slack_count += 1
                slack_index = self.n_orig + slack_count - 1
                self.slack_vars.append(slack_index)
                
                # Для ограничений >= также добавляем искусственную переменную
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(1.0)
                    else:
                        new_constraints[j].append(0.0)
                new_obj.append(self.M) # Большой штраф M
                artificial_count += 1
                self.artificial_vars.append(self.n_orig + slack_count + artificial_count - 1)
                
            elif constraint_type == '=':
                # Проверяем, есть ли уже базисная переменная для этого ограничения
                if i in existing_basis:
                    # Уже есть базисная переменная - не добавляем искусственную
                    pass
                else:
                    # Для ограничения типа = добавляем искусственную переменную
                    for j in range(self.m):
                        if j == i:
                            new_constraints[j].append(1.0)
                        else:
                            new_constraints[j].append(0.0)
                    new_obj.append(self.M)
                    artificial_count += 1
                    self.artificial_vars.append(self.n_orig + slack_count + artificial_count - 1)
        
        # Сохраняем модифицированные данные
        self.obj = np.array(new_obj, dtype=float)
        self.A = np.array(new_constraints, dtype=float)
        self.b = np.array(new_rhs, dtype=float)
        self.n_total = self.n_orig + slack_count + artificial_count
        
        # Формируем начальную симплекс-таблицу
        self._init_tableau()
    
    def _find_existing_basis_vars(self):
        """Находит переменные, которые уже являются базисными"""
        basis_vars = {}  # key: row_index, value: variable_index
        
        # Проверяем каждую переменную
        for j in range(self.n_orig):
            column = [self.constraints[i][j] for i in range(self.m)]
            
            # Проверяем, является ли переменная базисной
            # Базисная переменная: один коэффициент = 1, остальные = 0
            ones_count = 0
            one_row = -1
            all_zeros = True
            
            for i, val in enumerate(column):
                if abs(val - 1.0) < self.tol:
                    ones_count += 1
                    one_row = i
                elif abs(val) > self.tol:
                    all_zeros = False
            
            # Если ровно один коэффициент = 1 и все остальные = 0, то это базисная переменная
            if ones_count == 1 and all_zeros:
                basis_vars[one_row] = j
        
        return basis_vars
    
    def _init_tableau(self):
        """Формирование начальной симплекс-таблицы"""
        self.tableau = np.zeros((self.m + 1, self.n_total + 1), dtype=float)
        
        # Заполняем матрицу ограничений
        self.tableau[:self.m, :self.n_total] = self.A
        self.tableau[:self.m, -1] = self.b
        
        # Заполняем целевую функцию
        if self.is_min:
            self.tableau[-1, :self.n_total] = self.obj
        else:
            self.tableau[-1, :self.n_total] = -self.obj
        
        # Корректируем целевую функцию для искусственных переменных
        self._adjust_objective_for_artificial_vars()
        
        self.history.append(self.tableau.copy())
    
    def _adjust_objective_for_artificial_vars(self):
        """Корректировка целевой функции для учета искусственных переменных"""
        if not self.artificial_vars:
            return
            
        # Создаем временную целевую функцию для минимизации искусственных переменных
        temp_obj = np.zeros(self.n_total + 1, dtype=float)
        
        # Суммируем строки с искусственными переменными
        for art_var in self.artificial_vars:
            basis_row = self._find_basis_row(art_var)
            if basis_row != -1:
                temp_obj += self.tableau[basis_row, :]
        
        # Вычитаем временную целевую функцию из основной
        self.tableau[-1, :] -= self.M * temp_obj
    
    def _pivot_operation(self, pivot_row, pivot_col):
        """Операция поворота вокруг разрешающего элемента"""
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element # Нормализация: Делим разрешающую строку на разрешающий элемент
        
        for r in range(self.tableau.shape[0]): # Исключение: Обнуляем столбец в других строках
            if r != pivot_row:
                factor = self.tableau[r, pivot_col]
                self.tableau[r, :] -= factor * self.tableau[pivot_row, :]
    
    def _get_basis_vars(self): # Ищем столбцы с ровно одной 1 и остальными 0
        """Получение списка базисных переменных"""
        basis_vars = []
        for j in range(self.n_total):
            col = self.tableau[:self.m, j]
            if np.sum(np.isclose(col, 1.0, atol=1e-8)) == 1 and np.sum(np.isclose(col, 0.0, atol=1e-8)) == self.m - 1:
                basis_vars.append(j)
        return basis_vars
    
    def _find_basis_row(self, basis_var): # Ищем строку, где в столбце переменной стоит 1
        """Нахождение строки, в которой переменная является базисной"""
        for i in range(self.m):
            if np.isclose(self.tableau[i, basis_var], 1.0, atol=1e-8):
                return i
        return -1
    
    def _is_optimal(self): # проверка оптимальности
        """Проверка оптимальности текущего решения"""
        last_row = self.tableau[-1, :-1]
        
        if self.is_min:
            return np.all(last_row >= -self.tol) # Все коэффициенты ≥ 0
        else:
            return np.all(last_row <= self.tol) # Все коэффициенты ≤ 0
    
    def _select_pivot_column(self): # выбор столбца для ввода в базис
        """Выбор разрешающего столбца"""
        last_row = self.tableau[-1, :-1]
        
        if self.is_min:
            min_val = np.min(last_row)
            if min_val >= -self.tol:
                return None # Решение оптимально
            return np.argmin(last_row) # Самый отрицательный
        else:
            max_val = np.max(last_row)
            if max_val <= self.tol:
                return None
            return np.argmax(last_row)
    
    def _select_pivot_row(self, pivot_col): # выбор строки для вывода из базиса
        """Выбор разрешающей строки"""
        ratios = np.full(self.m, np.inf)
        
        for i in range(self.m):
            a_ij = self.tableau[i, pivot_col]
            b_i = self.tableau[i, -1]
            
            if a_ij > self.tol:
                ratios[i] = b_i / a_ij # Симплекс-отношение
        
        if np.all(ratios == np.inf): # Минимальное положительное отношение
            return None
        
        return np.argmin(ratios)
    
    def _print_tableau(self, step):
        """Печать симплекс-таблицы"""
        print(f"\nСимплекс-таблица (шаг {step}):")
        
        # Заголовки столбцов
        headers = []
        for i in range(self.n_orig):
            headers.append(f"x{i+1}")
        
        # Добавляем slack переменные
        slack_headers = []
        for i in range(len(self.slack_vars)):
            slack_headers.append(f"s{i+1}")
        
        # Добавляем искусственные переменные
        artificial_headers = []
        for i in range(len(self.artificial_vars)):
            artificial_headers.append(f"a{i+1}")
        
        all_headers = headers + slack_headers + artificial_headers + ["RHS"]
        
        col_width = 10
        
        # Печать заголовков
        header_line = "базис | " + " | ".join(f"{h:^{col_width}}" for h in all_headers)
        print(header_line)
        print("-" * len(header_line))
        
        # Печать строк ограничений
        basis_vars = self._get_basis_vars()
        for i in range(self.m):
            basis_var_idx = -1
            for j in basis_vars:
                if np.isclose(self.tableau[i, j], 1.0, atol=1e-8):
                    basis_var_idx = j
                    break
            
            basis_name = "???"
            if basis_var_idx < self.n_orig:
                basis_name = f"x{basis_var_idx+1}"
            elif basis_var_idx in self.slack_vars:
                slack_idx = self.slack_vars.index(basis_var_idx)
                basis_name = f"s{slack_idx+1}"
            elif basis_var_idx in self.artificial_vars:
                art_idx = self.artificial_vars.index(basis_var_idx)
                basis_name = f"a{art_idx+1}"
            
            row_data = [f"{self.tableau[i, j]:{col_width}.3f}" for j in range(self.n_total + 1)]
            row_line = f"{basis_name:^5} | " + " | ".join(row_data)
            print(row_line)
        
        # Печать целевой строки
        obj_data = [f"{self.tableau[-1, j]:{col_width}.3f}" for j in range(self.n_total + 1)]
        obj_line = " F    | " + " | ".join(obj_data)
        print(obj_line)
    
    def solve(self, max_steps=100, verbose=False):  # Добавляем параметр verbose
        """Решение задачи методом искусственного базиса"""
        step = 0
        if verbose:  # Выводим только если verbose=True
            print("Начальная симплекс-таблица:")
            self._print_tableau(step)
        
        # Фаза I: Удаление искусственных переменных из базиса
        if self.artificial_vars:
            if verbose:
                print("\n--- ФАЗА I: Удаление искусственных переменных из базиса ---")
            step = self._phase1(step, max_steps//2, verbose)  # Передаем verbose
            
            # Проверяем, остались ли искусственные переменные в базисе
            basis_vars = self._get_basis_vars()
            artificial_in_basis = any(art_var in basis_vars for art_var in self.artificial_vars)
            
            if artificial_in_basis:
                # Проверяем значение искусственных переменных
                for art_var in self.artificial_vars:
                    if art_var in basis_vars:
                        row = self._find_basis_row(art_var)
                        if abs(self.tableau[row, -1]) > self.tol:
                            raise Exception("Задача не имеет допустимого решения.")
        
        # Фаза II: Оптимизация исходной целевой функции
        if verbose:
            print("\n--- ФАЗА II: Оптимизация исходной целевой функции ---")
        step = self._phase2(step, max_steps, verbose)  # Передаем verbose
        
        # Получаем решение
        solution = self._get_solution()
        objective_value = self._get_objective_value()
        
        return solution, objective_value, self.history

    def _phase1(self, start_step, max_steps, verbose=False):
        """Фаза I: удаление искусственных переменных из базиса"""
        step = start_step
        
        for _ in range(max_steps):
            # Проверяем, есть ли еще искусственные переменные в базисе
            basis_vars = self._get_basis_vars()
            artificial_in_basis = any(art_var in basis_vars for art_var in self.artificial_vars)
            
            if not artificial_in_basis:
                if verbose:
                    print(f"Фаза I завершена на шаге {step}: искусственные переменные удалены из базиса")
                break
            
            # Выбираем искусственную переменную для удаления
            pivot_col = None
            pivot_row = None  # Инициализируем переменную
            
            for art_var in self.artificial_vars:
                if art_var in basis_vars:
                    # Ищем ненулевой коэффициент в строке этой искусственной переменной
                    row = self._find_basis_row(art_var)
                    for j in range(self.n_total):
                        if j not in self.artificial_vars and abs(self.tableau[row, j]) > self.tol:
                            pivot_col = j
                            pivot_row = row  # Устанавливаем pivot_row
                            break
                    if pivot_col is not None:
                        break
            
            if pivot_col is None:
                # Не удалось найти подходящий столбец - проверяем значение искусственной переменной
                for art_var in self.artificial_vars:
                    if art_var in basis_vars:
                        row = self._find_basis_row(art_var)
                        if abs(self.tableau[row, -1]) > self.tol:
                            raise Exception("Задача не имеет допустимого решения.")
                        else:
                            # Искусственная переменная равна 0 - можно удалить из базиса
                            # Ищем любой ненулевой столбец для поворота
                            for j in range(self.n_total):
                                if abs(self.tableau[row, j]) > self.tol:
                                    pivot_col = j
                                    pivot_row = row  # Устанавливаем pivot_row
                                    break
                if pivot_col is None:
                    raise Exception("Не удалось удалить искусственные переменные из базиса")
            
            # Если pivot_row не установлен, находим его через _select_pivot_row
            if pivot_row is None:
                pivot_row = self._select_pivot_row(pivot_col)
                if pivot_row is None:
                    raise Exception("Задача неограничена")
            
            # Выполняем операцию поворота
            self._pivot_operation(pivot_row, pivot_col)
            
            step += 1
            if verbose:
                print(f"\nШаг {step} (Фаза I):")
                self._print_tableau(step)
            self.history.append(self.tableau.copy())
        
        return step

    def _phase2(self, start_step, max_steps, verbose=False):  # Добавляем verbose
        """Фаза II: оптимизация исходной целевой функции"""
        step = start_step
        
        for _ in range(max_steps):
            if self._is_optimal():
                if verbose:
                    print(f"Фаза II завершена на шаге {step}: достигнуто оптимальное решение")
                break
            
            pivot_col = self._select_pivot_column()
            if pivot_col is None:
                if verbose:
                    print(f"Фаза II завершена на шаге {step}: решение оптимально")
                break
            
            pivot_row = self._select_pivot_row(pivot_col)
            if pivot_row is None:
                raise Exception("Задача неограничена")
            
            self._pivot_operation(pivot_row, pivot_col)
            
            step += 1
            if verbose:  # Выводим только если verbose=True
                print(f"\nШаг {step} (Фаза II):")
                self._print_tableau(step)
            self.history.append(self.tableau.copy())
        
        return step
    
    def _get_solution(self): #  извлечение решения
        """Получение решения задачи"""
        solution = np.zeros(self.n_orig, dtype=float)
        basis_vars = self._get_basis_vars()
        
        for basis_var in basis_vars:
            if basis_var < self.n_orig:
                basis_row = self._find_basis_row(basis_var)
                solution[basis_var] = self.tableau[basis_row, -1]
        
        return solution
    
    def _get_objective_value(self): # значение целевой функции
        """Получение значения целевой функции"""
        if self.is_min:
            return -self.tableau[-1, -1]
        else:
            return self.tableau[-1, -1]


def test_individual_task():
    """Тестирование на индивидуальном задании"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ НА ИНДИВИДУАЛЬНОМ ЗАДАНИИ")
    print("=" * 70)
    
    # Данные из индивидуального задания
    obj_coeffs = [2, -3, 3, 1]
    constraints = [
        [2, 1, -1, 1],  # 2x1 + x2 - x3 + x4 = 24
        [1, 2, 2, 0],   # x1 + 2x2 + 2x3 <= 22
        [1, -1, 1, 0]    # x1 - x2 + x3 >= 10
    ]
    rhs_values = [24, 22, 10]
    constraint_types = ['=', '<=', '>=']
    
    print("Исходная задача:")
    print("Целевая функция: 2x1 - 3x2 + 3x3 + x4 → min")
    print("Ограничения:")
    print("  2x1 + x2 - x3 + x4 = 24")
    print("  x1 + 2x2 + 2x3 <= 22")
    print("  x1 - x2 + x3 >= 10")
    print("  x1, x2, x3, x4 >= 0")
    
    # Создаем решатель
    solver = ArtificialBasisSolver(
        obj_coeffs=obj_coeffs,
        constraints=constraints, 
        rhs_values=rhs_values,
        constraint_types=constraint_types,
        is_min=True,
        M=10000
    )
    
    # Решаем задачу
    try:
        solution, objective_value, history = solver.solve()
        
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ РЕШЕНИЯ")
        print("=" * 70)
        
        print("Оптимальное решение:")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.6f}")
        
        print(f"Значение целевой функции: {objective_value:.6f}")
        
        # Сравнение с ожидаемым результатом
        expected_solution = [34/3, 4/3, 0, 0]
        expected_objective = 56/3
        
        print("\nОжидаемый результат:")
        for i, val in enumerate(expected_solution):
            print(f"  x{i+1} = {val:.6f}")
        print(f"Значение целевой функции: {expected_objective:.6f}")
        
        # Проверка точности
        tolerance = 1e-6
        solution_match = np.allclose(solution, expected_solution, atol=tolerance)
        objective_match = abs(objective_value - expected_objective) < tolerance
        
        print("\nПроверка точности:")
        print(f"  Решение совпадает: {solution_match}")
        print(f"  Целевая функция совпадает: {objective_match}")
        
        if solution_match and objective_match:
            print("  ✓ ТЕСТ ПРОЙДЕН УСПЕШНО!")
        else:
            print("  ✗ ТЕСТ НЕ ПРОЙДЕН!")
            
    except Exception as e:
        print(f"Ошибка при решении: {e}")
        import traceback
        traceback.print_exc()


def main_console():
    """Консольный интерфейс для решения произвольных задач"""
    print("\n" + "=" * 70)
    print("РЕШЕНИЕ ПРОИЗВОЛЬНОЙ ЗАДАЧИ")
    print("=" * 70)
    
    print("МЕТОД ИСКУССТВЕННОГО БАЗИСА (ЧАРНСА)")
    print("Решение задач линейного программирования")
    print("=" * 50)
    
    # Ввод данных
    try:
        n = int(input("Количество переменных: "))
        m = int(input("Количество ограничений: "))
        
        print("\nВведите коэффициенты целевой функции (через пробел):")
        obj_coeffs = list(map(float, input().split()))
        
        constraints = []
        rhs_values = []
        constraint_types = []
        
        print("\nВведите ограничения:")
        for i in range(m):
            print(f"Ограничение {i+1}:")
            coeffs = list(map(float, input("Коэффициенты (через пробел): ").split()))
            const_type = input("Тип ограничения (<=, =, >=): ").strip()
            rhs = float(input("Правая часть: "))
            
            constraints.append(coeffs)
            constraint_types.append(const_type)
            rhs_values.append(rhs)
        
        is_min_input = input("Минимизация? (y/n, по умолчанию y): ").strip().lower()
        is_min = is_min_input != 'n'
        
        M_input = input("Коэффициент M (по умолчанию 10000): ").strip()
        M = float(M_input) if M_input else 10000
        
        # Решение задачи
        solver = ArtificialBasisSolver(
            obj_coeffs=obj_coeffs,
            constraints=constraints,
            rhs_values=rhs_values,
            constraint_types=constraint_types,
            is_min=is_min,
            M=M
        )
        
        solution, objective_value, history = solver.solve()
        
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТ:")
        print("=" * 50)
        
        print("Оптимальное решение:")
        for i, val in enumerate(solution):
            print(f"x{i+1} = {val:.6f}")
        
        print(f"Значение целевой функции: {objective_value:.6f}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запуск тестирования на индивидуальном задании
    test_individual_task()
    
    # Запуск консольного интерфейса
    main_console()