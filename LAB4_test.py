import numpy as np

class TwoPhaseSimplexSolver:
    """
    Правильная реализация двухфазного симплекс-метода
    """
    
    def __init__(self, c, A, b, constraint_types, is_min=True, tol=1e-9):
        # Сохраняем исходные данные
        self.original_c = np.array(c, dtype=float)
        self.original_A = np.array(A, dtype=float)
        self.original_b = np.array(b, dtype=float)
        self.constraint_types = constraint_types
        self.is_min = is_min
        self.tol = tol
        
        # Размеры задачи
        self.n_vars = len(c)
        self.n_constraints = len(A)
        
        # Списки дополнительных переменных
        self.slack_vars = []
        self.surplus_vars = []
        self.artificial_vars = []
        
        # История
        self.history = []
        
        # Подготовка задачи
        self._prepare_problem()
    
    def _prepare_problem(self):
        """Подготовка задачи - приведение к каноническому виду"""
        print("Подготовка задачи...")
        
        # Создаем расширенную матрицу ограничений
        new_A = []
        for i in range(self.n_constraints):
            new_row = self.original_A[i].copy()
            new_A.append(new_row)
        
        new_b = self.original_b.copy()
        
        # Счетчики дополнительных переменных
        var_count = self.n_vars
        
        # Обрабатываем каждое ограничение
        for i in range(self.n_constraints):
            constraint_type = self.constraint_types[i]
            
            if constraint_type == '<=':
                # Добавляем slack-переменную
                for j in range(self.n_constraints):
                    if j == i:
                        new_A[j].append(1.0)
                    else:
                        new_A[j].append(0.0)
                self.slack_vars.append(var_count)
                var_count += 1
                
            elif constraint_type == '>=':
                # Добавляем surplus-переменную
                for j in range(self.n_constraints):
                    if j == i:
                        new_A[j].append(-1.0)
                    else:
                        new_A[j].append(0.0)
                self.surplus_vars.append(var_count)
                var_count += 1
                
                # Добавляем искусственную переменную
                for j in range(self.n_constraints):
                    if j == i:
                        new_A[j].append(1.0)
                    else:
                        new_A[j].append(0.0)
                self.artificial_vars.append(var_count)
                var_count += 1
                
            elif constraint_type == '=':
                # Добавляем искусственную переменную
                for j in range(self.n_constraints):
                    if j == i:
                        new_A[j].append(1.0)
                    else:
                        new_A[j].append(0.0)
                self.artificial_vars.append(var_count)
                var_count += 1
        
        # Обновляем данные
        self.A = np.array(new_A, dtype=float)
        self.b = new_b
        self.n_total = var_count
        
        print(f"Исходных переменных: {self.n_vars}")
        print(f"Slack-переменных: {len(self.slack_vars)}")
        print(f"Surplus-переменных: {len(self.surplus_vars)}")
        print(f"Искусственных переменных: {len(self.artificial_vars)}")
        print(f"Всего переменных: {self.n_total}")
    
    def _create_phase1_tableau(self):
        """Создание таблицы для этапа 1"""
        # Создаем таблицу: строки ограничений + строка целевой функции
        tableau = np.zeros((self.n_constraints + 1, self.n_total + 1))
        
        # Заполняем матрицу ограничений
        tableau[:self.n_constraints, :self.n_total] = self.A
        tableau[:self.n_constraints, -1] = self.b
        
        # Целевая функция этапа 1: минимизация суммы искусственных переменных
        phase1_obj = np.zeros(self.n_total)
        for art_var in self.artificial_vars:
            phase1_obj[art_var] = 1.0
        
        tableau[-1, :self.n_total] = phase1_obj
        
        return tableau
    
    def _create_phase2_tableau(self, phase1_tableau):
        """Создание таблицы для этапа 2 на основе результата этапа 1"""
        # Находим базисные переменные из этапа 1
        basis = self._find_basis(phase1_tableau)
        
        # Создаем новую таблицу без искусственных переменных
        keep_cols = [i for i in range(self.n_total) if i not in self.artificial_vars]
        keep_cols.append(self.n_total)  # RHS столбец
        
        tableau = phase1_tableau[:self.n_constraints, :][:, keep_cols]
        
        # Добавляем строку для целевой функции этапа 2
        new_tableau = np.zeros((self.n_constraints + 1, len(keep_cols)))
        new_tableau[:self.n_constraints, :] = tableau
        
        # Устанавливаем целевую функцию исходной задачи
        for j in range(self.n_vars):
            if self.is_min:
                new_tableau[-1, j] = -self.original_c[j]
            else:
                new_tableau[-1, j] = self.original_c[j]
        
        # Корректируем целевую функцию для базисных переменных
        for i, basis_var in enumerate(basis):
            if basis_var < self.n_vars:  # Только исходные переменные
                row = i
                # Находим коэффициент в целевой функции для базисной переменной
                if self.is_min:
                    c_basis = self.original_c[basis_var]
                else:
                    c_basis = -self.original_c[basis_var]
                
                # Вычитаем из целевой строки c_basis * строку базисной переменной
                new_tableau[-1, :] -= c_basis * new_tableau[row, :]
        
        return new_tableau
    
    def _find_basis(self, tableau):
        """Нахождение базисных переменных в таблице"""
        basis = []
        for i in range(self.n_constraints):
            for j in range(self.n_total):
                if abs(tableau[i, j] - 1.0) < self.tol:
                    # Проверяем, что в столбце только один ненулевой элемент
                    is_basis = True
                    for k in range(self.n_constraints):
                        if k != i and abs(tableau[k, j]) > self.tol:
                            is_basis = False
                            break
                    if is_basis and j not in basis:
                        basis.append(j)
                        break
        return basis
    
    def _print_tableau(self, tableau, step, phase, basis=None):
        """Печать симплекс-таблицы"""
        if basis is None:
            basis = self._find_basis(tableau)
        
        print(f"\n=== Симплекс-таблица (Этап {phase}, Шаг {step}) ===")
        
        # Формируем заголовки
        headers = []
        for i in range(self.n_vars):
            headers.append(f"x{i+1}")
        
        # Slack переменные
        for i in range(len(self.slack_vars)):
            headers.append(f"s{i+1}")
        
        # Surplus переменные
        for i in range(len(self.surplus_vars)):
            headers.append(f"u{i+1}")
        
        # Искусственные переменные (только для этапа 1)
        if phase == 1:
            for i in range(len(self.artificial_vars)):
                headers.append(f"a{i+1}")
        
        headers.append("RHS")
        
        # Печать заголовков
        header_line = "Базис | " + " | ".join(f"{h:>8}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        
        # Печать строк ограничений
        for i in range(self.n_constraints):
            basis_var = basis[i] if i < len(basis) else -1
            
            basis_name = "???"
            if basis_var < self.n_vars:
                basis_name = f"x{basis_var+1}"
            elif basis_var in self.slack_vars:
                idx = self.slack_vars.index(basis_var)
                basis_name = f"s{idx+1}"
            elif basis_var in self.surplus_vars:
                idx = self.surplus_vars.index(basis_var)
                basis_name = f"u{idx+1}"
            elif basis_var in self.artificial_vars:
                idx = self.artificial_vars.index(basis_var)
                basis_name = f"a{idx+1}"
            
            row_data = []
            for j in range(tableau.shape[1]):
                row_data.append(f"{tableau[i, j]:8.3f}")
            row_line = f"{basis_name:>5} | " + " | ".join(row_data)
            print(row_line)
        
        # Печать строки оценок
        delta_line = " Delta | " + " | ".join(f"{tableau[-1, j]:8.3f}" for j in range(tableau.shape[1]))
        print(delta_line)
    
    def _pivot(self, tableau, pivot_row, pivot_col, basis):
        """Операция поворота"""
        pivot_element = tableau[pivot_row, pivot_col]
        
        # Нормализуем разрешающую строку
        tableau[pivot_row, :] /= pivot_element
        
        # Обнуляем столбец в других строках
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
        
        # Обновляем базис
        if pivot_row < len(basis):
            basis[pivot_row] = pivot_col
    
    def _solve_phase(self, tableau, phase, max_iterations=100):
        """Решение фазы симплекс-метода"""
        basis = self._find_basis(tableau)
        step = 0
        
        self._print_tableau(tableau, step, phase, basis)
        self.history.append(tableau.copy())
        
        for iteration in range(max_iterations):
            # Проверка оптимальности
            if phase == 1:
                # Для этапа 1: минимизация суммы искусственных переменных
                optimal = np.all(tableau[-1, :-1] >= -self.tol)
            else:
                # Для этапа 2: минимизация или максимизация исходной задачи
                if self.is_min:
                    optimal = np.all(tableau[-1, :-1] >= -self.tol)
                else:
                    optimal = np.all(tableau[-1, :-1] <= self.tol)
            
            if optimal:
                print(f"\n✓ Этап {phase} завершен на шаге {step}")
                return tableau, basis
            
            # Выбор разрешающего столбца
            if phase == 1:
                # Для этапа 1: выбираем столбец с наименьшей оценкой
                pivot_col = np.argmin(tableau[-1, :-1])
            else:
                # Для этапа 2: в зависимости от типа задачи
                if self.is_min:
                    pivot_col = np.argmin(tableau[-1, :-1])
                else:
                    pivot_col = np.argmax(tableau[-1, :-1])
            
            # Проверка неограниченности
            if np.all(tableau[:-1, pivot_col] <= self.tol):
                raise ValueError(f"Задача неограничена на этапе {phase}")
            
            # Выбор разрешающей строки
            ratios = []
            for i in range(self.n_constraints):
                if tableau[i, pivot_col] > self.tol:
                    ratio = tableau[i, -1] / tableau[i, pivot_col]
                    ratios.append((i, ratio))
            
            if not ratios:
                raise ValueError(f"Нет допустимых разрешающих строк на этапе {phase}")
            
            # Выбираем строку с наименьшим положительным отношением
            pivot_row, min_ratio = min(ratios, key=lambda x: x[1])
            
            # Выполняем поворот
            self._pivot(tableau, pivot_row, pivot_col, basis)
            
            step += 1
            self._print_tableau(tableau, step, phase, basis)
            self.history.append(tableau.copy())
        
        raise ValueError(f"Достигнуто максимальное количество итераций на этапе {phase}")
    
    def solve(self, max_iterations=100):
        """Решение задачи двухфазным симплекс-методом"""
        print("=" * 70)
        print("РЕШЕНИЕ ДВУХЭТАПНЫМ СИМПЛЕКС-МЕТОДОМ")
        print("=" * 70)
        
        # Этап 1: Минимизация суммы искусственных переменных
        print(f"\n--- ЭТАП 1: Минимизация суммы искусственных переменных ---")
        
        phase1_tableau = self._create_phase1_tableau()
        
        # Корректируем целевую функцию этапа 1 для базисных искусственных переменных
        for i, art_var in enumerate(self.artificial_vars):
            if i < self.n_constraints:
                phase1_tableau[-1, :] -= phase1_tableau[i, :]
        
        try:
            phase1_tableau, phase1_basis = self._solve_phase(phase1_tableau, 1, max_iterations)
        except Exception as e:
            raise ValueError(f"Ошибка на этапе 1: {e}")
        
        # Проверка допустимости
        phase1_objective = phase1_tableau[-1, -1]
        if abs(phase1_objective) > self.tol:
            raise ValueError("Задача не имеет допустимого решения")
        
        # Этап 2: Решение исходной задачи
        print(f"\n--- ЭТАП 2: Решение исходной задачи ---")
        
        phase2_tableau = self._create_phase2_tableau(phase1_tableau)
        
        try:
            phase2_tableau, phase2_basis = self._solve_phase(phase2_tableau, 2, max_iterations)
        except Exception as e:
            raise ValueError(f"Ошибка на этапе 2: {e}")
        
        # Получение решения
        solution = self._get_solution(phase2_tableau, phase2_basis)
        objective_value = self._get_objective_value(phase2_tableau)
        
        return solution, objective_value
    
    def _get_solution(self, tableau, basis):
        """Получение оптимального решения"""
        solution = np.zeros(self.n_vars)
        
        for i, basis_var in enumerate(basis):
            if basis_var < self.n_vars:
                solution[basis_var] = tableau[i, -1]
        
        return solution
    
    def _get_objective_value(self, tableau):
        """Получение значения целевой функции"""
        if self.is_min:
            return tableau[-1, -1]
        else:
            return -tableau[-1, -1]


def solve_individual_task():
    """Решение индивидуального задания"""
    print("=" * 70)
    print("РЕШЕНИЕ ИНДИВИДУАЛЬНОЙ ЗАДАЧИ")
    print("=" * 70)
    
    # Данные индивидуального задания
    obj_coeffs = [2, -3, 3, 1]
    constraints = [
        [2, 1, -1, 1],  # 2x1 + x2 - x3 + x4 = 24
        [1, 2, 2, 0],   # x1 + 2x2 + 2x3 <= 22
        [1, -1, 1, 0]    # x1 - x2 + x3 >= 10
    ]
    rhs_values = [24, 22, 10]
    constraint_types = ['=', '<=', '>=']
    
    print("ПРЯМАЯ ЗАДАЧА:")
    print("Целевая функция: 2x1 - 3x2 + 3x3 + x4 → min")
    print("Ограничения:")
    print("  2x1 + x2 - x3 + x4 = 24")
    print("  x1 + 2x2 + 2x3 <= 22") 
    print("  x1 - x2 + x3 >= 10")
    print("  x1, x2, x3, x4 >= 0")
    
    # Решение прямой задачи
    print("\n" + "=" * 50)
    print("РЕШЕНИЕ ПРЯМОЙ ЗАДАЧИ")
    print("=" * 50)
    
    try:
        solver = TwoPhaseSimplexSolver(
            c=obj_coeffs,
            A=constraints,
            b=rhs_values,
            constraint_types=constraint_types,
            is_min=True
        )
        
        solution, objective_value = solver.solve(max_iterations=100)
        
        print(f"\n✓ ОПТИМАЛЬНОЕ РЕШЕНИЕ:")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.6f}")
        print(f"Значение целевой функции: {objective_value:.6f}")
        
        # Сравнение с ожидаемым результатом
        print("\n" + "=" * 50)
        print("СРАВНЕНИЕ С ОЖИДАЕМЫМ РЕЗУЛЬТАТОМ")
        print("=" * 50)
        
        expected_solution = [34/3, 4/3, 0, 0]
        expected_objective = 56/3
        
        print("ОЖИДАЕМЫЙ РЕЗУЛЬТАТ (из PDF):")
        for i, val in enumerate(expected_solution):
            print(f"  x{i+1} = {val:.6f}")
        print(f"  F = {expected_objective:.6f}")
        
        tolerance = 1e-6
        solution_match = np.allclose(solution, expected_solution, atol=tolerance)
        objective_match = abs(objective_value - expected_objective) < tolerance
        
        if solution_match and objective_match:
            print("✓ ТЕСТ ПРОЙДЕН УСПЕШНО! Результаты совпадают с ожидаемыми.")
        else:
            print("✗ ТЕСТ НЕ ПРОЙДЕН! Результаты не совпадают с ожидаемыми.")
            print("Расхождение по решению:", np.abs(solution - expected_solution))
            print("Расхождение по целевой функции:", abs(objective_value - expected_objective))
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    solve_individual_task()