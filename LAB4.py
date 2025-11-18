import numpy as np

class TwoPhaseSimplexSolver:
    """
    Исправленная реализация двухэтапного симплекс-метода
    """
    
    def __init__(self, obj_coeffs, constraints, rhs_values, constraint_types, is_min=True, tol=1e-9):
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
        self.orig_obj = np.array(obj_coeffs, dtype=float)
        self.constraints = [list(row) for row in constraints]
        self.rhs = list(rhs_values)
        self.constraint_types = constraint_types
        self.is_min = is_min
        self.tol = tol
        
        # Размеры задачи
        self.n_orig = len(obj_coeffs)
        self.m = len(constraints)
        
        # Списки для дополнительных переменных
        self.slack_vars = []
        self.surplus_vars = []
        self.artificial_vars = []
        
        # История
        self.history = []
        self.phase1_history = []
        
        # Подготовка задачи
        self._prepare_problem()
    
    def _prepare_problem(self):
        """Подготовка задачи к решению - приведение к каноническому виду"""
        print("Подготовка задачи...")
        
        # Копируем исходные данные
        new_constraints = [row[:] for row in self.constraints]
        new_rhs = self.rhs[:]
        
        # Счетчики дополнительных переменных
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        # Обрабатываем каждое ограничение
        for i in range(self.m):
            constraint_type = self.constraint_types[i]
            
            if constraint_type == '<=':
                # Добавляем slack-переменную
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(1.0)
                    else:
                        new_constraints[j].append(0.0)
                slack_count += 1
                self.slack_vars.append(self.n_orig + slack_count - 1)
                
            elif constraint_type == '>=':
                # Добавляем surplus-переменную
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(-1.0)
                    else:
                        new_constraints[j].append(0.0)
                surplus_count += 1
                surplus_index = self.n_orig + slack_count + surplus_count - 1
                self.surplus_vars.append(surplus_index)
                
                # Добавляем искусственную переменную
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(1.0)
                    else:
                        new_constraints[j].append(0.0)
                artificial_count += 1
                artificial_index = self.n_orig + slack_count + surplus_count + artificial_count - 1
                self.artificial_vars.append(artificial_index)
                
            elif constraint_type == '=':
                # Добавляем искусственную переменную
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(1.0)
                    else:
                        new_constraints[j].append(0.0)
                artificial_count += 1
                artificial_index = self.n_orig + slack_count + surplus_count + artificial_count - 1
                self.artificial_vars.append(artificial_index)
        
        # Обновляем данные
        self.A = np.array(new_constraints, dtype=float)
        self.b = np.array(new_rhs, dtype=float)
        self.n_total = self.n_orig + slack_count + surplus_count + artificial_count
        
        print(f"Добавлено slack-переменных: {len(self.slack_vars)}")
        print(f"Добавлено surplus-переменных: {len(self.surplus_vars)}")
        print(f"Добавлено искусственных переменных: {len(self.artificial_vars)}")
        print(f"Общее количество переменных: {self.n_total}")
        
        # Создаем целевую функцию для этапа 1
        self.phase1_obj = np.zeros(self.n_total, dtype=float)
        for art_var in self.artificial_vars:
            self.phase1_obj[art_var] = 1.0
    
    def _init_tableau_phase1(self):
        """Инициализация симплекс-таблицы для этапа 1"""
        print("Инициализация таблицы для этапа 1...")
        
        # Создаем таблицу: строки ограничений + строка целевой функции
        self.tableau = np.zeros((self.m + 1, self.n_total + 1), dtype=float)
        
        # Заполняем матрицу ограничений
        self.tableau[:self.m, :self.n_total] = self.A
        self.tableau[:self.m, -1] = self.b
        
        # Заполняем целевую функцию этапа 1
        self.tableau[-1, :self.n_total] = self.phase1_obj
        
        # Корректируем для базисных искусственных переменных
        self._adjust_objective_for_basis_phase1()
        
        self.phase1_history.append(self.tableau.copy())
    
    def _adjust_objective_for_basis_phase1(self):
        """Корректировка целевой функции для базисных искусственных переменных"""
        for art_var in self.artificial_vars:
            basis_row = self._find_basis_row(art_var)
            if basis_row != -1:
                self.tableau[-1, :] -= self.tableau[basis_row, :] * self.phase1_obj[art_var]
    
    def _find_basis_row(self, var_idx):
        """Находит строку, где переменная является базисной"""
        for i in range(self.m):
            if abs(self.tableau[i, var_idx] - 1.0) < self.tol:
                # Проверяем, что остальные элементы в столбце близки к 0
                is_basis = True
                for k in range(self.m):
                    if k != i and abs(self.tableau[k, var_idx]) > self.tol:
                        is_basis = False
                        break
                if is_basis:
                    return i
        return -1
    
    def _get_basis_vars(self):
        """Получает список базисных переменных"""
        basis_vars = []
        for j in range(self.n_total):
            col = self.tableau[:self.m, j]
            ones_count = np.sum(np.isclose(col, 1.0, atol=1e-8))
            zeros_count = np.sum(np.isclose(col, 0.0, atol=1e-8))
            
            if ones_count == 1 and zeros_count == self.m - 1:
                basis_vars.append(j)
        return basis_vars
    
    def _pivot_operation(self, pivot_row, pivot_col):
        """Операция поворота"""
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        if abs(pivot_element) < self.tol:
            raise Exception("Нулевой разрешающий элемент")
        
        # Нормализуем разрешающую строку
        self.tableau[pivot_row, :] /= pivot_element
        
        # Обнуляем столбец в других строках
        for r in range(self.tableau.shape[0]):
            if r != pivot_row:
                factor = self.tableau[r, pivot_col]
                self.tableau[r, :] -= factor * self.tableau[pivot_row, :]
    
    def _calculate_deltas_phase1(self):
        """Вычисление дельт для этапа 1"""
        deltas = np.zeros(self.n_total + 1, dtype=float)
        basis_vars = self._get_basis_vars()
        
        for j in range(self.n_total + 1):
            sum_val = 0.0
            for i, basis_var in enumerate(basis_vars):
                c_val = self.phase1_obj[basis_var]
                sum_val += c_val * self.tableau[i, j]
            
            if j < self.n_total:
                deltas[j] = sum_val - self.phase1_obj[j]
            else:
                deltas[j] = sum_val
        
        return deltas
    
    def _calculate_deltas_phase2(self):
        """Вычисление дельт для этапа 2"""
        deltas = np.zeros(self.n_total + 1, dtype=float)
        basis_vars = self._get_basis_vars()
        
        for j in range(self.n_total + 1):
            sum_val = 0.0
            for i, basis_var in enumerate(basis_vars):
                if basis_var < self.n_orig:
                    c_val = self.orig_obj[basis_var] if self.is_min else -self.orig_obj[basis_var]
                else:
                    c_val = 0.0
                sum_val += c_val * self.tableau[i, j]
            
            if j < self.n_total:
                if j < self.n_orig:
                    c_j = self.orig_obj[j] if self.is_min else -self.orig_obj[j]
                else:
                    c_j = 0.0
                deltas[j] = sum_val - c_j
            else:
                deltas[j] = sum_val
        
        return deltas
    
    def _print_tableau(self, step, phase):
        """Печать симплекс-таблицы"""
        print(f"\n=== Симплекс-таблица (Этап {phase}, Шаг {step}) ===")
        
        # Формируем заголовки
        headers = []
        for i in range(self.n_orig):
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
                basis_name = f"s{self.slack_vars.index(basis_var_idx)+1}"
            elif basis_var_idx in self.surplus_vars:
                basis_name = f"u{self.surplus_vars.index(basis_var_idx)+1}"
            elif basis_var_idx in self.artificial_vars:
                basis_name = f"a{self.artificial_vars.index(basis_var_idx)+1}"
            
            row_data = [f"{self.tableau[i, j]:8.3f}" for j in range(self.tableau.shape[1])]
            row_line = f"{basis_name:>5} | " + " | ".join(row_data)
            print(row_line)
        
        # Печать дельт
        if phase == 1:
            deltas = self._calculate_deltas_phase1()
        else:
            deltas = self._calculate_deltas_phase2()
        
        delta_line = " Delta | " + " | ".join(f"{d:8.3f}" for d in deltas)
        print(delta_line)
        
        # Выводим информацию о разрешающем элементе, если есть отрицательные дельты
        if phase == 2 and step > 0:
            if self.is_min:
                negative_deltas = [j for j in range(self.n_total) if deltas[j] < -self.tol]
                if negative_deltas:
                    pivot_col = negative_deltas[0]
                    print(f"Разрешающий столбец: {headers[pivot_col]} (Δ={deltas[pivot_col]:.3f})")
                    
                    # Находим разрешающую строку
                    ratios = []
                    for i in range(self.m):
                        if self.tableau[i, pivot_col] > self.tol:
                            ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                            ratios.append((i, ratio))
                    
                    if ratios:
                        pivot_row = min(ratios, key=lambda x: x[1])[0]
                        print(f"Разрешающая строка: {pivot_row+1}")
                        print(f"Разрешающий элемент: {self.tableau[pivot_row, pivot_col]:.3f}")
    
    def solve(self, max_steps=20):
        """Решение задачи двухэтапным симплекс-методом"""
        print("=" * 70)
        print("РЕШЕНИЕ ДВУХЭТАПНЫМ СИМПЛЕКС-МЕТОДОМ")
        print("=" * 70)
        
        # Этап 1: Минимизация суммы искусственных переменных
        print(f"\n--- ЭТАП 1: Минимизация суммы искусственных переменных ---")
        
        self._init_tableau_phase1()
        step = 0
        self._print_tableau(step, 1)
        
        for step in range(1, max_steps + 1):
            # Проверка оптимальности для этапа 1
            deltas = self._calculate_deltas_phase1()
            if np.all(deltas[:-1] >= -self.tol):
                print(f"\n✓ Этап 1 завершен на шаге {step-1}")
                break
            
            # Выбор разрешающего столбца (минимальная дельта)
            pivot_col = np.argmin(deltas[:-1])
            
            # Выбор разрешающей строки
            ratios = np.full(self.m, np.inf)
            for i in range(self.m):
                if self.tableau[i, pivot_col] > self.tol:
                    ratios[i] = self.tableau[i, -1] / self.tableau[i, pivot_col]
            
            if np.all(ratios == np.inf):
                raise Exception("Задача неограничена на этапе 1")
            
            pivot_row = np.argmin(ratios)
            
            # Поворот
            self._pivot_operation(pivot_row, pivot_col)
            self._print_tableau(step, 1)
            self.phase1_history.append(self.tableau.copy())
        
        # Проверка допустимости
        optimal_value_phase1 = -self.tableau[-1, -1]
        if optimal_value_phase1 > self.tol:
            raise Exception("Задача не имеет допустимого решения")
        
        # Этап 2: Решение исходной задачи
        print(f"\n--- ЭТАП 2: Решение исходной задачи ---")
        
        # Удаляем искусственные переменные из таблицы
        if self.artificial_vars:
            keep_cols = [i for i in range(self.n_total) if i not in self.artificial_vars]
            keep_cols.append(self.n_total)  # RHS
            self.tableau = self.tableau[:, keep_cols]
            self.n_total = len(keep_cols) - 1
        
        # Устанавливаем целевую функцию исходной задачи
        self.tableau[-1, :self.n_total] = 0
        for j in range(min(self.n_orig, self.n_total)):
            if self.is_min:
                self.tableau[-1, j] = self.orig_obj[j]
            else:
                self.tableau[-1, j] = -self.orig_obj[j]
        
        # Корректируем для базиса
        self._adjust_objective_for_basis_phase2()
        
        step = 0
        self._print_tableau(step, 2)
        
        for step in range(1, max_steps + 1):
            # Проверка оптимальности для этапа 2
            deltas = self._calculate_deltas_phase2()
            
            if self.is_min:
                optimal = np.all(deltas[:-1] >= -self.tol)
            else:
                optimal = np.all(deltas[:-1] <= self.tol)
            
            if optimal:
                print(f"\n✓ Этап 2 завершен на шаге {step-1}")
                break
            
            # Выбор разрешающего столбца
            if self.is_min:
                negative_deltas = [j for j in range(self.n_total) if deltas[j] < -self.tol]
                if not negative_deltas:
                    break
                pivot_col = negative_deltas[0]
            else:
                positive_deltas = [j for j in range(self.n_total) if deltas[j] > self.tol]
                if not positive_deltas:
                    break
                pivot_col = positive_deltas[0]
            
            # Выбор разрешающей строки
            ratios = np.full(self.m, np.inf)
            for i in range(self.m):
                if self.tableau[i, pivot_col] > self.tol:
                    ratios[i] = self.tableau[i, -1] / self.tableau[i, pivot_col]
            
            if np.all(ratios == np.inf):
                raise Exception("Задача неограничена на этапе 2")
            
            pivot_row = np.argmin(ratios)
            
            # Поворот
            self._pivot_operation(pivot_row, pivot_col)
            self._print_tableau(step, 2)
            self.history.append(self.tableau.copy())
        
        # Получение решения
        solution = self._get_solution()
        objective_value = self._get_objective_value()
        
        return solution, objective_value
    
    def _adjust_objective_for_basis_phase2(self):
        """Корректировка целевой функции для этапа 2"""
        basis_vars = self._get_basis_vars()
        
        for basis_var in basis_vars:
            if basis_var < self.n_orig:  # Только исходные переменные
                basis_row = self._find_basis_row(basis_var)
                if basis_row != -1:
                    if self.is_min:
                        c_basis = self.orig_obj[basis_var]
                    else:
                        c_basis = -self.orig_obj[basis_var]
                    
                    self.tableau[-1, :] -= c_basis * self.tableau[basis_row, :]
    
    def _get_solution(self):
        """Получение решения"""
        solution = np.zeros(self.n_orig, dtype=float)
        basis_vars = self._get_basis_vars()
        
        for basis_var in basis_vars:
            if basis_var < self.n_orig:
                basis_row = self._find_basis_row(basis_var)
                if basis_row != -1:
                    solution[basis_var] = self.tableau[basis_row, -1]
        
        return solution
    
    def _get_objective_value(self):
        """Получение значения целевой функции"""
        if self.is_min:
            return self.tableau[-1, -1]
        else:
            return -self.tableau[-1, -1]


class DualProblemBuilder:
    """Построитель двойственной задачи"""
    
    @staticmethod
    def build_dual_problem(obj_coeffs, constraints, rhs_values, constraint_types, is_min=True):
        """
        Построение двойственной задачи
        """
        n_vars = len(obj_coeffs)
        n_constraints = len(constraints)
        
        # Целевая функция двойственной задачи
        dual_obj = rhs_values
        
        # Матрица ограничений двойственной задачи (транспонированная)
        dual_constraints = []
        for j in range(n_vars):
            constraint_row = []
            for i in range(n_constraints):
                constraint_row.append(constraints[i][j])
            dual_constraints.append(constraint_row)
        
        # Правые части двойственной задачи
        dual_rhs = obj_coeffs
        
        # Типы ограничений двойственной задачи
        dual_constraint_types = []
        for i in range(n_vars):
            # Для задачи минимизации двойственная - максимизация
            # Ограничения двойственной: <= для неотрицательных переменных
            dual_constraint_types.append('<=' if is_min else '>=')
        
        # Тип оптимизации двойственной задачи
        dual_is_min = not is_min
        
        return dual_obj, dual_constraints, dual_rhs, dual_constraint_types, dual_is_min


def solve_individual_task():
    """Решение индивидуального задания"""
    print("=" * 70)
    print("РЕШЕНИЕ ИНДИВИДУАЛЬНОЙ ЗАДАЧИ")
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
        solver_direct = TwoPhaseSimplexSolver(
            obj_coeffs=obj_coeffs,
            constraints=constraints,
            rhs_values=rhs_values,
            constraint_types=constraint_types,
            is_min=True
        )
        
        solution_direct, objective_direct = solver_direct.solve(max_steps=10)
        
        print(f"\n✓ ОПТИМАЛЬНОЕ РЕШЕНИЕ ПРЯМОЙ ЗАДАЧИ:")
        for i, val in enumerate(solution_direct):
            print(f"  x{i+1} = {val:.6f}")
        print(f"Значение целевой функции: {objective_direct:.6f}")
        
    except Exception as e:
        print(f"✗ Ошибка при решении прямой задачи: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Построение и решение двойственной задачи
    print("\n" + "=" * 50)
    print("ПОСТРОЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ")
    print("=" * 50)
    
    try:
        dual_obj, dual_constraints, dual_rhs, dual_constraint_types, dual_is_min = \
            DualProblemBuilder.build_dual_problem(
                obj_coeffs, constraints, rhs_values, constraint_types, is_min=True
            )
        
        print("ДВОЙСТВЕННАЯ ЗАДАЧА:")
        print(f"Целевая функция: ", end="")
        obj_terms = []
        for i, coeff in enumerate(dual_obj):
            if abs(coeff) > 1e-10:
                sign = "+" if coeff >= 0 else "-"
                obj_terms.append(f"{sign} {abs(coeff):.1f}y{i+1}")
        obj_str = " ".join(obj_terms).lstrip("+ ")
        print(f"{obj_str} → {'min' if dual_is_min else 'max'}")
        
        print("Ограничения:")
        for i in range(len(dual_constraints)):
            constr_terms = []
            for j, coeff in enumerate(dual_constraints[i]):
                if abs(coeff) > 1e-10:
                    sign = "+" if coeff >= 0 else "-"
                    constr_terms.append(f"{sign} {abs(coeff):.1f}y{j+1}")
            constr_str = " ".join(constr_terms).lstrip("+ ")
            if not constr_str:
                constr_str = "0"
            print(f"  {constr_str} {dual_constraint_types[i]} {dual_rhs[i]}")
        
        print("\n" + "=" * 50)
        print("РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ")
        print("=" * 50)
        
        solver_dual = TwoPhaseSimplexSolver(
            obj_coeffs=dual_obj,
            constraints=dual_constraints,
            rhs_values=dual_rhs,
            constraint_types=dual_constraint_types,
            is_min=dual_is_min
        )
        
        solution_dual, objective_dual = solver_dual.solve(max_steps=10)
        
        print(f"\n✓ ОПТИМАЛЬНОЕ РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ:")
        for i, val in enumerate(solution_dual):
            print(f"  y{i+1} = {val:.6f}")
        print(f"Значение целевой функции: {objective_dual:.6f}")
        
    except Exception as e:
        print(f"✗ Ошибка при решении двойственной задачи: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Сравнение результатов
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    print("Прямая задача:")
    print(f"  F = {objective_direct:.6f}")
    print("Двойственная задача:")
    print(f"  G = {objective_dual:.6f}")
    
    tolerance = 1e-6
    if abs(objective_direct - objective_dual) < tolerance:
        print("✓ РЕШЕНИЕ ВЕРНОЕ! Значения целевых функций совпадают.")
    else:
        print("✗ РЕШЕНИЕ НЕВЕРНОЕ! Значения целевых функций не совпадают.")
    
    # Сравнение с ожидаемым результатом из PDF
    expected_solution = [34/3, 4/3, 0, 0]
    expected_objective = 56/3
    
    print(f"\nОЖИДАЕМЫЙ РЕЗУЛЬТАТ (из PDF):")
    for i, val in enumerate(expected_solution):
        print(f"  x{i+1} = {val:.6f}")
    print(f"  F = {expected_objective:.6f}")
    
    solution_match = np.allclose(solution_direct, expected_solution, atol=tolerance)
    objective_match = abs(objective_direct - expected_objective) < tolerance
    
    if solution_match and objective_match:
        print("✓ ТЕСТ ПРОЙДЕН УСПЕШНО! Результаты совпадают с ожидаемыми.")
    else:
        print("✗ ТЕСТ НЕ ПРОЙДЕН! Результаты не совпадают с ожидаемыми.")


if __name__ == "__main__":
    solve_individual_task()