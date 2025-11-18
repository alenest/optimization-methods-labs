import numpy as np
import os

class TwoPhaseSimplexSolver:
    """
    Реализация двухэтапного симплекс-метода (метод Данцига)
    """
    
    def __init__(self, obj_coeffs, constraints, rhs_values, constraint_types, is_min=True, tol=1e-9):
        """
        Инициализация решателя методом Данцига
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
        self.orig_obj = list(obj_coeffs)  # Исходная целевая функция
        self.constraints = [list(row) for row in constraints]  # Ограничения
        self.rhs = list(rhs_values)  # Правые части ограничений
        self.constraint_types = constraint_types  # Типы ограничений
        self.is_min = is_min  # Минимизация или максимизация
        self.tol = tol  # Допуск для численных сравнений
        
        # Размеры задачи
        self.n_orig = len(obj_coeffs)  # Количество исходных переменных
        self.m = len(constraints)  # Количество ограничений
        
        # Списки для дополнительных переменных
        self.slack_vars = []  # Slack-переменные
        self.artificial_vars = []  # Искусственные переменные
        
        # История итераций
        self.history = []  # История симплекс-таблиц
        self.phase1_history = []  # История первого этапа
        
        # Подготовка задачи к решению
        self._prepare_problem()  # Приведение к канонической форме
    
    def _prepare_problem(self):
        """Подготовка задачи: приведение к канонической форме и добавление искусственных переменных"""
        new_constraints = [row[:] for row in self.constraints]  # Копируем ограничения
        new_rhs = self.rhs[:]  # Копируем правые части
        new_obj = self.orig_obj[:]  # Копируем целевую функцию
        
        slack_count = 0  # Счетчик slack-переменных
        artificial_count = 0  # Счетчик искусственных переменных
        
        # Проверяем существующие базисные переменные
        existing_basis = self._find_existing_basis_vars()
        
        # Обрабатываем каждое ограничение
        for i in range(self.m):
            constraint_type = self.constraint_types[i]
            
            if constraint_type == '<=':
                # Для ограничения типа <= добавляем slack-переменную с коэффициентом +1
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(1.0)  # Единица в текущей строке
                    else:
                        new_constraints[j].append(0.0)  # Нули в других строках
                new_obj.append(0.0)  # В целевой функции коэффициент 0
                slack_count += 1
                # Сохраняем индекс добавленной переменной
                current_index = len(new_obj) - 1
                self.slack_vars.append(current_index)
                
            elif constraint_type == '>=':
                # Для ограничения типа >= добавляем slack-переменную с коэффициентом -1
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(-1.0)  # Минус единица в текущей строке
                    else:
                        new_constraints[j].append(0.0)  # Нули в других строках
                new_obj.append(0.0)  # В целевой функции коэффициент 0
                slack_count += 1
                # Сохраняем индекс добавленной переменной
                current_index = len(new_obj) - 1
                self.slack_vars.append(current_index)
                
                # Для ограничений >= также добавляем искусственную переменную
                for j in range(self.m):
                    if j == i:
                        new_constraints[j].append(1.0)  # Единица в текущей строке
                    else:
                        new_constraints[j].append(0.0)  # Нули в других строках
                new_obj.append(0.0)  # В исходной целевой функции коэффициент 0
                artificial_count += 1
                # Сохраняем индекс добавленной переменной
                current_index = len(new_obj) - 1
                self.artificial_vars.append(current_index)
                
            elif constraint_type == '=':
                # Проверяем, есть ли уже базисная переменная для этого ограничения
                if i in existing_basis:
                    # Уже есть базисная переменная - не добавляем искусственную
                    pass
                else:
                    # Для ограничения типа = добавляем искусственную переменную
                    for j in range(self.m):
                        if j == i:
                            new_constraints[j].append(1.0)  # Единица в текущей строке
                        else:
                            new_constraints[j].append(0.0)  # Нули в других строках
                    new_obj.append(0.0)  # В исходной целевой функции коэффициент 0
                    artificial_count += 1
                    # Сохраняем индекс добавленной переменной
                    current_index = len(new_obj) - 1
                    self.artificial_vars.append(current_index)
        
        # Сохраняем модифицированные данные
        self.obj = np.array(new_obj, dtype=float)  # Исходная целевая функция с нулями для доп. переменных
        self.A = np.array(new_constraints, dtype=float)  # Матрица ограничений
        self.b = np.array(new_rhs, dtype=float)  # Вектор правых частей
        self.n_total = len(new_obj)  # Общее количество переменных
        
        # Создаем целевую функцию для первого этапа (сумма искусственных переменных)
        self.phase1_obj = np.zeros(self.n_total, dtype=float)
        for art_var in self.artificial_vars:
            self.phase1_obj[art_var] = 1.0  # Коэффициент 1 для искусственных переменных
        
        print(f"Добавлено slack-переменных: {len(self.slack_vars)}")
        print(f"Добавлено искусственных переменных: {len(self.artificial_vars)}")
        print(f"Общее количество переменных: {self.n_total}")
    
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
    
    def _init_tableau_phase1(self):
        """Формирование начальной симплекс-таблицы для первого этапа"""
        # Создаем таблицу для первого этапа
        self.tableau = np.zeros((self.m + 1, self.n_total + 1), dtype=float)
        
        # Заполняем матрицу ограничений
        self.tableau[:self.m, :self.n_total] = self.A
        self.tableau[:self.m, -1] = self.b
        
        # Заполняем целевую функцию первого этапа (минимизация суммы искусственных переменных)
        self.tableau[-1, :self.n_total] = self.phase1_obj
        
        # Корректируем целевую функцию для базисных искусственных переменных
        self._adjust_objective_for_basis_phase1()
        
        self.phase1_history.append(self.tableau.copy())
    
    def _adjust_objective_for_basis_phase1(self):
        """Корректировка целевой функции первого этапа для базисных искусственных переменных"""
        # Для каждой базисной искусственной переменной вычитаем ее строку из целевой функции
        for art_var in self.artificial_vars:
            basis_row = self._find_basis_row(art_var)
            if basis_row != -1:
                self.tableau[-1, :] -= self.tableau[basis_row, :]
    
    def _init_tableau_phase2(self):
        """Формирование симплекс-таблицы для второго этапа"""
        # Удаляем столбцы искусственных переменных
        keep_columns = [i for i in range(self.n_total) if i not in self.artificial_vars]
        keep_columns.append(self.n_total)  # Добавляем столбец правых частей
        
        # Создаем новую таблицу без искусственных переменных
        phase2_tableau = self.tableau[:, keep_columns]
        
        # Обновляем количество переменных
        self.n_total_phase2 = len(keep_columns) - 1
        
        # Заменяем целевую функцию на исходную
        if self.is_min:
            # Для минимизации оставляем коэффициенты как есть
            phase2_tableau[-1, :self.n_total_phase2] = self.obj[keep_columns[:-1]]
        else:
            # Для максимизации меняем знак коэффициентов
            phase2_tableau[-1, :self.n_total_phase2] = -self.obj[keep_columns[:-1]]
        
        # Корректируем целевую функцию для текущего базиса
        self._adjust_objective_for_basis_phase2(phase2_tableau)
        
        return phase2_tableau
    
    def _adjust_objective_for_basis_phase2(self, tableau):
        """Корректировка целевой функции второго этапа для текущего базиса"""
        # Получаем текущие базисные переменные
        basis_vars = self._get_basis_vars()
        
        # Для каждой базисной переменной вычитаем из целевой функции ее строку, умноженную на коэффициент в целевой функции
        for basis_var in basis_vars:
            basis_row = self._find_basis_row(basis_var)
            if basis_row != -1 and basis_var < self.n_total_phase2:
                # Коэффициент в целевой функции для базисной переменной
                if self.is_min:
                    c_basis = self.obj[basis_var]
                else:
                    c_basis = -self.obj[basis_var]
                
                # Вычитаем строку базисной переменной, умноженную на ее коэффициент
                tableau[-1, :] -= c_basis * tableau[basis_row, :]
    
    def _pivot_operation(self, pivot_row, pivot_col):
        """Операция поворота вокруг разрешающего элемента"""
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        # Нормализация: Делим разрешающую строку на разрешающий элемент
        self.tableau[pivot_row, :] /= pivot_element
        
        # Исключение: Обнуляем столбец в других строках
        for r in range(self.tableau.shape[0]):
            if r != pivot_row:
                factor = self.tableau[r, pivot_col]
                self.tableau[r, :] -= factor * self.tableau[pivot_row, :]
    
    def _get_basis_vars(self):
        """Получение списка базисных переменных"""
        basis_vars = []
        n_vars = self.tableau.shape[1] - 1  # Количество переменных (без учета столбца правых частей)
        
        for j in range(n_vars):
            col = self.tableau[:self.m, j]
            # Проверяем, является ли столбец базисным (ровно одна 1, остальные 0)
            ones_count = np.sum(np.isclose(col, 1.0, atol=1e-8))
            zeros_count = np.sum(np.isclose(col, 0.0, atol=1e-8))
            
            if ones_count == 1 and zeros_count == self.m - 1:
                basis_vars.append(j)
        return basis_vars
    
    def _find_basis_row(self, basis_var):
        """Нахождение строки, в которой переменная является базисной"""
        n_vars = self.tableau.shape[1] - 1
        if basis_var >= n_vars:
            return -1
            
        for i in range(self.m):
            if np.isclose(self.tableau[i, basis_var], 1.0, atol=1e-8):
                return i
        return -1
    
    def _is_optimal_phase1(self):
        """Проверка оптимальности для первого этапа (минимизация суммы искусственных переменных)"""
        last_row = self.tableau[-1, :-1]  # Целевая строка без столбца правых частей
        
        # Для минимизации все коэффициенты должны быть >= 0
        return np.all(last_row >= -self.tol)
    
    def _is_optimal_phase2(self):
        """Проверка оптимальности для второго этапа"""
        last_row = self.tableau[-1, :-1]  # Целевая строка без столбца правых частей
        
        if self.is_min:
            # Для минимизации все коэффициенты должны быть >= 0
            return np.all(last_row >= -self.tol)
        else:
            # Для максимизации все коэффициенты должны быть <= 0
            return np.all(last_row <= self.tol)
    
    def _select_pivot_column_phase1(self):
        """Выбор разрешающего столбца для первого этапа"""
        last_row = self.tableau[-1, :-1]  # Целевая строка без столбца правых частей
        
        # Для минимизации выбираем самый отрицательный коэффициент
        min_val = np.min(last_row)
        if min_val >= -self.tol:
            return None  # Решение оптимально
        
        return np.argmin(last_row)  # Самый отрицательный столбец
    
    def _select_pivot_column_phase2(self):
        """Выбор разрешающего столбца для второго этапа"""
        last_row = self.tableau[-1, :-1]  # Целевая строка без столбца правых частей
        
        if self.is_min:
            # Для минимизации выбираем самый отрицательный коэффициент
            min_val = np.min(last_row)
            if min_val >= -self.tol:
                return None  # Решение оптимально
            return np.argmin(last_row)
        else:
            # Для максимизации выбираем самый положительный коэффициент
            max_val = np.max(last_row)
            if max_val <= self.tol:
                return None  # Решение оптимально
            return np.argmax(last_row)
    
    def _select_pivot_row(self, pivot_col):
        """Выбор разрешающей строки"""
        ratios = np.full(self.m, np.inf)  # Массив для симплекс-отношений
        
        for i in range(self.m):
            a_ij = self.tableau[i, pivot_col]  # Коэффициент в разрешающем столбце
            b_i = self.tableau[i, -1]  # Правая часть
            
            if a_ij > self.tol:  # Положительный коэффициент
                ratios[i] = b_i / a_ij  # Симплекс-отношение
        
        # Если все отношения бесконечны, задача неограничена
        if np.all(ratios == np.inf):
            return None
        
        # Выбираем строку с минимальным положительным отношением
        return np.argmin(ratios)
    
    def _print_tableau(self, step, phase, pivot_col=None):
        """Печать симплекс-таблицы с улучшенным форматированием"""
        print(f"\nСимплекс-таблица (этап {phase}, шаг {step}):")
        
        # Определяем количество переменных в текущей таблице
        n_vars = self.tableau.shape[1] - 1
        
        # Формируем заголовки столбцов
        all_headers = []
        
        # Добавляем исходные переменные
        for i in range(self.n_orig):
            all_headers.append(f"x{i+1}")
        
        # Добавляем slack-переменные
        for i in range(len(self.slack_vars)):
            all_headers.append(f"s{i+1}")
        
        # Добавляем искусственные переменные (только если они есть в текущей таблице)
        if phase == 1:
            for i in range(len(self.artificial_vars)):
                all_headers.append(f"a{i+1}")
        
        # Берем только те заголовки, которые соответствуют текущим переменным
        headers = all_headers[:n_vars]
        headers.extend(["bi", "bi/разр"])
        
        col_width = 10
        
        # Печать заголовков
        header_line = "   C   |  базис  | " + " | ".join(f"{h:^{col_width}}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        
        # Вычисляем симплекс-отношения для текущего разрешающего столбца
        ratios = np.full(self.m, np.inf)
        if pivot_col is not None:
            for i in range(self.m):
                a_ij = self.tableau[i, pivot_col]
                b_i = self.tableau[i, -1]
                if a_ij > self.tol:
                    ratios[i] = b_i / a_ij
        
        # Печать строк ограничений
        basis_vars = self._get_basis_vars()
        for i in range(self.m):
            basis_var_idx = -1
            for j in basis_vars:
                if np.isclose(self.tableau[i, j], 1.0, atol=1e-8):
                    basis_var_idx = j
                    break
            
            # Определяем коэффициент C для базисной переменной
            c_value = 0.0
            basis_name = "???"
            
            if basis_var_idx != -1:
                if basis_var_idx < self.n_orig:
                    basis_name = f"x{basis_var_idx+1}"
                    if phase == 1:
                        c_value = self.phase1_obj[basis_var_idx]
                    else:
                        c_value = self.obj[basis_var_idx] if self.is_min else -self.obj[basis_var_idx]
                elif basis_var_idx in self.slack_vars:
                    slack_idx = self.slack_vars.index(basis_var_idx)
                    basis_name = f"s{slack_idx+1}"
                    c_value = 0.0  # Slack переменные имеют коэффициент 0
                elif phase == 1 and basis_var_idx in self.artificial_vars:
                    art_idx = self.artificial_vars.index(basis_var_idx)
                    basis_name = f"a{art_idx+1}"
                    c_value = 1.0  # Искусственные переменные имеют коэффициент 1 в фазе 1
            
            # Формируем данные строки
            row_data = [f"{self.tableau[i, j]:{col_width}.3f}" for j in range(n_vars)]
            # Добавляем значение bi и bi/разр
            bi_value = f"{self.tableau[i, -1]:{col_width}.3f}"
            ratio_value = f"{ratios[i]:{col_width}.3f}" if ratios[i] != np.inf else " " * col_width
            
            row_line = f" {c_value:5.1f} | {basis_name:^7} | " + " | ".join(row_data) + f" | {bi_value} | {ratio_value}"
            print(row_line)
        
        # Печать строки дельт
        delta_data = [f"{self.tableau[-1, j]:{col_width}.3f}" for j in range(self.tableau.shape[1])]
        delta_line = "       |  delta  | " + " | ".join(delta_data)
        print(delta_line)
    
    def solve(self, max_steps=100):
        """Решение задачи двухэтапным симплекс-методом"""
        print("=" * 70)
        print("НАЧАЛО РЕШЕНИЯ МЕТОДОМ ДАНЦИГА")
        print("=" * 70)
        
        # Этап 1: Минимизация суммы искусственных переменных
        if self.artificial_vars:
            print("\n--- ЭТАП 1: Минимизация суммы искусственных переменных ---")
            print("Целевая функция: W = ", end="")
            art_vars_str = " + ".join([f"a{i+1}" for i in range(len(self.artificial_vars))])
            print(art_vars_str, "→ min")
            
            # Инициализация таблицы для первого этапа
            self._init_tableau_phase1()
            
            step = 0
            print("Начальная симплекс-таблица (этап 1):")
            self._print_tableau(step, 1)
            
            # Итерации первого этапа
            for step in range(1, max_steps + 1):
                if self._is_optimal_phase1():
                    print(f"\nЭтап 1 завершен на шаге {step-1}: достигнуто оптимальное решение")
                    break
                
                pivot_col = self._select_pivot_column_phase1()
                if pivot_col is None:
                    print(f"\nЭтап 1 завершен на шаге {step-1}: решение оптимально")
                    break
                
                pivot_row = self._select_pivot_row(pivot_col)
                if pivot_row is None:
                    raise Exception("Задача неограничена на этапе 1")
                
                # Выполняем операцию поворота
                self._pivot_operation(pivot_row, pivot_col)
                
                print(f"\nШаг {step} (этап 1):")
                self._print_tableau(step, 1, pivot_col)
                self.phase1_history.append(self.tableau.copy())
            
            # Проверяем результат первого этапа
            optimal_value_phase1 = -self.tableau[-1, -1]  # Значение целевой функции первого этапа
            
            # Если значение целевой функции первого этапа > 0, задача не имеет допустимого решения
            if optimal_value_phase1 > self.tol:
                raise Exception("Задача не имеет допустимого решения. Сумма искусственных переменных > 0.")
        else:
            print("\n--- Пропуск этапа 1: искусственные переменные не требуются ---")
            # Инициализация таблицы для первого этапа (без искусственных переменных)
            self._init_tableau_phase1()
        
        # Этап 2: Решение исходной задачи
        print("\n--- ЭТАП 2: Решение исходной задачи ---")
        print("Целевая функция: F = ", end="")
        obj_str = ""
        for i, coeff in enumerate(self.orig_obj):
            if coeff != 0:
                if obj_str and coeff > 0:
                    obj_str += " + "
                elif coeff < 0:
                    obj_str += " - "
                elif obj_str:
                    obj_str += " + "
                
                if abs(coeff) != 1:
                    obj_str += f"{abs(coeff)}"
                obj_str += f"x{i+1}"
        if not obj_str:
            obj_str = "0"
        print(obj_str, "→ min" if self.is_min else "→ max")
        
        # Создаем таблицу для второго этапа
        self.tableau = self._init_tableau_phase2()
        
        step = 0
        print("Начальная симплекс-таблица (этап 2):")
        self._print_tableau(step, 2)
        
        # Итерации второго этапа
        for step in range(1, max_steps + 1):
            if self._is_optimal_phase2():
                print(f"\nЭтап 2 завершен на шаге {step-1}: достигнуто оптимальное решение")
                break
            
            pivot_col = self._select_pivot_column_phase2()
            if pivot_col is None:
                print(f"\nЭтап 2 завершен на шаге {step-1}: решение оптимально")
                break
            
            pivot_row = self._select_pivot_row(pivot_col)
            if pivot_row is None:
                raise Exception("Задача неограничена на этапе 2")
            
            # Выполняем операцию поворота
            self._pivot_operation(pivot_row, pivot_col)
            
            print(f"\nШаг {step} (этап 2):")
            self._print_tableau(step, 2, pivot_col)
            self.history.append(self.tableau.copy())
        
        # Получаем решение
        solution = self._get_solution()
        objective_value = self._get_objective_value()
        
        return solution, objective_value, self.phase1_history + self.history
    
    def _get_solution(self):
        """Получение решения задачи"""
        solution = np.zeros(self.n_orig, dtype=float)  # Вектор решения для исходных переменных
        basis_vars = self._get_basis_vars()  # Текущие базисные переменные
        
        for basis_var in basis_vars:
            if basis_var < self.n_orig:  # Если базисная переменная - исходная
                basis_row = self._find_basis_row(basis_var)
                if basis_row != -1:
                    solution[basis_var] = self.tableau[basis_row, -1]  # Значение из столбца правых частей
        
        return solution
    
    def _get_objective_value(self):
        """Получение значения целевой функции"""
        if self.is_min:
            return -self.tableau[-1, -1]  # Для минимизации
        else:
            return self.tableau[-1, -1]  # Для максимизации


def test_individual_task():
    """Тестирование на индивидуальном задании"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ НА ИНДИВИДУАЛЬНОМ ЗАДАНИИ")
    print("=" * 70)
    
    # Данные из индивидуального задания
    obj_coeffs = [2, -3, 3, 1]  # Коэффициенты целевой функции
    constraints = [
        [2, 1, -1, 1],  # 2x1 + x2 - x3 + x4 = 24
        [1, 2, 2, 0],   # x1 + 2x2 + 2x3 <= 22
        [1, -1, 1, 0]    # x1 - x2 + x3 >= 10
    ]
    rhs_values = [24, 22, 10]  # Правые части ограничений
    constraint_types = ['=', '<=', '>=']  # Типы ограничений
    
    print("Исходная задача:")
    print("Целевая функция: 2x1 - 3x2 + 3x3 + x4 → min")
    print("Ограничения:")
    print("  2x1 + x2 - x3 + x4 = 24")
    print("  x1 + 2x2 + 2x3 <= 22")
    print("  x1 - x2 + x3 >= 10")
    print("  x1, x2, x3, x4 >= 0")
    
    # Создаем решатель
    solver = TwoPhaseSimplexSolver(
        obj_coeffs=obj_coeffs,
        constraints=constraints, 
        rhs_values=rhs_values,
        constraint_types=constraint_types,
        is_min=True
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
        expected_solution = [34/3, 4/3, 0, 0]  # Ожидаемое решение из PDF
        expected_objective = 56/3  # Ожидаемое значение целевой функции
        
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


def read_problem_from_file(filename):
    """Чтение задачи из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Удаляем пустые строки и строки с комментариями
        lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if len(lines) < 3:
            raise ValueError("Файл должен содержать как минимум 3 строки")
        
        # Первая строка: количество переменных и ограничений
        n, m = map(int, lines[0].split())
        
        # Вторая строка: коэффициенты целевой функции
        obj_coeffs = list(map(float, lines[1].split()))
        
        if len(obj_coeffs) != n:
            raise ValueError(f"Ожидается {n} коэффициентов целевой функции, получено {len(obj_coeffs)}")
        
        # Остальные строки: ограничения
        constraints = []
        rhs_values = []
        constraint_types = []
        
        for i in range(2, 2 + m):
            parts = lines[i].split()
            if len(parts) < n + 2:
                raise ValueError(f"Недостаточно данных в ограничении {i-1}")
            
            # Коэффициенты ограничения
            coeffs = list(map(float, parts[:n]))
            
            # Тип ограничения
            const_type = parts[n]
            if const_type not in ['<=', '=', '>=']:
                raise ValueError(f"Неизвестный тип ограничения: {const_type}")
            
            # Правая часть
            rhs = float(parts[n + 1])
            
            constraints.append(coeffs)
            constraint_types.append(const_type)
            rhs_values.append(rhs)
        
        return obj_coeffs, constraints, rhs_values, constraint_types
    
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return None, None, None, None


def solve_from_file():
    """Решение задачи из файла"""
    print("\n" + "=" * 50)
    print("РЕШЕНИЕ ЗАДАЧИ ИЗ ФАЙЛА")
    print("=" * 50)
    
    filename = input("Введите название файла или полный путь к файлу: ").strip()
    
    # Если файл не существует, попробуем добавить расширение .txt
    if not os.path.isfile(filename):
        if not filename.endswith('.txt'):
            filename_txt = filename + '.txt'
            if os.path.isfile(filename_txt):
                filename = filename_txt
            else:
                # Попробуем найти файл в текущей директории
                current_dir = os.getcwd()
                filename_in_current = os.path.join(current_dir, filename)
                if os.path.isfile(filename_in_current):
                    filename = filename_in_current
                elif not filename.endswith('.txt'):
                    filename_in_current_txt = os.path.join(current_dir, filename + '.txt')
                    if os.path.isfile(filename_in_current_txt):
                        filename = filename_in_current_txt
                    else:
                        print(f"Файл '{filename}' не найден.")
                        print(f"Текущая рабочая директория: {current_dir}")
                        return
                else:
                    print(f"Файл '{filename}' не найден.")
                    print(f"Текущая рабочая директория: {current_dir}")
                    return
    
    print(f"Чтение файла: {filename}")
    
    # Чтение задачи из файла
    obj_coeffs, constraints, rhs_values, constraint_types = read_problem_from_file(filename)
    
    if obj_coeffs is None:
        print("Не удалось прочитать задачу из файла.")
        return
    
    # Запрос типа оптимизации
    is_min_input = input("Минимизация? (y/n, по умолчанию y): ").strip().lower()
    is_min = is_min_input != 'n'
    
    # Вывод задачи
    print("\nЗадача из файла:")
    print("Целевая функция: ", end="")
    obj_str = ""
    for i, coeff in enumerate(obj_coeffs):
        if coeff != 0:
            if obj_str and coeff > 0:
                obj_str += " + "
            elif coeff < 0:
                obj_str += " - "
            elif obj_str:
                obj_str += " + "
            
            if abs(coeff) != 1:
                obj_str += f"{abs(coeff)}"
            obj_str += f"x{i+1}"
    if not obj_str:
        obj_str = "0"
    print(obj_str, "→ min" if is_min else "→ max")
    
    print("Ограничения:")
    for i in range(len(constraints)):
        constr_str = ""
        for j, coeff in enumerate(constraints[i]):
            if coeff != 0:
                if constr_str and coeff > 0:
                    constr_str += " + "
                elif coeff < 0:
                    constr_str += " - "
                elif constr_str:
                    constr_str += " + "
                
                if abs(coeff) != 1:
                    constr_str += f"{abs(coeff)}"
                constr_str += f"x{j+1}"
        if not constr_str:
            constr_str = "0"
        print(f"  {constr_str} {constraint_types[i]} {rhs_values[i]}")
    
    # Решение задачи
    try:
        solver = TwoPhaseSimplexSolver(
            obj_coeffs=obj_coeffs,
            constraints=constraints,
            rhs_values=rhs_values,
            constraint_types=constraint_types,
            is_min=is_min
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
        print(f"Ошибка при решении: {e}")
        import traceback
        traceback.print_exc()


def main_console():
    """Консольный интерфейс для решения произвольных задач"""
    print("\n" + "=" * 70)
    print("РЕШЕНИЕ ПРОИЗВОЛЬНОЙ ЗАДАЧИ МЕТОДОМ ДАНЦИГА")
    print("=" * 70)
    
    print("ДВУХЭТАПНЫЙ СИМПЛЕКС-МЕТОД (МЕТОД ДАНЦИГА)")
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
        
        # Решение задачи
        solver = TwoPhaseSimplexSolver(
            obj_coeffs=obj_coeffs,
            constraints=constraints,
            rhs_values=rhs_values,
            constraint_types=constraint_types,
            is_min=is_min
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


def main():
    """Главная функция программы"""
    # Запуск тестирования на индивидуальном задании
    test_individual_task()
    
    # Выбор способа ввода
    while True:
        print("\n" + "=" * 70)
        print("ВЫБЕРИТЕ СПОСОБ ВВОДА ДАННЫХ")
        print("=" * 70)
        print("1 - Ввод через консоль")
        print("2 - Ввод из файла")
        print("0 - Выход")
        
        choice = input("Ваш выбор: ").strip()
        
        if choice == '1':
            main_console()
        elif choice == '2':
            solve_from_file()
        elif choice == '0':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()