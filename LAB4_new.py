import numpy as np
from LAB2 import ArtificialBasisSolver

class DualProblemTransformer:
    """
    Класс для преобразования произвольной прямой задачи ЛП в двойственную задачу
    """
    
    def __init__(self, primal_obj_coeffs, primal_constraints, primal_rhs, primal_constraint_types, is_min=True):
        """
        Инициализация преобразователя двойственной задачи
        """
        self.primal_obj = primal_obj_coeffs
        self.primal_A = primal_constraints
        self.primal_b = primal_rhs
        self.primal_types = primal_constraint_types
        self.primal_is_min = is_min
        
        # Размеры задачи
        self.n_primal_vars = len(primal_obj_coeffs)
        self.n_primal_constraints = len(primal_rhs)
        
        # Результаты преобразования
        self.dual_obj = None
        self.dual_constraints = None
        self.dual_rhs = None
        self.dual_types = None
        self.dual_var_conditions = None
        
        # Результаты преобразования к неотрицательной форме
        self.dual_obj_nonnegative = None
        self.dual_constraints_nonnegative = None
        self.dual_rhs_nonnegative = None
        self.dual_types_nonnegative = None
        self.var_mapping = None
        
    def transform_to_dual(self):
        """
        Преобразование произвольной прямой задачи в двойственную
        Строго по правилам теории двойственности
        """
        print("\n" + "="*70)
        print("ПРЕОБРАЗОВАНИЕ ПРЯМОЙ ЗАДАЧИ В ДВОЙСТВЕННУЮ")
        print("="*70)
        
        if not self.primal_is_min:
            raise ValueError("Преобразование реализовано только для задач минимизации")
        
        # Для задачи минимизации двойственная - максимизация
        self.dual_is_min = False
        
        # Целевая функция двойственной = правые части прямой
        self.dual_obj = self.primal_b.copy()
        
        # Матрица ограничений двойственной = транспонированная матрица прямой
        self.dual_constraints = []
        for j in range(self.n_primal_vars):
            constraint_row = []
            for i in range(self.n_primal_constraints):
                constraint_row.append(self.primal_A[i][j])
            self.dual_constraints.append(constraint_row)
        
        # Правые части ограничений двойственной = коэффициенты целевой функции прямой
        self.dual_rhs = self.primal_obj.copy()
        
        # Все ограничения двойственной имеют тип '<=' для задачи минимизации
        self.dual_types = ['<='] * self.n_primal_vars
        
        # Условия на переменные двойственной определяются по типам ограничений прямой
        self.dual_var_conditions = []
        for constraint_type in self.primal_types:
            if constraint_type == '<=':
                self.dual_var_conditions.append('>=0')
            elif constraint_type == '>=':
                self.dual_var_conditions.append('<=0')
            else:  # '='
                self.dual_var_conditions.append('free')
        
        print("Двойственная задача после преобразования:")
        self._print_dual_intermediate()
    
    def _print_dual_intermediate(self):
        """Вывод промежуточной двойственной задачи"""
        print(f"Целевая функция: {self.dual_obj} → {'min' if self.dual_is_min else 'max'}")
        print("Ограничения:")
        for i, constraint in enumerate(self.dual_constraints):
            print(f"  {constraint} {self.dual_types[i]} {self.dual_rhs[i]}")
        print("Условия на переменные:")
        for i, condition in enumerate(self.dual_var_conditions):
            print(f"  y{i+1} {condition}")
    
    def convert_to_nonnegative_form(self):
        """
        Преобразование двойственной задачи к форме с неотрицательными переменными
        По точному алгоритму пользователя
        """
        print("\n" + "="*70)
        print("ПРЕОБРАЗОВАНИЕ К ФОРМЕ С НЕОТРИЦАТЕЛЬНЫМИ ПЕРЕМЕННЫМИ")
        print("="*70)
        
        # Новые переменные для неотрицательной формы
        new_obj = []
        new_constraints = [[] for _ in range(len(self.dual_constraints))]
        new_rhs = self.dual_rhs.copy()
        new_types = self.dual_types.copy()
        
        # Сопоставление старых переменных с новыми
        var_mapping = []
        
        print("Преобразование переменных:")
        
        for i, condition in enumerate(self.dual_var_conditions):
            if condition == 'free':
                # Свободная переменная: y = y+ - y-
                print(f"  y{i+1} (свободная) -> y{i+1}+ - y{i+1}-")
                var_mapping.append(('free', len(new_obj), len(new_obj) + 1))
                new_obj.extend([self.dual_obj[i], -self.dual_obj[i]])
                
                # Обновляем матрицу ограничений
                for j in range(len(self.dual_constraints)):
                    coeff = self.dual_constraints[j][i]
                    new_constraints[j].extend([coeff, -coeff])
                    
            elif condition == '>=0':
                # Переменная >=0 заменяется на отрицательную неотрицательную: y = -y'
                print(f"  y{i+1} >= 0 -> -y{i+1}' (y{i+1}' >= 0)")
                var_mapping.append(('>=0', len(new_obj)))
                new_obj.append(-self.dual_obj[i])
                
                for j in range(len(self.dual_constraints)):
                    coeff = self.dual_constraints[j][i]
                    new_constraints[j].append(-coeff)
                    
            elif condition == '<=0':
                # Переменная <=0 заменяется на неотрицательную: y = y'
                print(f"  y{i+1} <= 0 -> y{i+1}' (y{i+1}' >= 0)")
                var_mapping.append(('<=0', len(new_obj)))
                new_obj.append(self.dual_obj[i])
                
                for j in range(len(self.dual_constraints)):
                    coeff = self.dual_constraints[j][i]
                    new_constraints[j].append(coeff)
        
        # Сохраняем результаты
        self.dual_obj_nonnegative = new_obj
        self.dual_constraints_nonnegative = new_constraints
        self.dual_rhs_nonnegative = new_rhs
        self.dual_types_nonnegative = new_types
        self.var_mapping = var_mapping
        
        print(f"\nПосле преобразования: {len(new_obj)} переменных")
        
        print("\nПреобразованная двойственная задача:")
        self._print_nonnegative_dual()
    
    def _print_nonnegative_dual(self):
        """Вывод двойственной задачи в неотрицательной форме"""
        print(f"Целевая функция: {self.dual_obj_nonnegative} → {'min' if self.dual_is_min else 'max'}")
        print("Ограничения:")
        for i, constraint in enumerate(self.dual_constraints_nonnegative):
            print(f"  {constraint} {self.dual_types_nonnegative[i]} {self.dual_rhs_nonnegative[i]}")
        print("Условия неотрицательности: все переменные >= 0")
    
    def get_dual_problem_for_solver(self):
        """
        Возвращает двойственную задачу в форме, пригодной для решения симплекс-методом
        """
        # Для нашего решателя нужно преобразовать максимизацию в минимизацию
        if not self.dual_is_min:  # если двойственная - максимизация
            dual_obj_for_solver = [-coeff for coeff in self.dual_obj_nonnegative]
            is_min_for_solver = True
        else:
            dual_obj_for_solver = self.dual_obj_nonnegative
            is_min_for_solver = False
            
        return {
            'obj_coeffs': dual_obj_for_solver,
            'constraints': self.dual_constraints_nonnegative,
            'rhs_values': self.dual_rhs_nonnegative,
            'constraint_types': self.dual_types_nonnegative,
            'is_min': is_min_for_solver
        }
    
    def recover_dual_solution(self, nonnegative_solution):
        """
        Восстановление исходного решения двойственной задачи из решения в неотрицательной форме
        """
        original_solution = [0.0] * len(self.dual_var_conditions)
        var_index = 0
        
        for i, condition in enumerate(self.dual_var_conditions):
            if condition == 'free':
                # y = y+ - y-
                original_solution[i] = nonnegative_solution[var_index] - nonnegative_solution[var_index + 1]
                var_index += 2
            elif condition == '>=0':
                # y = -y'
                original_solution[i] = -nonnegative_solution[var_index]
                var_index += 1
            elif condition == '<=0':
                # y = y'
                original_solution[i] = nonnegative_solution[var_index]
                var_index += 1
                
        return original_solution


def test_transformer():
    """Тестирование преобразователя на индивидуальном задании"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ПРЕОБРАЗОВАТЕЛЯ НА ИНДИВИДУАЛЬНОМ ЗАДАНИИ")
    print("=" * 70)
    
    # Данные из индивидуального задания (прямая задача)
    primal_obj_coeffs = [2, -3, 3, 1]
    primal_constraints = [
        [2, 1, -1, 1],  # 2x1 + x2 - x3 + x4 = 24
        [1, 2, 2, 0],   # x1 + 2x2 + 2x3 <= 22  
        [1, -1, 1, 0]    # x1 - x2 + x3 >= 10
    ]
    primal_rhs_values = [24, 22, 10]
    primal_constraint_types = ['=', '<=', '>=']
    
    print("ИСХОДНАЯ (ПРЯМАЯ) ЗАДАЧА:")
    print("Целевая функция: 2x1 - 3x2 + 3x3 + x4 → min")
    print("Ограничения:")
    print("  2x1 + x2 - x3 + x4 = 24")
    print("  x1 + 2x2 + 2x3 <= 22") 
    print("  x1 - x2 + x3 >= 10")
    print("  x1, x2, x3, x4 >= 0")
    
    # Преобразование в двойственную задачу
    transformer = DualProblemTransformer(
        primal_obj_coeffs, 
        primal_constraints, 
        primal_rhs_values, 
        primal_constraint_types,
        is_min=True
    )
    
    transformer.transform_to_dual()
    transformer.convert_to_nonnegative_form()
    
    # Ожидаемые коэффициенты из алгоритма пользователя
    expected_obj = [24, -24, -22, 10]  # max: 24x1 - 24x2 - 22x3 + 10x4
    expected_constraints = [
        [2, -2, -1, 1],    # 2x1 - 2x2 - x3 + x4 <= 2
        [1, -1, -2, -1],   # x1 - x2 - 2x3 - x4 <= -3
        [-1, 1, -2, 1],    # -x1 + x2 - 2x3 + x4 <= 3
        [1, -1, 0, 0]      # x1 - x2 <= 1
    ]
    expected_rhs = [2, -3, 3, 1]
    expected_types = ['<=', '<=', '<=', '<=']
    
    # Сравнение полученных коэффициентов с ожидаемыми
    print("\n" + "="*70)
    print("СРАВНЕНИЕ С ОЖИДАЕМЫМИ КОЭФФИЦИЕНТАМИ")
    print("="*70)
    
    # Сравнение целевых функций
    print("Целевые функции:")
    print(f"  Полученная: {transformer.dual_obj_nonnegative}")
    print(f"  Ожидаемая:  {expected_obj}")
    
    obj_match = np.allclose(transformer.dual_obj_nonnegative, expected_obj, atol=1e-6)
    print(f"  Совпадение: {'✓' if obj_match else '✗'}")
    
    # Сравнение матрицы ограничений
    print("\nМатрица ограничений:")
    constraints_match = True
    for i, (expected, actual) in enumerate(zip(expected_constraints, transformer.dual_constraints_nonnegative)):
        match = np.allclose(actual, expected, atol=1e-6)
        constraints_match &= match
        print(f"  Ограничение {i+1}:")
        print(f"    Полученное: {actual}")
        print(f"    Ожидаемое:  {expected}")
        print(f"    Совпадение: {'✓' if match else '✗'}")
    
    # Сравнение правых частей
    print("\nПравые части:")
    print(f"  Полученные: {transformer.dual_rhs_nonnegative}")
    print(f"  Ожидаемые:  {expected_rhs}")
    
    rhs_match = np.allclose(transformer.dual_rhs_nonnegative, expected_rhs, atol=1e-6)
    print(f"  Совпадение: {'✓' if rhs_match else '✗'}")
    
    # Сравнение типов ограничений
    print("\nТипы ограничений:")
    print(f"  Полученные: {transformer.dual_types_nonnegative}")
    print(f"  Ожидаемые:  {expected_types}")
    
    types_match = transformer.dual_types_nonnegative == expected_types
    print(f"  Совпадение: {'✓' if types_match else '✗'}")
    
    # Итоговый результат
    print("\n" + "="*70)
    print("ИТОГОВОЕ СРАВНЕНИЕ:")
    all_match = obj_match and constraints_match and rhs_match and types_match
    if all_match:
        print("✓ ВСЕ КОЭФФИЦИЕНТЫ СОВПАДАЮТ!")
        print("Преобразование выполнено корректно!")
    else:
        print("✗ ЕСТЬ РАСХОЖДЕНИЯ!")
        if not obj_match:
            print("  - Целевые функции не совпадают")
        if not constraints_match:
            print("  - Матрица ограничений не совпадает") 
        if not rhs_match:
            print("  - Правые части не совпадают")
        if not types_match:
            print("  - Типы ограничений не совпадают")
    
    return all_match


def solve_primal_dual_problem(obj_coeffs, constraints, rhs_values, constraint_types, is_min=True):
    """
    Решение прямой и двойственной задачи с выводом результатов
    """
    print("\n" + "="*70)
    print("РЕШЕНИЕ ПРЯМОЙ И ДВОЙСТВЕННОЙ ЗАДАЧ")
    print("="*70)
    
    # Решение прямой задачи
    print("\n--- РЕШЕНИЕ ПРЯМОЙ ЗАДАЧИ ---")
    primal_solver = ArtificialBasisSolver(
        obj_coeffs=obj_coeffs,
        constraints=constraints,
        rhs_values=rhs_values,
        constraint_types=constraint_types,
        is_min=is_min,
        M=10000
    )
    
    try:
        primal_solution, primal_objective, primal_history = primal_solver.solve()
        
        print("РЕЗУЛЬТАТЫ РЕШЕНИЯ ПРЯМОЙ ЗАДАЧИ:")
        print("Оптимальное решение прямой задачи:")
        for i, val in enumerate(primal_solution):
            print(f"  x{i+1} = {val:.6f}")
        print(f"Значение целевой функции прямой задачи: {primal_objective:.6f}")
        
    except Exception as e:
        print(f"Ошибка при решении прямой задачи: {e}")
        return None, None, None, None
    
    # Преобразование в двойственную задачу
    print("\n--- ПРЕОБРАЗОВАНИЕ В ДВОЙСТВЕННУЮ ЗАДАЧУ ---")
    transformer = DualProblemTransformer(
        obj_coeffs, 
        constraints, 
        rhs_values, 
        constraint_types,
        is_min=is_min
    )
    
    transformer.transform_to_dual()
    transformer.convert_to_nonnegative_form()
    
    # Решение двойственной задачи
    print("\n--- РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ ---")
    dual_problem = transformer.get_dual_problem_for_solver()
    
    dual_solver = ArtificialBasisSolver(
        obj_coeffs=dual_problem['obj_coeffs'],
        constraints=dual_problem['constraints'],
        rhs_values=dual_problem['rhs_values'],
        constraint_types=dual_problem['constraint_types'],
        is_min=dual_problem['is_min'],
        M=10000
    )
    
    try:
        dual_solution_nonnegative, dual_objective, dual_history = dual_solver.solve()
        
        # Восстанавливаем исходные переменные двойственной задачи
        dual_solution_original = transformer.recover_dual_solution(dual_solution_nonnegative)
        
        # Корректируем значение целевой функции (так как мы преобразовали max в min)
        if not transformer.dual_is_min:
            dual_objective_original = -dual_objective
        else:
            dual_objective_original = dual_objective
            
        print("РЕЗУЛЬТАТЫ РЕШЕНИЯ ДВОЙСТВЕННОЙ ЗАДАЧИ:")
        print("Оптимальное решение двойственной задачи:")
        for i, val in enumerate(dual_solution_original):
            condition = transformer.dual_var_conditions[i]
            print(f"  y{i+1} = {val:.6f} ({condition})")
        print(f"Значение целевой функции двойственной задачи: {dual_objective_original:.6f}")
        
        return primal_solution, primal_objective, dual_solution_original, dual_objective_original
        
    except Exception as e:
        print(f"Ошибка при решении двойственной задачи: {e}")
        return primal_solution, primal_objective, None, None


def test_individual_task():
    """Тестирование на индивидуальном задании"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ НА ИНДИВИДУАЛЬНОМ ЗАДАНИИ")
    print("=" * 70)
    
    # Данные из индивидуального задания (прямая задача)
    primal_obj_coeffs = [2, -3, 3, 1]
    primal_constraints = [
        [2, 1, -1, 1],  # 2x1 + x2 - x3 + x4 = 24
        [1, 2, 2, 0],   # x1 + 2x2 + 2x3 <= 22  
        [1, -1, 1, 0]    # x1 - x2 + x3 >= 10
    ]
    primal_rhs_values = [24, 22, 10]
    primal_constraint_types = ['=', '<=', '>=']
    
    print("ИСХОДНАЯ (ПРЯМАЯ) ЗАДАЧА:")
    print("Целевая функция: 2x1 - 3x2 + 3x3 + x4 → min")
    print("Ограничения:")
    print("  2x1 + x2 - x3 + x4 = 24")
    print("  x1 + 2x2 + 2x3 <= 22") 
    print("  x1 - x2 + x3 >= 10")
    print("  x1, x2, x3, x4 >= 0")
    
    # Решение прямой и двойственной задачи
    primal_solution, primal_objective, dual_solution, dual_objective = solve_primal_dual_problem(
        primal_obj_coeffs, 
        primal_constraints, 
        primal_rhs_values, 
        primal_constraint_types,
        is_min=True
    )
    
    # Проверка теоремы двойственности
    if primal_solution is not None and dual_solution is not None:
        print("\n" + "="*70)
        print("ПРОВЕРКА ТЕОРЕМЫ ДВОЙСТВЕННОСТИ")
        print("="*70)
        
        print(f"Значение целевой функции прямой задачи: {primal_objective:.6f}")
        print(f"Значение целевой функции двойственной задачи: {dual_objective:.6f}")
        
        tolerance = 1e-6
        if abs(primal_objective - dual_objective) < tolerance:
            print("✓ ТЕОРЕМА ДВОЙСТВЕННОСТИ ПОДТВЕРЖДЕНА!")
            print("  Значения целевых функций прямой и двойственной задач совпадают.")
        else:
            print("✗ ТЕОРЕМА ДВОЙСТВЕННОСТИ НЕ ПОДТВЕРЖДЕНА!")
            print(f"  Разница: {abs(primal_objective - dual_objective):.10f}")

def read_problem_from_file(filename):
    """Чтение задачи из файла (импортировано из LAB3)"""
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
    """Решение задачи из файла (адаптировано из LAB3)"""
    import os
    
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
    
    # Решение прямой и двойственной задачи
    primal_solution, primal_objective, dual_solution, dual_objective = solve_primal_dual_problem(
        obj_coeffs, constraints, rhs_values, constraint_types, is_min
    )
    
    # Проверка теоремы двойственности
    if primal_solution is not None and dual_solution is not None:
        print("\n" + "="*70)
        print("ПРОВЕРКА ТЕОРЕМЫ ДВОЙСТВЕННОСТИ")
        print("="*70)
        
        print(f"Значение целевой функции прямой задачи: {primal_objective:.6f}")
        print(f"Значение целевой функции двойственной задачи: {dual_objective:.6f}")
        
        tolerance = 1e-6
        if abs(primal_objective - dual_objective) < tolerance:
            print("✓ ТЕОРЕМА ДВОЙСТВЕННОСТИ ПОДТВЕРЖДЕНА!")
            print("  Значения целевых функций прямой и двойственной задач совпадают.")
        else:
            print("✗ ТЕОРЕМА ДВОЙСТВЕННОСТИ НЕ ПОДТВЕРЖДЕНА!")
            print(f"  Разница: {abs(primal_objective - dual_objective):.10f}")


def main_console():
    """Консольный интерфейс для решения произвольных задач (адаптировано из LAB3)"""
    print("\n" + "=" * 70)
    print("РЕШЕНИЕ ПРОИЗВОЛЬНОЙ ЗАДАЧИ")
    print("=" * 70)
    
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
        
        # Вывод задачи
        print("\nВведенная задача:")
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
        for i in range(m):
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
        
        # Решение прямой и двойственной задачи
        primal_solution, primal_objective, dual_solution, dual_objective = solve_primal_dual_problem(
            obj_coeffs, constraints, rhs_values, constraint_types, is_min
        )
        
        # Проверка теоремы двойственности
        if primal_solution is not None and dual_solution is not None:
            print("\n" + "="*70)
            print("ПРОВЕРКА ТЕОРЕМЫ ДВОЙСТВЕННОСТИ")
            print("="*70)
            
            print(f"Значение целевой функции прямой задачи: {primal_objective:.6f}")
            print(f"Значение целевой функции двойственной задачи: {dual_objective:.6f}")
            
            tolerance = 1e-6
            if abs(primal_objective - dual_objective) < tolerance:
                print("✓ ТЕОРЕМА ДВОЙСТВЕННОСТИ ПОДТВЕРЖДЕНА!")
                print("  Значения целевых функций прямой и двойственной задач совпадают.")
            else:
                print("✗ ТЕОРЕМА ДВОЙСТВЕННОСТИ НЕ ПОДТВЕРЖДЕНА!")
                print(f"  Разница: {abs(primal_objective - dual_objective):.10f}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Главная функция программы"""
    print("ЛАБОРАТОРНАЯ РАБОТА 4: ДВОЙСТВЕННЫЕ ЗАДАЧИ")
    print("=" * 70)
    
    # Тестирование на индивидуальном задании
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