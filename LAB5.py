import numpy as np
from LAB2 import ArtificialBasisSolver

def is_integer(x, tol=1e-6):
    """Проверка, является ли число целым с заданной точностью"""
    return abs(x - round(x)) < tol

def gomory_cut(solver, integer_vars_indices):
    """
    Формирование отсечения Гомори для нецелой переменной
    """
    solution = solver._get_solution()
    
    # Собираем все нецелые переменные с их дробными частями
    non_integer_vars = []
    for var_idx in integer_vars_indices:
        if var_idx < len(solution) and not is_integer(solution[var_idx]):
            fraction = solution[var_idx] - np.floor(solution[var_idx])
            non_integer_vars.append((var_idx, fraction, solution[var_idx]))
    
    if not non_integer_vars:
        return None
    
    # Получаем базисные переменные по строкам
    basis_vars_by_row = get_basis_vars_by_row(solver)
    
    # Выбираем переменную с максимальной дробной частью
    var_idx, max_fraction, value = max(non_integer_vars, key=lambda x: x[1])
    print(f"Выбираем переменную x{var_idx+1} с максимальной дробной частью: {value:.6f} -> {max_fraction:.6f}")
    
    # Находим строку, соответствующую этой базисной переменной
    row_idx = -1
    for i, basis_var in enumerate(basis_vars_by_row):
        if basis_var == var_idx:
            row_idx = i
            break
    
    if row_idx == -1:
        print(f"Переменная x{var_idx+1} не в базисе, используем строку с максимальной дробной частью в решении")
        # Если переменная не в базисе, используем строку с максимальной дробной частью
        max_fraction = 0
        for i in range(solver.m):
            current_val = solver.tableau[i, -1]
            fraction = current_val - np.floor(current_val)
            if fraction > max_fraction:
                max_fraction = fraction
                row_idx = i
    
    if row_idx == -1:
        print("Не удалось найти подходящую строку для отсечения")
        return None
    
    # Получаем строку из симплекс-таблицы
    row = solver.tableau[row_idx, :].copy()
    rhs = row[-1]
    
    # Вычисляем дробные части
    fractional_rhs = rhs - np.floor(rhs)
    print(f"Строка {row_idx}: правая часть = {rhs:.6f}, дробная часть = {fractional_rhs:.6f}")
    
    # Формируем коэффициенты для нового ограничения
    cut_coeffs = []
    for j in range(solver.n_orig):
        coeff = row[j]
        fractional_coeff = coeff - np.floor(coeff)
        cut_coeffs.append(fractional_coeff)
        if abs(fractional_coeff) > 1e-6:
            print(f"  x{j+1}: {coeff:.6f} -> {fractional_coeff:.6f}")
    
    # Проверяем, что отсечение не вырождено
    if abs(fractional_rhs) < 1e-10 and all(abs(coeff) < 1e-10 for coeff in cut_coeffs):
        print("Вырожденное отсечение, пропускаем...")
        return None
    
    return cut_coeffs, fractional_rhs

def get_basis_vars_by_row(solver):
    """Определение базисных переменных по строкам"""
    basis_vars = []
    for i in range(solver.m):
        found = -1
        for j in range(solver.n_total):
            if np.isclose(solver.tableau[i, j], 1.0, atol=1e-8):
                # Проверяем, что в других строках этого столбца 0
                is_basis = True
                for k in range(solver.m):
                    if k != i and not np.isclose(solver.tableau[k, j], 0.0, atol=1e-8):
                        is_basis = False
                        break
                if is_basis:
                    found = j
                    break
        basis_vars.append(found)
    return basis_vars

def solve_with_simplex(obj_coeffs, constraints, rhs_values, constraint_types, is_min=True, max_steps=50):
    """Решает задачу симплекс-методом с правильным преобразованием максимизации"""
    if not is_min:
        # Преобразуем задачу максимизации в минимизацию
        obj_coeffs_min = [-coeff for coeff in obj_coeffs]
        solver = ArtificialBasisSolver(
            obj_coeffs=obj_coeffs_min,
            constraints=constraints,
            rhs_values=rhs_values,
            constraint_types=constraint_types,
            is_min=True,  # Всегда минимизация после преобразования
            M=100
        )
        
        solution, objective_value, history = solver.solve(max_steps=max_steps)
        
        # Корректируем значение целевой функции для максимизации
        objective_value = -objective_value
    else:
        solver = ArtificialBasisSolver(
            obj_coeffs=obj_coeffs,
            constraints=constraints,
            rhs_values=rhs_values,
            constraint_types=constraint_types,
            is_min=True,
            M=100
        )
        
        solution, objective_value, history = solver.solve(max_steps=max_steps)
    
    return solution, objective_value, solver

def read_problem_from_file(file_path):
    """
    Чтение задачи из файла
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    if len(lines) < 4:
        raise ValueError("Файл должен содержать как минимум 4 строки")
    
    # Читаем тип оптимизации
    optimization_type = lines[0].lower()
    is_min = optimization_type == 'min'
    
    # Читаем целевую функцию
    obj_coeffs = list(map(float, lines[1].split()))
    
    # Читаем целочисленные переменные
    integer_vars = list(map(int, lines[2].split()))
    integer_vars_indices = [idx - 1 for idx in integer_vars]
    
    # Читаем ограничения
    constraints = []
    rhs_values = []
    constraint_types = []
    
    for i in range(3, len(lines)):
        parts = lines[i].split()
        if len(parts) < len(obj_coeffs) + 2:
            raise ValueError(f"Недостаточно данных в ограничении {i-2}")
        
        coeffs = list(map(float, parts[:len(obj_coeffs)]))
        const_type = parts[len(obj_coeffs)]
        rhs = float(parts[len(obj_coeffs) + 1])
        
        constraints.append(coeffs)
        constraint_types.append(const_type)
        rhs_values.append(rhs)
    
    return obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min

def print_problem_info(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min):
    """Вывод информации о задаче"""
    print("=" * 70)
    print("ИНФОРМАЦИЯ О ЗАДАЧЕ:")
    print("=" * 70)
    
    obj_str = " + ".join([f"{coeff}x{i+1}" for i, coeff in enumerate(obj_coeffs)])
    optimization = "min" if is_min else "max"
    print(f"Целевая функция: {optimization} L(x) = {obj_str}")
    
    print("Ограничения:")
    for i, (coeffs, const_type, rhs) in enumerate(zip(constraints, constraint_types, rhs_values)):
        constraint_str = " + ".join([f"{coeff}x{j+1}" for j, coeff in enumerate(coeffs) if coeff != 0])
        print(f"  {constraint_str} {const_type} {rhs}")
    
    int_vars_str = ", ".join([f"x{idx+1}" for idx in integer_vars_indices])
    print(f"Целочисленные переменные: {int_vars_str}")
    print()

def input_problem_from_console():
    """Ввод задачи с консоли"""
    print("\nВВОД ДАННЫХ ЗАДАЧИ:")
    print("=" * 50)
    
    # Тип оптимизации
    optimization = input("Тип оптимизации (min/max): ").strip().lower()
    is_min = optimization == 'min'
    
    # Количество переменных
    n = int(input("Количество переменных: "))
    
    # Целевая функция
    print("Введите коэффициенты целевой функции (через пробел):")
    obj_coeffs = list(map(float, input().split()))
    
    # Целочисленные переменные
    print("Введите номера целочисленных переменных (через пробел, например: 1 3 5):")
    integer_vars = list(map(int, input().split()))
    integer_vars_indices = [idx - 1 for idx in integer_vars]
    
    # Количество ограничений
    m = int(input("Количество ограничений: "))
    
    # Ограничения
    constraints = []
    rhs_values = []
    constraint_types = []
    
    print("Введите ограничения в формате: коэффициенты тип правая_часть")
    print("Пример: 1 2 3 <= 10")
    
    for i in range(m):
        print(f"Ограничение {i+1}:")
        parts = input().split()
        
        if len(parts) < n + 2:
            print(f"Ошибка: ожидается {n} коэффициентов, тип ограничения и правая часть")
            continue
            
        coeffs = list(map(float, parts[:n]))
        const_type = parts[n]
        rhs = float(parts[n + 1])
        
        constraints.append(coeffs)
        constraint_types.append(const_type)
        rhs_values.append(rhs)
    
    return obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min

def get_test_problem():
    """Возвращает тестовую задачу"""
    obj_coeffs = [5, -6, -1, 3, -8]  # Максимизация
    constraints = [
        [-2, 1, 1, 4, 2],
        [1, 1, 0, -2, 1],
        [-8, 4, 5, 3, -1]
    ]
    rhs_values = [28, 31, 118]
    constraint_types = ['=', '=', '=']
    integer_vars_indices = [2, 3, 4]  # x₃, x₄, x₅ (индексы с 0)
    is_min = False  # Максимизация
    
    return obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min

def solve_integer_problem(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min=False):
    """
    Решение задачи методом Гомори
    """
    print("=" * 70)
    print("МЕТОД ГОМОРИ ДЛЯ ЦЕЛОЧИСЛЕННОГО ПРОГРАММИРОВАНИЯ")
    print("=" * 70)
    
    # Копируем исходные данные
    current_obj = obj_coeffs[:]
    current_constraints = [row[:] for row in constraints]
    current_rhs = rhs_values[:]
    current_types = constraint_types[:]
    
    iteration = 0
    max_iterations = 20
    
    while iteration < max_iterations:
        print(f"\n{'='*50}")
        print(f"ИТЕРАЦИЯ {iteration + 1}")
        print(f"{'='*50}")
        
        try:
            # Решаем задачу симплекс-методом
            print("\nЗапускаем симплекс-метод для текущей задачи...")
            solution, objective_value, solver = solve_with_simplex(
                current_obj, current_constraints, current_rhs, current_types, is_min, max_steps=100
            )
            
            print(f"\nРЕЗУЛЬТАТ СИМПЛЕКС-МЕТОДА:")
            for i, val in enumerate(solution):
                print(f"  x{i+1} = {val:.6f}")
            print(f"Значение целевой функции: {objective_value:.6f}")
            
            # Проверяем целочисленность указанных переменных
            all_integer = True
            non_integer_vars = []
            for var_idx in integer_vars_indices:
                if var_idx < len(solution):
                    if not is_integer(solution[var_idx]):
                        all_integer = False
                        fraction = solution[var_idx] - np.floor(solution[var_idx])
                        non_integer_vars.append((var_idx, fraction, solution[var_idx]))
                        print(f"  x{var_idx+1} = {solution[var_idx]:.6f} - НЕ ЦЕЛАЯ (дробная часть: {fraction:.6f})")
                    else:
                        print(f"  x{var_idx+1} = {solution[var_idx]:.6f} - целая")
            
            if all_integer:
                print("\n✓ Все указанные переменные целые! Решение найдено.")
                return solution, objective_value, iteration + 1
            
            # Формируем отсечение Гомори
            print("\nФОРМИРУЕМ ОТСЕЧЕНИЕ ГОМОРИ...")
            cut_result = gomory_cut(solver, integer_vars_indices)
            
            if cut_result is None:
                print("Не удалось сформировать отсечение")
                # Пробуем использовать другую нецелую переменную
                if non_integer_vars:
                    print("Пробуем использовать другую нецелую переменную...")
                    # Убираем первую переменную из списка и пробуем снова
                    temp_indices = [v[0] for v in non_integer_vars[1:]]
                    if temp_indices:
                        cut_result = gomory_cut(solver, temp_indices)
                
                if cut_result is None:
                    print("Не удалось сформировать отсечение ни для одной переменной")
                    break
                    
            cut_coeffs, cut_rhs = cut_result
            
            # Добавляем новое ограничение
            print(f"\nДОБАВЛЯЕМ НОВОЕ ОГРАНИЧЕНИЕ:")
            constraint_str = " + ".join([f"{coeff:.6f}*x{i+1}" 
                                       for i, coeff in enumerate(cut_coeffs) 
                                       if abs(coeff) > 1e-10])
            print(f"  {constraint_str} >= {cut_rhs:.6f}")
            
            # Добавляем новое ограничение к задаче
            current_constraints.append(cut_coeffs)
            current_rhs.append(cut_rhs)
            current_types.append('>=')
            
            iteration += 1
            print(f"\nНовая задача имеет {len(current_constraints)} ограничений")
            
        except Exception as e:
            print(f"Ошибка на итерации {iteration + 1}: {e}")
            # Если это первая итерация и произошла ошибка, прерываем выполнение
            if iteration == 0:
                raise e
            else:
                # На последующих итерациях возвращаем последнее найденное решение
                print("Возвращаем последнее найденное решение...")
                return solution, objective_value, iteration
    
    if iteration >= max_iterations:
        print(f"\nДостигнуто максимальное количество итераций ({max_iterations})")
    
    # Возвращаем последнее найденное решение
    return solution, objective_value, iteration

def main():
    """Главная функция с выбором способа ввода"""
    print("=" * 70)
    print("МЕТОД ГОМОРИ - РЕШЕНИЕ ЦЕЛОЧИСЛЕННЫХ ЗАДАЧ")
    print("=" * 70)
    
    while True:
        print("\nВыберите способ ввода данных:")
        print("1. Решить тестовую задачу")
        print("2. Ввести данные с консоли")
        print("3. Загрузить из файла")
        print("4. Выход")
        
        choice = input("Ваш выбор (1-4): ").strip()
        
        if choice == '1':
            # Тестовая задача
            print("\n" + "="*50)
            print("РЕШЕНИЕ ТЕСТОВОЙ ЗАДАЧИ")
            print("="*50)
            
            try:
                obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min = get_test_problem()
                print_problem_info(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min)
                
                solution, objective_value, iterations = solve_integer_problem(
                    obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min
                )
                
                # Вывод результатов
                print_results(solution, objective_value, iterations, integer_vars_indices)
                
            except Exception as e:
                print(f"Ошибка при решении тестовой задачи: {e}")
                continue
            
        elif choice == '2':
            # Ввод с консоли
            print("\n" + "="*50)
            print("ВВОД ДАННЫХ С КОНСОЛИ")
            print("="*50)
            
            try:
                obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min = input_problem_from_console()
                print_problem_info(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min)
                
                solution, objective_value, iterations = solve_integer_problem(
                    obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min
                )
                
                # Вывод результатов
                print_results(solution, objective_value, iterations, integer_vars_indices)
                
            except Exception as e:
                print(f"Ошибка при вводе данных: {e}")
                continue
                
        elif choice == '3':
            # Загрузка из файла
            print("\n" + "="*50)
            print("ЗАГРУЗКА ИЗ ФАЙЛА")
            print("="*50)
            
            try:
                file_path = input("Введите путь к файлу: ").strip()
                obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min = read_problem_from_file(file_path)
                print_problem_info(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min)
                
                solution, objective_value, iterations = solve_integer_problem(
                    obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min
                )
                
                # Вывод результатов
                print_results(solution, objective_value, iterations, integer_vars_indices)
                
            except FileNotFoundError:
                print("Ошибка: Файл не найден!")
                continue
            except Exception as e:
                print(f"Ошибка при решении задачи: {e}")
                continue
                
        elif choice == '4':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
            continue

def print_results(solution, objective_value, iterations, integer_vars_indices):
    """Вывод результатов решения"""
    print("\n" + "=" * 70)
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
    print("=" * 70)
    print("Оптимальное решение:")
    for i, val in enumerate(solution):
        print(f"  x{i+1} = {val:.6f}")
    print(f"Значение целевой функции: {objective_value:.6f}")
    print(f"Количество итераций метода Гомори: {iterations}")
    
    # Проверка целочисленности
    print("\nПроверка целочисленности:")
    all_integer = True
    for var_idx in integer_vars_indices:
        if var_idx < len(solution):
            is_int = is_integer(solution[var_idx])
            status = "ЦЕЛАЯ" if is_int else "НЕ ЦЕЛАЯ"
            print(f"  x{var_idx+1} = {solution[var_idx]:.6f} - {status}")
            if not is_int:
                all_integer = False
    
    if all_integer:
        print("\n✓ ЗАДАЧА РЕШЕНА - ВСЕ ПЕРЕМЕННЫЕ ЦЕЛЫЕ!")
    else:
        print("\n⚠ ЗАДАЧА НЕ РЕШЕНА - НЕ ВСЕ ПЕРЕМЕННЫЕ ЦЕЛЫЕ!")

if __name__ == "__main__":
    main()