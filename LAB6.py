import numpy as np
from collections import deque
from LAB2 import ArtificialBasisSolver

def is_integer(x, tol=1e-6):
    """Проверка, является ли число целым с заданной точностью"""
    return abs(x - round(x)) < tol

def validate_solution(solution, constraints, rhs_values, constraint_types, tol=1e-6):
    """Проверяет, что решение удовлетворяет всем ограничениям"""
    if solution is None:
        return False
        
    for i, (coeffs, const_type, rhs) in enumerate(zip(constraints, constraint_types, rhs_values)):
        left_side = sum(coeff * solution[j] for j, coeff in enumerate(coeffs))
        
        if const_type == '<=':
            if left_side > rhs + tol:
                return False
        elif const_type == '>=':
            if left_side < rhs - tol:
                return False
        elif const_type == '=':
            if abs(left_side - rhs) > tol:
                return False
    return True

class BranchAndBoundNode:
    """Узел в дереве ветвей и границ"""
    def __init__(self, constraints, rhs_values, constraint_types, depth=0, parent=None, branch_var=None, branch_value=None, branch_type=None):
        self.constraints = constraints
        self.rhs_values = rhs_values
        self.constraint_types = constraint_types
        self.depth = depth
        self.parent = parent
        self.branch_var = branch_var
        self.branch_value = branch_value
        self.branch_type = branch_type
        self.solution = None
        self.objective_value = None
        self.is_integer = False
        self.is_feasible = True
    
    def get_branch_info(self):
        """Информация о ветвлении для этого узла"""
        if self.branch_var is not None:
            return f"x{self.branch_var+1} {self.branch_type} {int(self.branch_value)} (глубина: {self.depth})"
        return f"корневая задача (глубина: {self.depth})"
    
    def get_state_hash(self):
        """Создает уникальный хэш для состояния задачи"""
        constraints_str = str([str(c) for c in self.constraints])
        rhs_str = str(self.rhs_values)
        types_str = str(self.constraint_types)
        return hash(constraints_str + rhs_str + types_str)

def solve_with_simplex(obj_coeffs, constraints, rhs_values, constraint_types, is_min=True, max_steps=50):
    """Решает задачу симплекс-методом без вывода таблиц"""
    try:
        if not is_min:
            obj_coeffs_min = [-coeff for coeff in obj_coeffs]
            solver = ArtificialBasisSolver(
                obj_coeffs=obj_coeffs_min,
                constraints=constraints,
                rhs_values=rhs_values,
                constraint_types=constraint_types,
                is_min=True,
                M=10000
            )
            
            solution, objective_value, history = solver.solve(max_steps=max_steps, verbose=False)
            objective_value = -objective_value
        else:
            solver = ArtificialBasisSolver(
                obj_coeffs=obj_coeffs,
                constraints=constraints,
                rhs_values=rhs_values,
                constraint_types=constraint_types,
                is_min=True,
                M=10000
            )
            
            solution, objective_value, history = solver.solve(max_steps=max_steps, verbose=False)
        
        # Проверяем, что решение удовлетворяет всем ограничениям
        if solution is not None and not validate_solution(solution, constraints, rhs_values, constraint_types):
            return None, None, None
            
        return solution, objective_value, solver
    except Exception as e:
        return None, None, None

class BranchAndBoundSolver:
    """Реализация метода ветвей и границ"""
    
    def __init__(self, obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min=True):
        self.obj_coeffs = obj_coeffs
        self.original_constraints = constraints
        self.original_rhs = rhs_values
        self.original_types = constraint_types
        self.integer_vars = integer_vars_indices
        self.is_min = is_min
        
        self.best_solution = None
        self.best_objective = float('-inf') if not is_min else float('inf')
        self.nodes_explored = 0
        self.visited_states = set()
        
    def solve(self, max_nodes=100):
        """Решение задачи методом ветвей и границ"""
        print("=" * 70)
        print("МЕТОД ВЕТВЕЙ И ГРАНИЦ")
        print("=" * 70)
        
        # Создаем корневой узел (Задача 0)
        root_node = BranchAndBoundNode(
            constraints=self.original_constraints[:],
            rhs_values=self.original_rhs[:],
            constraint_types=self.original_types[:],
            depth=0
        )
        
        root_hash = root_node.get_state_hash()
        self.visited_states.add(root_hash)
        
        # Используем очередь для обхода в ШИРИНУ
        queue = deque([root_node])
        self.nodes_explored = 0
        
        print("Начало решения.")
        print(f"Целочисленные переменные: {[f'x{i+1}' for i in self.integer_vars]}")
        print(f"Тип задачи: {'минимизация' if self.is_min else 'максимизация'}")
        print(f"Максимальное количество узлов: {max_nodes}")
        print(f"Стратегия обхода: в ширину (BFS)")
        print()
        
        while queue and self.nodes_explored < max_nodes:
            # Берем первый узел из очереди (FIFO)
            current_node = queue.popleft()
            self.nodes_explored += 1
            
            print(f"Узел {self.nodes_explored}: {current_node.get_branch_info()}")
            print("-" * 50)
            
            # Решаем задачу симплекс-методом
            solution, objective_value, solver = solve_with_simplex(
                self.obj_coeffs,
                current_node.constraints,
                current_node.rhs_values,
                current_node.constraint_types,
                self.is_min
            )
            
            # Проверяем допустимость решения
            if solution is None or objective_value is None:
                current_node.is_feasible = False
                print("    Область допустимых решений пуста")
                print()
                continue
                
            current_node.solution = solution
            current_node.objective_value = objective_value
            
            # Выводим решение
            print("    Решение симплекс-методом:")
            for i, val in enumerate(solution):
                print(f"      x{i+1} = {val:.6f}")
            print(f"    Целевая функция: {objective_value:.6f}")
            
            # Проверяем целочисленность указанных переменных
            is_all_integer = True
            candidate_vars = []
            
            for var_idx in self.integer_vars:
                if var_idx < len(solution) and not is_integer(solution[var_idx]):
                    is_all_integer = False
                    fraction = min(solution[var_idx] - np.floor(solution[var_idx]), 
                                 np.ceil(solution[var_idx]) - solution[var_idx])
                    candidate_vars.append((var_idx, fraction, solution[var_idx]))
            
            current_node.is_integer = is_all_integer
            
            # КРИТЕРИЙ ОТСЕЧЕНИЯ 1: Решение полностью целочисленное
            if is_all_integer:
                print("    Найдено целочисленное решение!")
                
                if self._is_better_solution(objective_value):
                    self.best_solution = solution
                    self.best_objective = objective_value
                    print(f"    Новое лучшее решение: {objective_value:.6f}")
                else:
                    print(f"    Решение {objective_value:.6f} не улучшает текущее лучшее {self.best_objective:.6f}")
                print()
                continue
            
            # КРИТЕРИЙ ОТСЕЧЕНИЯ 2: Решение хуже текущего лучшего целочисленного
            if self.best_solution is not None and self._is_worse_solution(objective_value):
                print(f"    Отсекаем - решение {objective_value:.6f} хуже текущего лучшего {self.best_objective:.6f}")
                print()
                continue
            
            # ВЕТВЛЕНИЕ: создаем две новые задачи
            if candidate_vars:
                # Выбираем переменную с максимальной дробной частью
                branching_var, max_fraction, branching_value = max(candidate_vars, key=lambda x: x[1])
                
                floor_val = np.floor(branching_value)
                ceil_val = np.ceil(branching_value)
                
                print(f"    Ветвление по x{branching_var+1} = {branching_value:.6f}:")
                print(f"      Задача 1: x{branching_var+1} <= {int(floor_val)}")
                print(f"      Задача 2: x{branching_var+1} >= {int(ceil_val)}")
                
                # Задача 1: x <= floor(value)
                left_constraints = current_node.constraints + [
                    [1 if i == branching_var else 0 for i in range(len(self.obj_coeffs))]
                ]
                left_rhs = current_node.rhs_values + [floor_val]
                left_types = current_node.constraint_types + ['<=']
                
                left_node = BranchAndBoundNode(
                    constraints=left_constraints,
                    rhs_values=left_rhs,
                    constraint_types=left_types,
                    depth=current_node.depth + 1,
                    parent=current_node,
                    branch_var=branching_var,
                    branch_value=floor_val,
                    branch_type='<='
                )
                
                # Проверяем уникальность состояния
                left_hash = left_node.get_state_hash()
                if left_hash not in self.visited_states:
                    self.visited_states.add(left_hash)
                    queue.append(left_node)
                else:
                    print(f"    Пропускаем дубликат задачи: x{branching_var+1} <= {int(floor_val)}")
                
                # Задача 2: x >= ceil(value)  
                right_constraints = current_node.constraints + [
                    [1 if i == branching_var else 0 for i in range(len(self.obj_coeffs))]
                ]
                right_rhs = current_node.rhs_values + [ceil_val]
                right_types = current_node.constraint_types + ['>=']
                
                right_node = BranchAndBoundNode(
                    constraints=right_constraints,
                    rhs_values=right_rhs,
                    constraint_types=right_types,
                    depth=current_node.depth + 1,
                    parent=current_node,
                    branch_var=branching_var,
                    branch_value=ceil_val,
                    branch_type='>='
                )
                
                # Проверяем уникальность состояния
                right_hash = right_node.get_state_hash()
                if right_hash not in self.visited_states:
                    self.visited_states.add(right_hash)
                    queue.append(right_node)
                else:
                    print(f"    Пропускаем дубликат задачи: x{branching_var+1} >= {int(ceil_val)}")
            else:
                print("    Нет кандидатов для ветвления")
            
            print()
        
        # Финальные результаты
        print("=" * 70)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 70)
        print(f"Исследовано узлов: {self.nodes_explored}")
        print(f"Уникальных состояний: {len(self.visited_states)}")
        
        if self.best_solution is not None:
            print("Найдено оптимальное целочисленное решение:")
            for i, val in enumerate(self.best_solution):
                print(f"  x{i+1} = {val:.6f}")
            print(f"Значение целевой функции: {self.best_objective:.6f}")
            
            # Проверка целочисленности
            all_integer = True
            for var_idx in self.integer_vars:
                if not is_integer(self.best_solution[var_idx]):
                    all_integer = False
                    break
            
            if all_integer:
                print("Все целочисленные переменные имеют целые значения!")
            else:
                print("Не все целочисленные переменные целые!")
                
            # Проверка ограничений
            if validate_solution(self.best_solution, self.original_constraints, self.original_rhs, self.original_types):
                print("Решение удовлетворяет всем ограничениям!")
            else:
                print("Решение НЕ удовлетворяет всем ограничениям!")
        else:
            print("Целочисленное решение не найдено")
        
        return self.best_solution, self.best_objective
    
    def _is_better_solution(self, objective_value):
        """Проверяет, лучше ли текущее решение"""
        if self.best_solution is None:
            return True
        
        if self.is_min:
            return objective_value < self.best_objective
        else:
            return objective_value > self.best_objective
    
    def _is_worse_solution(self, objective_value):
        """Проверяет, хуже ли текущее решение (для отсечения)"""
        if self.best_solution is None:
            return False
        
        if self.is_min:
            return objective_value >= self.best_objective
        else:
            return objective_value <= self.best_objective
        
def get_test_problem():
    """Тестовая задача из пятой лабораторной"""
    obj_coeffs = [5, -6, -1, 3, -8]
    constraints = [
        [-2, 1, 1, 4, 2],
        [1, 1, 0, -2, 1],
        [-8, 4, 5, 3, -1]
    ]
    rhs_values = [28, 31, 118]
    constraint_types = ['=', '=', '=']
    integer_vars_indices = [2, 3, 4]  # x₃, x₄, x₅
    is_min = False  # Максимизация
    
    return obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min

# НОВЫЕ ФУНКЦИИ ДЛЯ ВВОДА ИЗ ФАЙЛА И КОНСОЛИ

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

def print_problem_info(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min):
    """Вывод информации о задаче"""
    print("ИНФОРМАЦИЯ О ЗАДАЧЕ:")
    print("=" * 50)
    
    obj_str = " + ".join([f"{coeff}x{i+1}" for i, coeff in enumerate(obj_coeffs)])
    optimization = "min" if is_min else "max"
    print(f"Целевая функция: {optimization} {obj_str}")
    
    print("Ограничения:")
    for i, (coeffs, const_type, rhs) in enumerate(zip(constraints, constraint_types, rhs_values)):
        constraint_str = " + ".join([f"{coeff}x{j+1}" for j, coeff in enumerate(coeffs) if coeff != 0])
        print(f"  {constraint_str} {const_type} {rhs}")
    
    int_vars_str = ", ".join([f"x{idx+1}" for idx in integer_vars_indices])
    print(f"Целочисленные переменные: {int_vars_str}")
    print()



def main():
    """Главная функция с выбором способа ввода"""
    print("МЕТОД ВЕТВЕЙ И ГРАНИЦ - РЕШЕНИЕ ЦЕЛОЧИСЛЕННЫХ ЗАДАЧ")
    print()
    
    while True:
        print("\nВыберите способ ввода данных:")
        print("1. Решить тестовую задачу")
        print("2. Ввести данные с консоли")
        print("3. Загрузить из файла")
        print("4. Выход")
        
        choice = input("Ваш выбор (1-4): ").strip()
        
        if choice == '1':
            # Тестовая задача (не трогаем!)
            print("\n" + "="*50)
            print("РЕШЕНИЕ ТЕСТОВОЙ ЗАДАЧИ")
            print("="*50)
            
            obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min = get_test_problem()
            print_problem_info(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min)
            
            # Создаем решатель
            solver = BranchAndBoundSolver(
                obj_coeffs=obj_coeffs,
                constraints=constraints,
                rhs_values=rhs_values,
                constraint_types=constraint_types,
                integer_vars_indices=integer_vars_indices,
                is_min=is_min
            )
            
            # Решаем задачу
            solution, objective_value = solver.solve(max_nodes=100)
            
        elif choice == '2':
            # Ввод с консоли
            print("\n" + "="*50)
            print("ВВОД ДАННЫХ С КОНСОЛИ")
            print("="*50)
            
            try:
                obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min = input_problem_from_console()
                print_problem_info(obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min)
                
                # Создаем решатель
                solver = BranchAndBoundSolver(
                    obj_coeffs=obj_coeffs,
                    constraints=constraints,
                    rhs_values=rhs_values,
                    constraint_types=constraint_types,
                    integer_vars_indices=integer_vars_indices,
                    is_min=is_min
                )
                
                # Решаем задачу
                solution, objective_value = solver.solve(max_nodes=100)
                
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
                
                # Создаем решатель
                solver = BranchAndBoundSolver(
                    obj_coeffs=obj_coeffs,
                    constraints=constraints,
                    rhs_values=rhs_values,
                    constraint_types=constraint_types,
                    integer_vars_indices=integer_vars_indices,
                    is_min=is_min
                )
                
                # Решаем задачу
                solution, objective_value = solver.solve(max_nodes=100)
                
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

if __name__ == "__main__":
    main()