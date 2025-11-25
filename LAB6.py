import numpy as np
from collections import deque
from LAB2 import ArtificialBasisSolver

def is_integer(x, tol=1e-6):
    """Проверка, является ли число целым с заданной точностью"""
    return abs(x - round(x)) < tol

class BranchAndBoundNode:
    """Узел в дереве ветвей и границ"""
    def __init__(self, constraints, rhs_values, constraint_types, depth=0, parent=None, branch_var=None, branch_value=None, branch_type=None):
        self.constraints = constraints
        self.rhs_values = rhs_values
        self.constraint_types = constraint_types
        self.depth = depth
        self.parent = parent
        self.branch_var = branch_var  # Переменная, по которой ветвились
        self.branch_value = branch_value  # Значение ветвления
        self.branch_type = branch_type  # Тип ветвления ('<=', '>=')
        self.solution = None
        self.objective_value = None
        self.is_integer = False
        self.is_feasible = True
    
    def __str__(self):
        info = f"Узел {self.depth}"
        if self.branch_var is not None:
            info += f" | x{self.branch_var+1} {self.branch_type} {self.branch_value}"
        return info

def solve_with_simplex(obj_coeffs, constraints, rhs_values, constraint_types, is_min=True, max_steps=50):
    """Решает задачу симплекс-методом"""
    if not is_min:
        # Преобразуем задачу максимизации в минимизацию
        obj_coeffs_min = [-coeff for coeff in obj_coeffs]
        solver = ArtificialBasisSolver(
            obj_coeffs=obj_coeffs_min,
            constraints=constraints,
            rhs_values=rhs_values,
            constraint_types=constraint_types,
            is_min=True,
            M=10000
        )
        
        solution, objective_value, history = solver.solve(max_steps=max_steps)
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
        
        solution, objective_value, history = solver.solve(max_steps=max_steps)
    
    return solution, objective_value, solver

class BranchAndBoundSolver:
    """Реализация метода ветвей и границ"""
    
    def __init__(self, obj_coeffs, constraints, rhs_values, constraint_types, integer_vars_indices, is_min=True):
        self.obj_coeffs = obj_coeffs
        self.original_constraints = constraints
        self.original_rhs = rhs_values
        self.original_types = constraint_types
        self.integer_vars = integer_vars_indices
        self.is_min = is_min
        
        # Лучшее найденное решение
        self.best_solution = None
        self.best_objective = float('inf') if is_min else -float('inf')
        self.nodes_explored = 0
        
    def solve(self, max_nodes=100):
        """Решение задачи методом ветвей и границ"""
        print("=" * 70)
        print("МЕТОД ВЕТВЕЙ И ГРАНИЦ")
        print("=" * 70)
        
        # Очередь узлов для исследования
        queue = deque()
        
        # Создаем корневой узел
        root_node = BranchAndBoundNode(
            constraints=self.original_constraints[:],
            rhs_values=self.original_rhs[:],
            constraint_types=self.original_types[:],
            depth=0
        )
        
        queue.append(root_node)
        self.nodes_explored = 0
        
        print(f"Корневой узел создан. Целочисленные переменные: {[f'x{i+1}' for i in self.integer_vars]}")
        
        while queue and self.nodes_explored < max_nodes:
            current_node = queue.popleft()
            self.nodes_explored += 1
            
            print(f"\n--- Исследуем {current_node} ---")
            
            # Решаем задачу для текущего узла
            try:
                solution, objective_value, solver = solve_with_simplex(
                    self.obj_coeffs,
                    current_node.constraints,
                    current_node.rhs_values,
                    current_node.constraint_types,
                    self.is_min
                )
                
                current_node.solution = solution
                current_node.objective_value = objective_value
                
                # Проверяем допустимость решения
                if solution is None:
                    current_node.is_feasible = False
                    print("  ✗ Задача недопустима")
                    continue
                
                print(f"  Решение: {[f'x{i+1}={val:.3f}' for i, val in enumerate(solution)]}")
                print(f"  Целевая функция: {objective_value:.6f}")
                
                # Проверяем границы
                if self._is_worse_than_best(objective_value):
                    print("  ✗ Отсекаем по границе")
                    continue
                
                # Проверяем целочисленность
                is_integer_solution = True
                non_integer_vars = []
                
                for var_idx in self.integer_vars:
                    if not is_integer(solution[var_idx]):
                        is_integer_solution = False
                        fraction = solution[var_idx] - np.floor(solution[var_idx])
                        non_integer_vars.append((var_idx, fraction, solution[var_idx]))
                        print(f"  x{var_idx+1} = {solution[var_idx]:.6f} - нецелое")
                
                current_node.is_integer = is_integer_solution
                
                if is_integer_solution:
                    print("  ✓ Целочисленное решение найдено!")
                    self._update_best_solution(solution, objective_value)
                    continue
                
                # Ветвление
                if non_integer_vars:
                    # Выбираем переменную для ветвления (с максимальной дробной частью)
                    branch_var, max_fraction, value = max(non_integer_vars, key=lambda x: min(x[1], 1-x[1]))
                    
                    print(f"  Ветвление по x{branch_var+1} = {value:.6f}")
                    print(f"  Дробная часть: {max_fraction:.6f}")
                    
                    # Создаем два дочерних узла
                    floor_val = np.floor(value)
                    ceil_val = np.ceil(value)
                    
                    # Узел с ограничением x <= floor(value)
                    left_constraints = current_node.constraints + [[1 if i == branch_var else 0 for i in range(len(self.obj_coeffs))]]
                    left_rhs = current_node.rhs_values + [floor_val]
                    left_types = current_node.constraint_types + ['<=']
                    
                    left_node = BranchAndBoundNode(
                        constraints=left_constraints,
                        rhs_values=left_rhs,
                        constraint_types=left_types,
                        depth=current_node.depth + 1,
                        parent=current_node,
                        branch_var=branch_var,
                        branch_value=floor_val,
                        branch_type='<='
                    )
                    
                    # Узел с ограничением x >= ceil(value)
                    right_constraints = current_node.constraints + [[1 if i == branch_var else 0 for i in range(len(self.obj_coeffs))]]
                    right_rhs = current_node.rhs_values + [ceil_val]
                    right_types = current_node.constraint_types + ['>=']
                    
                    right_node = BranchAndBoundNode(
                        constraints=right_constraints,
                        rhs_values=right_rhs,
                        constraint_types=right_types,
                        depth=current_node.depth + 1,
                        parent=current_node,
                        branch_var=branch_var,
                        branch_value=ceil_val,
                        branch_type='>='
                    )
                    
                    # Добавляем узлы в очередь
                    queue.append(left_node)
                    queue.append(right_node)
                    
                    print(f"  Созданы узлы: x{branch_var+1} <= {floor_val} и x{branch_var+1} >= {ceil_val}")
                else:
                    print("  ! Не найдены нецелые переменные для ветвления")
                    
            except Exception as e:
                print(f"  ✗ Ошибка при решении: {e}")
                current_node.is_feasible = False
        
        print(f"\n{'='*70}")
        print("РЕЗУЛЬТАТЫ МЕТОДА ВЕТВЕЙ И ГРАНИЦ")
        print(f"{'='*70}")
        print(f"Исследовано узлов: {self.nodes_explored}")
        
        if self.best_solution is not None:
            print("Найдено целочисленное решение:")
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
                print("✓ Все целочисленные переменные имеют целые значения!")
            else:
                print("⚠ Не все целочисленные переменные целые!")
        else:
            print("Целочисленное решение не найдено")
        
        return self.best_solution, self.best_objective
    
    def _is_worse_than_best(self, objective_value):
        """Проверяет, хуже ли текущее значение целевой функции, чем лучшее найденное"""
        if self.best_solution is None:
            return False
        
        if self.is_min:
            return objective_value > self.best_objective
        else:
            return objective_value < self.best_objective
    
    def _update_best_solution(self, solution, objective_value):
        """Обновляет лучшее найденное решение"""
        if self.best_solution is None:
            self.best_solution = solution
            self.best_objective = objective_value
        else:
            if self.is_min:
                if objective_value < self.best_objective:
                    self.best_solution = solution
                    self.best_objective = objective_value
            else:
                if objective_value > self.best_objective:
                    self.best_solution = solution
                    self.best_objective = objective_value

def get_test_problem():
    """Возвращает тестовую задачу из пятой лабораторной"""
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

def main():
    """Главная функция"""
    print("=" * 70)
    print("МЕТОД ВЕТВЕЙ И ГРАНИЦ - РЕШЕНИЕ ЦЕЛОЧИСЛЕННЫХ ЗАДАЧ")
    print("=" * 70)
    
    # Используем тестовую задачу из пятой лабораторной
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
    solution, objective_value = solver.solve(max_nodes=50)
    
    print(f"\n{'='*70}")
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
    print(f"{'='*70}")
    
    if solution is not None:
        print("Оптимальное целочисленное решение:")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.6f}")
        print(f"Значение целевой функции: {objective_value:.6f}")
    else:
        print("Целочисленное решение не найдено в пределах ограничений")

if __name__ == "__main__":
    main()