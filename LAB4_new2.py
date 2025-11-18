import numpy as np

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


def test_individual_task():
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


def main():
    """Главная функция"""
    print("ЛАБОРАТОРНАЯ РАБОТА 4: ДВОЙСТВЕННЫЕ ЗАДАЧИ")
    print("=" * 70)
    print("ПРОВЕРКА КОРРЕКТНОСТИ ПРЕОБРАЗОВАТЕЛЯ")
    print("=" * 70)
    
    # Тестирование преобразователя на индивидуальном задании
    test_individual_task()


if __name__ == "__main__":
    main()