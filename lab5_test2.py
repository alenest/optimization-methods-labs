from dataclasses import dataclass
from typing import List, Tuple
import sys

TOL = 1e-12


# ---------- Ввод и валидация ----------
AUTO_EXAMPLE = {
    "mode": "min",
    "n": 6,
    "m": 3,
    "C": [9.0, 0.0, 2.0, 0.0, 0.0, -6.0],
    "A": [
        [6.0, 2.0, -3.0, 0.0, 0.0, 6.0],
        [-9.0, 0.0, 2.0, 1.0, 0.0, -2.0],
        [3.0, 0.0, 3.0, 0.0, 1.0, -4.0],
    ],
    "B": [18.0, 24.0, 36.0],
}


def read_mode() -> bool:
    s = (
        input("Введите 'max' для максимизации или 'min' для минимизации: ")
        .strip()
        .lower()
    )
    if s not in ("max", "min"):
        print("Ошибка: введите 'max' или 'min'.")
        sys.exit(1)
    return s == "max"


def read_int(prompt: str) -> int:
    try:
        v = int(input(prompt).strip())
        if v <= 0:
            raise ValueError
        return v
    except Exception:
        print("Ошибка: требуется положительное целое число.")
        sys.exit(1)


def parse_floats_line(expected: int, prompt: str) -> List[float]:
    line = input(prompt).strip().replace(",", " ")
    parts = [p for p in line.split() if p != ""]
    if len(parts) != expected:
        print(f"Ошибка: ожидалось {expected} чисел, получено {len(parts)}.")
        sys.exit(1)
    try:
        return [float(x) for x in parts]
    except Exception:
        print("Ошибка: не удалось распарсить числа.")
        sys.exit(1)


# ---------- Вспомогательные функции ----------
@dataclass
class SimplexResult:
    x: List[float]
    objective: float
    basis: List[str]
    var_names: List[str]
    tableau: List[List[float]]


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]


def solve_linear_system(matrix: List[List[float]], rhs: List[float]) -> List[float]:
    n = len(matrix)
    if any(len(row) != n for row in matrix) or len(rhs) != n:
        raise ValueError("Некорректные размеры системы уравнений.")
    # Создаем расширенную матрицу для метода Гаусса-Жордана
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < TOL:
            raise ValueError(
                "Матрица базиса вырождена, не удается восстановить двойственное решение."
            )
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][-1] for i in range(n)]


def format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6g}"


def format_linear_expression(coeffs: List[float], names: List[str]) -> str:
    terms = []
    for coeff, name in zip(coeffs, names):
        if abs(coeff) < 1e-12:
            continue
        sign = "-" if coeff < 0 else "+"
        abs_coeff = abs(coeff)
        if abs(abs_coeff - 1.0) < 1e-12:
            term_body = name
        else:
            term_body = f"{format_number(abs_coeff)}{name}"
        if not terms:
            prefix = "-" if coeff < 0 else ""
        else:
            prefix = f" {sign} "
        terms.append(prefix + term_body)
    if not terms:
        return "0"
    expr = "".join(terms)
    return expr.strip()


def format_vector(prefix: str, values: List[float]) -> str:
    return ", ".join(
        f"{prefix}{i + 1} = {format_number(val)}" for i, val in enumerate(values)
    )


def build_canonical_problem(
    A: List[List[float]], B: List[float], C: List[float], maximize: bool
) -> Tuple[List[List[float]], List[float], List[str]]:
    m = len(A)
    n = len(C)
    var_names = [f"x{j + 1}" for j in range(n)] + [f"s{i + 1}" for i in range(m)]

    canonical_matrix: List[List[float]] = []
    for i in range(m):
        row = [float(A[i][j]) for j in range(n)]
        slack = [0.0] * m
        slack[i] = 1.0
        row.extend(slack)
        canonical_matrix.append(row)

    if maximize:
        canonical_obj = [-float(ci) for ci in C] + [0.0] * m
    else:
        canonical_obj = [float(ci) for ci in C] + [0.0] * m
    return canonical_matrix, canonical_obj, var_names


def describe_original_primal(
    C: List[float],
    A: List[List[float]],
    B: List[float],
    maximize: bool,
) -> str:
    var_names = [f"x{j + 1}" for j in range(len(C))]
    objective_prefix = "maximize" if maximize else "minimize"
    lines = []
    lines.append(objective_prefix + " f = " + format_linear_expression(C, var_names))
    lines.append("subject to:")
    for i, row in enumerate(A):
        expr = format_linear_expression(row, var_names)
        lines.append(f"  {expr} <= {format_number(B[i])}")
    lines.append("  " + ", ".join(var_names) + " >= 0")
    return "\n".join(lines)


def describe_canonical_primal(
    canonical_obj: List[float],
    canonical_matrix: List[List[float]],
    B: List[float],
    var_names: List[str],
) -> str:
    lines = []
    lines.append("minimize f = " + format_linear_expression(canonical_obj, var_names))
    lines.append("subject to:")
    for i, row in enumerate(canonical_matrix):
        expr = format_linear_expression(row, var_names)
        lines.append(f"  {expr} = {format_number(B[i])}")
    lines.append("  " + ", ".join(var_names) + " >= 0")
    return "\n".join(lines)


def describe_canonical_dual(
    canonical_obj: List[float],
    canonical_matrix: List[List[float]],
    B: List[float],
) -> str:
    m = len(B)
    var_names = [f"y{i + 1}" for i in range(m)]
    lines = []
    lines.append("maximize g = " + format_linear_expression(B, var_names))
    lines.append("subject to:")
    transposed = transpose(canonical_matrix)
    for j, column in enumerate(transposed):
        expr = format_linear_expression(column, var_names)
        rhs = format_number(canonical_obj[j])
        lines.append(f"  {expr} <= {rhs}")
    lines.append("  y1, ..., y" + str(m) + " принадлежат R (переменные свободны)")
    return "\n".join(lines)


def describe_dual_original(
    A: List[List[float]],
    B: List[float],
    C: List[float],
    maximize: bool,
) -> str:
    m = len(B)
    dual_vars = [f"y{i + 1}" for i in range(m)]
    lines = []
    if maximize:
        lines.append("minimize g = " + format_linear_expression(B, dual_vars))
        lines.append("subject to:")
        columns = transpose(A)
        for j, column in enumerate(columns):
            expr = format_linear_expression(column, dual_vars)
            rhs = format_number(C[j])
            lines.append(f"  {expr} >= {rhs}")
        lines.append("  " + ", ".join(dual_vars) + " >= 0")
    else:
        # For минимизация с ограничениями <= выводим формулировку через каноническую двойственную
        lines.append(
            "Двойственная формулировка приводится для канонической формы (y свободны):"
        )
    return "\n".join(lines)


def compute_dual_solution(
    canonical_matrix: List[List[float]],
    canonical_obj: List[float],
    B: List[float],
    basis: List[str],
    var_names: List[str],
) -> Tuple[List[float], float]:
    m = len(B)
    index_map = {name: idx for idx, name in enumerate(var_names)}
    basis_indices = []
    for name in basis:
        if name not in index_map:
            raise ValueError(
                f"Переменная {name} не найдена при восстановлении двойственного решения."
            )
        basis_indices.append(index_map[name])

    basis_matrix = [
        [canonical_matrix[row][col_idx] for col_idx in basis_indices]
        for row in range(m)
    ]
    c_B = [canonical_obj[col_idx] for col_idx in basis_indices]
    dual = solve_linear_system(transpose(basis_matrix), c_B)
    dual_obj = sum(B[i] * dual[i] for i in range(m))
    return dual, dual_obj


def extract_full_solution(result: SimplexResult, m: int) -> List[float]:
    total_vars = len(result.var_names)
    full_solution = [0.0] * total_vars
    index_map = {name: idx for idx, name in enumerate(result.var_names)}
    for row_idx in range(m):
        name = result.basis[row_idx]
        value = result.tableau[row_idx][-1]
        full_solution[index_map[name]] = value
    return full_solution


# ---------- Вывод симплекс-таблицы ----------
def print_table(
    tableau: List[List[float]],
    var_names: List[str],
    basis: List[str],
    m: int,
    n: int,
    iter_num: int,
    entering_col: int = None,
):
    """
    Печатает симплекс-таблицу в формате как на картинке.
    entering_col - индекс входящего столбца (или None, если нет).
    """
    cols = len(var_names)  # всего переменных (x + s)
    headers = ["{xj0}", "{bi}", "{Ci0}"] + var_names + ["bi/ais"]

    # ширины колонок
    col_widths = [8, 10, 10] + [10] * cols + [10]

    line_sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    print("\n" + line_sep)
    title = f" Итерация {iter_num} "
    print("|" + title.center(sum(col_widths) + len(col_widths) - 1) + "|")
    print(line_sep)

    # заголовок
    row = "|"
    for h, w in zip(headers, col_widths):
        row += h.center(w) + "|"
    print(row)
    print(line_sep)

    # строки ограничений
    for i in range(m):
        row = "|"
        # базис
        row += basis[i].center(col_widths[0]) + "|"
        # bi
        row += f"{tableau[i][-1]: {col_widths[1]}.6f}" + "|"
        # Ci0 (коэффициент цели для базисной переменной)
        ci0 = 0.0
        if basis[i].startswith("x"):
            idx = int(basis[i][1:]) - 1
            if 0 <= idx < n:
                ci0 = tableau[-1][idx] * -1
        row += f"{ci0: {col_widths[2]}.6f}" + "|"

        # переменные
        for j in range(cols):
            row += f"{tableau[i][j]: {col_widths[3]}.6f}" + "|"

        # bi/ais
        if entering_col is not None:
            a = tableau[i][entering_col]
            if a > TOL:
                ratio = tableau[i][-1] / a
                row += f"{ratio: {col_widths[-1]}.6f}" + "|"
            else:
                row += "—".center(col_widths[-1]) + "|"
        else:
            row += "—".center(col_widths[-1]) + "|"

        print(row)
    print(line_sep)

    # строка цели
    row = "|"
    row += "Z".center(col_widths[0]) + "|"
    row += " ".center(col_widths[1]) + "|"  # bi пусто
    row += " ".center(col_widths[2]) + "|"  # Ci0 пусто
    for j in range(cols):
        row += f"{tableau[m][j]: {col_widths[3]}.6f}" + "|"
    row += f"{tableau[m][-1]: {col_widths[-1]}.6f}" + "|"
    print(row)

    print(line_sep)


# ---------- Симплекс ----------
def simplex(
    A: List[List[float]],
    B: List[float],
    C: List[float],
    maximize: bool,
    max_iters: int = 500,
    show_tables: bool = True,
) -> SimplexResult:
    m = len(A)
    n = len(A[0]) if m > 0 else 0

    if any(b < -TOL for b in B):
        print("Внимание: найдены отрицательные свободные члены B.")
        print("Текущая версия требует B >= 0.")
        sys.exit(1)

    sign_fix = 1.0
    if not maximize:
        C = [-ci for ci in C]
        sign_fix = -1.0

    cols = n + m + 1
    tableau = [[0.0] * cols for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            tableau[i][j] = float(A[i][j])
        tableau[i][n + i] = 1.0
        tableau[i][-1] = float(B[i])

    for j in range(n):
        tableau[m][j] = -float(C[j])

    var_names = [f"x{j + 1}" for j in range(n)] + [f"s{j + 1}" for j in range(m)]
    basis = [f"s{i + 1}" for i in range(m)]

    iter_num = 0
    if show_tables:
        print_table(tableau, var_names, basis, m, n, iter_num, None)

    while True:
        iter_num += 1
        if iter_num > max_iters:
            print("Превышено макс. число итераций.")
            break

        obj_row = tableau[m][:-1]
        min_val = min(obj_row)
        if min_val >= -TOL:
            print("Критерий оптимальности выполнен.")
            break

        pivot_col = min(range(len(obj_row)), key=lambda j: (obj_row[j], j))
        entering = var_names[pivot_col]

        ratios = []
        for i in range(m):
            a = tableau[i][pivot_col]
            if a > TOL:
                ratios.append(tableau[i][-1] / a)
            else:
                ratios.append(float("inf"))
        min_ratio = min(ratios)
        if min_ratio == float("inf"):
            print("Задача неограничена (unbounded).")
            sys.exit(1)
        pivot_row = ratios.index(min_ratio)
        leaving = basis[pivot_row]

        if show_tables:
            print(
                f"\nИтерация {iter_num}: входящая = {entering}, выходящая = {leaving}"
            )

        pivot_val = tableau[pivot_row][pivot_col]
        if abs(pivot_val) < TOL:
            print("Числовая ошибка: опорный элемент близок к нулю.")
            sys.exit(1)
        inv_pivot = 1.0 / pivot_val
        for j in range(cols):
            tableau[pivot_row][j] *= inv_pivot

        for i in range(m + 1):
            if i == pivot_row:
                continue
            factor = tableau[i][pivot_col]
            if abs(factor) < 1e-15:
                continue
            for j in range(cols):
                tableau[i][j] -= factor * tableau[pivot_row][j]

        basis[pivot_row] = entering

        if show_tables:
            print_table(tableau, var_names, basis, m, n, iter_num, pivot_col)

    x = [0.0] * n
    for i in range(m):
        name = basis[i]
        if name.startswith("x"):
            idx = int(name[1:]) - 1
            if 0 <= idx < n:
                x[idx] = tableau[i][-1]

    z = tableau[m][-1] * sign_fix
    final_tableau = [row[:] for row in tableau]
    result = SimplexResult(
        x=x,
        objective=z,
        basis=basis[:],
        var_names=var_names[:],
        tableau=final_tableau,
    )
    return result


# ---------- Основная логика ----------
def main():
    print("=" * 60)
    print("      РЕШАТЕЛЬ СИМПЛЕКС-МЕТОДА")
    print("=" * 60)
    input_choice = (
        input(
            "Выберите режим ввода данных:\n"
            "  1) ручной ввод\n"
            "  2) пример\n"
            "Ваш выбор (1/2, по умолчанию 1): "
        )
        .strip()
        .lower()
    )

    if input_choice == "2":
        maximize = AUTO_EXAMPLE["mode"] == "max"
        n = AUTO_EXAMPLE["n"]
        m = AUTO_EXAMPLE["m"]
        C = AUTO_EXAMPLE["C"][:]
        A = [row[:] for row in AUTO_EXAMPLE["A"]]
        B = AUTO_EXAMPLE["B"][:]
        print("\nИспользуется демонстрационный пример:")
        print(f"  Тип задачи: {'максимизация' if maximize else 'минимизация'}")
        print(
            "  Коэффициенты целевой функции C:", " ".join(format_number(c) for c in C)
        )
        print("  Матрица A:")
        for idx, row in enumerate(A, start=1):
            print(f"    A[{idx},:] = " + " ".join(format_number(v) for v in row))
        print("  Вектор B:", " ".join(format_number(b) for b in B))
    else:
        maximize = read_mode()
        n = read_int("Введите количество переменных n (>0): ")
        m = read_int("Введите количество ограничений m (>0): ")

        C = parse_floats_line(
            n, f"Введите {n} коэффициентов целевой функции C (через пробел): "
        )
        print(
            f"Введите матрицу A ({m} строк по {n} чисел). Каждую строку нажмите Enter:"
        )
        A = []
        for i in range(m):
            row = parse_floats_line(n, f"  A[{i + 1},:] = ")
            A.append(row)
        B = parse_floats_line(
            m, f"Введите {m} свободных коэффициентов B (через пробел): "
        )

    show_tables_answer = (
        input("Показывать промежуточные симплекс-таблицы? (y/n, по умолчанию y): ")
        .strip()
        .lower()
    )
    show_tables = show_tables_answer not in {"n", "no", "н", "нет"}

    canonical_matrix, canonical_obj, canonical_var_names = build_canonical_problem(
        A, B, C, maximize
    )
    initial_basis = [f"s{i + 1}" for i in range(m)]

    print("\nПеред запуском симплекс-метода:")
    print(" - Целевая функция приведена к минимизации (каноническая форма).")
    print(
        f" - Ограничения переписаны в виде равенств с добавлением переменных s1..s{m}."
    )
    print(" - Начальный базис: " + ", ".join(initial_basis))

    print("\nИсходная прямая задача:")
    print(describe_original_primal(C, A, B, maximize))

    print("\nПрямая задача (каноническая форма):")
    print(
        describe_canonical_primal(
            canonical_obj, canonical_matrix, B, canonical_var_names
        )
    )

    print("\nДвойственная задача (для исходной формулировки):")
    print(describe_dual_original(A, B, C, maximize))
    if not maximize:
        print("\nДвойственная задача (каноническая форма):")
        print(describe_canonical_dual(canonical_obj, canonical_matrix, B))

    print("\nЗапуск симплекс-метода...")
    result = simplex(A, B, C, maximize=maximize, max_iters=500, show_tables=show_tables)

    full_solution = extract_full_solution(result, m)

    try:
        dual_solution, dual_value = compute_dual_solution(
            canonical_matrix,
            canonical_obj,
            B,
            result.basis,
            result.var_names,
        )
    except ValueError as exc:
        dual_solution = None
        dual_value = None
        print(f"Предупреждение: {exc}")
    else:
        if maximize:
            dual_solution = [-val for val in dual_solution]
            dual_solution = [0.0 if abs(val) < 1e-12 else val for val in dual_solution]
            dual_value = sum(B[i] * dual_solution[i] for i in range(m))
        else:
            dual_solution = [0.0 if abs(val) < 1e-12 else val for val in dual_solution]

    canonical_value = sum(
        canonical_obj[j] * full_solution[j] for j in range(len(full_solution))
    )

    print("\n" + "=" * 20 + " РЕЗУЛЬТАТ " + "=" * 20)
    print("Прямая задача:")
    print("  " + format_vector("x", result.x))
    print(
        f"  Оптимум целевой функции = {format_number(result.objective)} "
        f"({'максимизация' if maximize else 'минимизация'})"
    )
    print(
        "  Значение целевой функции в канонической форме (минимизация) = "
        + format_number(canonical_value)
    )

    if dual_solution is not None and dual_value is not None:
        dual_problem_type = "минимизация" if maximize else "максимизация"
        if not maximize:
            dual_problem_type += " (каноническая форма)"
        print("\nДвойственная задача:")
        print("  " + format_vector("y", dual_solution))
        print(
            "  Оптимум целевой функции двойственной задачи = "
            + format_number(dual_value)
            + f" ({dual_problem_type})"
        )
    else:
        print("\nДвойственная задача: не удалось восстановить оптимальное решение.")

    print("=" * 60)


if __name__ == "__main__":
    main()
