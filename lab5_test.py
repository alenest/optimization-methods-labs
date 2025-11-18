import math
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

TOL = 1e-9
MAX_SIMPLEX_ITERS = 200
MAX_GOMORY_CUTS = 25


def read_mode() -> bool:
    s = input("Тип задачи (min/max): ").strip().lower()
    if s not in ("min", "max"):
        print("Ошибка: введите 'min' или 'max'.")
        sys.exit(1)
    return s == "max"


def read_int(prompt: str) -> int:
    try:
        value = int(input(prompt).strip())
        if value <= 0:
            raise ValueError
        return value
    except Exception:
        print("Ошибка: требуется положительное целое число.")
        sys.exit(1)


def read_yes_no(prompt: str) -> bool:
    ans = input(prompt + " [y/n]: ").strip().lower()
    return ans in ("y", "yes", "д", "да")


def read_integer_indices(n: int) -> List[int]:
    line = input(
        "Введите индексы переменных, которые должны быть целыми (через пробел). "
        "Пустая строка — все переменные непрерывны.\n  Integers = "
    )
    if not line.strip():
        return []
    parts = [p for p in line.replace(",", " ").split() if p]
    try:
        idxs = [int(p) - 1 for p in parts]
    except ValueError:
        print("Ошибка: индексы целых переменных должны быть целыми числами.")
        sys.exit(1)
    for idx in idxs:
        if idx < 0 or idx >= n:
            print("Ошибка: индекс целой переменной вне диапазона [1, n].")
            sys.exit(1)
    if len(set(idxs)) != len(idxs):
        print("Ошибка: повторяющиеся индексы целых переменных.")
        sys.exit(1)
    return sorted(idxs)


def parse_floats_line(expected: int, prompt: str) -> List[float]:
    line = input(prompt).strip().replace(",", " ")
    parts = [p for p in line.split() if p]
    if len(parts) != expected:
        print(f"Ошибка: ожидалось {expected} чисел, получено {len(parts)}.")
        sys.exit(1)
    try:
        return [float(x) for x in parts]
    except Exception:
        print("Ошибка: не удалось распарсить числа.")
        sys.exit(1)


def read_basis_indices(m: int, n: int) -> List[int]:
    line = input(
        f"Введите {m} номеров базисных переменных (от 1 до {n}, через пробел): "
    )
    parts = [p for p in line.strip().split() if p]
    if len(parts) != m:
        print("Ошибка: количество номеров базиса не совпадает с числом ограничений.")
        sys.exit(1)
    try:
        basis = [int(p) - 1 for p in parts]
    except ValueError:
        print("Ошибка: индексы базиса должны быть целыми.")
        sys.exit(1)
    if any(idx < 0 or idx >= n for idx in basis):
        print("Ошибка: индексы базиса выходят за пределы [1, n].")
        sys.exit(1)
    if len(set(basis)) != len(basis):
        print("Ошибка: базис содержит повторяющиеся индексы.")
        sys.exit(1)
    return basis


def solve_square_system(matrix: List[List[float]], rhs: List[float]) -> Optional[List[float]]:
    m = len(matrix)
    if m == 0:
        return []
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(m):
        pivot = None
        max_val = 0.0
        for row in range(col, m):
            val = abs(aug[row][col])
            if val > max_val + TOL:
                max_val = val
                pivot = row
        if pivot is None or max_val < TOL:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for j in range(col, m + 1):
            aug[col][j] /= pivot_val
        for row in range(m):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < TOL:
                continue
            for j in range(col, m + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][m] for i in range(m)]


def auto_select_basis(A: List[List[float]], b: List[float]) -> Optional[List[int]]:
    m = len(A)
    n = len(A[0]) if A else 0
    for combo in combinations(range(n), m):
        submatrix = [[A[row][col] for col in combo] for row in range(m)]
        solution = solve_square_system(submatrix, b)
        if solution is None:
            continue
        if all(val >= -1e-8 for val in solution):
            return list(combo)
    return None


def describe_basis(indices: List[int]) -> str:
    return " ".join(f"x{idx + 1}" for idx in indices)


def frac_part(value: float) -> float:
    floor_val = math.floor(value + TOL)
    frac = value - floor_val
    if frac < 0:
        frac += 1.0
    if frac < TOL or 1.0 - frac < TOL:
        return 0.0
    return frac


def is_integer(value: float) -> bool:
    return abs(value - round(value)) <= 1e-7


class SimplexError(Exception):
    """Исключение симплекс-решателя."""


@dataclass
class GomoryCut:
    index: int
    rhs: float
    terms: List[Tuple[str, float]]

    def to_text(self) -> str:
        if not self.terms:
            left = "0"
        else:
            parts = [f"{coeff:.4f}*{name}" for name, coeff in self.terms]
            left = " + ".join(parts)
        return f"{left} >= {self.rhs:.4f}"


class SimplexTableau:
    def __init__(
        self,
        A: List[List[float]],
        b: List[float],
        costs: List[float],
        basis: List[int],
        var_names: List[str],
        verbose: bool = False,
    ):
        self.m = len(A)
        self.n = len(A[0]) if A else 0
        self.verbose = verbose
        self.var_names = var_names[:]
        self.costs = costs[:]
        self.basis = basis[:]
        if self.m > 0:
            row_len = len(A[0])
            for idx, row in enumerate(A):
                if len(row) != row_len:
                    raise SimplexError(f"Строка A[{idx + 1}] имеет некорректную длину.")
        if len(b) != self.m:
            raise SimplexError("Размерность B не совпадает с числом ограничений.")
        if len(self.var_names) != self.n:
            raise SimplexError("Количество имён переменных не совпадает с числом столбцов A.")
        self.tableau = [row[:] + [b[i]] for i, row in enumerate(A)]
        self.tableau.append([0.0] * (self.n + 1))
        self.iteration = 0
        if len(self.costs) != self.n:
            raise SimplexError("Размерность коэффициентов цели не совпадает с числом переменных.")
        if len(self.basis) != self.m:
            raise SimplexError("Количество базисных переменных отличается от числа ограничений.")
        self._canonicalize_basis()
        self._recompute_obj_row()
        if self.verbose:
            self._print_table(None)

    # ---------- Базовые операции ----------
    def _canonicalize_basis(self) -> None:
        for i in range(self.m):
            pivot_col = self.basis[i]
            pivot_val = self.tableau[i][pivot_col]
            if abs(pivot_val) < TOL:
                swap_row = None
                for r in range(i + 1, self.m):
                    if abs(self.tableau[r][pivot_col]) > TOL:
                        swap_row = r
                        break
                if swap_row is None:
                    raise SimplexError(
                        f"Колонка x{pivot_col + 1} не может образовать базис (нулевой столбец)."
                    )
                self.tableau[i], self.tableau[swap_row] = (
                    self.tableau[swap_row],
                    self.tableau[i],
                )
                self.basis[i], self.basis[swap_row] = self.basis[swap_row], self.basis[i]
                pivot_val = self.tableau[i][pivot_col]
            factor = pivot_val
            for c in range(self.n + 1):
                self.tableau[i][c] /= factor
            for r in range(self.m):
                if r == i:
                    continue
                factor = self.tableau[r][pivot_col]
                if abs(factor) < TOL:
                    continue
                for c in range(self.n + 1):
                    self.tableau[r][c] -= factor * self.tableau[i][c]

    def _recompute_obj_row(self) -> None:
        obj_row = [-self.costs[j] for j in range(self.n)] + [0.0]
        for i in range(self.m):
            var_idx = self.basis[i]
            coeff = self.costs[var_idx]
            if abs(coeff) < TOL:
                continue
            for j in range(self.n):
                obj_row[j] += coeff * self.tableau[i][j]
            obj_row[-1] += coeff * self.tableau[i][-1]
        self.tableau[self.m] = obj_row

    def pivot(self, pivot_row: int, pivot_col: int) -> None:
        pivot_val = self.tableau[pivot_row][pivot_col]
        if abs(pivot_val) < TOL:
            raise SimplexError("Попытка деления на нулевой опорный элемент.")
        for c in range(self.n + 1):
            self.tableau[pivot_row][c] /= pivot_val
        for r in range(self.m + 1):
            if r == pivot_row:
                continue
            factor = self.tableau[r][pivot_col]
            if abs(factor) < TOL:
                continue
            for c in range(self.n + 1):
                self.tableau[r][c] -= factor * self.tableau[pivot_row][c]
        self.basis[pivot_row] = pivot_col
        self.iteration += 1

    # ---------- Прямой симплекс ----------
    def _ratio_test(self, pivot_col: int) -> int:
        best_row = None
        best_value = float("inf")
        for i in range(self.m):
            coeff = self.tableau[i][pivot_col]
            if coeff > TOL:
                ratio = self.tableau[i][-1] / coeff
                if ratio < best_value - 1e-15:
                    best_value = ratio
                    best_row = i
        return best_row

    def run_primal_simplex(self, max_iters: int) -> str:
        while max_iters > 0:
            max_iters -= 1
            obj_row = self.tableau[self.m][:-1]
            min_value = min(obj_row)
            if min_value >= -TOL:
                return "optimal"
            pivot_col = min(range(self.n), key=lambda j: (obj_row[j], j))
            if obj_row[pivot_col] >= -TOL:
                return "optimal"
            pivot_row = self._ratio_test(pivot_col)
            if pivot_row is None:
                return "unbounded"
            self.pivot(pivot_row, pivot_col)
            if self.verbose:
                self._print_table(pivot_col)
        return "iteration_limit"

    # ---------- Двойственный симплекс ----------
    def run_dual_simplex(self, max_iters: int) -> bool:
        while max_iters > 0:
            max_iters -= 1
            rhs_values = [self.tableau[i][-1] for i in range(self.m)]
            min_rhs = min(rhs_values)
            if min_rhs >= -TOL:
                return True
            pivot_row = rhs_values.index(min_rhs)
            row = self.tableau[pivot_row]
            candidates = []
            for j in range(self.n):
                coeff = row[j]
                if coeff < -TOL:
                    ratio = self.tableau[self.m][j] / coeff
                    candidates.append((ratio, j))
            if not candidates:
                return False
            pivot_col = min(candidates)[1]
            self.pivot(pivot_row, pivot_col)
            if self.verbose:
                self._print_table(pivot_col)
        return False

    # ---------- Утилиты ----------
    def extract_solution(self, count: int) -> List[float]:
        values = [0.0] * self.n
        for i, var_idx in enumerate(self.basis):
            values[var_idx] = self.tableau[i][-1]
        return values[:count]

    def objective_value(self) -> float:
        return self.tableau[self.m][-1]

    def find_fractional_row(self, integer_mask: List[bool]) -> Tuple[int, float]:
        """
        Возвращает индекс строки с наибольшей дробной частью rhs среди базисных целых переменных.
        """
        best_row = -1
        best_frac = 0.0
        for i in range(self.m):
            var_idx = self.basis[i]
            if var_idx >= len(integer_mask) or not integer_mask[var_idx]:
                continue
            rhs = self.tableau[i][-1]
            frac = frac_part(rhs)
            if frac > best_frac + 1e-9:
                best_frac = frac
                best_row = i
        return best_row, best_frac

    def add_gomory_cut(self, row_idx: int, cut_idx: int) -> GomoryCut:
        rhs_value = self.tableau[row_idx][-1]
        rhs_frac = frac_part(rhs_value)
        if rhs_frac < TOL:
            raise SimplexError("Попытка построить сечение по целой строке.")
        old_n = self.n
        coeffs = [frac_part(self.tableau[row_idx][j]) for j in range(old_n)]
        terms = []
        for j in range(old_n):
            coeff = coeffs[j]
            if coeff > TOL:
                terms.append((self.var_names[j], coeff))
        new_var_name = f"g{cut_idx}"
        self._append_variable(new_var_name, 0.0)
        new_row = [-coeff for coeff in coeffs] + [1.0]
        new_row.append(-rhs_frac)
        self.tableau.insert(self.m, new_row)
        self.basis.append(self.n - 1)
        self.m += 1
        self._recompute_obj_row()
        cut = GomoryCut(cut_idx, rhs_frac, terms)
        if self.verbose:
            print(f"Добавлено отсечение №{cut_idx}: {cut.to_text()}")
            self._print_table(None)
        return cut

    def _append_variable(self, name: str, cost: float) -> None:
        self.var_names.append(name)
        self.costs.append(cost)
        self.n += 1
        for row in self.tableau:
            row.insert(self.n - 1, 0.0)

    def _print_table(self, entering_col: int) -> None:
        headers = ["Базис", "b"] + self.var_names
        widths = [10, 14] + [14] * self.n
        sep = "+" + "+".join("-" * w for w in widths) + "+"
        print("\n" + sep)
        title = f" Итерация {self.iteration} "
        print("|" + title.center(sum(widths) + len(widths) - 1) + "|")
        print(sep)
        header_row = "|"
        for text, w in zip(headers, widths):
            header_row += text.center(w) + "|"
        print(header_row)
        print(sep)
        for i in range(self.m):
            row = "|"
            basis_name = self.var_names[self.basis[i]]
            row += basis_name.center(widths[0]) + "|"
            row += f"{self.tableau[i][-1]: {widths[1]}.6f}" + "|"
            for j in range(self.n):
                row += f"{self.tableau[i][j]: {widths[2]}.6f}" + "|"
            print(row)
        print(sep)
        obj_row = "|"
        obj_row += "Z".center(widths[0]) + "|"
        obj_row += " ".center(widths[1]) + "|"
        for j in range(self.n):
            obj_row += f"{self.tableau[self.m][j]: {widths[2]}.6f}" + "|"
        obj_row += f"{self.tableau[self.m][-1]: {widths[2]}.6f}" + "|"
        print(obj_row)
        print(sep)


def gomory_method(
    A: List[List[float]],
    b: List[float],
    c: List[float],
    basis: List[int],
    maximize: bool,
    integer_vars: List[int],
    verbose: bool,
) -> Tuple[str, List[float], float, List[GomoryCut], str]:
    var_names = [f"x{i + 1}" for i in range(len(c))]
    original_n = len(c)
    costs_for_simplex = c[:] if maximize else [-ci for ci in c]
    tableau = SimplexTableau(A, b, costs_for_simplex, basis, var_names, verbose)
    integer_mask = [False] * original_n
    for idx in integer_vars:
        integer_mask[idx] = True
    cuts: List[GomoryCut] = []
    for cut_idx in range(1, MAX_GOMORY_CUTS + 1):
        status = tableau.run_primal_simplex(MAX_SIMPLEX_ITERS)
        if status == "unbounded":
            return status, [], 0.0, cuts, "Задача неограничена."
        if status == "iteration_limit":
            return status, [], 0.0, cuts, "Превышен лимит итераций симплекс-метода."
        solution = tableau.extract_solution(original_n)
        if all(is_integer(solution[idx]) for idx in integer_vars):
            objective_value = tableau.objective_value() if maximize else -tableau.objective_value()
            return "optimal", solution, objective_value, cuts, ""
        frac_row, frac_val = tableau.find_fractional_row(integer_mask)
        if frac_row == -1:
            objective_value = tableau.objective_value() if maximize else -tableau.objective_value()
            return "optimal", solution, objective_value, cuts, ""
        cut = tableau.add_gomory_cut(frac_row, cut_idx)
        cuts.append(cut)
        if not tableau.run_dual_simplex(MAX_SIMPLEX_ITERS):
            return "infeasible", [], 0.0, cuts, "Получено противоречие после добавления отсечения."
    return (
        "iteration_limit",
        [],
        0.0,
        cuts,
        "Достигнут лимит количества отсечений без нахождения целочисленного решения.",
    )


def main() -> None:
    print("=" * 60)
    print("        ГОМОРИ: ЦЕЛОЧИСЛЕННЫЙ СИМПЛЕКС-РЕШАТЕЛЬ")
    print("=" * 60)
    print("Выберите режим:")
    print("  1 — ручной ввод")
    print("  2 — демонстрационный пример из задания")
    mode = input("Ваш выбор (1/2): ").strip()
    if mode == "2":
        print("\nЗапуск встроенного примера...")
        maximize = True
        n, m = 5, 3
        c = [26.0, 0.0, -30.0, 47.0, -70.0]
        A = [
            [4.0, 0.0, -2.0, 12.0, -7.0],
            [2.0, 14.0, 2.0, 4.0, -7.0],
            [6.0, 28.0, 0.0, 9.0, -21.0],
        ]
        b = [145.0, 121.0, 105.0]
        basis = [0, 2, 4]  # x1, x3, x5 образуют начальный базис
        integer_vars = [2, 3, 4]  # x3, x4, x5 целочисленные
        verbose = True
    else:
        maximize = read_mode()
        n = read_int("Введите количество переменных n (>0): ")
        m = read_int("Введите количество ограничений m (>0): ")
        print("Коэффициенты целевой функции:")
        c = parse_floats_line(n, "  C = ")
        print("Введите матрицу A (m строк по n значений).")
        A = []
        for i in range(m):
            row = parse_floats_line(n, f"  A[{i + 1},:] = ")
            A.append(row)
        b = parse_floats_line(m, f"Введите {m} правых частей B (>=0): ")
        if any(value < -TOL for value in b):
            print("Ошибка: правая часть должна удовлетворять B >= 0.")
            sys.exit(1)
        auto_basis = auto_select_basis(A, b)
        if auto_basis is not None:
            basis = auto_basis
            print("Автоматически выбран допустимый базис:", describe_basis(basis))
        else:
            print("Автоматически подобрать базис не удалось. Введите вручную.")
            basis = read_basis_indices(m, n)
        integer_vars = read_integer_indices(n)
        verbose = read_yes_no("Печать симплекс-таблиц на каждом шаге?")
    status, solution, optimum, cuts, message = gomory_method(
        A, b, c, basis, maximize, integer_vars, verbose
    )
    print("\n" + "=" * 20 + " РЕЗУЛЬТАТ " + "=" * 20)
    if status != "optimal":
        print("Решение не найдено.")
        if message:
            print(message)
        print("=" * 60)
        return
    if cuts:
        print("Добавленные отсечения:")
        for cut in cuts:
            print(f"  #{cut.index}: {cut.to_text()}")
    else:
        print("Отсечения не потребовались.")
    print("Оптимальное решение (x):")
    for idx, value in enumerate(solution, start=1):
        print(f"  x{idx} = {value:.6f}")
    print(f"Оптимум целевой функции ({'max' if maximize else 'min'}) = {optimum:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except SimplexError as exc:
        print(f"Ошибка симплекс-метода: {exc}")
        sys.exit(1)
