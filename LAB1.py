try:
    import numpy as np
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    if "numpy" in str(e):
        print("Установите NumPy командой: pip install numpy")
    exit(1)


class SimplexSolver:
    def __init__(self, obj_coeffs, constraints, rhs_values, is_min=True, tol=1e-9):
        if not obj_coeffs:
            raise ValueError("Целевая функция не может быть пустой.")
        if not constraints:
            raise ValueError("Ограничения не могут быть пустыми.")
        if len(rhs_values) != len(constraints):
            raise ValueError("Количество свободных членов должно равняться количеству ограничений.")

        self.orig_obj = np.array(obj_coeffs, dtype=float)
        self.obj = np.array(obj_coeffs, dtype=float)
        self.constraints = np.array(constraints, dtype=float)
        self.rhs = np.array(rhs_values, dtype=float)

        if self.constraints.ndim != 2:
            raise ValueError("Матрица ограничений должна быть двумерной.")
        if self.constraints.shape[1] != len(obj_coeffs):
            raise ValueError("Число переменных в ограничениях не совпадает с количеством в ЦФ.")

        self.rows, self.cols = self.constraints.shape

        if self.rows < 0 or self.cols < 0:
            raise ValueError("Отрицательные размеры не допускаются.")

        self.tolerance = tol
        self.is_min = is_min
        self.tableau = None
        self.history = []
        self._init_tableau()

    def _init_tableau(self):
        """Формируем исходную симплекс-таблицу"""
        try:
            self.tableau = np.zeros((self.rows + 1, self.cols + self.rows + 1), dtype=float)
        except Exception as e:
            raise ValueError(f"Ошибка создания таблицы: {str(e)}")

        # матрица A
        self.tableau[: self.rows, : self.cols] = self.constraints
        # единичная матрица для базисных переменных
        self.tableau[: self.rows, self.cols : self.cols + self.rows] = np.eye(self.rows)
        # свободные члены
        self.tableau[: self.rows, -1] = self.rhs

        # строка целевой функции
        if self.is_min:
            self.tableau[-1, : self.cols] = self.obj
        else:
            self.tableau[-1, : self.cols] = -self.obj

        self.history.append(self.tableau.copy())

    def _pivot_operation(self, pivot_row, pivot_col):
        """Выполняет перестроение таблицы по разрешающему элементу"""
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element
        for r in range(self.tableau.shape[0]):
            if r != pivot_row:
                self.tableau[r, :] -= (
                    self.tableau[r, pivot_col] * self.tableau[pivot_row, :]
                )

    def solve(self, display=False, max_steps=200):
        """Основной цикл решения симплекс-методом. Возвращает решение, значение ЦФ и историю."""
        step = 0
        while True:
            last_row = self.tableau[-1, :-1]
            pivot_col = int(np.argmin(last_row))
            if last_row[pivot_col] >= -self.tolerance:
                break

            # тест отношения для выбора строки
            ratios = np.full(self.rows, np.inf)
            for i in range(self.rows):
                a_ij = self.tableau[i, pivot_col]
                if a_ij > self.tolerance:
                    ratios[i] = self.tableau[i, -1] / a_ij
            pivot_row = int(np.argmin(ratios))
            if ratios[pivot_row] == np.inf:
                raise Exception("Задача не имеет ограничений (unbounded).")

            self._pivot_operation(pivot_row, pivot_col)
            self.history.append(self.tableau.copy())
            step += 1
            if step > max_steps:
                raise Exception("Превышено максимальное число итераций.")

        # определяем значения переменных
        solution_vars = np.zeros(self.cols, dtype=float)
        for j in range(self.cols):
            col_vector = self.tableau[: self.rows, j]
            if np.isclose(col_vector, 1.0, atol=1e-8).sum() == 1 and np.isclose(
                col_vector, 0.0, atol=1e-8
            ).sum() == (self.rows - 1):
                row_index = int(np.where(np.isclose(col_vector, 1.0, atol=1e-8))[0][0])
                solution_vars[j] = self.tableau[row_index, -1]

        objective_value = float(np.dot(self.orig_obj, solution_vars))
        return solution_vars, objective_value, self.history


class SimplexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Симплекс-метод решения задач линейного программирования")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.n_var = tk.IntVar(value=2)
        self.m_var = tk.IntVar(value=2)
        self.problem_type = tk.StringVar(value="min")  # по умолчанию минимизация

        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.Frame(notebook)
        notebook.add(input_frame, text="Ввод данных")

        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="Результаты")

        self.create_input_tab(input_frame)
        self.create_result_tab(result_frame)

    def create_input_tab(self, parent):
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        title_label = ttk.Label(scrollable_frame,
                                text="Решение задачи линейного программирования симплекс-методом",
                                font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=10)

        ttk.Label(scrollable_frame, text="Количество переменных (n):").grid(row=1, column=0, sticky=tk.W, pady=5)
        n_entry = ttk.Entry(scrollable_frame, textvariable=self.n_var, width=10)
        n_entry.grid(row=1, column=1, pady=5, padx=5)

        ttk.Label(scrollable_frame, text="Количество ограничений (m):").grid(row=1, column=2, sticky=tk.W, pady=5)
        m_entry = ttk.Entry(scrollable_frame, textvariable=self.m_var, width=10)
        m_entry.grid(row=1, column=3, pady=5, padx=5)

        ttk.Button(scrollable_frame, text="Создать поля ввода",
                   command=self.create_input_fields).grid(row=2, column=0, columnspan=4, pady=10)

        self.input_fields_frame = ttk.Frame(scrollable_frame)
        self.input_fields_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W)

        ttk.Label(scrollable_frame, text="Тип задачи:").grid(row=4, column=0, sticky=tk.W, pady=10)
        ttk.Radiobutton(scrollable_frame, text="Максимизация", variable=self.problem_type, value="max").grid(row=4,
                                                                                                              column=1,
                                                                                                              sticky=tk.W)
        ttk.Radiobutton(scrollable_frame, text="Минимизация", variable=self.problem_type, value="min").grid(row=4,
                                                                                                              column=2,
                                                                                                              sticky=tk.W)

        ttk.Button(scrollable_frame, text="Решить задачу",
                   command=self.solve_problem).grid(row=5, column=0, columnspan=4, pady=20)

    def create_input_fields(self):
        for widget in self.input_fields_frame.winfo_children():
            widget.destroy()

        n = self.n_var.get()
        m = self.m_var.get()

        if n <= 0 or m <= 0:
            messagebox.showerror("Ошибка", "Количество переменных и ограничений должно быть положительным.")
            return

        ttk.Label(self.input_fields_frame, text="Коэффициенты целевой функции:").grid(row=0, column=0, columnspan=n,
                                                                                      pady=5)

        self.c_entries = []
        for i in range(n):
            ttk.Label(self.input_fields_frame, text=f"x{i + 1}:").grid(row=1, column=i, padx=5)
            entry = ttk.Entry(self.input_fields_frame, width=8)
            entry.grid(row=2, column=i, padx=5, pady=5)
            self.c_entries.append(entry)

        ttk.Label(self.input_fields_frame, text="Ограничения (коэффициенты и свободный член):").grid(row=3, column=0,
                                                                                                    columnspan=n + 1,
                                                                                                    pady=10)

        self.a_entries = []
        self.b_entries = []

        for i in range(m):
            ttk.Label(self.input_fields_frame, text=f"Ограничение {i + 1}:").grid(row=4 + i, column=0, sticky=tk.W,
                                                                                  padx=5, pady=2)

            a_row = []
            for j in range(n):
                entry = ttk.Entry(self.input_fields_frame, width=8)
                entry.grid(row=4 + i, column=1 + j, padx=2, pady=2)
                a_row.append(entry)
            self.a_entries.append(a_row)

            ttk.Label(self.input_fields_frame, text="≤").grid(row=4 + i, column=1 + n, padx=5)

            b_entry = ttk.Entry(self.input_fields_frame, width=8)
            b_entry.grid(row=4 + i, column=2 + n, padx=2, pady=2)
            self.b_entries.append(b_entry)

    def create_result_tab(self, parent):
        self.result_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=80, height=30)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_text.config(state=tk.DISABLED)

    def solve_problem(self):
        try:
            n = self.n_var.get()
            m = self.m_var.get()

            if n <= 0 or m <= 0:
                messagebox.showerror("Ошибка", "Количество переменных и ограничений должно быть положительным.")
                return

            C = [float(entry.get().strip() or 0.0) for entry in self.c_entries]

            A = []
            B = []
            for i in range(m):
                a_row = [float(self.a_entries[i][j].get().strip() or 0.0) for j in range(n)]
                A.append(a_row)
                B.append(float(self.b_entries[i].get().strip() or 0.0))

            if any(len(row) != n for row in A):
                raise ValueError("Некорректная длина строки ограничений.")

            is_min = self.problem_type.get() == "min"

            solver = SimplexSolver(C, A, B, is_min=is_min)
            solution, objective_value, history = solver.solve()

            self.show_results(solution, objective_value, history, C, is_min)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")


    def show_results(self, solution, objective_value, history, obj_coeffs, is_min):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        self.result_text.insert(tk.END, "=" * 60 + "\n")
        self.result_text.insert(tk.END, "РЕЗУЛЬТАТЫ РЕШЕНИЯ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")

        self.result_text.insert(tk.END, "ОПТИМАЛЬНОЕ РЕШЕНИЕ:\n")
        for i, val in enumerate(solution):
            self.result_text.insert(tk.END, f"x{i + 1} = {val:.6f}\n")

        self.result_text.insert(tk.END, f"\nОПТИМУМ ЦЕЛЕВОЙ ФУНКЦИИ: {objective_value:.6f}\n\n")

        self.result_text.insert(tk.END, "=" * 60 + "\n")
        self.result_text.insert(tk.END, "ПРОМЕЖУТОЧНЫЕ СИМПЛЕКС-ТАБЛИЦЫ\n")
        self.result_text.insert(tk.END, "=" * 60 + "\n\n")

        n = len(obj_coeffs)
        m = len(history[0]) - n - 1

        for idx, tableau in enumerate(history):
            self.result_text.insert(tk.END, f"=== Шаг {idx} ===\n")

            # Заголовки
            headers = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["RHS"]
            col_width = 10
            header_line = "".join(f"{h:^{col_width}}" for h in headers)
            self.result_text.insert(tk.END, header_line + "\n")
            self.result_text.insert(tk.END, "-" * len(header_line) + "\n")

            # Строки ограничений
            for i in range(m):
                row_data = []
                for j in range(n + m + 1):
                    row_data.append(f"{tableau[i, j]:8.3f}")
                row_line = "".join(f"{d:^{col_width}}" for d in row_data)
                self.result_text.insert(tk.END, f"Ур{i+1:<2} {row_line}\n")

            # Строка целевой функции
            func_data = []
            for j in range(n + m + 1):
                func_data.append(f"{tableau[m, j]:8.3f}")
            func_line = "".join(f"{d:^{col_width}}" for d in func_data)
            self.result_text.insert(tk.END, f"Функц {''.join(func_line)}\n")

            # Текущие коэффициенты
            c_current = tableau[m, :n]
            self.result_text.insert(tk.END, f"Текущие коэффициенты c: {c_current}\n")

            # Если не последняя таблица — выводим разрешающий элемент
            if idx < len(history) - 1:
                last_row = tableau[m, :-1]
                pivot_col = int(np.argmin(last_row))
                ratios = np.full(m, np.inf)
                for i in range(m):
                    if tableau[i, pivot_col] > 1e-9:
                        ratios[i] = tableau[i, -1] / tableau[i, pivot_col]
                pivot_row = int(np.argmin(ratios))
                pivot_val = tableau[pivot_row, pivot_col]
                self.result_text.insert(tk.END, f"Разрешающий элемент: строка {pivot_row}, столбец {pivot_col} (значение {pivot_val:.6f})\n")

            self.result_text.insert(tk.END, "\n")

        # Итоговая таблица
        self.result_text.insert(tk.END, "=== Конечная таблица ===\n")
        tableau = history[-1]
        headers = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["RHS"]
        col_width = 10
        header_line = "".join(f"{h:^{col_width}}" for h in headers)
        self.result_text.insert(tk.END, header_line + "\n")
        self.result_text.insert(tk.END, "-" * len(header_line) + "\n")

        for i in range(m):
            row_data = []
            for j in range(n + m + 1):
                row_data.append(f"{tableau[i, j]:8.3f}")
            row_line = "".join(f"{d:^{col_width}}" for d in row_data)
            self.result_text.insert(tk.END, f"Ур{i+1:<2} {row_line}\n")

        func_data = []
        for j in range(n + m + 1):
            func_data.append(f"{tableau[m, j]:8.3f}")
        func_line = "".join(f"{d:^{col_width}}" for d in func_data)
        self.result_text.insert(tk.END, f"Функц {''.join(func_line)}\n")

        c_current = tableau[m, :n]
        self.result_text.insert(tk.END, f"Текущие коэффициенты c: {c_current}\n\n")

        self.result_text.insert(tk.END, f"Найденное решение x: {solution}\n")
        self.result_text.insert(tk.END, f"Значение целевой функции (в исходной постановке): {objective_value}\n")

        self.result_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        try:
            n = int(input("Сколько переменных (n): "))
            m = int(input("Сколько ограничений (m): "))

            C = list(map(float, input("Коэффициенты целевой функции (через пробел): ").split()))
            if len(C) != n:
                raise ValueError(f"Ожидалось {n} коэффициентов, получено {len(C)}")

            A = []
            for i in range(m):
                row = list(map(float, input(f"Строка {i + 1}: ").split()))
                if len(row) != n:
                    raise ValueError(f"Ожидалось {n} чисел в строке {i + 1}, получено {len(row)}")
                A.append(row)

            B = list(map(float, input("Введите свободные члены ограничений: ").split()))
            if len(B) != m:
                raise ValueError(f"Ожидалось {m} свободных членов, получено {len(B)}")

            minimize_input = input("Минимизация? (y/n, по умолчанию y): ").strip().lower()
            is_min = minimize_input != 'n'  # по умолчанию минимизация

            print("\n=== Решение задачи ===")
            solver = SimplexSolver(C, A, B, is_min=is_min)
            solution, objective_value, history = solver.solve(display=False)

            print("\n=== ОПТИМАЛЬНОЕ РЕШЕНИЕ ===")
            for i, val in enumerate(solution):
                print(f"x{i + 1} = {val:.6f}")
            print(f"\nОптимум целевой функции: {objective_value:.6f}\n")

            print("=== ИСТОРИЯ ТАБЛИЦ ===")
            n_vars = len(C)
            m_cons = len(A)

            for idx, tableau in enumerate(history):
                print(f"\n=== Шаг {idx} ===")
                headers = [f"x{i+1}" for i in range(n_vars)] + [f"s{i+1}" for i in range(m_cons)] + ["RHS"]
                print("".join(f"{h:>8}" for h in headers))
                print("-" * (9 * len(headers)))

                for i in range(m_cons):
                    row = tableau[i]
                    print(f"Ур{i+1:<2} " + "".join(f"{x:8.3f}" for x in row))

                func_row = tableau[m_cons]
                print(f"Функц " + "".join(f"{x:8.3f}" for x in func_row))

                c_current = func_row[:n_vars]
                print(f"Текущие коэффициенты c: {c_current}")

                if idx < len(history) - 1:
                    last_row = func_row[:-1]
                    pivot_col = int(np.argmin(last_row))
                    ratios = np.full(m_cons, np.inf)
                    for i in range(m_cons):
                        if tableau[i, pivot_col] > 1e-9:
                            ratios[i] = tableau[i, -1] / tableau[i, pivot_col]
                    pivot_row = int(np.argmin(ratios))
                    pivot_val = tableau[pivot_row, pivot_col]
                    print(f"Разрешающий элемент: строка {pivot_row}, столбец {pivot_col} (значение {pivot_val:.6f})")

            print("\n=== Конечная таблица ===")
            final_tableau = history[-1]
            headers = [f"x{i+1}" for i in range(n_vars)] + [f"s{i+1}" for i in range(m_cons)] + ["RHS"]
            print("".join(f"{h:>8}" for h in headers))
            print("-" * (9 * len(headers)))

            for i in range(m_cons):
                row = final_tableau[i]
                print(f"Ур{i+1:<2} " + "".join(f"{x:8.3f}" for x in row))

            func_row = final_tableau[m_cons]
            print(f"Функц " + "".join(f"{x:8.3f}" for x in func_row))
            print(f"Текущие коэффициенты c: {func_row[:n_vars]}")

            print(f"\nНайденное решение x: {solution}")
            print(f"Значение целевой функции (в исходной постановке): {objective_value}")

        except Exception as e:
            print(f"Ошибка: {e}")
    else:
        root = tk.Tk()
        app = SimplexGUI(root)
        root.mainloop()