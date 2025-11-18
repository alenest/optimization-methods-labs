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


if name == "main":
    main()