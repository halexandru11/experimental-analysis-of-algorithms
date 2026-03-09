from ortools.sat.python import cp_model


def main():
    max_gas = 50  # Nitrogen is the limit
    max_chloride = 40  # Chlorine is the limit

    model = cp_model.CpModel()

    gas = model.new_int_var(0, max_gas, "gas")
    chloride = model.new_int_var(0, max_chloride, "chloride")

    model.add(gas + chloride <= 50)  # Nitrogen is the limit
    model.add(3 * gas + 4 * chloride <= 180)  # Hydrogen is the limit
    model.add(chloride <= 40)  # Chlorine is the limit

    objective = [
        cp_model.LinearExpr.term(gas, 40),  # Gas profits 40€
        cp_model.LinearExpr.term(chloride, 50),  # Chloride profits 50€
    ]
    model.maximize(cp_model.LinearExpr.sum(objective))

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL:
        print(f"Gas:       {solver.value(gas):4}")
        print(f"Chloride:  {solver.value(chloride):4}")
        print(f"Objective: {round(solver.objective_value):4}")
    else:
        print("The problem does not have an optimal solution.")


if __name__ == "__main__":
    main()
