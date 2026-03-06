from ortools.sat.python import cp_model


def get_weights():
    return [48, 30, 42, 36, 36, 48, 42, 42, 36, 24, 30, 30, 42, 36, 36]


def get_values():
    return [10, 30, 25, 50, 35, 30, 15, 40, 30, 35, 45, 10, 20, 30, 25]


def get_knapsack_capacities():
    return [100, 100, 100, 100, 100]


def main() -> None:
    knapsack_capacities = get_knapsack_capacities()
    weights = get_weights()
    values = get_values()
    assert len(weights) == len(values)

    model = cp_model.CpModel()

    # Variables.
    # x[i, k] = 1 if item i is packed in knapsack k.
    x = {}
    for i in range(len(weights)):
        for k in range(len(knapsack_capacities)):
            x[i, k] = model.new_bool_var(f"x_{i}_{k}")

    # Constraints.
    # Each item is assigned to at most one knapsack.
    for i in range(len(weights)):
        model.add_at_most_one(x[i, k] for k in range(len(knapsack_capacities)))

    # The amount packed in each knapsack cannot exceed its capacity.
    for k in range(len(knapsack_capacities)):
        model.add(
            sum(x[i, k] * weights[i] for i in range(len(weights)))
            <= knapsack_capacities[k]
        )

    # Objective.
    # maximize total value of packed items.
    objective = []
    for i in range(len(weights)):
        for k in range(len(knapsack_capacities)):
            objective.append(cp_model.LinearExpr.term(x[i, k], values[i]))
    model.maximize(cp_model.LinearExpr.sum(objective))

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL:
        print(f"Total packed value: {solver.objective_value}")
        total_weight = 0
        for k in range(len(knapsack_capacities)):
            print(f"knapsack {k}")
            knapsack_weight = 0
            knapsack_value = 0
            for i in range(len(weights)):
                if solver.value(x[i, k]) > 0:
                    print(f"Item:{i:2}   weight:{weights[i]:3}   value:{values[i]:3}")
                    knapsack_weight += weights[i]
                    knapsack_value += values[i]
            print(f"Packed knapsack weight: {knapsack_weight:3}")
            print(f"Packed knapsack value:  {knapsack_value:3}\n")
            total_weight += knapsack_weight
        print(f"Total packed weight: {total_weight}")
    else:
        print("The problem does not have an optimal solution.")


if __name__ == "__main__":
    main()
