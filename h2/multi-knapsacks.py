from ortools.sat.python import cp_model


def get_data(filename: str):
    with open(filename) as f:
        knapsack_capacities = list(map(int, f.readline().strip().split(" ")))
        weights = list(map(int, f.readline().strip().split(" ")))
        values = list(map(int, f.readline().strip().split(" ")))

    assert len(weights) == len(values)
    return knapsack_capacities, weights, values


def main():
    knapsack_capacities, weights, values = get_data("./multi-knapsacks-data/03.txt")

    model = cp_model.CpModel()

    # x[i, k] = 1 if item i is packed in knapsack k.
    x = {}
    for i in range(len(weights)):
        for k in range(len(knapsack_capacities)):
            x[i, k] = model.new_bool_var(f"x_{i}_{k}")

    # Each item i is assigned to at most one knapsack.
    for i in range(len(weights)):
        model.add_at_most_one(x[i, k] for k in range(len(knapsack_capacities)))

    # The amount packed in each knapsack cannot exceed its capacity.
    for k in range(len(knapsack_capacities)):
        model.add(
            sum(x[i, k] * weights[i] for i in range(len(weights)))
            <= knapsack_capacities[k]
        )

    # maximize total value of packed items.
    objective = []
    for i in range(len(weights)):
        for k in range(len(knapsack_capacities)):
            objective.append(cp_model.LinearExpr.term(x[i, k], values[i]))
    model.maximize(cp_model.LinearExpr.sum(objective))

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    # print the solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
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
        print(f"Solution: {status.__str__().split('.')[1]}")
        print(f"Total packed weight: {total_weight}")
        print(f"Total packed value:  {round(solver.objective_value)}")
    else:
        print("The problem does not have an optimal solution.")


if __name__ == "__main__":
    main()
