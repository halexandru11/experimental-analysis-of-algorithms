from ortools.sat.python import cp_model

ITEM_REPEATED_MAX = 3


def get_data(filename: str):
    with open(filename) as f:
        knapsack_capacities = list(map(int, f.readline().strip().split(" ")))
        weights = list(map(int, f.readline().strip().split(" ")))
        values = list(map(int, f.readline().strip().split(" ")))

    assert len(weights) == len(values)
    return knapsack_capacities, weights, values


def main():
    knapsack_capacities, weights, values = get_data("./multi-knapsacks-data/07.txt")

    model = cp_model.CpModel()

    # x[i, k] how many items i are packed in knapsack k.
    x = {}
    for i in range(len(weights)):
        for k in range(len(knapsack_capacities)):
            x[i, k] = model.new_int_var(0, ITEM_REPEATED_MAX, f"x_{i}_{k}")

    # Each item i is used at most ITEM_REPEATED_MAX times
    for i in range(len(weights)):
        model.add(
            sum(x[i, k] for k in range(len(knapsack_capacities))) <= ITEM_REPEATED_MAX
        )

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
                count = solver.value(x[i, k])
                if count > 0:
                    print(
                        f"{count:1}x Item_{i:02}   weight:{weights[i]:3}   value:{values[i]:3}"
                    )
                    knapsack_weight += count * weights[i]
                    knapsack_value += count * values[i]
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
