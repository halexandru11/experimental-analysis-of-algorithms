from ortools.sat.python import cp_model


def get_countries():
    return [
        "Belgium",
        "Denmark",
        "France",
        "Germany",
        "Luxembourg",
        "Netherlands",
        "Switzerland",
    ]


def get_colors():
    return ["blue", "white", "yellow", "green"]


def get_neighbors():
    return [
        # same colors
        ("Denmark", "Germany"),
        # different colors
        ("Belgium", "France"),
        ("Belgium", "Germany"),
        ("Belgium", "Luxembourg"),
        ("Belgium", "Netherlands"),
        ("France", "Germany"),
        ("France", "Luxembourg"),
        ("Germany", "Luxembourg"),
        ("Germany", "Netherlands"),
        ("Luxembourg", "Netherlands"),
        # new
        ("Switzerland", "France"),
        ("Switzerland", "Germany"),
    ]


def main():
    model = cp_model.CpModel()

    countries = get_countries()
    colors = get_colors()

    country_vars = {}
    for c in countries:
        country_vars[c] = model.new_int_var(0, len(colors) - 1, c)

    neighbors = get_neighbors()
    same_color_neighbors = neighbors[:1]
    different_color_neighbors = neighbors[1:]
    for c1, c2 in same_color_neighbors:
        model.add(country_vars[c1] == country_vars[c2])
    for c1, c2 in different_color_neighbors:
        model.add(country_vars[c1] != country_vars[c2])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found:")
        for c in countries:
            print(f"    {c}: {colors[solver.Value(country_vars[c])]}")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
