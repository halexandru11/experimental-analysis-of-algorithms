import random

from ortools.sat.python import cp_model

BOARD_SIZE = 4
BLOCKED = 3
# min cell to leave available on each row/column
MIN_IN_FILE = 2


def print_blocked_board(blocked_cells: list[list[int]]):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if blocked_cells.__contains__([row, col]):
                print("x", end=" ")
            else:
                print("_", end=" ")
        print()


def get_blocked_cells():
    for _ in range(1000):
        queensorder = [[r, c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)]
        random.shuffle(queensorder)

        rows = [[r for r in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        cols = [[c for c in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        answer = []

        for [r, c] in queensorder:
            if len(rows[r]) > MIN_IN_FILE and len(cols[c]) > MIN_IN_FILE:
                rows[r].remove(c)
                cols[c].remove(r)
                answer.append([r, c])
                if len(answer) == BLOCKED:
                    answer.sort()
                    return answer

    return []


class NQueenSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, queens: list[cp_model.IntVar], blocked: list[list[int]]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__queens = queens
        self.__blocked = blocked
        self.__solution_count = 1

    @property
    def solution_count(self) -> int:
        return self.__solution_count

    def on_solution_callback(self):
        print(f"Solution {self.__solution_count}:")
        self.__solution_count += 1

        all_queens = range(len(self.__queens))
        for i in all_queens:
            for j in all_queens:
                if self.value(self.__queens[j]) == i:
                    assert not self.__blocked.__contains__([i, j])
                    # There is a queen in column j, row i.
                    print("Q", end=" ")
                elif self.__blocked.__contains__([i, j]):
                    print("x", end=" ")
                else:
                    print("_", end=" ")
            print()
        print()


def main():
    model = cp_model.CpModel()

    queens = [
        model.new_int_var(0, BOARD_SIZE - 1, f"x_{col}") for col in range(BOARD_SIZE)
    ]

    # All rows must be different.
    model.add_all_different(queens)

    # No two queens can be on the same diagonal.
    model.add_all_different(queens[col] + col for col in range(BOARD_SIZE))
    model.add_all_different(queens[col] - col for col in range(BOARD_SIZE))

    blocked_cells = get_blocked_cells()
    for [row, col] in blocked_cells:
        model.add(queens[col] != row)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solution_printer = NQueenSolutionPrinter(queens, blocked_cells)
    status = solver.Solve(model, solution_printer)

    if status == cp_model.INFEASIBLE:
        print("No Solution found")
        print_blocked_board(blocked_cells)


if __name__ == "__main__":
    main()
