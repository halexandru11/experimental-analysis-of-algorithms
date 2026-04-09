import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


class InstrumentedBinarySearchableSet:
    def __init__(self) -> None:
        self.arrays: list[list[int]] = []
        self.n = 0

    def insert_with_cost(self, x: int) -> int:
        """Insert x and return actual operation cost in element-work units."""
        carry = [x]
        i = 0
        cost = 1  # creating the singleton carry / inserting the new key

        while True:
            if i >= len(self.arrays):
                self.arrays.append([])

            if not self.arrays[i]:
                self.arrays[i] = carry
                break

            # Merging two arrays of equal length L costs 2L work.
            cost += len(self.arrays[i]) + len(carry)
            carry = self._merge(self.arrays[i], carry)
            self.arrays[i] = []
            i += 1

        self.n += 1
        return cost

    def search_with_cost(self, x: int) -> tuple[bool, int]:
        """Search x and return (found, comparison-count cost)."""
        total_cost = 0
        for arr in self.arrays:
            if not arr:
                continue
            found, bs_cost = self._binary_search_with_cost(arr, x)
            total_cost += bs_cost
            if found:
                return True, total_cost
        return False, total_cost

    @staticmethod
    def _binary_search_with_cost(arr: list[int], x: int) -> tuple[bool, int]:
        left = 0
        right = len(arr) - 1
        cost = 0

        while left <= right:
            mid = (left + right) // 2
            cost += 1
            if arr[mid] == x:
                return True, cost
            if arr[mid] < x:
                left = mid + 1
            else:
                right = mid - 1

        return False, cost

    @staticmethod
    def _merge(a: list[int], b: list[int]) -> list[int]:
        merged: list[int] = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                merged.append(a[i])
                i += 1
            else:
                merged.append(b[j])
                j += 1
        merged.extend(a[i:])
        merged.extend(b[j:])
        return merged


def generate_insert_data(num_inserts: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    ds = InstrumentedBinarySearchableSet()
    x_vals: list[int] = []
    insert_costs: list[int] = []

    # Insert unique random values so the internal arrays stay sorted by key.
    values = rng.sample(range(1, 20 * num_inserts + 1), num_inserts)
    for i, v in enumerate(values, start=1):
        c = ds.insert_with_cost(v)
        x_vals.append(i)
        insert_costs.append(c)

    return x_vals, insert_costs


def generate_search_data(max_n: int, points: int, seed: int) -> tuple[list[int], list[float], list[float]]:
    rng = random.Random(seed)
    ns = sorted(set(max(2, int((i / points) * max_n)) for i in range(1, points + 1)))

    avg_costs: list[float] = []
    log2_sq_values: list[float] = []

    for n in ns:
        ds = InstrumentedBinarySearchableSet()
        inserted = rng.sample(range(1, 50 * n + 1), n)
        for v in inserted:
            ds.insert_with_cost(v)

        # Query values outside inserted range to approximate worst-case misses.
        total = 0
        q = 200
        base = 50 * n + 10
        for t in range(q):
            x = base + t
            _, c = ds.search_with_cost(x)
            total += c

        avg_cost = total / q
        avg_costs.append(avg_cost)
        log2_sq_values.append((math.log2(n) ** 2) if n > 1 else 0.0)

    return ns, avg_costs, log2_sq_values


def plot_insert_complexity(x_vals: list[int], insert_costs: list[int], output_path: Path) -> None:
    cumulative_actual = []
    running = 0
    for c in insert_costs:
        running += c
        cumulative_actual.append(running)

    n_log_n = [i * math.log2(i) if i > 1 else 0.0 for i in x_vals]
    # Scale n log n to visually compare with cumulative observed cost.
    scale = cumulative_actual[-1] / n_log_n[-1] if n_log_n[-1] > 0 else 1.0
    ref = [scale * v for v in n_log_n]

    amortized_running = [cumulative_actual[i] / x_vals[i] for i in range(len(x_vals))]
    log_ref = [math.log2(i) if i > 1 else 0.0 for i in x_vals]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(x_vals, cumulative_actual, linewidth=2, label="Cumulative INSERT cost")
    axes[0].plot(x_vals, ref, "--", linewidth=2, label="Reference: scaled n log n")
    axes[0].set_title("Exercise 2 INSERT: cumulative growth (expected O(n log n))")
    axes[0].set_ylabel("Total cost")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_vals, insert_costs, linewidth=1.2, label="Per-insert actual cost (spiky)")
    axes[1].plot(x_vals, amortized_running, linewidth=2, label="Running avg cost per insert")
    axes[1].plot(x_vals, log_ref, "--", linewidth=1.8, label="Reference: log2 n")
    axes[1].set_title("Exercise 2 INSERT: amortized behavior (expected O(log n))")
    axes[1].set_xlabel("Number of insertions")
    axes[1].set_ylabel("Cost")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_search_complexity(ns: list[int], avg_costs: list[float], log2_sq_values: list[float], output_path: Path) -> None:
    scale = avg_costs[-1] / log2_sq_values[-1] if log2_sq_values[-1] > 0 else 1.0
    ref = [scale * v for v in log2_sq_values]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(ns, avg_costs, linewidth=2, label="Measured avg SEARCH cost (miss queries)")
    ax.plot(ns, ref, "--", linewidth=2, label="Reference: scaled (log2 n)^2")
    ax.set_title("Exercise 2 SEARCH: empirical trend vs O((log n)^2)")
    ax.set_xlabel("n elements")
    ax.set_ylabel("Comparison cost")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exercise 2 complexity plots.")
    parser.add_argument("--num-inserts", type=int, default=30000)
    parser.add_argument("--search-max-n", type=int, default=30000)
    parser.add_argument("--search-points", type=int, default=45)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="Directory where plot images are saved",
    )
    args = parser.parse_args()

    x_vals, insert_costs = generate_insert_data(args.num_inserts, args.seed)
    ns, avg_costs, log2_sq_values = generate_search_data(
        args.search_max_n, args.search_points, args.seed + 1
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    insert_path = args.out_dir / "ex2_insert_complexity.png"
    search_path = args.out_dir / "ex2_search_complexity.png"

    plot_insert_complexity(x_vals, insert_costs, insert_path)
    plot_search_complexity(ns, avg_costs, log2_sq_values, search_path)

    print("Exercise 2 plot generation summary")
    print(f"INSERT samples: {len(x_vals)}")
    print(f"SEARCH n-points: {len(ns)}")
    print(f"Final running avg INSERT cost: {sum(insert_costs) / len(insert_costs):.4f}")
    print(f"Final avg SEARCH cost: {avg_costs[-1]:.4f}")
    print(f"Saved: {insert_path}")
    print(f"Saved: {search_path}")


if __name__ == "__main__":
    main()