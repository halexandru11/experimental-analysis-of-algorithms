import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class StepRecord:
    op_index: int
    op_type: str
    actual_cost: float
    accounting_charge: float
    potential_amortized_cost: float
    potential_value: float
    bank_balance: float
    size: int
    capacity: int


class DynamicTableSimulator:
    def __init__(self) -> None:
        self.size = 0
        self.capacity = 1

    @staticmethod
    def potential(size: int, capacity: int) -> float:
        if size >= capacity / 2:
            return 2 * size - capacity
        return capacity / 2 - size

    def insert(self) -> float:
        copy_cost = 0
        if self.size == self.capacity:
            copy_cost = self.size
            self.capacity *= 2
        self.size += 1
        return 1 + copy_cost

    def delete(self) -> float:
        if self.size == 0:
            raise ValueError("DELETE called on empty table")

        self.size -= 1
        actual_cost = 1

        if self.capacity > 1 and self.size <= self.capacity // 4:
            self.capacity //= 2
            actual_cost += self.size

        return actual_cost


def run_random_simulation(num_ops: int, insert_probability: float, seed: int) -> list[StepRecord]:
    rng = random.Random(seed)
    sim = DynamicTableSimulator()

    records: list[StepRecord] = []
    bank_balance = 0.0

    phase_switch = 20000
    for op_index in range(1, num_ops + 1):
        phi_before = sim.potential(sim.size, sim.capacity)

        if op_index <= phase_switch:
            current_insert_prob = insert_probability
        else:
            # After phase switch: increase randomness and adapt to load factor so
            # the table experiences both growth and shrink cycles.
            alpha = sim.size / sim.capacity if sim.capacity > 0 else 0.0
            if alpha > 0.60:
                base_prob = 0.25  # delete-biased while too full
            elif alpha < 0.30:
                base_prob = 0.75  # insert-biased while too empty
            else:
                base_prob = 0.50  # balanced region
            current_insert_prob = min(0.90, max(0.10, base_prob + rng.uniform(-0.15, 0.15)))

        do_insert = sim.size == 0 or rng.random() < current_insert_prob
        if do_insert:
            op_type = "INSERT"
            actual_cost = sim.insert()
            accounting_charge = 3.0
        else:
            op_type = "DELETE"
            actual_cost = sim.delete()
            accounting_charge = 2.0

        phi_after = sim.potential(sim.size, sim.capacity)
        potential_amortized_cost = actual_cost + (phi_after - phi_before)

        bank_balance += accounting_charge - actual_cost

        records.append(
            StepRecord(
                op_index=op_index,
                op_type=op_type,
                actual_cost=actual_cost,
                accounting_charge=accounting_charge,
                potential_amortized_cost=potential_amortized_cost,
                potential_value=phi_after,
                bank_balance=bank_balance,
                size=sim.size,
                capacity=sim.capacity,
            )
        )

    return records


def plot_accounting(records: list[StepRecord], output_path: Path) -> None:
    x = [r.op_index for r in records]
    cumulative_actual = []
    cumulative_charge = []
    running_actual = 0.0
    running_charge = 0.0

    for r in records:
        running_actual += r.actual_cost
        running_charge += r.accounting_charge
        cumulative_actual.append(running_actual)
        cumulative_charge.append(running_charge)

    upper_bound_3n = [3 * i for i in x]
    bank_balance = [r.bank_balance for r in records]
    per_op_actual = [r.actual_cost for r in records]
    per_op_charge = [r.accounting_charge for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(x, cumulative_actual, label="Cumulative actual cost", linewidth=2)
    axes[0].plot(x, cumulative_charge, label="Cumulative charged credits", linewidth=2)
    axes[0].plot(x, upper_bound_3n, "--", label="Reference: 3n", linewidth=2)
    axes[0].set_title("Accounting Method: cumulative behavior")
    axes[0].set_ylabel("Cost")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, per_op_actual, label="Per-op actual cost (squiggly)", linewidth=1.5)
    axes[1].plot(x, per_op_charge, label="Per-op charged credits", linewidth=1.5)
    axes[1].set_title("Accounting Method: per-operation costs")
    axes[1].set_ylabel("Per-op cost")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, bank_balance, color="green", label="Credit bank balance", linewidth=2)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("Accounting Method: bank never going negative shows charges are valid")
    axes[2].set_xlabel("Operation index")
    axes[2].set_ylabel("Saved credits")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_potential(records: list[StepRecord], output_path: Path) -> None:
    x = [r.op_index for r in records]
    cumulative_actual = []
    cumulative_amortized = []
    running_actual = 0.0
    running_amortized = 0.0

    for r in records:
        running_actual += r.actual_cost
        running_amortized += r.potential_amortized_cost
        cumulative_actual.append(running_actual)
        cumulative_amortized.append(running_amortized)

    potential_values = [r.potential_value for r in records]
    per_op_actual = [r.actual_cost for r in records]
    per_op_amortized = [r.potential_amortized_cost for r in records]

    observed_constant = max(per_op_amortized)
    observed_line = [observed_constant * i for i in x]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(x, cumulative_actual, label="Cumulative actual cost", linewidth=2)
    axes[0].plot(x, cumulative_amortized, label="Cumulative amortized (potential)", linewidth=2)
    axes[0].plot(
        x,
        observed_line,
        "--",
        label=f"Reference: ({observed_constant:.2f})n",
        linewidth=2,
    )
    axes[0].set_title("Potential Method: cumulative behavior")
    axes[0].set_ylabel("Cost")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, per_op_actual, label="Per-op actual cost (squiggly)", linewidth=1.5)
    axes[1].plot(x, per_op_amortized, label="Per-op amortized cost", linewidth=1.5)
    axes[1].set_title("Potential Method: per-operation costs")
    axes[1].set_ylabel("Per-op cost")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, potential_values, color="purple", label="Potential value", linewidth=2)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("Potential function stays non-negative")
    axes[2].set_xlabel("Operation index")
    axes[2].set_ylabel("Phi(T,m)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(records: list[StepRecord]) -> None:
    total_actual = sum(r.actual_cost for r in records)
    total_accounting = sum(r.accounting_charge for r in records)
    total_potential_amortized = sum(r.potential_amortized_cost for r in records)

    min_bank = min(r.bank_balance for r in records)
    min_phi = min(r.potential_value for r in records)
    max_amortized = max(r.potential_amortized_cost for r in records)
    num_inserts = sum(1 for r in records if r.op_type == "INSERT")
    num_deletes = len(records) - num_inserts

    num_expansions = 0
    num_contractions = 0
    prev_capacity = 1
    for r in records:
        if r.capacity > prev_capacity:
            num_expansions += 1
        elif r.capacity < prev_capacity:
            num_contractions += 1
        prev_capacity = r.capacity

    print("Simulation summary")
    print(f"Operations: {len(records)}")
    print(f"INSERT operations: {num_inserts}")
    print(f"DELETE operations: {num_deletes}")
    print(f"Expansions (doublings): {num_expansions}")
    print(f"Contractions (halvings): {num_contractions}")
    print(f"Total actual cost: {total_actual:.2f}")
    print(f"Total accounting charged: {total_accounting:.2f}")
    print(f"Total potential amortized: {total_potential_amortized:.2f}")
    print(f"Minimum credit bank balance: {min_bank:.2f}")
    print(f"Minimum potential value: {min_phi:.2f}")
    print(f"Max per-op potential amortized cost: {max_amortized:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate graphs for dynamic table amortized analysis (accounting + potential)."
    )
    parser.add_argument(
        "--ops",
        type=int,
        default=50000,
        help="Number of operations (first 20000 use base behavior, rest use higher randomness)",
    )
    parser.add_argument(
        "--insert-prob",
        type=float,
        default=0.55,
        help="Probability of INSERT when table is non-empty",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="Directory where plot images are saved",
    )
    args = parser.parse_args()

    records = run_random_simulation(args.ops, args.insert_prob, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accounting_path = args.out_dir / "dynamic_table_accounting.png"
    potential_path = args.out_dir / "dynamic_table_potential.png"

    plot_accounting(records, accounting_path)
    plot_potential(records, potential_path)
    print_summary(records)

    print(f"Saved: {accounting_path}")
    print(f"Saved: {potential_path}")


if __name__ == "__main__":
    main()