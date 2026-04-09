import os
import time

from test_benchmarks import BenchmarkRunner, parse_inp_file


def main() -> None:
    data_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\data"
    results_dir = r"c:\experimental-analysis-of-algorithms\water-distribution-networks\Memetic_GA\Attempt_004\results"

    runner = BenchmarkRunner(data_dir, results_dir)

    network_file = "BIN.inp"
    filepath = os.path.join(data_dir, network_file)
    network = parse_inp_file(filepath)

    t0 = time.time()

    best_meme, _, _, _, _ = runner.run_memetic_ga(
        network_file,
        filepath,
        network,
        population_size=70,
        max_generations=250,
        local_search_intensity=0.8,
        use_strict_paper_objective=False,
        enable_early_stopping=False,
        seed=42,
    )
    eval_meme = runner.evaluate_solution(
        network_file,
        filepath,
        best_meme,
        network,
        apply_paper_polish=True,
    )

    best_std, _, _, _, _ = runner.run_standard_ga(
        network_file,
        filepath,
        network,
        population_size=70,
        max_generations=250,
        use_strict_paper_objective=False,
        enable_early_stopping=False,
        seed=42,
    )
    eval_std = runner.evaluate_solution(
        network_file,
        filepath,
        best_std,
        network,
        apply_paper_polish=False,
    )

    elapsed = time.time() - t0

    print("--- BIN 250 GEN SUMMARY ---")
    print(
        f"Memetic fitness={eval_meme['fitness']:.2f}, "
        f"cost={eval_meme['cost']:.2f}, paper_score={eval_meme['paper_score']}"
    )
    print(
        f"Standard fitness={eval_std['fitness']:.2f}, "
        f"cost={eval_std['cost']:.2f}, paper_score={eval_std['paper_score']}"
    )

    improvement = (
        (eval_std["cost"] - eval_meme["cost"]) / eval_std["cost"] * 100.0
        if eval_std["cost"] > 0
        else 0.0
    )
    print(f"Cost improvement (Memetic vs Standard)={improvement:.2f}%")
    print(f"Elapsed seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
