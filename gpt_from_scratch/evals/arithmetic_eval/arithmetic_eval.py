from gpt_from_scratch.evals.arithmetic_eval.sample_generation import (
    generate_solved_problem,
    convert_solved_problem_to_sample,
)

import math
from typing import Any

import evalugator.evals
import evalugator.evals_utils

import pandas as pd


# TODO(bschoen): Handle `output.parsed` == `None` (only possible when parsing is like MCQ)
def _is_correct(row: Any) -> bool:

    actual_answer = row["output_parsed"]
    expected_answer = row["output_correct"]

    # chosen somewhat arbitrarily to reflect close enough values
    sig_figs = 2

    # compare floats (failing if cannot convert to float)
    try:
        return math.isclose(
            round(float(actual_answer), sig_figs),
            round(float(expected_answer), sig_figs),
        )
    except ValueError:
        return False


# TODO(bschoen): Note in practice you'd want creation and running as separate classes, plus loadable samples
def create_and_run_eval(model: str, max_depth: int, num_problems: int) -> pd.DataFrame:

    print("Generating samples...")
    solved_problems = [
        generate_solved_problem(max_depth=max_depth) for _ in range(num_problems)
    ]

    samples = [convert_solved_problem_to_sample(x) for x in solved_problems]

    print("Creating eval...")
    eval_spec = evalugator.evals_utils.get_eval_spec(samples)

    simple_eval = evalugator.evals.SimpleEval(model=model, eval_spec=eval_spec)

    # calls:
    #  self.step(...)
    #  return self.result(...)
    #
    # note: default is to include a canary
    #
    print("Running eval...")
    eval_result = simple_eval.run(comment="no applicable canary, just for pipe flush")

    print("Eval complete, converting results to output format...")
    df = pd.json_normalize(eval_result.model_dump()["sample_results"])

    # fix scores per `is_close``
    df["score"] = df.apply(lambda row: _is_correct(row), axis="columns").astype(float)

    print(f"Completed eval of {len(df)} samples")
    return df
