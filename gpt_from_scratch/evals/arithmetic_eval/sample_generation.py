import random
import dataclasses
import textwrap

import evalugator.structs

import hashlib
import json
import dataclasses


# some example problem generation
def _generate_complex_problem(
    max_depth: int, current_depth: int = 0
) -> tuple[str, int]:
    """
    Recursively generate a complex mathematical problem as a string expression,
    also returning the maximum depth reached.

    Args:
        max_depth (int, optional): The maximum depth of the expression tree. Defaults to 3.
        current_depth (int, optional): The current depth in the expression tree. Defaults to 0.

    Returns:
        tuple: A tuple where the first element is a string representing a mathematical expression,
               and the second element is an integer representing the maximum depth reached.

    Example:
        >>> generate_complex_problem(max_depth=2)
        ('((2 + 3) * (4 - 5))', 2)
    """
    # chosen arbitrarily, this is the probability we stop recursing at this step
    # before reaching max depth
    STOP_PROBABILITY = 0.3

    if current_depth >= max_depth or random.random() < STOP_PROBABILITY:
        return str(random.randint(1, 10)), current_depth

    operations = ["+", "-", "*", "/", "**"]  # Added '**' for exponentiation
    operation = random.choice(operations)

    left_expr, left_depth = _generate_complex_problem(max_depth, current_depth + 1)

    if operation == "**":
        # Limit the exponent to avoid extremely large numbers
        right_expr = str(random.randint(2, 4))
        # Since we're not recursing for the right side, depth is current depth + 1
        right_depth = current_depth + 1
    else:
        right_expr, right_depth = _generate_complex_problem(
            max_depth, current_depth + 1
        )

    max_depth_reached = max(left_depth, right_depth)

    return f"({left_expr} {operation} {right_expr})", max_depth_reached


@dataclasses.dataclass(frozen=True)
class SolvedProblem:
    problem: str
    max_problem_depth: int
    solution: float


def generate_solved_problem(max_depth: int) -> SolvedProblem:
    """Generate a complex mathematical problem and compute its solution.

    Example:
        >>> generate_solved_problem()
        SolvedProblem(problem='((2 + 3) * (4 - 5))', max_problem_depth=2, solution=-5.0)
    """
    # TODO(bschoen): Don't have forever while loops
    while True:

        problem, max_problem_depth = _generate_complex_problem(max_depth=max_depth)

        try:
            # note: normally we'd want something besides `eval`, but this is not model inputs
            # print(f"Evaluating: {problem}")
            solution = eval(problem)

            if isinstance(solution, (int, float)) and -1e10 < solution < 1e10:
                return SolvedProblem(
                    problem=problem,
                    max_problem_depth=max_problem_depth,
                    solution=solution,
                )
            else:
                print(f"Invalid solution: {solution} for {problem}, attempting again")

        except (ValueError, SyntaxError, OverflowError, ZeroDivisionError) as e:
            # If there's an error or the result is too large, generate a new problem
            print(
                f"Encountered: {type(e)} during problem generation for problem "
                f"{problem}, attempting again"
            )
            continue


# TODO(bschoen): Another point for needing a custom type is
#                that we need to check `is_close`
def convert_solved_problem_to_sample(
    solved_problem: SolvedProblem,
) -> evalugator.structs.QA:
    """
    Convert `SolvedProblem` dataclass into evalugator sample

    Note:
        Currently provides `split` keys to later split on via
        `evalugator.eval_results_parse.split_evalresult`:
         - max_depth
         - problem_length

    Note:
        Splits can be computed after the fact with
        `evalugator.eval_results_parse.annotate_evalresult_with_splits`

    """

    system_prompt = textwrap.dedent(
        """
        Respond ONLY with the answer to the given arithmetic problem.

        Example:
        
        <user_input>
        (((2 - 4) - (6 / 3)) + ((1 ** 2) * 3))
        </user_input>

        <assistant_output>
        -1.0
        </assistant_output>
    """
    )

    return evalugator.structs.QA(
        # TODO(bschoen): In reality would want to hash contents of sample, or whatevers not supposed to change
        id=f"id={solved_problem}",
        type="QA",
        # these are fields later accessible via `evalugator.eval_results_parse.split_evalresult`
        splits={
            "max_problem_depth": str(solved_problem.max_problem_depth),
            "problem_length": str(len(solved_problem.problem)),
        },
        # note: this can also just be text if no system prompt supported
        body=[
            evalugator.structs.Message(
                role="system",
                content=system_prompt,
            ),
            evalugator.structs.Message(
                role="user",
                content=f"Problem: {solved_problem.problem}",
            ),
        ],
        ideal_answers=[str(solved_problem.solution)],
        comments=None,
    )
