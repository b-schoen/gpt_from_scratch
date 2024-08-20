import random
import dataclasses
import argparse


# some example problem generation
def generate_complex_problem(max_depth: int, current_depth: int = 0) -> tuple[str, int]:
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

    left_expr, left_depth = generate_complex_problem(max_depth, current_depth + 1)

    if operation == "**":
        # Limit the exponent to avoid extremely large numbers
        right_expr = str(random.randint(2, 4))
        # Since we're not recursing for the right side, depth is current depth + 1
        right_depth = current_depth + 1
    else:
        right_expr, right_depth = generate_complex_problem(max_depth, current_depth + 1)

    max_depth_reached = max(left_depth, right_depth)

    return f"({left_expr} {operation} {right_expr})", max_depth_reached


@dataclasses.dataclass(frozen=True)
class SolvedProblem:
    problem: str
    max_problem_depth: int
    solution: float


def generate_solved_problem(max_depth: int) -> SolvedProblem:
    """
    Generate a complex mathematical problem and compute its solution.

    Returns:
        Tuple[str, float]: A tuple where the first element is the string representation
                           of the mathematical problem, and the second element is the
                           calculated solution.

    Example:
        >>> generate_problem()
        ('((2 + 3) * (4 - 5))', -5.0)
    """
    # TODO(bschoen): Don't have forever while loops
    while True:

        problem, max_problem_depth = generate_complex_problem(max_depth=max_depth)

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
