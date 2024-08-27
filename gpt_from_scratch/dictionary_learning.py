"""
Note: See `simple_dictionary_learning.ipynb` annotated implementation


This is primarily here to avoid cluttering the few notebooks where it is used.

"""

import torch

import dataclasses


def orthogonal_matching_pursuit(
    X: torch.Tensor,
    D: torch.Tensor,
    n_nonzero: int,
) -> torch.Tensor:
    """
    OMP is used for the sparse coding step.

    Its role is to find a sparse representation of the input data using the current dictionary.

    * Input: Data matrix X, current dictionary D
    * Output: Sparse coefficient matrix α
    * Goal: Solve the problem X ≈ Dα, where α is sparse

    """

    # Get the number of samples and dictionary atoms
    n_samples = X.size(1)
    n_components = D.size(1)

    # Initialize the sparse coefficient matrix
    alpha = torch.zeros(n_components, n_samples)

    # Iterate over each sample
    for i in range(n_samples):

        # Initialize the residual as the current sample
        residual = X[:, i].clone()

        # Initialize an empty support set (indices of selected atoms)
        support = []

        # Iterate n_nonzero times to select atoms
        for _ in range(n_nonzero):

            # Compute correlations between residual and dictionary atoms
            correlations = torch.abs(torch.mv(D.t(), residual))

            # Select the atom with the highest correlation
            new_atom = torch.argmax(correlations).item()

            # Add the selected atom to the support set
            support.append(new_atom)

            # Get the subdictionary of selected atoms
            D_support = D[:, support]

            # Solve the least squares problem to find coefficients
            # Returns: NamedTuple(solution, residuals, rank, singular_values)
            least_squares_result = torch.linalg.lstsq(D_support, X[:, i].unsqueeze(1))

            # print(f'{least_squares_result=}')
            alpha_support = least_squares_result.solution

            # Ensure alpha_support is a 1D tensor with at least one element
            alpha_support = alpha_support.flatten()

            # Update the residual by subtracting the current approximation
            residual = X[:, i] - D_support @ alpha_support

        # Update the sparse coefficient matrix with the found coefficients
        alpha[support, i] = alpha_support

    # Return the sparse coefficient matrix
    return alpha


def calculate_reconstruction_error(
    X: torch.Tensor,
    D: torch.Tensor,
    alpha: torch.Tensor,
) -> float:
    """
    Literally just ||X - Dα||²

    """
    # Calculate reconstruction error
    X_reconstructed = torch.mm(D, alpha)

    mse = torch.nn.functional.mse_loss(X, X_reconstructed)

    return mse.item()


def calculate_sparsity_loss(alpha: torch.Tensor) -> float:
    """How many of the sparse coefficients have a mean > 0"""

    # essentially how "spread out" is the representation of this sample across the basis
    return torch.mean((alpha != 0).float())


@dataclasses.dataclass
class KSVDResult:
    D: torch.Tensor
    alpha: torch.Tensor


def k_svd(
    X: torch.Tensor,
    n_components: int,
    n_nonzero: int,
    n_iterations: int,
    eval_frequency: int,
) -> KSVDResult:
    """
    K-SVD is the main algorithm for dictionary learning.

    It iteratively updates the dictionary to better represent the input data sparsely.

    * Input: Data matrix X
    * Output: Updated dictionary D and sparse coefficient matrix α
    * Goal: Find D and α that minimize ||X - Dα||² subject to sparsity constraints on α

    """

    print(f"Learning K-SVD for {n_iterations=}...")

    # Get the dimensions of the input data
    n_features, n_samples = X.size()

    # Initialize the dictionary with random values
    D = torch.randn(n_features, n_components)

    # Normalize each dictionary atom to have unit L2 norm
    D = torch.nn.functional.normalize(D, p=2, dim=0)

    # Main K-SVD iteration loop
    for step in range(n_iterations):

        # Sparse coding step: use OMP to find sparse representations
        # Here we use OMP to essentially find the representation of each sample in our new basis `D`
        # print(f'Learning omp for step {step=} for {n_iterations=}')
        alpha = orthogonal_matching_pursuit(X, D, n_nonzero)

        # Dictionary update step: update each atom individually
        for j in range(n_components):

            # Find which samples use the current atom
            I = alpha[j, :] != 0

            if torch.sum(I) == 0:
                continue

            # Temporarily remove the contribution of the current atom
            D[:, j] = 0

            # Compute the residual error for samples using this atom
            E = X[:, I] - torch.mm(D, alpha[:, I])

            # Add back the error caused by setting the current atom to zero
            # Ensure proper broadcasting by using outer product
            E += torch.outer(D[:, j], alpha[j, I])

            # Perform SVD on the error matrix
            U, s, V = torch.svd(E)

            # Update the atom with the first left singular vector
            D[:, j] = U[:, 0]

            # Update the coefficients for this atom
            alpha[j, I] = s[0] * V[:, 0]

        # show reconstruction error each `eval_frequency`
        if ((step + 1) % eval_frequency) == 0:

            mse_loss = calculate_reconstruction_error(X, D, alpha)
            sparsity_loss = calculate_sparsity_loss(alpha)

            print(
                f"[{step + 1}/{n_iterations}] Reconstruction Error (MSE Loss): "
                f"{mse_loss:.4f}, Sparsity loss: {sparsity_loss:.4f}"
            )

    # Return the learned dictionary and sparse coefficients
    return KSVDResult(D, alpha)
