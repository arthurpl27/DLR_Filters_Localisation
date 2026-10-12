import numpy as np


def gaspari_cohn(r):
    r = np.asarray(r, dtype=float)

    gc = np.piecewise(r, [(0 <= r) & (r <= 1), (1 < r) & (r <= 2), r > 2],
           [lambda x: 1 - (5/3)*x**2 + (5/8)*x**3 + (1/2)*x**4 - (1/4)*x**5,
               lambda x: 4 - 5*x + (5/3)*x**2 + (5/8)*x**3 - (1/2)*x**4 + (1/12)*x**5 - 2/(3*x), 0])
    return gc


def covariance_loc_gc(radius, C, row_positions, col_positions=None, domain_size=None):
    """
    Gaspari-Cohn localisation using physical Euclidean distances.

    Parameters
    ----------
    radius : float
        Localisation radius in physical units.

    C : ndarray, shape (m, n)
        Matrix to localise.

    row_positions : ndarray
        Physical locations associated with the rows of C.
        Shape:
            (m,)      for a 1D mesh
            (m, p)    for a p-dimensional mesh

    col_positions : ndarray, optional
        Physical locations associated with the columns of C.
        If None, the same positions as the rows are used.

    domain_size : float or array-like, optional
        Size of the periodic domain.
        If None, ordinary Euclidean distance is used.

    Returns
    -------
    C_loc : ndarray
        Gaspari-Cohn localised matrix.
    """

    m, n = C.shape

    row_positions = np.asarray(row_positions, dtype=float)

    if col_positions is None:
        col_positions = row_positions

    col_positions = np.asarray(col_positions, dtype=float)

    # Convert 1D coordinates from shape (m,) to shape (m,1)
    if row_positions.ndim == 1:
        row_positions = row_positions[:, None]

    if col_positions.ndim == 1:
        col_positions = col_positions[:, None]

    if row_positions.shape[0] != m:
        raise ValueError("Number of row positions must equal C.shape[0].")

    if col_positions.shape[0] != n:
        raise ValueError("Number of column positions must equal C.shape[1].")

    if row_positions.shape[1] != col_positions.shape[1]:
        raise ValueError("Row and column positions must have the same spatial dimension.")

    # Pairwise physical displacement with [row location, column location, spatial coordinate]
    difference = (row_positions[:, None, :] - col_positions[None, :, :])

    """if domain_size is not None:

        domain_size = np.asarray(domain_size, dtype=float)
        difference = np.abs(difference)
        difference = np.minimum(difference, domain_size - difference)"""

    distance = np.linalg.norm(difference, axis=2)    # Euclidean distance
    rho = gaspari_cohn(distance / radius)

    return rho * C