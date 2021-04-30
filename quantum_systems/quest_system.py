from quantum_systems import QuantumSystem, GeneralOrbitalSystem


class QuestSystem(QuantumSystem):
    r"""Quantum system containing orbital matrix elements, i.e., we only keep
    the spatial orbitals as they are degenerate in each spin direction. We have

    .. math:: \psi(x, t) = \psi(\mathbf{r}, t) \sigma(m_s),

    where :math:`x = (\mathbf{r}, m_s)` is a generalized coordinate of position
    and spin, and :math:`\sigma(m_s)` is either :math:`\alpha(m_s)` or
    :math:`\beta(m_s)` as the two-dimensional spin basis states. This means
    that we only store :math:`\psi(\mathbf{r}, t)`.

    Parameters
    ----------
    n : int
        Number of particles.  Internally ``SpatialOrbitalSystem`` converts
        ``n`` to ``n // 2`` such that ``n`` denotes the number of occupied
        basis functions (half the number of particles). See example in doctest
        below.
    basis_set : BasisSet
        Spatial orbital basis set without explicit spin-dependence.

    See Also
    -------
    QuantumSystem

    Example
    -------
    >>> n = 4 # Four particles
    >>> l = 20 # Twenty basis functions
    >>> dim = 2
    >>> from quantum_systems import RandomBasisSet
    >>> spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    >>> spas.n == n // 2
    True
    """

    def __init__(self, n, basis_set, **kwargs):
        assert (
            n % 2 == 0
        ), "n must be divisable by 2 to be a closed-shell system"

        assert not basis_set.includes_spin, (
            f"{self.__class__.__name__} only supports basis sets without "
            + "spin-dependence."
        )

        super().__init__(n, basis_set, **kwargs)


    def compute_reference_energy(self):
        r"""Function computing the reference energy in an orbital system.
        This is given by

        .. math:: E_0 = \langle \Phi_0 \rvert \hat{H} \lvert \Phi_0 \rangle
            = 2 h^{i}_{i} + 2 u^{ij}_{ij} - u^{ij}_{ji},

        where :math:`\lvert \Phi_0 \rangle` is the reference determinant, and
        :math:`i, j` are occupied indices.

        Returns
        -------
        complex
            The reference energy.
        """

        o, v = self.o, self.v
        
 
        return self.np.trace(self.h[o, o]) + 0.5 * self.np.trace(
             self.np.trace(self.u[o, o, o, o], axis1=1, axis2=3)
        )

    def construct_fock_matrix(self, h, u, f=None):
        r"""Function setting up the restricted Fock matrix in a closed-shell
        orbital basis.

        In an orbital basis we compute:

        .. math:: f^{p}_{q} = h^{p}_{q} + 2 u^{pi}_{qi} - u^{pi}_{iq},

        where :math:`p, q, r, s, ...` run over all indices and :math:`i, j, k,
        l, ...` correspond to the occupied indices. The two-body elements are
        assumed to not be anti-symmetric.

        Parameters
        ----------
        h : np.ndarray
            The one-body matrix elements.
        u : np.ndarray
            The two-body matrix elements.
        f : np.ndarray
            An empty array of the same shape as `h` to be filled with the Fock
            matrix elements. Default is `None` which means that we allocate a
            new matrix.

        Returns
        -------
        np.ndarray
            The filled Fock matrix.
        """
        np = self.np
        o, v = (self.o, self.v) 
        f = np.zeros_like(h)

        f += np.einsum("piqi -> pq", u[:, o, :, o])
        
        f = f + h
         
        return f

    def change_to_hf_basis(self, *args, **kwargs):
        # TODO: Change to RHF-basis.
        raise NotImplementedError("There is currently no RHF implementation")
