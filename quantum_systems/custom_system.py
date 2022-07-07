import warnings

from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)


def construct_pyscf_system_ao(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=True,
    np=None,
    **kwargs,
):
    """Convenience function setting up an atom or a molecule from PySCF as a
    ``QuantumSystem``.

    Parameters
    ----------
    molecule : str
        String describing the atom or molecule. This gets passed to PySCF which
        means that we support all the same string options as PySCF.
    basis : str
        String describing the basis set. PySCF determines which options are
        available.
    add_spin : bool
        Whether or not to return a ``SpatialOrbitalSystem`` (``False``) or a
        ``GeneralOrbitalSystem`` (``True``). Default is ``True``.
    anti_symmetrize : bool
        Whether or not to anti-symmetrize the two-body elements in a
        ``GeneralOrbitalSystem``. This only applies if ``add_spin = True``.
        Default is ``True``.
    np : module
        Array- and linear algebra module.

    Returns
    -------
    SpatialOrbitalSystem, GeneralOrbitalSystem
        Depending on the choice of ``add_spin`` we return a
        ``SpatialOrbitalSystem`` (``add_spin = False``), or a
        ``GeneralOrbitalSystem`` (``add_spin = True``).

    See Also
    -------
    PySCF

    Example
    -------
    >>> # Set up the Beryllium atom centered at (0, 0, 0)
    >>> system = construct_pyscf_system_ao(
    ...     "be 0 0 0", basis="cc-pVDZ", add_spin=False
    ... )
    >>> # Compare the number of occupied basis functions
    >>> system.n == 4 // 2
    True
    >>> gos = system.construct_general_orbital_system()
    >>> gos.n == 4
    True
    """

    import pyscf

    if np is None:
        import numpy as np

    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.build(atom=molecule, basis=basis, **kwargs)
    nuclear_repulsion_energy = mol.energy_nuc()

    n = mol.nelectron
    l = mol.nao

    # n_a = (mol.nelectron + mol.spin) // 2
    # n_b = n_a - mol.spin

    # assert n_b == n - n_a

    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
    position = mol.intor("int1e_r").reshape(3, l, l)

    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = position
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(n, bs)

    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )


def construct_pyscf_system_rhf(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=True,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    **kwargs,
):
    """Convenience function setting up a closed-shell atom or a molecule from
    PySCF as a ``QuantumSystem`` in RHF-basis using PySCF's RHF-solver.

    Parameters
    ----------
    molecule : str
        String describing the atom or molecule. This gets passed to PySCF which
        means that we support all the same string options as PySCF.
    basis : str
        String describing the basis set. PySCF determines which options are
        available.
    add_spin : bool
        Whether or not to return a ``SpatialOrbitalSystem`` (``False``) or a
        ``GeneralOrbitalSystem`` (``True``). Default is ``True``.
    anti_symmetrize : bool
        Whether or not to anti-symmetrize the two-body elements in a
        ``GeneralOrbitalSystem``. This only applies if ``add_spin = True``.
        Default is ``True``.
    np : module
        Array- and linear algebra module.

    Returns
    -------
    SpatialOrbitalSystem, GeneralOrbitalSystem
        Depending on the choice of ``add_spin`` we return a
        ``SpatialOrbitalSystem`` (``add_spin = False``), or a
        ``GeneralOrbitalSystem`` (``add_spin = True``).

    See Also
    -------
    PySCF

    Example
    -------
    >>> # Set up the Beryllium atom centered at (0, 0, 0)
    >>> system = construct_pyscf_system_rhf(
    ...     "be 0 0 0", basis="cc-pVDZ", add_spin=False
    ... ) # doctest.ELLIPSIS
    converged SCF energy = -14.5723...
    >>> # Compare the number of occupied basis functions
    >>> system.n == 4 // 2
    True
    >>> gos = system.construct_general_orbital_system()
    >>> gos.n == 4
    True
    >>> system = construct_pyscf_system_rhf(
    ...     "be 0 0 0", basis="cc-pVDZ"
    ... ) # doctest.ELLIPSIS
    converged SCF energy = -14.5723...
    >>> system.n == gos.n
    True
    """

    import pyscf

    if np is None:
        import numpy as np

    # Build molecule in AO-basis
    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = cart
    mol.build(atom=molecule, basis=basis, **kwargs)
    nuclear_repulsion_energy = mol.energy_nuc()

    n = mol.nelectron
    assert (
        n % 2 == 0
    ), "We require closed shell, with an even number of particles"

    l = mol.nao

    hf = pyscf.scf.RHF(mol)
    hf_energy = hf.kernel()

    if not hf.converged:
        warnings.warn("RHF calculation did not converge")

    if verbose:
        print(f"RHF energy: {hf.e_tot}")

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)

    C = np.asarray(hf.mo_coeff)

    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
    position = mol.intor("int1e_r").reshape(3, l, l)

    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = position
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(n, bs)
    system.change_basis(C)

    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )

def construct_quest_system(
    n,
    l,
    h,
    u,
    nuclear_repulsion_energy,
    dip_int, 
    mom_int,
    anti_symmetrize=False,
    np=None,
    verbose=False,
    **kwargs,
):
  
    if np is None:
        import numpy as np

    bs = BasisSet(l, dim=3, np=np, includes_spin=True, anti_symmetrized_u=True)
    bs.h = h
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = -1*dip_int
    bs.momentum = mom_int

    system = GeneralOrbitalSystem(n, bs)   

    return system

def construct_quest_system_rhf(
    n,
    l,
    h,
    s,
    u,
    mo_coef,
    nuclear_repulsion_energy,
    dip_int, 
    anti_symmetrize=False,
    np=None,
    verbose=False,
    **kwargs,
):
    if np is None:
        import numpy as np

    bs = BasisSet(l, dim=3, np=np, includes_spin=False, anti_symmetrized_u=False)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = -1*dip_int

    system = SpatialOrbitalSystem(n, bs)
    system.change_basis(mo_coef)
 
    return system


def construct_quest_system_rhf_mom(
    n,
    l,
    h,
    s,
    u,
    mo_coef,
    nuclear_repulsion_energy,
    dip_int, 
    mom_int,
    anti_symmetrize=False,
    np=None,
    verbose=False,
    **kwargs,
):
    if np is None:
        import numpy as np

    bs = BasisSet(l, dim=3, np=np, includes_spin=False, anti_symmetrized_u=False)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = -1*dip_int
    bs.momentum = mom_int

    system = SpatialOrbitalSystem(n, bs)
    system.change_basis(mo_coef)
 
    return system
