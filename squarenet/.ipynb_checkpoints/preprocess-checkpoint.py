from __future__ import annotations

import numpy as np
from typing import Optional, Sequence, Union, Tuple, List

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def to_conventional_standard(
    structure: Structure,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
) -> Structure:
    """Convert to a conventional standard structure and wrap fractional coords into [0,1)."""
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    s_conv = sga.get_conventional_standard_structure()
    return Structure(
        lattice=s_conv.lattice,
        species=[site.specie for site in s_conv.sites],
        coords=np.mod(s_conv.frac_coords, 1.0),
        coords_are_cartesian=False,
        site_properties=s_conv.site_properties,
    )


def make_supercell(
    structure: Structure,
    supercell: Union[Sequence[int], Sequence[Sequence[int]]],
) -> Structure:
    s = structure.copy()
    M = np.array(supercell, dtype=int)
    if M.ndim == 1:
        if M.size != 3:
            raise ValueError("Diagonal supercell must be a 3-element sequence like [na, nb, nc].")
        M = np.diag(M)
    elif M.shape != (3, 3):
        raise ValueError("Supercell matrix must be 3x3 for non-diagonal transforms.")
    s.make_supercell(M, in_place=True)
    return Structure(
        lattice=s.lattice,
        species=[site.specie for site in s.sites],
        coords=np.mod(s.frac_coords, 1.0),
        coords_are_cartesian=False,
        site_properties=s.site_properties,
    )


def make_symmetric_supercell(
    structure: Structure,
    supercell: Union[Sequence[int], Sequence[Sequence[int]]],
) -> Tuple[Structure, Optional[List[int]]]:
    """Create a supercell and recenter Cartesian coords so they are roughly symmetric about 0."""
    s = structure.copy()
    M = np.array(supercell, dtype=int)
    if M.ndim == 1:
        if M.size != 3:
            raise ValueError("Diagonal supercell must be a 3-element sequence like [na, nb, nc].")
        M = np.diag(M)
    elif M.shape != (3, 3):
        raise ValueError("Supercell matrix must be 3x3 for non-diagonal transforms.")

    mapping = None
    try:
        mapping = s.make_supercell(M, in_place=True)
    except TypeError:
        s.make_supercell(M, in_place=True)

    a_p, b_p, c_p = s.lattice.matrix
    center_shift_cart = 0.5 * (a_p + b_p + c_p)
    new_cart = s.cart_coords - center_shift_cart

    s_sym = Structure(
        lattice=s.lattice,
        species=[site.specie for site in s.sites],
        coords=new_cart,
        coords_are_cartesian=True,
        site_properties=s.site_properties,
    )
    return s_sym, mapping


def prepare_structure(
    s_in: Structure,
    *,
    to_conventional: bool = True,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    supercell: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    sym_supercell: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = (3, 3, 3),
) -> Tuple[Structure, Structure, Structure]:
    """Preprocess a structure for geometry screening."""
    s_source = s_in.copy()
    s_conv = to_conventional_standard(s_source, symprec=symprec, angle_tolerance=angle_tolerance) if to_conventional else s_source

    if sym_supercell is not None:
        s_final, _ = make_symmetric_supercell(s_conv, sym_supercell)
    elif supercell is not None:
        s_final = make_supercell(s_conv, supercell)
    else:
        s_final = s_conv

    return s_source, s_conv, s_final
