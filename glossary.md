

# Compact field glossary (SquarePlaneResult)

## Identity / grouping
- `axis`: Axis normal to the plane (`"a"`, `"b"`, `"c"`).
- `plane_id`: Index of the plane group along that axis.
- `plane_center_frac`: Median fractional coordinate of plane group along `axis`.
- `species`: Candidate square-net species (element symbol).
- `n_sites`: Number of sites of `species` in this plane group.

## Labels
- `passes`: Base geometric square-net pass (from squareness score fraction).
- `passes2`: Secondary pass (passes + optional numeric/composition/bonding constraints).

## Squareness scores (per-site scores aggregated)
- `pass_fraction`: Fraction of sites with `score >= score_threshold`.
- `mean_score`: Mean squareness score over sites.
- `median_score`: Median squareness score over sites.
- `min_score`: Minimum squareness score over sites.
- `max_score`: Maximum squareness score over sites.

## Intralayer distance features
- `nn_intra_min`: Minimum in-plane nearest-neighbor distance (Å) among `species` sites.
- `nn_intra_mean`: Mean in-plane nearest-neighbor distance (Å) among `species` sites.

## Layer separation / tolerance ratio
- `tol_ratio_any`: `nn_intra_min / min_adj` where `min_adj` depends on `adjacent_by`.

## Adjacent plane (closest-by-atom)
- `min_adj_dist_any_atom`: Minimum atom-to-atom distance (Å) to nearest adjacent plane (prev/next).
- `closest_by_atom_side`: `"prev"` or `"next"` chosen by atom distance.
- `closest_by_atom_plane_id`: Adjacent plane id chosen by atom distance.
- `closest_by_atom_plane_center_frac`: Adjacent plane center (fractional along axis).
- `closest_by_atom_plane_species_counts`: Species histogram of that adjacent plane.
- `closest_by_atom_plane_major_species`: Most common species in that adjacent plane.
- `closest_by_atom_plane_major_fraction`: Fraction of adjacent plane that is the major species.

## Adjacent plane (closest-by-plane-spacing)
- `min_adj_dist_any_plane`: Atom-to-atom distance (Å) to plane chosen by plane spacing.
- `closest_by_plane_side`: `"prev"` or `"next"` chosen by plane spacing.
- `closest_by_plane_plane_id`: Adjacent plane id chosen by plane spacing.
- `closest_by_plane_plane_center_frac`: Adjacent plane center (fractional along axis).
- `closest_by_plane_plane_species_counts`: Species histogram of that adjacent plane.
- `closest_by_plane_plane_major_species`: Most common species in that adjacent plane.
- `closest_by_plane_plane_major_fraction`: Fraction of adjacent plane that is the major species.
- `closest_by_plane_sep_frac`: Fractional separation between plane centers.
- `closest_by_plane_sep_ang`: Plane separation in Å (from reciprocal lattice scaling).

## Squareness vector diagnostics
- `uv_len_err_mean`: Mean relative |u|-|v| mismatch (lower is more square).
- `uv_ang_deg_mean`: Mean angle(u,v) in degrees (near 90 is more square).
- `uv_ang_err_mean`: Mean |angle(u,v) − 90| in degrees.
- `u_len_min`, `u_len_max`: Min/max chosen u-vector length (Å) across sites.
- `v_len_min`, `v_len_max`: Min/max chosen v-vector length (Å) across sites.
- `uv_len_err_min`, `uv_len_err_max`: Min/max length mismatch across sites.
- `uv_ang_deg_min`, `uv_ang_deg_max`: Min/max angle(u,v) across sites.

## Co-plane composition (within the plane group)
- `coplane_species_counts`: Species histogram of the entire plane group.
- `has_coplane_other_species`: True if species besides `species` occur in the plane group.
- `coplane_other_species_counts`: Histogram of non-`species` occupants in plane group.

## Bond filter diagnostic
- `has_out_of_plane_same_species_bond`: True if CrystalNN finds any same-`species` bonded neighbor out of plane.

## CrystalNN nearest-neighbor features
- `cnn_in_plane_nn_dist`: Shortest CrystalNN bond length (Å) classified in-plane.
- `cnn_in_plane_nn_species`: Neighbor species for `cnn_in_plane_nn_dist`.
- `cnn_out_of_plane_nn_dist`: Shortest CrystalNN bond length (Å) classified out-of-plane.
- `cnn_out_of_plane_nn_species`: Neighbor species for `cnn_out_of_plane_nn_dist`.

## CrystalNN coordination summaries
- `cnn_cn_mean`: Mean total CrystalNN neighbor count per site.
- `cnn_cn_in_plane_mean`: Mean in-plane CrystalNN neighbor count per site.
- `cnn_cn_out_of_plane_mean`: Mean out-of-plane CrystalNN neighbor count per site.

## CrystalNN bonded-species histograms
- `cnn_in_plane_bonded_species_counts`: Counts of bonded neighbor species classified in-plane (summed over sites).
- `cnn_out_of_plane_bonded_species_counts`: Counts of bonded neighbor species classified out-of-plane (summed over sites).

## Oxidation state summaries (square-net species sites)
- `square_species_oxi_state_mean`: Mean oxidation state for the `species` sites (if decorated).
- `square_species_oxi_state_std`: Std dev of oxidation state for the `species` sites.

## CrystalNN bond-angle features
- `cnn_in_plane_bond_angle_deg_mean`: Mean “best” in-plane bond angle (deg) per site (closest to 90° among shortest bonds).
- `cnn_in_plane_bond_angle_deg_std`: Std dev of that in-plane best angle.
- `cnn_in_plane_bond_angle_err90_mean`: Mean |best angle − 90°|.

- `cnn_out_of_plane_tilt_angle_deg_mean`: Mean tilt of shortest out-of-plane bond relative to plane (0=in plane, 90=perpendicular).
- `cnn_out_of_plane_tilt_angle_deg_std`: Std dev of that tilt.

- `cnn_out_of_plane_pair_angle_deg_mean`: Mean angle between two shortest out-of-plane bonds (deg).
- `cnn_out_of_plane_pair_angle_deg_std`: Std dev of that out-of-plane pair angle.
