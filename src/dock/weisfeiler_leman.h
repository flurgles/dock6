#ifndef WEISFEILER_LEMAN_H
#define WEISFEILER_LEMAN_H 

#include <vector>
class DOCKMol;

// +++++++++++++++++++++++++++++++++++++++++
// WL_RMSD — Weisfeiler-Leman guided symmetry-corrected RMSD.
//
// Uses WL color refinement (graph automorphism partitioning) to identify
// equivalent atom positions in a molecule, then finds the optimal
// permutation within each orbit to minimize RMSD between two conformers.
//
// This is faster than Hungarian RMSD (O(n^3)) because WL partitions atoms
// into small equivalence groups (typically 2–6 atoms for drug-like
// molecules), and trying all permutations within each group is O(k!·k)
// where k is the group size.
//
// Unlike Hungarian (which matches by DOCK type label only and can
// produce graph-breaking assignments), WL respects the molecular graph
// structure — only automorphic atoms are allowed to permute.
//
// Reference: Weisfeiler, B. & Leman, A. A. (1968).

class WL_RMSD {

  public:
    WL_RMSD() {}
    ~WL_RMSD() {}

    // Compute WL-guided symmetry-corrected RMSD between two conformers.
    // Returns RMSD over heavy atoms, using optimal within-orbit permutations.
    // Returns -1000.0 if atom counts differ.
    double  calc_WL_RMSD(DOCKMol & refmol, DOCKMol & mol);

    // Weighted variant: same as calc_WL_RMSD but each atom's contribution is
    // scaled by weights[atom_idx] (when non-null) and normalized by the sum
    // of weights instead of the heavy atom count. Used by growth-time pruning
    // where atoms in later growth layers carry higher weight.
    //
    // Accepts pre-computed WL colors (from wl_color_refine) so the caller can
    // compute colors once per growth layer and reuse across all pairwise
    // comparisons — critical for the O(n^2) pruning loop.
    // Returns -1000.0 if atom counts differ.
    double  calc_WL_RMSD_weighted(DOCKMol & refmol, DOCKMol & mol,
                                  const std::vector<int> & colors,
                                  const double * weights = nullptr);

    // Weisfeiler-Leman color refinement on heavy atoms.
    // colors[i] = -1 for hydrogen/inactive atoms, else WL orbit color.
    // When active_only=true, also checks atom_active_flags (for pruning).
    void    wl_color_refine(DOCKMol & mol, std::vector<int> & colors,
                            bool active_only = false);
};

#endif  // WEISFEILER_LEMAN_H
