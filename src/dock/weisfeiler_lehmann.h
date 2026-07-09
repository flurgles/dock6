#ifndef WEISFEILER_LEHMANN_H
#define WEISFEILER_LEHMANN_H 

#include <vector>
class DOCKMol;

// +++++++++++++++++++++++++++++++++++++++++
// WL_RMSD — Weisfeiler-Lehman guided symmetry-corrected RMSD.
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
// Reference: Weisfeiler, B. & Lehman, A. A. (1968).

class WL_RMSD {

  public:
    WL_RMSD() {}
    ~WL_RMSD() {}

    // Compute WL-guided symmetry-corrected RMSD between two conformers.
    // Returns RMSD over heavy atoms, using optimal within-orbit permutations.
    // Returns -1000.0 if atom counts differ.
    double  calc_WL_RMSD(DOCKMol & refmol, DOCKMol & mol);

  private:
    // Weisfeiler-Lehman color refinement on heavy active atoms.
    // colors[i] = -1 for hydrogen/inactive atoms, else WL orbit color.
    void    wl_color_refine(DOCKMol & mol, std::vector<int> & colors);
};

#endif  // WEISFEILER_LEHMANN_H
