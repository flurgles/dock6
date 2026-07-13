#include "weisfeiler_leman.h"
#include "dockmol.h"
#include <map>
#include <algorithm>
#include <cmath>
#include <utility>  // pair

using namespace std;

// +++++++++++++++++++++++++++++++++++++++++
void
WL_RMSD::wl_color_refine(DOCKMol & mol, vector<int> & colors,
                          bool active_only)
{
    int N = mol.num_atoms;

    // ----- Step 1: Build adjacency for heavy atoms -----
    vector<vector<int>> adj(N);
    for (int i = 0; i < mol.num_bonds; i++) {
        if (!mol.bond_active_flags[i]) continue;
        int u = mol.bonds_origin_atom[i];
        int v = mol.bonds_target_atom[i];
        bool is_heavy = mol.amber_at_heavy_flag[u] && mol.amber_at_heavy_flag[v];
        bool is_active = !active_only ||
                         (mol.atom_active_flags[u] && mol.atom_active_flags[v]);
        if (is_heavy && is_active) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }

    // ----- Step 2: Initial colors from DOCK atom type -----
    map<string, int> type_to_color;
    int next_color = 0;
    vector<int> cur(N, -1);  // -1 = hydrogen / skip

    for (int i = 0; i < N; i++) {
        bool is_heavy = mol.amber_at_heavy_flag[i];
        bool is_active = !active_only || mol.atom_active_flags[i];
        if (is_heavy && is_active) {
            auto it = type_to_color.find(mol.atom_types[i]);
            if (it == type_to_color.end()) {
                type_to_color[mol.atom_types[i]] = next_color;
                cur[i] = next_color;
                next_color++;
            } else {
                cur[i] = it->second;
            }
        }
    }

    // ----- Step 3: WL color refinement -----
    bool changed = true;
    int iterations = 0;
    const int MAX_ITER = N > 0 ? N : 1;

    while (changed && iterations < MAX_ITER) {
        changed = false;
        iterations++;

        // Map: (color, sorted_neighbor_colors) -> new_color
        map<pair<int, vector<int>>, int> refine_map;
        next_color = 0;
        vector<int> nxt(N, -1);

        for (int i = 0; i < N; i++) {
            if (cur[i] < 0) continue;

            // Collect neighbor colors
            vector<int> ncols;
            for (int nb : adj[i]) {
                if (cur[nb] >= 0)
                    ncols.push_back(cur[nb]);
            }
            sort(ncols.begin(), ncols.end());

            auto key = make_pair(cur[i], move(ncols));
            auto it = refine_map.find(key);
            if (it == refine_map.end()) {
                refine_map.emplace(move(key), next_color);
                nxt[i] = next_color;
                next_color++;
            } else {
                nxt[i] = it->second;
            }
        }

        // Check convergence
        for (int i = 0; i < N; i++) {
            if (cur[i] >= 0 && cur[i] != nxt[i]) {
                changed = true;
                cur = move(nxt);
                break;
            }
        }
    }

    colors = move(cur);
}


// +++++++++++++++++++++++++++++++++++++++++
double
WL_RMSD::calc_WL_RMSD(DOCKMol & refmol, DOCKMol & mol)
{
    if (refmol.num_atoms != mol.num_atoms)
        return -1000.0;

    // Compute WL colors from reference molecule, then delegate to the
    // weighted core (no weights ⇒ uniform RMSD over heavy atoms).
    vector<int> colors;
    wl_color_refine(refmol, colors);
    return calc_WL_RMSD_weighted(refmol, mol, colors, nullptr);
}


// +++++++++++++++++++++++++++++++++++++++++
// Weighted WL RMSD core. Accepts pre-computed WL colors so that growth-time
// pruning can compute colors once per layer and reuse them across the O(n^2)
// pairwise comparisons (wl_color_refine is expensive — heap allocs, map ops,
// per-iteration vectors — and must NOT be called per pair).
//
// When weights is non-null, each atom's distance² is scaled by weights[i]
// and the result is normalized by the sum of weights instead of the count.
// Atoms with colors[i] < 0 (hydrogen/inactive) are skipped entirely.
double
WL_RMSD::calc_WL_RMSD_weighted(DOCKMol & refmol, DOCKMol & mol,
                               const vector<int> & colors,
                               const double * weights)
{
    if (refmol.num_atoms != mol.num_atoms)
        return -1000.0;

    int N = refmol.num_atoms;

    // ----- Guard: colors must match the molecule size -----
    if ((int)colors.size() != N)
        return -1000.0;

    // ----- Group heavy-atom indices by WL color -----
    // colors[i] >= 0  → heavy atom with a WL orbit color
    // colors[i] < 0   → hydrogen or inactive — skip
    map<int, vector<int>> color_groups;
    for (int i = 0; i < N; i++) {
        if (colors[i] >= 0) {
            color_groups[colors[i]].push_back(i);
        }
    }

    // ----- For each color group, find optimal permutation -----
    // The RMSD is separable by color group because atoms from different
    // groups never swap (they are in different graph orbits). So we can
    // minimize each group independently and sum the contributions.
    double total_sum_sq = 0.0;
    double total_weight = 0.0;
    int heavy_count = 0;

    for (auto & entry : color_groups) {
        vector<int> & idx = entry.second;  // atom indices in this group
        int k = (int)idx.size();

        if (k == 1) {
            // Single atom — identity mapping
            int i = idx[0];
            double w = weights ? weights[i] : 1.0;
            double dx = refmol.x[i] - mol.x[i];
            double dy = refmol.y[i] - mol.y[i];
            double dz = refmol.z[i] - mol.z[i];
            total_sum_sq += w * (dx*dx + dy*dy + dz*dz);
            total_weight += w;
            heavy_count++;
            continue;
        }

        // Multiple atoms in same orbit — try all permutations.
        // k is typically 2–6 for drug-like molecules (e.g., 4 α-carbons
        // in naphthalene, 2 carboxylate oxygens, 6 benzene carbons).
        // Worst-case k=6 gives 720 permutations — fast per RMSD call.
        vector<int> perm(k);
        for (int p = 0; p < k; p++) perm[p] = p;

        double best_sum = 1e30;
        double group_weight = 0.0;

        do {
            double sum = 0.0;
            double gw = 0.0;
            for (int p = 0; p < k; p++) {
                int ri = idx[p];             // reference atom
                int ti = idx[perm[p]];       // target atom (same orbit)
                double w = weights ? weights[ri] : 1.0;
                double dx = refmol.x[ri] - mol.x[ti];
                double dy = refmol.y[ri] - mol.y[ti];
                double dz = refmol.z[ri] - mol.z[ti];
                sum += w * (dx*dx + dy*dy + dz*dz);
                gw += w;
            }
            if (sum < best_sum) {
                best_sum = sum;
                group_weight = gw;
            }
        } while (next_permutation(perm.begin(), perm.end()));

        total_sum_sq += best_sum;
        total_weight += group_weight;
        heavy_count += k;
    }

    // Hydrogen atoms are excluded — HA_RMSD* only includes heavy atoms.
    // All heavy atoms already have colors[i] >= 0 and were handled above.

    if (weights) {
        if (total_weight <= 0.0) return -1000.0;
        return sqrt(total_sum_sq / total_weight);
    } else {
        if (heavy_count == 0) return -1000.0;
        return sqrt(total_sum_sq / heavy_count);
    }
}
