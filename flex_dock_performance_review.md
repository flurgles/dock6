# Flex Docking Code Review - Performance Bottlenecks
## Date: 2026-06-21

### Architecture (from dock.cpp)
Flex docking (method=1) uses anchor-and-grow via Master_Conformer_Search.
Main loop: prepare_molecule() → next_anchor() → submit_anchor_orientation() → grow_periphery() → next_conformer()

### Key Functions and Their Costs

#### 1. copy_molecule() [dockmol.cpp:1198-1417]
- Copies ~40+ per-atom arrays (x, y, z, atom_types, atom_names, charges, amber_at_id, amber_at_radius, amber_at_well_depth, amber_at_heavy_flag, atom_active_flags, bond_active_flags, etc.)
- Copies ~15 per-bond arrays
- Copies 10+ scalar fields (strings, floats, ints)
- Calls allocate_arrays() first
- Called per torsion angle in segment_torsion_drive, per seed, per anchor
- Also copies footprints vector, child_list

#### 2. copy_molecule_shallow() [dockmol.cpp:1419]
- Fewer fields copied than full copy_molecule
- Still copies x/y/z, atom_types, atom_names, charges, etc.
- Missing: flag_acceptor, flag_donator, charges, subst_names, ring_flags, well_depth, radius, etc.

#### 3. copy_crds() [dockmol.cpp ~1498]
- Only copies x[i], y[i], z[i] for all atoms
- THIS is what we need for torsion drive - only coordinates change!

#### 4. activate_layer_segment() [conf_gen_ag.cpp:1554-1608]
- Calls reset_active_lists() which zeros ALL atom_active_flags and bond_active_flags
- Then re-enables flags for active layers by iterating over layer_segments
- Then counts active atoms/bonds by scanning ALL atoms/bonds
- Called once per torsion copy in segment_torsion_drive

#### 5. reset_active_lists() [conf_gen_ag.cpp:1610]
- Zeros num_atoms booleans + num_bonds booleans + 2 ints
- O(num_atoms + num_bonds) per call

#### 6. segment_torsion_drive() [conf_gen_ag.cpp:1380-1460]
For each torsion angle:
  a. copy_molecule(new_conf.structure, conf.structure) ← FULL DEEP COPY
  b. set_torsion() ← only changes coordinates
  c. activate_layer_segment() ← resets and rebuilds flags
  d. copy_molecule(return_list[...].structure, new_conf.structure) ← ANOTHER FULL DEEP COPY

So per torsion angle: 2 full copy_molecule() + 1 activate_layer_segment()

#### 7. Clustering/pruning [conf_gen_ag.cpp ~1220-1250]
O(n²) pairwise calc_layer_rmsd() comparisons

### Performance Bottleneck Hierarchy (by impact)

1. **copy_molecule() called excessively** - The #1 bottleneck.
   - segment_torsion_drive calls it TWICE per torsion: once to copy parent, once to push to return_list
   - Only x/y/z change between torsions (set_torsion rotates coordinates)
   - All ~40 atom arrays and ~15 bond arrays are redundantly copied
   - Should use copy_crds() or a minimal copy for torsion drive

2. **activate_layer_segment() does redundant work**
   - Resets ALL flags then rebuilds from scratch every call
   - Only the NEWLY activated segment's flags change between consecutive calls
   - Could use incremental update instead of full reset+rebuild

3. **Clustering is O(n²)** - pairwise RMSD for all conformers

4. **Per-conform minimization** - simplex minimize for every growth tree node

### Optimization Opportunities

#### Quick Wins (low risk, high impact):
A. **Replace copy_molecule with copy_crds in segment_torsion_drive**
   - For the torsion drive loop, only coordinates change
   - Use copy_crds() instead of copy_molecule() → copies 3 arrays instead of 60+
   - Need to also copy atom_active_flags since activate_layer_segment modifies them
   - Estimated speedup: 10-20x reduction in copy cost per torsion

B. **Incremental activate_layer_segment**
   - Instead of reset+rebuild, track which flags were set and only toggle the delta
   - Or: pass parent's active flags and only add/remove the new segment

C. **Reserve vectors** - Some vectors already reserved, but seed lists could be pre-sized

#### Medium Wins:
D. **Skip b4min_seeds when print_growth_tree is off** - unnecessary copies
E. **Cluster with hash-based RMSD** instead of pairwise

#### Notes:
- The scoring hotpath (calc_inter_energy_grid) is already optimized - trilinear interpolation per atom
- copy_molecule_shallow() exists but still copies more than needed for torsion drive
- DOCKMol has ~60+ member arrays for a typical 30-atom ligand
