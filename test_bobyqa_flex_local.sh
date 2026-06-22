#!/bin/bash
# Flex docking test - runs in CWD, links files from DT100
set -e

SYSTEM="1A28"
DT100_SYS="/Users/user/dock6/DT100/zzz.DT100_systems/$SYSTEM"
PARAMDIR="/Users/user/dock6/DT100/zzz.parameters"
DOCKBIN="/Users/user/dock6/dock6/bin/dock6"

# Create workspace
mkdir -p "flex_${SYSTEM}_workspace"
cd "flex_${SYSTEM}_workspace"
rm -f *.mol2 *.defn *.tbl *.sph

# Link files
ln -sf "${DT100_SYS}/001.files/${SYSTEM}.lig.am1bcc.mol2" ./
ln -sf "${DT100_SYS}/001.files/${SYSTEM}.rec.clean.mol2" ./
ln -sf "${DT100_SYS}/001.files/${SYSTEM}.rec.clust.close.sph" ./
ln -sf "${PARAMDIR}/vdw_AMBER_parm99.defn" ./vdw.defn
ln -sf "${PARAMDIR}/flex.defn" ./flex.defn
ln -sf "${PARAMDIR}/flex_drive.tbl" ./flex_drive.tbl

GRID_PREFIX="../flex_${SYSTEM}_grid/${SYSTEM}.rec"

# Check if grid exists
if [ ! -f "../flex_${SYSTEM}_grid/${SYSTEM}.rec.nrg" ]; then
  echo "Grid not found, checking ${DT100_SYS}/002.grid_gen/"
  ls "${DT100_SYS}/002.grid_gen/" 2>/dev/null | head -5
fi

cat > flex_simplex.in << 'TEMPLATE'
conformer_search_type                                        flex
write_fragment_libraries                                     no
user_specified_anchor                                        no
limit_max_anchors                                            no
min_anchor_size                                              5
pruning_use_clustering                                       yes
pruning_max_orients                                          1000
pruning_clustering_cutoff                                    100
pruning_conformer_score_cutoff                               100.0
pruning_conformer_score_scaling_factor                       1.0
use_clash_overlap                                            no
write_growth_tree                                            no
use_internal_energy                                          yes
internal_energy_rep_exp                                      12
internal_energy_cutoff                                       100.0
ligand_atom_file                                             1A28.lig.am1bcc.mol2
limit_max_ligands                                            no
skip_molecule                                                no
read_mol_solvation                                           no
calculate_rmsd                                               no
use_database_filter                                          no
orient_ligand                                                yes
automated_matching                                           yes
receptor_site_file                                           1A28.rec.clust.close.sph
max_orientations                                             1000
critical_points                                              no
chemical_matching                                            no
use_ligand_spheres                                           no
bump_filter                                                  yes
bump_grid_prefix                                             ../002.grid_gen/1A28.rec
score_molecules                                              yes
contact_score_primary                                        no
grid_score_primary                                           yes
grid_score_rep_rad_scale                                     1.0
grid_score_vdw_scale                                         1.0
grid_score_es_scale                                          1.0
grid_score_grid_prefix                                       ../002.grid_gen/1A28.rec
grid_score_grid_suffix                                       .nrg
grid_score_grid_read                                         yes
multi_grid_score_secondary                                   no
dock3.5_score_secondary                                      no
continuous_score_secondary                                   no
footprint_similarity_score_secondary                         no
pharmacophore_score_secondary                                no
descriptor_score_secondary                                   no
gbsa_zou_score_secondary                                     no
gbsa_hawkins_score_secondary                                 no
SASA_score_secondary                                         no
amber_score_secondary                                        no
minimize_ligand                                              yes
simplex_max_iterations                                       1000
simplex_tors_premin_iterations                               0
simplex_max_cycles                                           1
simplex_score_converge                                       0.1
simplex_cycle_converge                                       1.0
simplex_trans_step                                           1.0
simplex_rot_step                                             0.5
simplex_tors_step                                            10.0
simplex_random_seed                                          0
simplex_restraint_min                                        no
atom_model                                                   all
vdw_defn_file                                                ./vdw.defn
flex_defn_file                                               ./flex.defn
flex_drive_file                                              ./flex_drive.tbl
ligand_outfile_prefix                                        output
minimizer_type                                               simplex
write_orientations                                           no
num_scored_poses                                             100
write_primary_pose                                           yes
write_secondary_pose                                         no
write_entire_pose                                            yes
rank_ligands                                                 no
TEMPLATE

echo "Running flex simplex..."
$DOCKBIN -v -i flex_simplex.in -o flex_simplex.out 2>/dev/null
echo "Simplex exit: $?"
grep "Grid_Score:" flex_simplex.out 2>/dev/null | tail -3
grep "Total elapsed" flex_simplex.out 2>/dev/null | tail -1
grep "Molecules Processed" flex_simplex.out 2>/dev/null | tail -1
