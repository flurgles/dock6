#!/bin/bash
# Flex docking test for BOBYQA on a single DT100 system
# Usage: bash test_bobyqa_flex.sh <system_name>

DOCKHOME="/Users/user/dock6"
DOCKBIN="${DOCKHOME}/dock6/bin/dock6"
DT100_SYS="${DOCKHOME}/DT100/zzz.DT100_systems"
PARAMDIR="${DOCKHOME}/DT100/zzz.parameters"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <system_name>"
    exit 1
fi
SYSTEM="$1"

WORKDIR="${DT100_SYS}/${SYSTEM}/bobyqa_flex_test"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Copy required files
ln -sf "${DT100_SYS}/${SYSTEM}/001.files/${SYSTEM}.lig.am1bcc.mol2" ./
ln -sf "${DT100_SYS}/${SYSTEM}/001.files/${SYSTEM}.rec.clean.mol2" ./
ln -sf "${DT100_SYS}/${SYSTEM}/001.files/${SYSTEM}.rec.clust.close.sph" ./
ln -sf "${PARAMDIR}/vdw_AMBER_parm99.defn" ./vdw.defn
ln -sf "${PARAMDIR}/flex.defn" ./flex.defn
ln -sf "${PARAMDIR}/flex_drive.tbl" ./flex_drive.tbl

run_flex_test() {
    local config_name="$1"
    local minimizer_type="$2"
    local extra_params="$3"
    local input_file="${SYSTEM}_${config_name}.in"
    local output_file="${SYSTEM}_${config_name}.out"
    
    cp flex_template.in "$input_file"
    echo "minimizer_type                                               $minimizer_type" >> "$input_file"
    echo "$extra_params" >> "$input_file"
    
    $DOCKBIN -v -i "$input_file" -o "$output_file" 2>/dev/null
    
    if [ -f "$output_file" ]; then
        local score=$(grep "Grid_Score:" "$output_file" | tail -1 | awk '{print $2}')
        local time=$(grep "Total elapsed time:" "$output_file" | tail -1 | awk '{print $4}' | sed 's/seconds//')
        if [ -z "$time" ]; then
            time=$(grep "Elapsed time for docking:" "$output_file" | tail -1 | awk '{print $6}' | sed 's/seconds//')
        fi
        echo "${SYSTEM},${config_name},${score:-FAILED},${time:-0},SUCCESS"
    else
        echo "${SYSTEM},${config_name},FAILED,0,FAILED"
    fi
}

# Create flex template
cat > flex_template.in << 'TEMPLATE'
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
ligand_atom_file                                             ${SYSTEM}.lig.am1bcc.mol2
limit_max_ligands                                            no
skip_molecule                                                no
read_mol_solvation                                           no
calculate_rmsd                                               no
use_database_filter                                          no
orient_ligand                                                yes
automated_matching                                           yes
receptor_site_file                                           ${SYSTEM}.rec.clust.close.sph
max_orientations                                             1000
critical_points                                              no
chemical_matching                                            no
use_ligand_spheres                                           no
bump_filter                                                  yes
bump_grid_prefix                                             ../002.grid_gen/${SYSTEM}.rec
score_molecules                                              yes
contact_score_primary                                        no
grid_score_primary                                           yes
grid_score_rep_rad_scale                                     1.0
grid_score_vdw_scale                                         1.0
grid_score_es_scale                                          1.0
grid_score_grid_prefix                                       ../002.grid_gen/${SYSTEM}.rec
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
simplex_initial_score_converge                               5
simplex_random_seed                                          0
simplex_restraint_min                                        no
atom_model                                                   all
vdw_defn_file                                                ./vdw.defn
flex_defn_file                                               ./flex.defn
flex_drive_file                                              ./flex_drive.tbl
ligand_outfile_prefix                                        output
write_orientations                                           no
num_scored_poses                                             100
write_primary_pose                                           yes
write_secondary_pose                                         no
write_entire_pose                                            yes
write_solvent_params                                         no
write_fragment_libraries                                     no
write_hbonds                                                no
rank_ligands                                                 no
TEMPLATE

echo "=== Flex tests for ${SYSTEM} ==="
run_flex_test "flex_simplex" "simplex" ""
run_flex_test "flex_bobyqa_default" "bobyqa" "bobyqa_restarts_per_torsion                                   5"
run_flex_test "flex_bobyqa_block_diag" "bobyqa" "bobyqa_hessian_mode                                           block_diag
bobyqa_restarts_per_torsion                                   5"
run_flex_test "flex_bobyqa_full_quad" "bobyqa" "bobyqa_hessian_mode                                           full_quad
bobyqa_restarts_per_torsion                                   5"
echo "=== Done with flex tests for ${SYSTEM} ==="
