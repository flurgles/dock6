#!/bin/bash
#                                                                    #
#                        Copyright UCSF, 2026                        #
#                                                                    #
# Verify GPU grid correctness against existing CPU grids.
#
# For each system named on the command line (or all 649 completed systems):
#   1. Generate a GPU grid (if not already present) in 002.grid_gen/
#   2. Run rigid singlepoint scoring using both CPU and GPU grids
#   3. Compare Grid_Score, Grid_vdw_energy, Grid_es_energy
#   4. Report passing or failing systems
#
# Both runs place output files in 002.grid_gen/ with distinct prefixes
# to avoid clobbering each other or the CPU grid files.
#
# Usage:
#   ./verify_gpu_grids.sh [-j N] [system1 system2 ...]
#
# Options:
#   -j N     Run up to N verification jobs concurrently (default: 1)
#
# Without system names, verifies all 649 completed systems.
# Systems are expected to have pre-existing CPU d0.05 grids at:
#   {ROOT}/zzz.SB2025_systems/{SYSTEM}/002.grid_gen/{SYSTEM}_d0.05.rec.nrg

ROOT=/Users/user/dock6/SB2025_Pnc_v2
GRID_DIR=$ROOT/zzz.SB2025_systems  # each system has 002.grid_gen/ under here
DIST_DIR=$ROOT/zzz.distribution
PARAM_DIR=$ROOT/zzz.parameters
DOCK=/Users/user/dock6/dock6/bin/dock6
GPU_GRID=$(cd "$(dirname "$0")" && pwd)/grid

# Singlepoint input template (rigid, no minimization, grid_score_primary)
# Placeholders: {LIGAND} {GRID_PREFIX} {OUT_PREFIX} {VDW_DEFN} {FLEX_DEFN} {FLEX_DRIVE}
SP_TEMPLATE='conformer_search_type                                        rigid
use_internal_energy                                          no
ligand_atom_file                                             {LIGAND}
limit_max_ligands                                            no
skip_molecule                                                no
read_mol_solvation                                           no
calculate_rmsd                                               yes
use_rmsd_reference_mol                                       yes
rmsd_reference_filename                                      {LIGAND}
use_database_filter                                          no
orient_ligand                                                no
bump_filter                                                  no
score_molecules                                              yes
contact_score_primary                                        no
grid_score_primary                                           yes
grid_score_rep_rad_scale                                     1
grid_score_vdw_scale                                         1
grid_score_es_scale                                          1
grid_lig_efficiency                                          no
grid_score_grid_prefix                                       {GRID_PREFIX}
minimize_ligand                                              no
atom_model                                                   all
vdw_defn_file                                                {VDW_DEFN}
flex_defn_file                                               {FLEX_DEFN}
flex_drive_file                                              {FLEX_DRIVE}
ligand_outfile_prefix                                        {OUT_PREFIX}
write_mol_solvation                                          no
write_orientations                                           no
num_final_scored_poses                                       1
num_preclustered_conformers                                  1
write_conformations                                          no
cluster_conformations                                        no
score_threshold                                              100.0
rank_ligands                                                 no
'

verify_one() {
    local sys=$1
    local gendir=$2
    local distdir=$3
    local dockbin=$4
    local gridbin=$5

    local grid_dir="$gendir/$sys/002.grid_gen"
    local dist_sys="$distdir/$sys"
    local ligand="$dist_sys/${sys}.lig.am1bcc.mol2"

    # Skip if no CPU grid exists
    if [ ! -f "$grid_dir/${sys}_d0.05.rec.nrg" ]; then
        return 0  # silently skip — no CPU reference
    fi

    # Skip if no ligand file
    if [ ! -f "$ligand" ]; then
        echo "  [SKIP] $sys: no ligand file"
        return 0
    fi

    echo "  [TEST] $sys"

    # --- Step 1: Generate GPU grid if needed ---
    local gpu_grid_prefix="$grid_dir/${sys}_gpu.rec"
    if [ ! -f "${gpu_grid_prefix}.nrg" ]; then
        # DOCK 4.0.1 grid binary requires all paths in the .in file to be
        # filenames (not paths) located in the current working directory.
        # Symlink the necessary files into grid_dir, then use bare filenames.
        local receptor="${dist_sys}/${sys}.rec.clean.mol2"
        local box="${dist_sys}/box.pdb"
        local vdw_defn="${PARAM_DIR}/vdw_AMBER_parm99.defn"
        local chem_defn="${PARAM_DIR}/chem.defn"

        ln -sf "$receptor" "$grid_dir/${sys}.rec.clean.mol2"
        ln -sf "$box"      "$grid_dir/box.pdb"
        ln -sf "$vdw_defn"  "$grid_dir/vdw_AMBER_parm99.defn"
        ln -sf "$chem_defn" "$grid_dir/chem.defn"

        local grid_in="${sys}_gpu_grid.in"
        local grid_out="${sys}_gpu_grid.out"

        cat > "$grid_dir/$grid_in" <<- GRIDEOF
compute_grids                  yes
grid_spacing                   0.3
output_molecule                no
contact_score                  no
chemical_score                 no
energy_score                   yes
energy_cutoff_distance         999
atom_model                     a
attractive_exponent            6
repulsive_exponent             9
distance_dielectric            yes
dielectric_factor              4
allow_non_integral_charges     yes
bump_filter                    yes
bump_overlap                   0.75
receptor_file                  ${sys}.rec.clean.mol2
box_file                       box.pdb
vdw_definition_file            vdw_AMBER_parm99.defn
chemical_definition_file       chem.defn
score_grid_prefix              ./${sys}_gpu.rec
grid_soft_delta                0.05
GRIDEOF

        # Run GPU grid from the grid directory (everything uses filenames)
        (cd "$grid_dir" && "$gridbin" -i "$grid_in" -o "$grid_out" 2>/dev/null)
        if [ ! -f "${gpu_grid_prefix}.nrg" ]; then
            echo "  [FAIL] $sys: GPU grid generation failed (see $grid_out)"
            return 1
        fi
    fi

    # --- Step 2: Run CPU singlepoint ---
    local cpu_out_prefix="$grid_dir/${sys}_cpu_d0.05_sp"
    local cpu_in="$grid_dir/${sys}_cpu_d0.05_sp.in"
    local cpu_out="$grid_dir/${sys}_cpu_d0.05_sp.out"

    local cpu_grid_prefix="$grid_dir/${sys}_d0.05.rec"
    echo "$SP_TEMPLATE" | sed \
        -e "s|{LIGAND}|$ligand|g" \
        -e "s|{GRID_PREFIX}|$cpu_grid_prefix|g" \
        -e "s|{OUT_PREFIX}|$cpu_out_prefix|g" \
        -e "s|{VDW_DEFN}|$PARAM_DIR/vdw_AMBER_parm99.defn|g" \
        -e "s|{FLEX_DEFN}|$PARAM_DIR/flex.defn|g" \
        -e "s|{FLEX_DRIVE}|$PARAM_DIR/flex_drive.tbl|g" \
        > "$cpu_in"

    "$dockbin" -i "$cpu_in" -o "$cpu_out" 2>/dev/null

    # --- Step 3: Run GPU singlepoint ---
    local gpu_out_prefix="$grid_dir/${sys}_gpu_d0.05_sp"
    local gpu_in="$grid_dir/${sys}_gpu_d0.05_sp.in"
    local gpu_out="$grid_dir/${sys}_gpu_d0.05_sp.out"

    echo "$SP_TEMPLATE" | sed \
        -e "s|{LIGAND}|$ligand|g" \
        -e "s|{GRID_PREFIX}|${gpu_grid_prefix}|g" \
        -e "s|{OUT_PREFIX}|$gpu_out_prefix|g" \
        -e "s|{VDW_DEFN}|$PARAM_DIR/vdw_AMBER_parm99.defn|g" \
        -e "s|{FLEX_DEFN}|$PARAM_DIR/flex.defn|g" \
        -e "s|{FLEX_DRIVE}|$PARAM_DIR/flex_drive.tbl|g" \
        > "$gpu_in"

    "$dockbin" -i "$gpu_in" -o "$gpu_out" 2>/dev/null

    # --- Step 4: Extract and compare scores ---
    local cpu_score=$(grep -E '^\s+Grid_Score:' "$cpu_out" | awk '{print $2}')
    local gpu_score=$(grep -E '^\s+Grid_Score:' "$gpu_out" | awk '{print $2}')
    local cpu_vdw=$(grep -E 'Grid_vdw_energy:' "$cpu_out" | awk '{print $2}')
    local gpu_vdw=$(grep -E 'Grid_vdw_energy:' "$gpu_out" | awk '{print $2}')
    local cpu_es=$(grep -E 'Grid_es_energy:' "$cpu_out" | awk '{print $2}')
    local gpu_es=$(grep -E 'Grid_es_energy:' "$gpu_out" | awk '{print $2}')

    if [ -z "$cpu_score" ] || [ -z "$gpu_score" ]; then
        echo "  [FAIL] $sys: missing scores in dock output"
        return 1
    fi

    # Compare with tolerance 0.001 kcal/mol
    local tol=0.001
    local pass=1
    local diff_score=$(awk "BEGIN { d = $cpu_score - $gpu_score; if (d < 0) d = -d; printf \"%.6f\", d }")
    local diff_vdw=$(awk "BEGIN { d = $cpu_vdw - $gpu_vdw; if (d < 0) d = -d; printf \"%.6f\", d }")
    local diff_es=$(awk "BEGIN { d = $cpu_es - $gpu_es; if (d < 0) d = -d; printf \"%.6f\", d }")

    local score_ok=$(awk "BEGIN { print ($diff_score <= $tol) ? 1 : 0 }")
    local vdw_ok=$(awk "BEGIN { print ($diff_vdw <= $tol) ? 1 : 0 }")
    local es_ok=$(awk "BEGIN { print ($diff_es <= $tol) ? 1 : 0 }")

    if [ "$score_ok" = 1 ] && [ "$vdw_ok" = 1 ] && [ "$es_ok" = 1 ]; then
        echo "  [PASS] $sys: score diff=$diff_score vdw diff=$diff_vdw es diff=$diff_es"
        return 0
    else
        echo "  [FAIL] $sys: CPU=($cpu_score,$cpu_vdw,$cpu_es) GPU=($gpu_score,$gpu_vdw,$gpu_es)"
        echo "         diffs: score=$diff_score vdw=$diff_vdw es=$diff_es"
        return 1
    fi
}

# --- Main ---

# Parse options
concurrency=1
while getopts "j:h" opt; do
    case $opt in
        j) concurrency=$OPTARG ;;
        h) echo "Usage: $0 [-j N] [system1 system2 ...]"
           echo "  -j N    parallel verification jobs (default: 1)"
           echo "  Without args, verifies all 649 completed systems."
           exit 0 ;;
        *) exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ $# -eq 0 ]; then
    # Auto-detect all completed systems
    systems=()
    for d in "$GRID_DIR"/*/002.grid_gen/*_d0.05.rec.nrg; do
        sys=$(basename "$(dirname "$(dirname "$d")")")
        systems+=("$sys")
    done
else
    systems=("$@")
fi

total=${#systems[@]}
echo "Verifying GPU grid correctness on $total system(s) ..."
echo "Concurrency: $concurrency"
echo "Grid output dirs: 002.grid_gen/ under each system"
echo "CPU singlepoint:  {sys}_cpu_d0.05_sp.in/.out/.mol2"
echo "GPU singlepoint:  {sys}_gpu_d0.05_sp.in/.out/.mol2"
echo ""

# Export for parallel subshells
export -f verify_one
export ROOT GRID_DIR DIST_DIR PARAM_DIR DOCK GPU_GRID

if [ "$concurrency" -gt 1 ]; then
    # Run in parallel using xargs
    printf '%s\0' "${systems[@]}" | xargs -0 -n 1 -P "$concurrency" \
        bash -c 'verify_one "$1" "$GRID_DIR" "$DIST_DIR" "$DOCK" "$GPU_GRID"' _
else
    pass=0
    fail=0
    for sys in "${systems[@]}"; do
        if verify_one "$sys" "$GRID_DIR" "$DIST_DIR" "$DOCK" "$GPU_GRID"; then
            ((pass++))
        else
            ((fail++))
        fi
    done
    echo ""
    echo "=== Results: $pass passed, $fail failed (out of $total) ==="
fi
