#ifndef HUNGARIAN_H
#define HUNGARIAN_H 

#include <algorithm>
#include <iostream>
#include <string>
#include "dockmol.h"
using namespace std;

class Hungarian_RMSD {

  public:
    Hungarian_RMSD() : current_mat_size(0), matrix(NULL), matrix_original(NULL), matrix_case(NULL),
                      row_count(NULL), column_count(NULL), row_assigned(NULL), column_assigned(NULL),
                      matrix_match(NULL) {}
    ~Hungarian_RMSD() { free_buffers(); }

    double  calc_Hungarian_RMSD(DOCKMol &, DOCKMol &);    // calculates hungarian rmsd (standard, unweighted)
    // Weighted variant: same as above but each atom's contribution is scaled by ref_weights[atom_idx].
    // Used by pruning layer-weighted RMSD where atoms in later growth layers get higher weight.
    double  calc_Hungarian_RMSD(DOCKMol &, DOCKMol &, const double *);
    std::pair <double, int>  calc_Hungarian_RMSD_dissimilar(DOCKMol &, DOCKMol &);    // calculates hungarian rmsd for two dissimilar molecules

  private:
    double  MAX;			// A large int used to find minimum values
    double  total_assignment;		// The sum of assignments for all atom types
    double  rmsd;			// rmsd, calculated from total_assignment
    int     num_unique;			// number of unique atom types in DOCKMol object
    int     num_type;			// instances of a certain atom type in DOCKMol object
    int     current_mat_size;		// current allocated size of all matrix/array buffers


    string  *unique_atom_types;		// list of unique atom types in DOCKMol object passed to this function
    double  **matrix;			// cost matrix of size NUM x NUM
    double  **matrix_original;		// A backup copy of the original matrix
    int     *row_count;			// row_count[i] = number of 0s to occur in row i
    int     *column_count;		// column_count[j] = number of 0s to occur in column j
    int     *row_assigned;		// flag to remember if a row has been assigned or crossed out
    int     *column_assigned;		// flag to remember if a column has been assigned or crossed out
    int     **matrix_case;		// A matrix of flags (0 || 1 || 2)
					    // 0 = uncovered, decrease by minimum value of all 0s
					    // 1 = covered, ignore
					    // 2 = covered twice, increase by minimum value of all 0s
    int     *matrix_match;		// If matrix_match[i] = j, then worker i is assigned job j


    void    initialize();		// Allocate/reuse memory for matrices; only allocs if num_type > current_mat_size
    void    clear();			// No-op during per-type iterations (buffers reused).
    void    free_buffers();		// Actually deallocate all buffers (destructor + initialize realloc path).

    void    reset_match();		// Resets all matches to -100 (null value)
    void    reset_row_assigned();	// Resets all of row_assigned to 0
    void    reset_column_assigned();	// Resets all of column_assigned to 0
    void    reset_row_count();		// Resets all of row_count to 0
    void    reset_column_count();	// Resets all of column_count to 0
    void    reset_case();		// Resets the value of every cell in matrix_case to 0

    void    hungarian();		// an O(n^4) implementation of the Hungarian algorithm to solve minimum assignment
    void    reduce_matrix();		// Backs up the matrix[] as matrix_original[][], then reduces matrix
    int     assign_jobs();		// Assigns workers to jobs, loops until all are assigned, return # of assignments
    void    draw_line();		// Figures out minimum number of lines required to cross out all 0s
    void    update_matrix();		// Updates matrix[][] according to matrix_case[][]

    double  sum_assignment();		// Iterates through matrix_match[] and sums costs from matrix_original[][]
    double  sum_assignment_dissimilar(int);	// Iterates through matrix_match[] and sums costs from matrix_original[][]

};

#endif  // HUNGARIAN_H
