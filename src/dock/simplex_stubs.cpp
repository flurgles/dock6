#include "minimizer.h"


/* ================================================================== */
/*  C1 Batch Queue Infrastructure (stubs — removed in P1)              */
/* ================================================================== */

void enable_gpu_batch_mode(bool enabled)
{
    /* P1: batch-queue mode removed — simplex runs end-to-end per call. */
}


int flush_gpu_batch(Minimizer &min)
{
    return 0;
}
