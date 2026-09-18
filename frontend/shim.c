// Batch entry points for the ctypes binding
#include "frontend.h"
#include <stddef.h>

size_t fe_state_size(void) { return sizeof(fe_t); }
size_t fe_out_size(void)   { return sizeof(fe_out_t); }
int    fe_spec_version(void) { return FE_SPEC_VERSION; }
int    fe_block_samples(void) { return FE_BLOCK_SAMPLES; }
int    fe_sample_rate(void) { return FE_SAMPLE_RATE; }

void fe_run(fe_t *fe, const int16_t *s, int n_blocks, fe_out_t *out)
{
    for (int i = 0; i < n_blocks; i++) {
        fe_block(fe, s + (size_t)i * FE_BLOCK_SAMPLES, FE_BLOCK_SAMPLES, &out[i]);
    }
}
