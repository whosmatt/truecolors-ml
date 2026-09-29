// Batch entry points for the melflux ctypes binding
#include "melflux.h"
#include <stddef.h>

size_t melflux_state_size(void) { return sizeof(melflux_t); }
int    melflux_bands(void) { return MEL_BANDS; }

void melflux_run(melflux_t *m, const int16_t *s, int n_blocks, float *out)
{
    for (int i = 0; i < n_blocks; i++) {
        melflux_block(m, s + (size_t)i * MEL_HALF, MEL_HALF, out + (size_t)i * MEL_BANDS);
    }
}
