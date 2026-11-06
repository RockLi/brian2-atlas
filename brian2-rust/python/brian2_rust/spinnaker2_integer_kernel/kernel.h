#ifndef ATLAS_S2_INTEGER_H
#define ATLAS_S2_INTEGER_H
#include <stdint.h>
/* Host-testable, bounded neuron kernel. No SpiNNaker2 transport/firmware ABI.
 * Banks are row-major input/readout/recurrent. Thresholds: hidden then output.
 * State: [batch][3: I,V,Q][hidden+output]. Trace: [batch][tick][3][cell].
 * Input: [batch][tick][input]. Callers own correctly sized buffers.
 * Every argument is validated before state/trace mutation. Return 0 on success.
 */
int atlas_s2_integer_run(int ni, int nh, int no, int batch, int steps,
    const int32_t *wi, const int32_t *wo, const int32_t *wr,
    const int32_t *threshold, const uint8_t *input, int32_t *state, int32_t *trace);
#endif
