{# USES_VARIABLES { _queue_capsule } #}
{% extends 'common_group.pyx' %}

{% block template_support_code %}
from cpython.pycapsule cimport PyCapsule_GetPointer
from libcpp.vector cimport vector
from cython.operator cimport dereference
# We declare minimal C++ interface here - only methods we actually want to use
# This avoids importing the full SpikeQueue wrapper and its dependencies
cdef extern from "spikequeue.h":
    cdef cppclass CSpikeQueue:
        vector[int32_t]* peek();
        void advance();
{% endblock %}

{% block maincode %}
     # Extract the raw C++ object pointer from the Python capsule.
    cdef object _queue_capsule_object = _queue_capsule
    cdef CSpikeQueue* _queue_pointer = <CSpikeQueue*>PyCapsule_GetPointer(_queue_capsule_object, "CSpikeQueue")

    # Now we call the C++ peek method directly to get the current spike vector.
    # This returns a pointer to std::vector<int32_t> containing synapse IDs
    # that are ready for processing in the current time step.
    cdef vector[int32_t]* _spike_vector = _queue_pointer.peek()
    cdef size_t _num_spikes = dereference(_spike_vector).size()

    # Early exit for empty queue - avoid all processing overhead
    if _num_spikes == 0:
        _queue_pointer.advance()
        return

    # Access the underlying raw data pointer of the vector
    cdef int32_t* _spike_data = &dereference(_spike_vector)[0]

    # scalar code
    _vectorisation_idx = 1
    {{ scalar_code | autoindent }}


    cdef size_t _spike_cursor = 0
    cdef int32_t _synapse_id
    while _spike_cursor < _num_spikes:
        _synapse_id = _spike_data[_spike_cursor]

        _idx = _synapse_id
        _vectorisation_idx = _idx

        {{ vector_code | autoindent }}

        _spike_cursor += 1

    # Move the queue forward to the next time step
    _queue_pointer.advance()

{% endblock %}
