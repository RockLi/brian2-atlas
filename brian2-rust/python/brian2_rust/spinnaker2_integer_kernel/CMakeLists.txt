cmake_minimum_required(VERSION 3.16)
# This is the host reference extension, not an S2 PE firmware application.
if(COMMAND snn_add_application)
  message(FATAL_ERROR "Atlas integer kernel has no SpiNNaker2 firmware transport adapter")
endif()
project(atlas_s2_integer_host C)
add_library(atlas_s2_integer_host SHARED kernel.c)
set_target_properties(atlas_s2_integer_host PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)
