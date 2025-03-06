# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\ED_cPSD_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\ED_cPSD_autogen.dir\\ParseCache.txt"
  "ED_cPSD_autogen"
  )
endif()
