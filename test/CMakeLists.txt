configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Only swage's own tools are listed as targets; FileCheck/not/count come from
# the LLVM install tree (built with LLVM_INSTALL_UTILS=ON) and are found via
# the lit tool substitutions, so an install-tree LLVM works.
set(SWAGE_TEST_DEPENDS
  swage-opt
)

add_lit_testsuite(check-swage "Running the Swage regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SWAGE_TEST_DEPENDS}
)
set_target_properties(check-swage PROPERTIES FOLDER "Tests")

add_lit_testsuites(SWAGE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SWAGE_TEST_DEPENDS})
