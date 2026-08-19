add_mlir_public_c_api_library(SwageCAPI
  Dialects.cpp

  LINK_LIBS PUBLIC
  MLIRSwage
)
