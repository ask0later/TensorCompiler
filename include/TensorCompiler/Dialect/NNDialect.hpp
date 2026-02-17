#pragma once

#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "NNDialect.h.inc"

#define GET_OP_CLASSES
#include "NNOps.h.inc"