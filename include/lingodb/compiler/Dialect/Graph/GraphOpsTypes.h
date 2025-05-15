#ifndef LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSTYPES_H
#define LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSTYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

using namespace lingodb::compiler::dialect::subop;

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.h.inc"

#endif // LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPSTYPES_H
