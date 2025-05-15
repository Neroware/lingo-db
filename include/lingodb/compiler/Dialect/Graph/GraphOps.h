#ifndef LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPS_H
#define LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsTypes.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypes.h"

#include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.h"
#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOps.h.inc"

#endif // LINGODB_COMPILER_DIALECT_GRAPH_GRAPHOPS_H
