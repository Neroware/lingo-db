#include "lingodb/compiler/Dialect/Graph/GraphDialect.h"
#include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include <llvm/ADT/TypeSwitch.h>

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"

void lingodb::compiler::dialect::graph::GraphDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/compiler/Dialect/Graph/GraphOpsAttributes.cpp.inc"

      >();
}