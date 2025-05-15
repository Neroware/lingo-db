#include "lingodb/compiler/Dialect/Graph/GraphDialect.h"

#include "lingodb/compiler/Dialect/Graph/GraphOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace lingodb::compiler::dialect::graph;

void GraphDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "lingodb/compiler/Dialect/Graph/GraphOps.cpp.inc"
    
          >();
    
    registerTypes();
    registerAttrs();
}
#include "lingodb/compiler/Dialect/Graph/GraphOpsDialect.cpp.inc"