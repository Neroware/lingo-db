#include "lingodb/compiler/Dialect/Graph/GraphDialect.h"
#include "lingodb/compiler/Dialect/Graph/GraphOps.h"
#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace lingodb::compiler::dialect::graph;
namespace {
   using namespace lingodb::compiler::dialect;
   static mlir::LogicalResult parseStateMembers(mlir::AsmParser& parser, subop::StateMembersAttr& stateMembersAttr) {
      if (parser.parseLSquare()) return mlir::failure();
      std::vector<mlir::Attribute> names;
      std::vector<mlir::Attribute> types;
      while (true) {
         if (!parser.parseOptionalRSquare()) { break; }
         llvm::StringRef colName;
         mlir::Type t;
         if (parser.parseKeyword(&colName) || parser.parseColon() || parser.parseType(t)) { return mlir::failure(); }
         names.push_back(parser.getBuilder().getStringAttr(colName));
         types.push_back(mlir::TypeAttr::get(t));
         if (!parser.parseOptionalComma()) { continue; }
         if (parser.parseRSquare()) { return mlir::failure(); }
         break;
      }
      stateMembersAttr = subop::StateMembersAttr::get(parser.getContext(), parser.getBuilder().getArrayAttr(names), parser.getBuilder().getArrayAttr(types));
      return mlir::success();
   }
   static void printStateMembers(mlir::AsmPrinter& p, subop::StateMembersAttr stateMembersAttr) {
      p << "[";
      auto first = true;
      for (size_t i = 0; i < stateMembersAttr.getNames().size(); i++) {
         auto name = mlir::cast<mlir::StringAttr>(stateMembersAttr.getNames()[i]).str();
         auto type = mlir::cast<mlir::TypeAttr>(stateMembersAttr.getTypes()[i]).getValue();
         if (first) {
            first = false;
         } else {
            p << ", ";
         }
         p << name << " : " << type;
      }
      p << "]";
   }
} // namespace

subop::StateMembersAttr graph::GraphType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getNodeMembers().getNames().begin(), getNodeMembers().getNames().end());
   names.insert(names.end(), getEdgeMembers().getNames().begin(), getEdgeMembers().getNames().end());
   types.insert(types.end(), getNodeMembers().getTypes().begin(), getNodeMembers().getTypes().end());
   types.insert(types.end(), getEdgeMembers().getTypes().begin(), getEdgeMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr graph::NodeRefType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getNodeMembers().getNames().begin(), getNodeMembers().getNames().end());
   names.insert(names.end(), getIncomingMembers().getNames().begin(), getIncomingMembers().getNames().end());
   names.insert(names.end(), getOutgoingMembers().getNames().begin(), getOutgoingMembers().getNames().end());
   names.insert(names.end(), getPropertyMembers().getNames().begin(), getPropertyMembers().getNames().end());
   types.insert(types.end(), getNodeMembers().getTypes().begin(), getNodeMembers().getTypes().end());
   types.insert(types.end(), getIncomingMembers().getTypes().begin(), getIncomingMembers().getTypes().end());
   types.insert(types.end(), getOutgoingMembers().getTypes().begin(), getOutgoingMembers().getTypes().end());
   types.insert(types.end(), getPropertyMembers().getTypes().begin(), getPropertyMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr graph::EdgeRefType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getEdgeMembers().getNames().begin(), getEdgeMembers().getNames().end());
   names.insert(names.end(), getFromMembers().getNames().begin(), getFromMembers().getNames().end());
   names.insert(names.end(), getToMembers().getNames().begin(), getToMembers().getNames().end());
   names.insert(names.end(), getPropertyMembers().getNames().begin(), getPropertyMembers().getNames().end());
   types.insert(types.end(), getEdgeMembers().getTypes().begin(), getEdgeMembers().getTypes().end());
   types.insert(types.end(), getFromMembers().getTypes().begin(), getFromMembers().getTypes().end());
   types.insert(types.end(), getToMembers().getTypes().begin(), getToMembers().getTypes().end());
   types.insert(types.end(), getPropertyMembers().getTypes().begin(), getPropertyMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.cpp.inc"
void lingodb::compiler::dialect::graph::GraphDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/Graph/GraphOpsTypes.cpp.inc"

      >();
}
// #include "lingodb/compiler/Dialect/Graph/GraphOpsTypeInterfaces.cpp.inc"