#include "lingodb/compiler/Dialect/Graph/GraphOps.h"

#include "lingodb/compiler/Dialect/Graph/GraphDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace lingodb::compiler::dialect;

namespace {
tuples::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
}
ParseResult parseCustRef(OpAsmParser& parser, tuples::ColumnRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}

void printCustRef(OpAsmPrinter& p, mlir::Operation* op, tuples::ColumnRefAttr attr) {
   p << attr.getName();
}
ParseResult parseCustRefArr(OpAsmParser& parser, ArrayAttr& attr) {
   ArrayAttr parsedAttr;
   std::vector<Attribute> attributes;
   if (parser.parseAttribute(parsedAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
   }
   for (auto a : parsedAttr) {
      SymbolRefAttr parsedSymbolRefAttr = mlir::dyn_cast<SymbolRefAttr>(a);
      tuples::ColumnRefAttr attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
      attributes.push_back(attr);
   }
   attr = ArrayAttr::get(parser.getBuilder().getContext(), attributes);
   return success();
}

void printCustRefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      tuples::ColumnRefAttr parsedSymbolRefAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(a);
      p << parsedSymbolRefAttr.getName();
   }
   p << "]";
}
ParseResult parseCustDef(OpAsmParser& parser, tuples::ColumnDefAttr& attr) {
   SymbolRefAttr attrSymbolAttr;
   if (parser.parseAttribute(attrSymbolAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   std::string attrName(attrSymbolAttr.getLeafReference().getValue());
   if (parser.parseLParen()) { return failure(); }
   DictionaryAttr dictAttr;
   if (parser.parseAttribute(dictAttr)) { return failure(); }
   mlir::ArrayAttr fromExisting;
   if (parser.parseRParen()) { return failure(); }
   if (parser.parseOptionalEqual().succeeded()) {
      if (parseCustRefArr(parser, fromExisting)) {
         return failure();
      }
   }
   attr = getColumnManager(parser).createDef(attrSymbolAttr, fromExisting);
   auto propType = mlir::dyn_cast<TypeAttr>(dictAttr.get("type")).getValue();
   attr.getColumn().type = propType;
   return success();
}
void printCustDef(OpAsmPrinter& p, mlir::Operation* op, tuples::ColumnDefAttr attr) {
   p << attr.getName();
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   const tuples::Column& relationalAttribute = attr.getColumn();
   relAttrDefProps.push_back({mlir::StringAttr::get(context, "type"), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, relAttrDefProps) << ")";
   Attribute fromExisting = attr.getFromExisting();
   if (fromExisting) {
      ArrayAttr fromExistingArr = mlir::dyn_cast_or_null<ArrayAttr>(fromExisting);
      p << "=";
      printCustRefArr(p, op, fromExistingArr);
   }
}

void mapResults(mlir::IRMapping& mapping, mlir::Operation* from, mlir::Operation* to) {
   for (auto i = 0ul; i < from->getNumResults(); i++) {
      mapping.map(from->getResult(i), to->getResult(i));
   }
}

} // namespace 

void graph::ScanGraphOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getGraph() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getElem().getColumn().type);
      setElemAttr(transformer.createReplacementColumn(getElemAttr(), newRefType));
   }
}
void graph::ScanGraphOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
mlir::Operation* graph::ScanGraphOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ScanNodeSetOp>(this->getLoc(), mapping.lookupOrDefault(getGraph()), columnMapping.clone(getElem()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   
   return newOp;
}

void graph::ScanNodeSetOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getNodeSet() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getElem().getColumn().type);
      setElemAttr(transformer.createReplacementColumn(getElemAttr(), newRefType));
   }
}
void graph::ScanNodeSetOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
mlir::Operation* graph::ScanNodeSetOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ScanNodeSetOp>(this->getLoc(), mapping.lookupOrDefault(getNodeSet()), columnMapping.clone(getElem()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}

void graph::ScanEdgeSetOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getEdgeSet() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getElem().getColumn().type);
      setElemAttr(transformer.createReplacementColumn(getElemAttr(), newRefType));
   }
}
void graph::ScanEdgeSetOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
mlir::Operation* graph::ScanEdgeSetOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ScanNodeSetOp>(this->getLoc(), mapping.lookupOrDefault(getEdgeSet()), columnMapping.clone(getElem()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/Graph/GraphOps.cpp.inc"