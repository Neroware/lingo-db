#ifndef LINGODB_DIALECT_GRAPH_OPS
#define LINGODB_DIALECT_GRAPH_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/Graph/GraphDialect.td"

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

class Graph_Op<string mnemonic, list<Trait> traits = []> : Op<Graph_Dialect, mnemonic, traits> {
    let hasCustomAssemblyFormat = 1;
}

def CreateGraphOp : Graph_Op<"subop.create_graph", [Pure,StateCreator]> {
    let summary = "create new graph";
    let results = (outs Graph : $res);
    let assemblyFormat = [{ type($res) attr-dict }];
}

def ScanGraphOp : Graph_Op<"subop.scan_graph", [Pure,SubOperator,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";
    let arguments = (ins AnyType : $graph, ColumnDefAttr: $nodeSet, ColumnDefAttr: $edgeSet);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{ $graph `:` type($graph) custom<CustDef>($nodeSet) `,` custom<CustDef>($edgeSet) attr-dict }];
    let extraClassDeclaration = [{
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

def ScanNodeSetOp : Graph_Op<"subop.scan_node_set", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";
    let arguments = (ins AnyType : $node_set, ColumnDefAttr: $elem);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{ $node_set `:` type($node_set) custom<CustDef>($elem) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getElem(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}
def ScanEdgeSetOp : Graph_Op<"subop.scan_edge_set", [Pure,SubOperator,ReferenceProducer,DeclareOpInterfaceMethods<StateUsingSubOperator>]> {
    let summary = "scan state (sequentially)";
    let arguments = (ins AnyType : $edge_set, ColumnDefAttr: $elem);
    let results = (outs TupleStream : $res);
    let assemblyFormat = [{ $edge_set `:` type($edge_set) custom<CustDef>($elem) attr-dict }];
    let extraClassDeclaration = [{
        dialect::tuples::ColumnDefAttr getProducedReference(){ return getElem(); }
        mlir::Operation* cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, lingodb::compiler::dialect::subop::ColumnMapping& columnMapping);
    }];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Iterating over nodes and edges
////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO

////////////////////////////////////////////////////////////////////////////////////////////////////
// Properties
////////////////////////////////////////////////////////////////////////////////////////////////////

// def GetNodePropertyOp : Graph_Op<"get_node_property", [Pure]> {
//     let summary = "get node property operation";

//     let arguments = (ins Node:$node,PropertyRefAttr:$propRef);
//     let results = (outs AnyType : $res);
//     let assemblyFormat = [{  $node $propRef `:` type($res) attr-dict }];
// }

// def GetEdgePropertyOp : Graph_Op<"get_edge_property", [Pure]> {
//     let summary = "get edge property operation";

//     let arguments = (ins Node:$node,PropertyRefAttr:$propRef);
//     let results = (outs AnyType : $res);
//     let assemblyFormat = [{  $node $propRef `:` type($res) attr-dict }];
// }

#endif // LINGODB_DIALECT_GRAPH_OPS