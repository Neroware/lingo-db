#ifndef LINGODB_DIALECT_GRAPH_DIALECT
#define LINGODB_DIALECT_GRAPH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/SubOperator/SubOperatorBase.td"

//===----------------------------------------------------------------------===//
// Graph dialect definition.
//===----------------------------------------------------------------------===//

def Graph_Dialect : Dialect {
    let name = "graph";
    let summary = "A dialect for universal graph processing";
    let cppNamespace = "::lingodb::compiler::dialect::graph";
    let useDefaultTypePrinterParser=1;
    let useDefaultAttributePrinterParser=1;
    let extraClassDeclaration = [{
            private:
            void registerTypes();
            void registerAttrs();
        }];
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class Graph_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Graph_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def NodeSetIterator : Graph_Type<"NodeSetIterator", "node_set_iterator"> {
    let summary = "node set iterator";
}
def EdgeSetIterator : Graph_Type<"EdgeSetIterator", "edge_set_iterator"> {
    let summary = "edge set iterator";
}

def NodeSet : Graph_Type<"NodeSet", "node_set", [State]> {
    let summary = "node set";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def EdgeSet : Graph_Type<"EdgeSet", "edge_set", [State]> {
    let summary = "edge set";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}

def Graph : Graph_Type<"Graph", "graph", [State]> {
    let summary = "universal graph type";
    let parameters = (ins "StateMembersAttr":$nodeMembers, "StateMembersAttr":$edgeMembers);
    let assemblyFormat = "`<` custom<StateMembers>($nodeMembers) `,` custom<StateMembers>($edgeMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}

def NodeRef : Graph_Type<"NodeRef", "node_ref", [State]> {
    let summary = "graph node reference";
    let parameters = (ins "StateMembersAttr":$nodeMembers, "StateMembersAttr":$incomingMembers, "StateMembersAttr":$outgoingMembers, "StateMembersAttr":$propertyMembers);
    let assemblyFormat = "`<` custom<StateMembers>($nodeMembers) `,` custom<StateMembers>($incomingMembers) `,` custom<StateMembers>($outgoingMembers) `,` custom<StateMembers>($propertyMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def EdgeRef : Graph_Type<"EdgeRef", "edge_ref", [State]> {
    let summary = "graph edge reference";
    let parameters = (ins "StateMembersAttr":$edgeMembers, "StateMembersAttr":$fromMembers, "StateMembersAttr":$toMembers, "StateMembersAttr":$propertyMembers);
    let assemblyFormat = "`<` custom<StateMembers>($edgeMembers) `,` custom<StateMembers>($fromMembers) `,` custom<StateMembers>($toMembers) `,` custom<StateMembers>($propertyMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class Graph_Attr<string name> : AttrDef<Graph_Dialect, name>;

// Currently needed because otherwise attribute part of the codegen won't run causing a compiler error
def GraphDummyAttr : Graph_Attr<"GraphDummyAttr"> {
    let mnemonic = "graph_dummy_attr";
}

#endif // LINGODB_DIALECT_GRAPH_DIALECT