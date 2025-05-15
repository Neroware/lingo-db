#ifndef LINGODB_DIALECT_SubOperator_IR_BASE
#define LINGODB_DIALECT_SubOperator_IR_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "lingodb/compiler/Dialect/TupleStream/TupleStreamBase.td"
include "lingodb/compiler/Dialect/SubOperator/SubOpInterfaces.td"


//===----------------------------------------------------------------------===//
// SubOperator dialect definition.
//===----------------------------------------------------------------------===//

def SubOperator_Dialect : Dialect {
    let name = "subop";
    let summary = "A dialect around sub-operators that manipulate a tuplestream, but state is managed explicitly";
    let cppNamespace = "::lingodb::compiler::dialect::subop";
        let extraClassDeclaration = [{
            private:
            void registerTypes();
            void registerAttrs();
            MemberManager memberManager;
            public:
            MemberManager& getMemberManager(){
                return memberManager;
            }
        }];
    let useDefaultTypePrinterParser=1;
    let useDefaultAttributePrinterParser=1;
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class SubOperator_Attr<string name> : AttrDef<SubOperator_Dialect, name>;

def StateMembers : SubOperator_Attr<"StateMembers"> {
    let mnemonic = "state_members";
    let parameters = (ins "mlir::ArrayAttr":$names,"mlir::ArrayAttr":$types);
    let assemblyFormat ="`<` $names `:` $types `>`";
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class SubOperator_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<SubOperator_Dialect, name,traits> {
    let mnemonic = typeMnemonic;
}
def ThreadLocal : SubOperator_Type<"ThreadLocal", "thread_local"> {
    let summary = "wrapper around any State that makes it thread local";
    let parameters = (ins "State":$wrapped);
    let assemblyFormat = "`<` $wrapped `>`";
}
def Table : SubOperator_Type<"Table", "table",[State]> {
    let summary = "external table";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def LocalTable : SubOperator_Type<"LocalTable", "local_table",[State]> {
    let summary = "local table";
    let parameters = (ins "StateMembersAttr":$members, "mlir::ArrayAttr": $columnNames);
    let assemblyFormat = "`<` custom<StateMembers>($members) `,` $columnNames `>`";
}
def ResultTable : SubOperator_Type<"ResultTable", "result_table",[State]> {
    let summary = "reference to table";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}
def SimpleState : SubOperator_Type<"SimpleState", "simple_state",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
        let extraClassDeclaration = [{
            StateMembersAttr getKeyMembers(){
            return lingodb::compiler::dialect::subop::StateMembersAttr::get(getContext(), mlir::ArrayAttr::get(getContext(), {}), mlir::ArrayAttr::get(getContext(), {}));
            }
            StateMembersAttr getValueMembers();
        }];
}
def Map : SubOperator_Type<"Map", "map",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers, "bool": $withLock  );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers)custom<WithLock>($withLock) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
        bool hasLock(){return getWithLock();}
    }];
}
def Hashmap : SubOperator_Type<"HashMap", "hashmap",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers, "bool": $withLock  );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers)custom<WithLock>($withLock) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
        bool hasLock(){return getWithLock();}
    }];
}
def PreAggrHtFragment : SubOperator_Type<"PreAggrHtFragment", "optimistic_ht_fragment",[State,LookupAbleState]> {
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers, "bool": $withLock  );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers)  custom<WithLock>($withLock)  `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
        bool hasLock(){return getWithLock();}
    }];
}
def PreAggrHt : SubOperator_Type<"PreAggrHt", "optimistic_ht",[State,LookupAbleState]> {
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers, "bool": $withLock  );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers)  custom<WithLock>($withLock) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
        bool hasLock(){return getWithLock();}
    }];
}
def HashMultimap : SubOperator_Type<"HashMultiMap", "hashmultimap",[State,LookupAbleState]> {
    let summary = "simple state with multiple members";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def MultiMap : SubOperator_Type<"MultiMap", "multimap",[State,LookupAbleState]> {
    let summary = "multi map: key -> [values]";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
        StateMembersAttr getMembers();
    }];
}
def ExternalHashIndex : SubOperator_Type<"ExternalHashIndex", "externalhashindex",[State,LookupAbleState]> {
    let summary = "external hash index: key -> [values]";
    let parameters = (ins "StateMembersAttr": $keyMembers, "StateMembersAttr": $valueMembers);
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
    let extraClassDeclaration = [{
       StateMembersAttr getMembers();
    }];
}
def Buffer : SubOperator_Type<"Buffer", "buffer", [State]> {
    let summary = "growing buffer type";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}

def Array : SubOperator_Type<"Array", "array", [State]> {
    let summary = "fixed sized array";
    let parameters = (ins "StateMembersAttr":$members);
    let assemblyFormat = "`<` custom<StateMembers>($members) `>`";
}

def ContinuousView : SubOperator_Type<"ContinuousView", "continuous_view", [State]> {
    let summary = "continuous view type";
    let parameters = (ins "State":$based_on);
    let assemblyFormat = "`<` $based_on `>`";
     let extraClassDeclaration = [{
         StateMembersAttr getMembers(){return getBasedOn().getMembers();}
     }];
}

def Heap : SubOperator_Type<"Heap", "heap", [State]> {
    let summary = "heap type";
    let parameters = (ins "StateMembersAttr":$members, "uint32_t": $max_elements);
    let assemblyFormat = "`<` $max_elements `,` custom<StateMembers>($members) `>`";
}
def SortedView : SubOperator_Type<"SortedView", "sorted_view", [State]> {
    let summary = "sorted view type";
    let parameters = (ins "State":$based_on);
    let assemblyFormat = "`<` $based_on `>`";
     let extraClassDeclaration = [{
         StateMembersAttr getMembers(){return getBasedOn().getMembers();}
     }];
}
def HashIndexedView : SubOperator_Type<"HashIndexedView", "hash_indexed_view", [State,LookupAbleState]> {
    let summary = "sorted view type";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
     let extraClassDeclaration = [{
        StateMembersAttr getMembers();
     }];
}

def SegmentTreeView : SubOperator_Type<"SegmentTreeView", "segment_tree_view", [State,LookupAbleState]> {
    let summary = "segment tree view type";
    let parameters = (ins "StateMembersAttr":$keyMembers,"StateMembersAttr":$valueMembers );
    let assemblyFormat = "`<` custom<StateMembers>($keyMembers) `,` custom<StateMembers>($valueMembers) `>`";
     let extraClassDeclaration = [{
        StateMembersAttr getMembers();
     }];
}

def Entry : SubOperator_Type<"Entry", "entry"> {
    let summary = "entry of some state";
    let parameters = (ins "mlir::Type":$t);
    let assemblyFormat = "`<` $t `>`";
}
def GenericEntryRef : SubOperator_Type<"EntryRef", "entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "State":$state);
    let assemblyFormat = "`<` $state `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return false;}
        bool canBeOffset(){return false;}
        bool hasLock(){return getState().hasLock();}
        StateMembersAttr getMembers() { return getState().getMembers(); }
    }];
}
def TableEntryRef : SubOperator_Type<"TableEntryRef", "table_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of table";
    let parameters = (ins "StateMembersAttr": $tableColumns);
    let assemblyFormat = "`<` custom<StateMembers>($tableColumns) `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return false;}
        bool isStable(){return false;}
        bool canBeOffset(){return false;}
        bool hasLock(){return false;}
        StateMembersAttr getMembers() { return getTableColumns(); }
    }];
}
def GenericLookupEntryRef : SubOperator_Type<"LookupEntryRef", "lookup_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "LookupAbleState":$state);
    let assemblyFormat = "`<` $state `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return false;}
        bool canBeOffset(){return false;}
        bool hasLock(){return mlir::cast<subop::State>(getState()).hasLock();}
        StateMembersAttr getMembers(){ return getState().getValueMembers();}
    }];
}
def ContinuousEntryRef : SubOperator_Type<"ContinuousEntryRef", "continous_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "State":$state);
    let assemblyFormat = "`<` $state `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        bool hasLock(){return getState().hasLock();}
        StateMembersAttr getMembers(){ return getState().getMembers();}
    }];
}
def MultiMapEntryRef : SubOperator_Type<"MultiMapEntryRef", "multi_map_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "MultiMapType":$multi_map);
    let assemblyFormat = "`<` $multi_map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        bool hasLock(){return getMultiMap().hasLock();}
        StateMembersAttr getMembers(){ return getMultiMap().getMembers();}
    }];
}
def MapEntryRef : SubOperator_Type<"MapEntryRef", "map_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "MapType":$map);
    let assemblyFormat = "`<` $map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
        bool hasLock(){return getMap().hasLock();}
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getMap().getMembers();} }];
}
def HashMapEntryRef : SubOperator_Type<"HashMapEntryRef", "hash_map_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "HashMapType":$hash_map);
    let assemblyFormat = "`<` $hash_map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
        bool hasLock(){return getHashMap().hasLock();}
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getHashMap().getMembers();} }];
}
def PreAggrHTEntryRef : SubOperator_Type<"PreAggrHTEntryRef", "optimistic_ht_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "PreAggrHtType":$hash_map);
    let assemblyFormat = "`<` $hash_map `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
        bool hasLock(){return getHashMap().hasLock();}
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getHashMap().getMembers();} }];
}
def ExternalHashIndexEntryRef : SubOperator_Type<"ExternalHashIndexEntryRef", "external_hash_index_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "ExternalHashIndexType":$external_hash_index);
    let assemblyFormat = "`<` $external_hash_index `>`";
    let extraClassDeclaration = [{
       bool isReadable(){return true;}
       bool isWriteable(){return true;}
       bool isStable(){return true;}
       bool canBeOffset(){return true;}
       bool hasLock(){return false;}
       StateMembersAttr getMembers();
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getExternalHashIndex().getMembers();} }];
}
def HashMultiMapEntryRef : SubOperator_Type<"HashMultiMapEntryRef", "hash_multimap_entry_ref",[StateEntryReference]> {
    let summary = "reference to entry of some state";
    let parameters = (ins "HashMultiMapType":$hash_multimap);
    let assemblyFormat = "`<` $hash_multimap `>`";
    let extraClassDeclaration = [{
        bool isReadable(){return true;}
        bool isWriteable(){return true;}
        bool isStable(){return true;}
        bool canBeOffset(){return true;}
        StateMembersAttr getMembers();
        bool hasLock(){return getHashMultimap().hasLock();}
    }];
    let extraClassDefinition= [{ StateMembersAttr $cppClass::getMembers(){ return getHashMultimap().getMembers();} }];
}
def EntryList : SubOperator_Type<"List", "list"> {
    let summary = "list type";
    let parameters = (ins "StateEntryReference":$t);
    let assemblyFormat = "`<` $t `>`";
}
def OptionalType : SubOperator_Type<"Optional", "optional"> {
    let summary = "optional type";
    let parameters = (ins "StateEntryReference":$t);
    let assemblyFormat = "`<` $t `>`";
}

#endif // LINGODB_DIALECT_SubOperator_IR_BASE
