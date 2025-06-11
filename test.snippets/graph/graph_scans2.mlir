// ./build/mlir-db-opt test.snippets/graph/graph_scans.mlir

module {
    func.func @main() {
    	%subop_result = subop.execution_group (){
            
            %g = graph.subop.create_graph !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>]>
            %g_scan = graph.subop.scan_graph %g : !graph.graph<[vx : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>],[ex : !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>]> @nodes::@set({type = !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]>}), @edges::@set({type = !graph.edge_set<[ex_it : !graph.edge_set_iterator<["all"]>]>})
            %vx = subop.nested_map %g_scan [@nodes::@set] (%arg0, %arg1){
                %node_stream = graph.subop.scan_node_set %arg1 : !graph.node_set<[vx_it : !graph.node_set_iterator<["all"]>]> @nodes::@ref({type = !graph.node_ref<[node_id : i64],[incoming : !graph.edge_set<[incoming_it : !graph.edge_set_iterator<["incoming"]>]>],[outgoing : !graph.edge_set<[outgoing_it : !graph.edge_set_iterator<["outgoing"]>]>],[property : i64]>})
                tuples.return %node_stream : !tuples.tuplestream
            }
            %result_nodes = subop.gather %vx @nodes::@ref {node_id => @nodes::@id({type = i64})}

            %0 = subop.create !subop.result_table<[int64p0 : i64]>
            subop.materialize %result_nodes {@nodes::@id => int64p0}, %0 : !subop.result_table<[int64p0 : i64]>
            %res = subop.create_from ["int64"] %0 : !subop.result_table<[int64p0 : i64]> -> !subop.local_table<[int64p0 : i64], ["int64"]>
            subop.execution_group_return %res : !subop.local_table<[int64p0 : i64], ["int64"]>
        
        } -> !subop.table<[int64n0 : i64]>
        subop.set_result 0 %subop_result : !subop.table<[int64n0 : i64]>
        return
    }
}