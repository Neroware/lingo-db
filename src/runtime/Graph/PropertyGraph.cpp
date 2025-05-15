#include "lingodb/runtime/Graph/PropertyGraph.h"
#include <cassert>

namespace lingodb::runtime::graph {

PropertyGraph::NodeEntry* PropertyGraph::addNode() {
    NodeEntry* node;
    if (unusedNodeEntries.empty()) {
        node = nodes.getPtr(nodeEntryCount++);
    }
    else {
        node = unusedNodeEntries.back();
        unusedNodeEntries.pop_back();
    }
    assert(!node->inUse && "should not happen");
    node->inUse = true;
    node->nextRelationship = nullptr;
    node->property = 0;
    return node;
}
PropertyGraph::RelationshipEntry* PropertyGraph::addRelationship(NodeEntry* from, NodeEntry* to) {
    RelationshipEntry* rel;
    if (unusedRelEntries.empty()) {
        rel = relationships.getPtr(relEntryCount++);
    }
    else {
        rel = unusedRelEntries.back();
        unusedRelEntries.pop_back();
    }
    assert(!rel->inUse && "should not happen");
    rel->inUse = true;
    rel->firstNode = from;
    rel->secondNode = to;

    // TODO add references for relationship chain!

    return rel;
}

} // lingodb::runtime::graph

// TODO Property Graph implementation