#include "lingodb/runtime/Graph/PropertyGraph.h"
#include <cassert>

namespace lingodb::runtime::graph {

node_id_t PropertyGraph::getNodeId(NodeEntry* node) const {
    return (node - nodes.ptr) / sizeof(NodeEntry);
}
PropertyGraph::NodeEntry* PropertyGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
relationship_id_t PropertyGraph::getRelationshipId(RelationshipEntry* rel) const {
    return (rel - relationships.ptr) / sizeof(RelationshipEntry);
}
PropertyGraph::RelationshipEntry* PropertyGraph::getRelationship(relationship_id_t rel) const {
    return relationships.ptr + rel;
}
node_id_t PropertyGraph::addNode() {
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
    node->nextRelationship = -1;
    node->property = 0;
    return getNodeId(node);
}
relationship_id_t PropertyGraph::addRelationship(node_id_t from, node_id_t to) {
    RelationshipEntry* rel;
    NodeEntry *fromNode = getNode(from), *toNode = getNode(to);
    if (unusedRelEntries.empty()) {
        rel = relationships.getPtr(relEntryCount++);
    }
    else {
        rel = unusedRelEntries.back();
        unusedRelEntries.pop_back();
    }
    relationship_id_t relId = getRelationshipId(rel);
    rel->inUse = true;
    rel->firstNode = from;
    rel->secondNode = to;
    rel->firstNextRelation = rel->firstPrevRelation = rel->secondNextRelation = rel->secondPrevRelation = -1;
    if (fromNode->nextRelationship >= 0) {
        RelationshipEntry* fromNodeRelChain = getRelationship(fromNode->nextRelationship);
        if (fromNodeRelChain->firstNode == from) {
            fromNodeRelChain->firstPrevRelation = relId;
            rel->firstNextRelation = fromNode->nextRelationship;   
        }
        else {
            fromNodeRelChain->secondPrevRelation = relId;
            rel->firstNextRelation = fromNode->nextRelationship;
        }
    }
    fromNode->nextRelationship = relId;
    if (toNode->nextRelationship >= 0) {
        RelationshipEntry* toNodeRelChain = getRelationship(toNode->nextRelationship);
        if (toNodeRelChain->firstNode == from) {
            toNodeRelChain->firstPrevRelation = relId;
            rel->firstNextRelation = toNode->nextRelationship;   
        }
        else {
            toNodeRelChain->secondPrevRelation = relId;
            rel->firstNextRelation = toNode->nextRelationship;
        }
    }
    toNode->nextRelationship = relId;
    return relId;
}

} // lingodb::runtime::graph

// TODO Property Graph implementation