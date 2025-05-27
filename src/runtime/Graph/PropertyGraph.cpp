#include "lingodb/runtime/Graph/PropertyGraph.h"
#include <cassert>

namespace lingodb::runtime {

node_id_t PropertyGraph::getNodeId(NodeEntry* node) const {
    return node - nodes.ptr;
}
PropertyGraph::NodeEntry* PropertyGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
edge_id_t PropertyGraph::getRelationshipId(RelationshipEntry* rel) const {
    return rel - relationships.ptr;
}
PropertyGraph::RelationshipEntry* PropertyGraph::getRelationship(edge_id_t rel) const {
    return relationships.ptr + rel;
}
node_id_t PropertyGraph::addNode() {
    NodeEntry* node;
    if (unusedNodeEntries.empty()) {
        node = nodes.getPtr(nodeBufferSize++);
    }
    else {
        node = unusedNodeEntries.back();
        unusedNodeEntries.pop_back();
    }
    assert(!node->inUse && "should not happen");
    node_id_t nodeId = getNodeId(node);
    node->inUse = true;
    node->graph = this;
    node->id = nodeId;
    node->nextRelationship = -1;
    node->property = 0;
    return nodeId;
}
edge_id_t PropertyGraph::addRelationship(node_id_t from, node_id_t to) {
    RelationshipEntry* rel;
    NodeEntry *fromNode = getNode(from), *toNode = getNode(to);
    if (unusedRelEntries.empty()) {
        rel = relationships.getPtr(relBufferSize++);
    }
    else {
        rel = unusedRelEntries.back();
        unusedRelEntries.pop_back();
    }
    edge_id_t relId = getRelationshipId(rel);
    rel->inUse = true;
    rel->graph = this;
    rel->id = relId;
    rel->firstNode = from;
    rel->secondNode = to;
    rel->type = 0;
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
        if (toNodeRelChain->firstNode == to) {
            toNodeRelChain->firstPrevRelation = relId;
            rel->secondNextRelation = toNode->nextRelationship;   
        }
        else {
            toNodeRelChain->secondPrevRelation = relId;
            rel->secondNextRelation = toNode->nextRelationship;
        }
    }
    toNode->nextRelationship = relId;
    return relId;
}
node_id_t PropertyGraph::removeNode(node_id_t node) {
    assert(false && "not impelemented"); // TODO implement
}
edge_id_t PropertyGraph::removeRelationship(edge_id_t rel) {
    assert(false && "not impelemented"); // TODO implement
}
void PropertyGraph::setNodeProperty(node_id_t id, int64_t value) {
    getNode(id)->property = value;
}
int64_t PropertyGraph::getNodeProperty(node_id_t id) const {
    return getNode(id)->property;
}
void PropertyGraph::setRelationshipProperty(edge_id_t id, int64_t value) {
    getRelationship(id)->property = value;
}
int64_t PropertyGraph::getRelationshipProperty(edge_id_t id) const {
    return getRelationship(id)->property;
}
PropertyGraph* PropertyGraph::create(size_t initialNodeCapacity, size_t initialRelationshipCapacity) {
    return new PropertyGraph(initialNodeCapacity, initialRelationshipCapacity);
}
PropertyGraph* PropertyGraph::createTestGraph() {
    auto g = new PropertyGraph(16, 256);
    for (int i = 0; i < 6; i++) {
        g->addNode();
    }
    g->addRelationship(0, 2);
    g->addRelationship(1, 0);
    g->addRelationship(1, 2);
    g->addRelationship(1, 4);
    g->addRelationship(2, 4);
    g->addRelationship(2, 3);
    g->setRelationshipProperty(0, 111);
    g->setRelationshipProperty(2, 222);
    g->setRelationshipProperty(3, 333);
    g->setRelationshipProperty(4, 444);
    g->setRelationshipProperty(5, 555);
    return g;
}
void PropertyGraph::destroy(PropertyGraph* graph) {
    delete graph;
}

class PropertyGraphNodeSetIterator : public BufferIterator {
    GraphNodeSet& nodeSet;
    bool valid;

    public:
    PropertyGraphNodeSetIterator(GraphNodeSet& nodeSet) 
        : nodeSet(nodeSet), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return nodeSet.getGraph()->getNodeBuffer(); }
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in PropertyGraph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // PropertyGraphNodeSetIterator
class PropertyGraphEdgeSetIterator : public BufferIterator {
    GraphEdgeSet& edgeSet;
    bool valid;

    public:
    PropertyGraphEdgeSetIterator(GraphEdgeSet& edgeSet) 
        : edgeSet(edgeSet), valid(true) {}
    bool isValid() override { return valid; }
    void next() override { valid = false; }
    Buffer getCurrentBuffer() override { return edgeSet.getGraph()->getRelationshipBuffer(); }
    void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void* contextPtr) override {
        // TODO No parallelism in PropertyGraph iterators yet...
        auto buffer = getCurrentBuffer();
        forEachChunk(buffer, contextPtr);
    }
}; // PropertyGraphEdgeSetIterator
BufferIterator* PropertyGraph::PropertyGraphNodeSet::createIterator() {
    return new PropertyGraphNodeSetIterator(*this);
}
BufferIterator* PropertyGraph::PropertyGraphRelationshipSet::createIterator() {
    return new PropertyGraphEdgeSetIterator(*this);
}

} // lingodb::runtime::graph