#include "lingodb/runtime/Graph/PropertyGraph.h"
#include <cassert>

namespace lingodb::runtime {

node_id_t PropertyGraph::getNodeId(NodeEntry* node) const {
    return node - nodes.ptr;
}
PropertyGraph::NodeEntry* PropertyGraph::getNode(node_id_t node) const {
    return nodes.ptr + node;
}
relationship_id_t PropertyGraph::getRelationshipId(RelationshipEntry* rel) const {
    return rel - relationships.ptr;
}
PropertyGraph::RelationshipEntry* PropertyGraph::getRelationship(relationship_id_t rel) const {
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
    node->inUse = true;
    node->nextRelationship = -1;
    node->property = 0;
    return getNodeId(node);
}
relationship_id_t PropertyGraph::addRelationship(node_id_t from, node_id_t to) {
    RelationshipEntry* rel;
    NodeEntry *fromNode = getNode(from), *toNode = getNode(to);
    if (unusedRelEntries.empty()) {
        rel = relationships.getPtr(relBufferSize++);
    }
    else {
        rel = unusedRelEntries.back();
        unusedRelEntries.pop_back();
    }
    relationship_id_t relId = getRelationshipId(rel);
    rel->inUse = true;
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
relationship_id_t PropertyGraph::removeRelationship(relationship_id_t rel) {
    assert(false && "not impelemented"); // TODO implement
}
void PropertyGraph::setNodeProperty(node_id_t id, int64_t value) {
    getNode(id)->property = value;
}
int64_t PropertyGraph::getNodeProperty(node_id_t id) const {
    return getNode(id)->property;
}
void PropertyGraph::setRelationshipProperty(relationship_id_t id, int64_t value) {
    getRelationship(id)->property = value;
}
int64_t PropertyGraph::getRelationshipProperty(relationship_id_t id) const {
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

struct PropertyGraph::AllNodesIterator : NodeSetIterator {
    PropertyGraph* graph;
    node_id_t node;
    AllNodesIterator(PropertyGraph* graph) : graph(graph), node(0) {}
    bool isValid() override {
        return node >= 0 && node < graph->nodeBufferSize && graph->getNode(node)->inUse;
    }
    void next() override {
        if (!isValid())
            return;
        while (node < graph->nodeBufferSize) {
            if (graph->getNode(node++)->inUse)
                break;
        }
    }
    node_id_t operator*() override {
        return node;
    }
    PropertyGraph* getPropertyGraph() override {
        return graph;
    }
};
NodeSet* PropertyGraph::createNodeSet() {
    return new PropertyGraphNodeSet(this);
}

struct PropertyGraph::AllEdgesIterator : EdgeSetIterator {
    PropertyGraph* graph;
    relationship_id_t rel;
    AllEdgesIterator(PropertyGraph* graph) : graph(graph), rel(0) {}
    bool isValid() override {
        return rel >= 0 && rel < graph->relBufferSize && graph->getRelationship(rel)->inUse;
    }
    void next() override {
        if (!isValid())
            return;
        while (rel < graph->relBufferSize) {
            if (graph->getRelationship(rel++)->inUse)
                break;
        }
    }
    relationship_id_t operator*() override {
        return rel;
    }
    PropertyGraph* getPropertyGraph() override {
        return graph;
    }
};
EdgeSet* PropertyGraph::createEdgeSet() {
    return new PropertyGraphEdgeSet(this);
}

struct PropertyGraph::LinkedRelationshipsIterator : EdgeSetIterator {
    enum Mode { All, Incoming, Outgoing };
    PropertyGraph* graph;
    node_id_t node;
    relationship_id_t rel;
    Mode mode;
    LinkedRelationshipsIterator(PropertyGraph* graph, node_id_t node, Mode mode = Mode::All) 
        : graph(graph), node(node), rel(graph->getNode(node)->nextRelationship), mode(mode) {
            while (isValidPtr() && !isValidMode()) {
                RelationshipEntry* relPtr = graph->getRelationship(rel);
                rel = relPtr->firstNode == node ? relPtr->firstNextRelation : relPtr->secondNextRelation;
            }
        }
    bool isValid() override {
        return isValidPtr() && isValidMode();
    }
    void next() override {
        do {
            RelationshipEntry* relPtr = graph->getRelationship(rel);
            rel = relPtr->firstNode == node ? relPtr->firstNextRelation : relPtr->secondNextRelation;
        } while (isValidPtr() && !isValidMode());
    }
    relationship_id_t operator*() override {
        return rel;
    }
    PropertyGraph* getPropertyGraph() override {
        return graph;
    }
    private:
    bool isValidPtr() {
        return rel >= 0 && rel < graph->relBufferSize && graph->getRelationship(rel)->inUse;
    }
    bool isValidMode() {
        if (mode == Mode::Outgoing) {
            return graph->getRelationship(rel)->firstNode == node;
        }
        if (mode == Mode::Incoming) {
            return graph->getRelationship(rel)->secondNode == node;
        }
        return true;
    }
};
EdgeSet* PropertyGraph::createConnectedEdgeSet(node_id_t node) {
    return new PropertyGraphLinkedRelationshipsSet(
        this, node, PropertyGraphLinkedRelationshipsSet::Mode::All);
}
EdgeSet* PropertyGraph::createIncomingEdgeSet(node_id_t node) {
    return new PropertyGraphLinkedRelationshipsSet(
        this, node, PropertyGraphLinkedRelationshipsSet::Mode::Incoming);
}
EdgeSet* PropertyGraph::createOutgoingEdgeSet(node_id_t node) {
    return new PropertyGraphLinkedRelationshipsSet(
        this, node, PropertyGraphLinkedRelationshipsSet::Mode::Outgoing);
}

NodeSetIterator* PropertyGraphNodeSet::createIterator() {
    return new PropertyGraph::AllNodesIterator(graph);
}
EdgeSetIterator* PropertyGraphEdgeSet::createIterator() {
    return new PropertyGraph::AllEdgesIterator(graph);
}
EdgeSetIterator* PropertyGraphLinkedRelationshipsSet::createIterator() {
    return new PropertyGraph::LinkedRelationshipsIterator(
        graph, node, (PropertyGraph::LinkedRelationshipsIterator::Mode) mode);
}

} // lingodb::runtime::graph

// TODO Property Graph implementation