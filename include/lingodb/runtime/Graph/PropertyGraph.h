#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/helpers.h"

namespace lingodb::runtime::graph {
typedef int64_t node_id_t;
typedef int64_t relationship_id_t;
// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
class PropertyGraph;
struct NodeSetIterator {
    virtual bool isValid() = 0;
    virtual void next() = 0;

    virtual PropertyGraph* getPropertyGraph() = 0;
    static bool isIteratorValid(NodeSetIterator* iterator);
    static void iteratorNext(NodeSetIterator* iterator);

    static PropertyGraph* iteratorGetPropertyGraph(NodeSetIterator* iterator);
    static void destroy(NodeSetIterator* iterator);
    static void iterate(NodeSetIterator* iterator, void (*forEachChunk)(PropertyGraph*, node_id_t), void*);
    virtual ~NodeSetIterator() {}
}; // NodeSetIterator
struct EdgeSetIterator {
    virtual bool isValid() = 0;
    virtual void next() = 0;

    virtual PropertyGraph* getPropertyGraph() = 0;
    static bool isIteratorValid(EdgeSetIterator* iterator);
    static void iteratorNext(EdgeSetIterator* iterator);

    static PropertyGraph* iteratorGetPropertyGraph(EdgeSetIterator* iterator);
    static void destroy(EdgeSetIterator* iterator);
    static void iterate(EdgeSetIterator* iterator, void (*forEachChunk)(PropertyGraph*, relationship_id_t), void*);
    virtual ~EdgeSetIterator() {}
}; // EdgeSetIterator
class PropertyGraph {
    private:
    struct NodeEntry {
        bool inUse;
        relationship_id_t nextRelationship;
        int64_t property; // TODO for now we only support a single node property of type i64
    }; // NodeEntry
    struct RelationshipEntry {
        bool inUse;
        node_id_t firstNode;
        node_id_t secondNode;
        int64_t type;
        relationship_id_t firstPrevRelation;
        relationship_id_t firstNextRelation;
        relationship_id_t secondPrevRelation;
        relationship_id_t secondNextRelation;
        int64_t property; // TODO for now we only support a single edge property of type i64
    }; // RelationshipEntry
    runtime::LegacyFixedSizedBuffer<NodeEntry> nodes;
    runtime::LegacyFixedSizedBuffer<RelationshipEntry> relationships;
    std::vector<NodeEntry*> unusedNodeEntries;
    std::vector<RelationshipEntry*> unusedRelEntries;
    PropertyGraph(size_t maxNodeCapacity, size_t maxRelCapacity) : nodes(maxNodeCapacity), relationships(maxRelCapacity) {}

    size_t nodeEntryCount = 0;
    size_t relEntryCount = 0;

    node_id_t getNodeId(NodeEntry* node) const;
    relationship_id_t getRelationshipId(RelationshipEntry* rel) const;
    NodeEntry* getNode(node_id_t node) const;
    RelationshipEntry* getRelationship(relationship_id_t rel) const;

    public:
    node_id_t addNode();
    relationship_id_t addRelationship(node_id_t from, node_id_t to);

    node_id_t removeNode(node_id_t node);
    relationship_id_t removeRelationship(relationship_id_t rel);

    void setNodeProperty(node_id_t id, int64_t value);
    int64_t getNodeProperty(node_id_t id) const;
    void setRelationshipProperty(relationship_id_t id, int64_t value);
    int64_t getRelationshipProperty(relationship_id_t id) const;

    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static void destroy(PropertyGraph*);

    NodeSetIterator* createNodeSetIterator() const;
    EdgeSetIterator* createEdgeSetIterator() const;
    EdgeSetIterator* createConnectedEdgeSetIterator(node_id_t node) const;
    EdgeSetIterator* createIncomingEdgeSetIterator(node_id_t node) const;
    EdgeSetIterator* createOutgoingEdgeSetIterator(node_id_t node) const;

}; // PropertyGraph
} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H