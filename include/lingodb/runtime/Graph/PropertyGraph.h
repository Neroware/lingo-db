#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/helpers.h"

namespace lingodb::runtime::graph {
// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
class PropertyGraph;
struct PropertyGraphSetIterator {
    virtual bool isValid() = 0;
    virtual void next() = 0;
    virtual void* operator*() = 0;

    virtual PropertyGraph* getPropertyGraph() = 0;
    static bool isIteratorValid(PropertyGraphSetIterator* iterator);
    static void iteratorNext(PropertyGraphSetIterator* iterator);

    static PropertyGraph iteratorGetPropertyGraph(PropertyGraphSetIterator* iterator);
    static void destroy(PropertyGraphSetIterator* iterator);
    static void iterate(PropertyGraphSetIterator* iterator, void (*forEachChunk)(PropertyGraph*, void*), void*);
    virtual ~PropertyGraphSetIterator() {}
};
class PropertyGraph {
    struct NodeEntry;
    struct RelationshipEntry;
    struct EdgeSetIterator;
    struct NodeSetIterator;
    struct ConnectedNodesIterator;
    struct IncomingNodesIterator;
    struct OutgoingNodesIterator;
    struct NodeEntry {
        bool inUse;
        RelationshipEntry* nextRelationship;
        int64_t property; // TODO for now we only support a single node property of type i64
    }; // NodeEntry
    struct RelationshipEntry {
        bool inUse;
        NodeEntry* firstNode;
        NodeEntry* secondNode;
        int64_t type;
        RelationshipEntry* firstPrevRelation;
        RelationshipEntry* firstNextRelation;
        RelationshipEntry* secondPrevRelation;
        RelationshipEntry* secondNextRelation;
        int64_t property; // TODO for now we only support a single edge property of type i64
        bool firstInChain;
    }; // RelationshipEntry
    runtime::LegacyFixedSizedBuffer<NodeEntry> nodes;
    runtime::LegacyFixedSizedBuffer<RelationshipEntry> relationships;
    std::vector<NodeEntry*> unusedNodeEntries;
    std::vector<RelationshipEntry*> unusedRelEntries;
    PropertyGraph(size_t maxNodeCapacity, size_t maxRelCapacity) : nodes(maxNodeCapacity), relationships(maxRelCapacity) {}

    private:
    size_t nodeEntryCount = 0;
    size_t relEntryCount = 0;

    public:
    NodeEntry* addNode();
    RelationshipEntry* addRelationship(NodeEntry* from, NodeEntry* to);

    void removeNode(NodeEntry* node);
    void removeRelationship(RelationshipEntry* rel);
    bool removeNodeById(int64_t node);
    bool removeRelationshipById(int64_t rel);

    int64_t getNodeId(NodeEntry* node);
    int64_t getRelationshipId(RelationshipEntry* rel);
    NodeEntry* getNodeFromId(int64_t nodeId);
    RelationshipEntry* getRelFromId(int64_t relId);

    bool setNodeProperty(int64_t id, int64_t value);
    int64_t getNodeProperty(int64_t id);
    bool setRelationshipProperty(int64_t id, int64_t value);
    int64_t getRelationshipProperty(int64_t id);

    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static void destroy(PropertyGraph*);

    PropertyGraphSetIterator* createNodeSetIterator();
    PropertyGraphSetIterator* createEdgeSetIterator();
    PropertyGraphSetIterator* createConnectedEdgeSetIterator(NodeEntry* node);
    PropertyGraphSetIterator* createIncomingEdgeSetIterator(NodeEntry* node);
    PropertyGraphSetIterator* createOutgoingEdgeSetIterator(NodeEntry* node);

}; // PropertyGraph
} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H