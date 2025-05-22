#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Graph/GraphSet.h"

namespace lingodb::runtime {
typedef int64_t node_id_t;
typedef int64_t relationship_id_t;
// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
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

    node_id_t nodeBufferSize = 0;
    relationship_id_t relBufferSize = 0;

    node_id_t getNodeId(NodeEntry* node) const;
    relationship_id_t getRelationshipId(RelationshipEntry* rel) const;
    NodeEntry* getNode(node_id_t node) const;
    RelationshipEntry* getRelationship(relationship_id_t rel) const;

    public:
    struct AllNodesIterator;
    struct AllEdgesIterator;
    struct LinkedRelationshipsIterator;

    node_id_t addNode();
    relationship_id_t addRelationship(node_id_t from, node_id_t to);

    node_id_t removeNode(node_id_t node);
    relationship_id_t removeRelationship(relationship_id_t rel);

    void setNodeProperty(node_id_t id, int64_t value);
    int64_t getNodeProperty(node_id_t id) const;
    void setRelationshipProperty(relationship_id_t id, int64_t value);
    int64_t getRelationshipProperty(relationship_id_t id) const;

    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static PropertyGraph* createTestGraph();
    static void destroy(PropertyGraph*);

    // TODO Store node/edge set states as members in PropertyGraph and free them when graph is destroyed
    // The reason is that a graph is a state containing node/edge sets, which themselves are states!
    // Return pointer to member in PropertyGraph

    NodeSet* createNodeSet();
    EdgeSet* createEdgeSet();
    EdgeSet* createConnectedEdgeSet(node_id_t node);
    EdgeSet* createIncomingEdgeSet(node_id_t node);
    EdgeSet* createOutgoingEdgeSet(node_id_t node);

}; // PropertyGraph
struct PropertyGraphNodeSet : NodeSet {
    PropertyGraph* graph;
    PropertyGraphNodeSet(PropertyGraph* graph) : graph(graph) {}
    NodeSetIterator* createIterator() override;
}; // PropertyGraphNodeSet
struct PropertyGraphEdgeSet : EdgeSet {
    PropertyGraph* graph;
    PropertyGraphEdgeSet(PropertyGraph* graph) : graph(graph) {}
    EdgeSetIterator* createIterator() override;
}; // PropertyGraphEdgeSet
struct PropertyGraphLinkedRelationshipsSet : EdgeSet {
    enum Mode { All, Incoming, Outgoing };
    PropertyGraph* graph;
    node_id_t node;
    Mode mode;
    PropertyGraphLinkedRelationshipsSet(PropertyGraph* graph, node_id_t node, Mode mode) 
        : graph(graph), node(node), mode(mode) {}
    EdgeSetIterator* createIterator() override;
}; // PropertyGraphLinkedRelationshipsSet
} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H