#ifndef LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H
#define LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H

#include "lingodb/runtime/Graph/GraphSets.h"

namespace lingodb::runtime {
// Implementation of a native property graph following Graph Databases, 2nd Edition by Ian Robinson, Jim Webber & Emil Eifrem
// See: https://www.oreilly.com/library/view/graph-databases-2nd/9781491930885/ (Figure 6-4)
class PropertyGraph {
    private:
    struct NodeEntry {
        bool inUse;
        PropertyGraph* graph;
        edge_id_t nextRelationship;
        int64_t property; // TODO for now we only support a single node property of type i64
    }; // NodeEntry
    struct RelationshipEntry {
        bool inUse;
        PropertyGraph* graph;
        node_id_t firstNode;
        node_id_t secondNode;
        int64_t type;
        edge_id_t firstPrevRelation;
        edge_id_t firstNextRelation;
        edge_id_t secondPrevRelation;
        edge_id_t secondNextRelation;
        int64_t property; // TODO for now we only support a single edge property of type i64
    }; // RelationshipEntry
    struct PropertyGraphNodeSet : GraphNodeSet {
        PropertyGraph* graph;
        PropertyGraphNodeSet(PropertyGraph* graph) : graph(graph) {}
        PropertyGraph* getGraph() override { return graph; }
        BufferIterator* createIterator() override;
    }; // PropertyGraphNodeSet
    struct PropertyGraphRelationshipSet : GraphEdgeSet {
        PropertyGraph* graph;
        PropertyGraphRelationshipSet(PropertyGraph* graph) : graph(graph) {}
        PropertyGraph* getGraph() override { return graph; }
        BufferIterator* createIterator() override;
    }; // PropertyGraphRelationshipSet
    struct PropertyGraphNodeLinkedRelationshipsSet : GraphNodeLinkedEdgesSet {
        PropertyGraph* graph;
        PropertyGraphNodeLinkedRelationshipsSet(PropertyGraph* graph) : graph(graph) {}
        PropertyGraph* getGraph() override { return graph; }
        void* getFirstEdge(node_id_t node) override { return (void*) getGraph()->getNode(node)->nextRelationship; }
        void* getEdgesBuf() override { return getGraph()->relationships.ptr; }
    }; // PropertyGraphNodeLinkedRelationshipsSet
    runtime::LegacyFixedSizedBuffer<NodeEntry> nodes;
    runtime::LegacyFixedSizedBuffer<RelationshipEntry> relationships;
    std::vector<NodeEntry*> unusedNodeEntries;
    std::vector<RelationshipEntry*> unusedRelEntries;
    PropertyGraphNodeSet nodeSet;
    PropertyGraphRelationshipSet edgeSet;
    PropertyGraphNodeLinkedRelationshipsSet connectionsSet;
    PropertyGraph(size_t maxNodeCapacity, size_t maxRelCapacity) 
        : nodes(maxNodeCapacity), relationships(maxRelCapacity), nodeSet(this), edgeSet(this), connectionsSet(this) {}

    node_id_t nodeBufferSize = 0;
    edge_id_t relBufferSize = 0;

    node_id_t getNodeId(NodeEntry* node) const;
    edge_id_t getRelationshipId(RelationshipEntry* rel) const;
    NodeEntry* getNode(node_id_t node) const;
    RelationshipEntry* getRelationship(edge_id_t rel) const;

    public:
    node_id_t addNode();
    edge_id_t addRelationship(node_id_t from, node_id_t to);

    node_id_t removeNode(node_id_t node);
    edge_id_t removeRelationship(edge_id_t rel);

    void setNodeProperty(node_id_t id, int64_t value);
    int64_t getNodeProperty(node_id_t id) const;
    void setRelationshipProperty(edge_id_t id, int64_t value);
    int64_t getRelationshipProperty(edge_id_t id) const;

    static PropertyGraph* create(size_t initialNodeCapacity, size_t initialRelationshipCapacity);
    static PropertyGraph* createTestGraph();
    static void destroy(PropertyGraph*);

    GraphNodeSet* getNodeSet() { return &nodeSet; }
    GraphEdgeSet* getEdgeSet() { return &edgeSet; };
    GraphNodeLinkedEdgesSet* getNodeLinkedEdgeSet() { return &connectionsSet; }

    Buffer getNodeBuffer() { return Buffer{(size_t) nodeBufferSize, (uint8_t*) nodes.ptr }; }
    Buffer getRelationshipBuffer() { return Buffer{(size_t) relBufferSize, (uint8_t*) relationships.ptr }; }

}; // PropertyGraphLinkedRelationshipsSet
} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H