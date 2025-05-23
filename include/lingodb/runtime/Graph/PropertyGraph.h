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
        PropertyGraphNodeLinkedRelationshipsSet(PropertyGraph* graph, Mode mode) 
            : GraphNodeLinkedEdgesSet(mode), graph(graph) {}
        PropertyGraph* getGraph() override { return graph; }
        int64_t getMode() override { return (int64_t) mode; }
        void* getNodeRef(node_id_t node) override { return (void*) getGraph()->getNode(node); }
    }; // PropertyGraphNodeLinkedRelationshipsSet
    runtime::LegacyFixedSizedBuffer<NodeEntry> nodes;
    runtime::LegacyFixedSizedBuffer<RelationshipEntry> relationships;
    std::vector<NodeEntry*> unusedNodeEntries;
    std::vector<RelationshipEntry*> unusedRelEntries;
    PropertyGraphNodeSet nodeSet;
    PropertyGraphRelationshipSet edgeSet;
    PropertyGraphNodeLinkedRelationshipsSet nodeConnectionsSet;
    PropertyGraphNodeLinkedRelationshipsSet nodeIncomingSet;
    PropertyGraphNodeLinkedRelationshipsSet nodeOutgoingSet;
    PropertyGraph(size_t maxNodeCapacity, size_t maxRelCapacity) 
        : nodes(maxNodeCapacity), relationships(maxRelCapacity), nodeSet(this), edgeSet(this), 
        nodeConnectionsSet(this, GraphNodeLinkedEdgesSet::All), 
        nodeIncomingSet(this, GraphNodeLinkedEdgesSet::Incoming),
        nodeOutgoingSet(this, GraphNodeLinkedEdgesSet::Outgoing) {}

    node_id_t nodeBufferSize = 0;
    relationship_id_t relBufferSize = 0;

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
    static PropertyGraph* createTestGraph();
    static void destroy(PropertyGraph*);

    GraphNodeSet* getNodeSet() { return &nodeSet; }
    GraphEdgeSet* getEdgeSet() { return &edgeSet; };
    GraphNodeLinkedEdgesSet* getNodeConnectedEdgeSet() { return &nodeConnectionsSet; }
    GraphNodeLinkedEdgesSet* getNodeIncomingEdgeSet() { return &nodeIncomingSet; }
    GraphNodeLinkedEdgesSet* getNodeOutgoingEdgeSet() { return &nodeOutgoingSet; }

    Buffer getNodeBuffer() { return Buffer{(size_t) nodeBufferSize, (uint8_t*) nodes.ptr }; }
    Buffer getRelationshipBuffer() { return Buffer{(size_t) relBufferSize, (uint8_t*) relationships.ptr }; }

}; // PropertyGraphLinkedRelationshipsSet
} // lingodb::runtime::graph

#endif // LINGODB_RUNTIME_GRAPH_PROPERTYGRAPH_H