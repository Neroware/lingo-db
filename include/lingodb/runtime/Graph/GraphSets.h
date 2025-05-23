#ifndef LINGODB_RUNTIME_GRAPH_GRAPHSET_H
#define LINGODB_RUNTIME_GRAPH_GRAPHSET_H

#include "lingodb/runtime/helpers.h"
#include "lingodb/runtime/Buffer.h"

namespace lingodb::runtime {

typedef int64_t node_id_t;
typedef int64_t relationship_id_t;

class PropertyGraph;
struct GraphNodeSet {
    virtual PropertyGraph* getGraph() = 0;
    virtual BufferIterator* createIterator() = 0;
    virtual ~GraphNodeSet() {}
}; // GraphNodeSet
struct GraphEdgeSet {
    virtual PropertyGraph* getGraph() = 0;
    virtual BufferIterator* createIterator() = 0;
    virtual ~GraphEdgeSet() {}
}; // GraphEdgeSet
struct GraphNodeLinkedEdgesSet {
    enum Mode { All, Incoming, Outgoing };
    Mode mode;
    GraphNodeLinkedEdgesSet(Mode mode) : mode(mode) {}
    virtual PropertyGraph* getGraph() = 0;
    virtual void* getNodeRef(node_id_t node) = 0;
    virtual int64_t getMode() = 0;
    virtual ~GraphNodeLinkedEdgesSet() {}
}; // GraphNodeLinkedEdgesSet

} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_GRAPH_GRAPHSET_H