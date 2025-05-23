#ifndef LINGODB_RUNTIME_GRAPH_GRAPHSET_H
#define LINGODB_RUNTIME_GRAPH_GRAPHSET_H

#include "lingodb/runtime/helpers.h"

namespace lingodb::runtime {

typedef int64_t node_id_t;
typedef int64_t relationship_id_t;

class PropertyGraph;
struct NodeSetIterator;
struct EdgeSetIterator;
struct NodeSet {
    virtual NodeSetIterator* createIterator() = 0;
    static void destroy(NodeSet* nodeSet);
    virtual ~NodeSet() {}
}; // NodeSet
struct EdgeSet {
    virtual EdgeSetIterator* createIterator() = 0;
    static void destroy(EdgeSet* edgeSet);
    virtual ~EdgeSet() {}
}; // EdgeSet
struct NodeSetIterator {
    virtual bool isValid() = 0;
    virtual void next() = 0;
    virtual node_id_t operator*() = 0;

    virtual PropertyGraph* getPropertyGraph() = 0;
    static bool isIteratorValid(NodeSetIterator* iterator);
    static void iteratorNext(NodeSetIterator* iterator);

    static PropertyGraph* iteratorGetPropertyGraph(NodeSetIterator* iterator);
    static void destroy(NodeSetIterator* iterator);
    static void iterate(NodeSetIterator* iterator, void (*forEachChunk)(PropertyGraph*, node_id_t));
    virtual ~NodeSetIterator() {}
}; // NodeSetIterator
struct EdgeSetIterator {
    virtual bool isValid() = 0;
    virtual void next() = 0;
    virtual relationship_id_t operator*() = 0;

    virtual PropertyGraph* getPropertyGraph() = 0;
    static bool isIteratorValid(EdgeSetIterator* iterator);
    static void iteratorNext(EdgeSetIterator* iterator);

    static PropertyGraph* iteratorGetPropertyGraph(EdgeSetIterator* iterator);
    static void destroy(EdgeSetIterator* iterator);
    static void iterate(EdgeSetIterator* iterator, void (*forEachChunk)(PropertyGraph*, relationship_id_t));
    virtual ~EdgeSetIterator() {}
}; // EdgeSetIterator

} // namespace lingodb::runtime

#endif // LINGODB_RUNTIME_GRAPH_GRAPHSET_H