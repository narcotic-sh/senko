#include "hdbscan.hpp"

#include <algorithm>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <limits>

namespace senko_hdbscan {
namespace {

constexpr uint64_t kWorkspaceAlignment = 16u;
constexpr int kKdLeafSize = 13;
constexpr int kKdBoruvkaThreshold = 1024;

struct MstEdge {
  int32_t from;
  int32_t to;
  double distance;
};

struct CondensedEdge {
  int32_t parent;
  int32_t child;
  double lambda;
  int32_t child_size;
};

struct KdNode {
  int32_t start;
  int32_t end;
  int32_t left;
  int32_t right;
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

bool add_allocation(uint64_t* cursor, uint64_t count, uint64_t item_size,
                    uint64_t alignment = kWorkspaceAlignment) {
  if (count != 0 && item_size > UINT64_MAX / count) return false;
  const uint64_t bytes = count * item_size;
  const uint64_t start = align_up(*cursor, alignment);
  if (start < *cursor || bytes > UINT64_MAX - start) return false;
  *cursor = start + bytes;
  return true;
}

bool valid_arguments(int count, int dimension, int min_samples,
                     int min_cluster_size) {
  return count >= 2 && dimension >= 1 && min_samples >= 1 &&
         min_cluster_size >= 2;
}

uint64_t kd_node_capacity(uint64_t count) {
  const uint64_t minimum_leaves =
      (count + static_cast<uint64_t>(kKdLeafSize) - 1u) /
      static_cast<uint64_t>(kKdLeafSize);
  // Repeated median splits produce fewer than twice the theoretical minimum
  // number of leaves. Four nodes per minimum leaf plus the root is therefore a
  // conservative bound for the complete binary tree.
  return minimum_leaves * 4u + 1u;
}

uint64_t calculate_workspace_bytes(int count, int dimension, int min_samples,
                                   int min_cluster_size) {
  if (!valid_arguments(count, dimension, min_samples, min_cluster_size)) {
    return 0;
  }
  const uint64_t n = static_cast<uint64_t>(count);
  const uint64_t d = static_cast<uint64_t>(dimension);
  const uint64_t k = static_cast<uint64_t>(
      min_samples < count ? min_samples : count - 1);
  const uint64_t node_count = n * 2u - 1u;
  const uint64_t cluster_capacity = n * 2u + 1u;
  uint64_t cursor = 0;

  // Input conversion and exact core-distance heaps.
  if (!add_allocation(&cursor, n * d, sizeof(double)) ||
      !add_allocation(&cursor, n * k, sizeof(double)) ||
      !add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, n, sizeof(double))) {
    return 0;
  }

  // Exact implicit Prim MST.
  if (!add_allocation(&cursor, n - 1u, sizeof(MstEdge)) ||
      !add_allocation(&cursor, n - 1u, sizeof(MstEdge)) ||
      !add_allocation(&cursor, n, sizeof(double)) ||
      !add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, n, sizeof(uint8_t))) {
    return 0;
  }

  // Scalable exact KD-tree/Boruvka provider. The exact Prim buffers above
  // remain available as a small-input correctness oracle.
  const uint64_t kd_nodes = kd_node_capacity(n);
  if (!add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, kd_nodes, sizeof(KdNode)) ||
      !add_allocation(&cursor, kd_nodes * d * 2u, sizeof(double)) ||
      !add_allocation(&cursor, kd_nodes, sizeof(double)) ||
      !add_allocation(&cursor, kd_nodes, sizeof(int32_t)) ||
      !add_allocation(&cursor, k, sizeof(double)) ||
      !add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, n, sizeof(int32_t)) ||
      !add_allocation(&cursor, n, sizeof(double))) {
    return 0;
  }

  // Single-linkage hierarchy and its union-find.
  if (!add_allocation(&cursor, n - 1u, sizeof(int32_t)) ||
      !add_allocation(&cursor, n - 1u, sizeof(int32_t)) ||
      !add_allocation(&cursor, n - 1u, sizeof(int32_t)) ||
      !add_allocation(&cursor, n - 1u, sizeof(double)) ||
      !add_allocation(&cursor, node_count, sizeof(int32_t)) ||
      !add_allocation(&cursor, node_count, sizeof(int32_t))) {
    return 0;
  }

  // Condensation. Two queues are needed because pruning a subtree occurs
  // while walking the breadth-first order of the complete hierarchy.
  if (!add_allocation(&cursor, n * 2u, sizeof(CondensedEdge)) ||
      !add_allocation(&cursor, node_count, sizeof(int32_t)) ||
      !add_allocation(&cursor, node_count, sizeof(int32_t)) ||
      !add_allocation(&cursor, node_count, sizeof(uint8_t)) ||
      !add_allocation(&cursor, node_count, sizeof(int32_t))) {
    return 0;
  }

  // Stability, EOM selection and final condensed-tree union-find.
  if (!add_allocation(&cursor, cluster_capacity, sizeof(double)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(double)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(int32_t)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(int32_t)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(int32_t)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(uint8_t)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(int32_t)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(int32_t)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(int32_t)) ||
      !add_allocation(&cursor, cluster_capacity, sizeof(int32_t))) {
    return 0;
  }
  return align_up(cursor, kWorkspaceAlignment);
}

class Workspace {
 public:
  Workspace(void* memory, uint32_t size)
      : memory_(static_cast<uint8_t*>(memory)), size_(size), cursor_(0) {}

  template <typename T>
  T* allocate(uint64_t count) {
    if (count != 0 && sizeof(T) > UINT64_MAX / count) return nullptr;
    const uint64_t bytes = count * sizeof(T);
    const uint64_t start = align_up(cursor_, kWorkspaceAlignment);
    if (start < cursor_ || start > size_ || bytes > size_ - start) {
      return nullptr;
    }
    cursor_ = start + bytes;
    return reinterpret_cast<T*>(memory_ + start);
  }

 private:
  uint8_t* memory_;
  uint64_t size_;
  uint64_t cursor_;
};

double squared_euclidean(const double* left, const double* right,
                         int dimension) {
  double result = 0.0;
  int column = 0;
  const int unrolled_end = dimension - dimension % 4;
  for (; column < unrolled_end; column += 4) {
    const double difference0 = left[column] - right[column];
    const double difference1 = left[column + 1] - right[column + 1];
    const double difference2 = left[column + 2] - right[column + 2];
    const double difference3 = left[column + 3] - right[column + 3];
    result += difference0 * difference0 + difference1 * difference1 +
              difference2 * difference2 + difference3 * difference3;
  }
  for (; column < dimension; ++column) {
    const double difference = left[column] - right[column];
    result += difference * difference;
  }
  return result;
}

void max_heap_push(double* heap, int32_t* size, int capacity, double value) {
  if (*size < capacity) {
    int position = *size;
    *size += 1;
    while (position > 0) {
      const int parent = (position - 1) / 2;
      if (heap[parent] >= value) break;
      heap[position] = heap[parent];
      position = parent;
    }
    heap[position] = value;
    return;
  }
  if (value >= heap[0]) return;

  int position = 0;
  while (true) {
    const int left = position * 2 + 1;
    if (left >= capacity) break;
    const int right = left + 1;
    int larger = left;
    if (right < capacity && heap[right] > heap[left]) larger = right;
    if (heap[larger] <= value) break;
    heap[position] = heap[larger];
    position = larger;
  }
  heap[position] = value;
}

struct KdTreeView {
  const double* points;
  int count;
  int dimension;
  int32_t* indices;
  KdNode* nodes;
  double* bounds;
  int node_capacity;
  int node_count;
};

double* node_min_bounds(KdTreeView* tree, int node) {
  return tree->bounds +
         static_cast<uint64_t>(node) * tree->dimension * 2u;
}

const double* node_min_bounds(const KdTreeView& tree, int node) {
  return tree.bounds +
         static_cast<uint64_t>(node) * tree.dimension * 2u;
}

double* node_max_bounds(KdTreeView* tree, int node) {
  return node_min_bounds(tree, node) + tree->dimension;
}

const double* node_max_bounds(const KdTreeView& tree, int node) {
  return node_min_bounds(tree, node) + tree.dimension;
}

int build_kd_node(KdTreeView* tree, int start, int end) {
  if (tree->node_count >= tree->node_capacity || start >= end) return -1;
  const int node = tree->node_count++;
  KdNode& result = tree->nodes[node];
  result = {start, end, -1, -1};
  double* minimum = node_min_bounds(tree, node);
  double* maximum = node_max_bounds(tree, node);
  const double* first =
      tree->points +
      static_cast<uint64_t>(tree->indices[start]) * tree->dimension;
  for (int column = 0; column < tree->dimension; ++column) {
    minimum[column] = first[column];
    maximum[column] = first[column];
  }
  for (int cursor = start + 1; cursor < end; ++cursor) {
    const double* point =
        tree->points +
        static_cast<uint64_t>(tree->indices[cursor]) * tree->dimension;
    for (int column = 0; column < tree->dimension; ++column) {
      if (point[column] < minimum[column]) minimum[column] = point[column];
      if (point[column] > maximum[column]) maximum[column] = point[column];
    }
  }
  if (end - start <= kKdLeafSize) return node;

  int split_dimension = 0;
  double largest_span = maximum[0] - minimum[0];
  for (int column = 1; column < tree->dimension; ++column) {
    const double span = maximum[column] - minimum[column];
    if (span > largest_span) {
      largest_span = span;
      split_dimension = column;
    }
  }
  const int middle = start + (end - start) / 2;
  const double* points = tree->points;
  const int dimension = tree->dimension;
  std::nth_element(
      tree->indices + start, tree->indices + middle, tree->indices + end,
      [points, dimension, split_dimension](int32_t left, int32_t right) {
        const double left_value =
            points[static_cast<uint64_t>(left) * dimension + split_dimension];
        const double right_value =
            points[static_cast<uint64_t>(right) * dimension + split_dimension];
        return left_value < right_value ||
               (left_value == right_value && left < right);
      });
  result.left = build_kd_node(tree, start, middle);
  result.right = build_kd_node(tree, middle, end);
  return result.left >= 0 && result.right >= 0 ? node : -1;
}

bool build_kd_tree(const double* points, int count, int dimension,
                   int32_t* indices, KdNode* nodes, double* bounds,
                   int node_capacity, KdTreeView* result) {
  for (int row = 0; row < count; ++row) indices[row] = row;
  *result = {points, count, dimension, indices, nodes, bounds, node_capacity,
             0};
  return build_kd_node(result, 0, count) == 0;
}

double point_box_distance_squared(const double* point,
                                  const KdTreeView& tree, int node) {
  const double* minimum = node_min_bounds(tree, node);
  const double* maximum = node_max_bounds(tree, node);
  double result = 0.0;
  for (int column = 0; column < tree.dimension; ++column) {
    double difference = 0.0;
    if (point[column] < minimum[column]) {
      difference = minimum[column] - point[column];
    } else if (point[column] > maximum[column]) {
      difference = point[column] - maximum[column];
    }
    result += difference * difference;
  }
  return result;
}

void query_core_neighbors(const KdTreeView& tree, int query_index, int node,
                          int neighbor_count, double* heap,
                          int32_t* heap_size) {
  const KdNode& current = tree.nodes[node];
  const double* query =
      tree.points + static_cast<uint64_t>(query_index) * tree.dimension;
  const double maximum_distance =
      *heap_size == neighbor_count
          ? heap[0]
          : std::numeric_limits<double>::infinity();
  if (point_box_distance_squared(query, tree, node) >= maximum_distance) {
    return;
  }
  if (current.left < 0) {
    for (int cursor = current.start; cursor < current.end; ++cursor) {
      const int candidate = tree.indices[cursor];
      if (candidate == query_index) continue;
      const double distance_squared = squared_euclidean(
          query,
          tree.points +
              static_cast<uint64_t>(candidate) * tree.dimension,
          tree.dimension);
      max_heap_push(heap, heap_size, neighbor_count, distance_squared);
    }
    return;
  }

  const double left_distance =
      point_box_distance_squared(query, tree, current.left);
  const double right_distance =
      point_box_distance_squared(query, tree, current.right);
  const int first =
      left_distance <= right_distance ? current.left : current.right;
  const int second =
      left_distance <= right_distance ? current.right : current.left;
  query_core_neighbors(tree, query_index, first, neighbor_count, heap,
                       heap_size);
  query_core_neighbors(tree, query_index, second, neighbor_count, heap,
                       heap_size);
}

bool calculate_core_distances_kd(const KdTreeView& tree, int min_samples,
                                 double* query_heap,
                                 double* core_distances) {
  for (int row = 0; row < tree.count; ++row) {
    int32_t heap_size = 0;
    query_core_neighbors(tree, row, 0, min_samples, query_heap, &heap_size);
    if (heap_size != min_samples) return false;
    core_distances[row] = sqrt(query_heap[0]);
  }
  return true;
}

void calculate_core_distances(const double* points, int count, int dimension,
                              int min_samples, double* heaps,
                              int32_t* heap_sizes, double* core_distances) {
  memset(heap_sizes, 0, static_cast<size_t>(count) * sizeof(int32_t));
  for (int left = 0; left < count; ++left) {
    const double* left_point =
        points + static_cast<uint64_t>(left) * dimension;
    for (int right = left + 1; right < count; ++right) {
      const double distance_squared = squared_euclidean(
          left_point, points + static_cast<uint64_t>(right) * dimension,
          dimension);
      max_heap_push(heaps + static_cast<uint64_t>(left) * min_samples,
                    heap_sizes + left, min_samples, distance_squared);
      max_heap_push(heaps + static_cast<uint64_t>(right) * min_samples,
                    heap_sizes + right, min_samples, distance_squared);
    }
  }
  for (int row = 0; row < count; ++row) {
    core_distances[row] =
        sqrt(heaps[static_cast<uint64_t>(row) * min_samples]);
  }
}

int build_exact_mst(const double* points, int count, int dimension,
                    const double* core_distances, MstEdge* edges,
                    double* current_distances, int32_t* current_sources,
                    uint8_t* in_tree) {
  const double infinity = std::numeric_limits<double>::infinity();
  for (int row = 0; row < count; ++row) {
    current_distances[row] = infinity;
    current_sources[row] = 1;
    in_tree[row] = 0;
  }

  int current_node = 0;
  for (int edge_index = 0; edge_index < count - 1; ++edge_index) {
    in_tree[current_node] = 1;
    double next_distance = std::numeric_limits<double>::max();
    int next_source = 0;
    int next_node = -1;
    const double current_core = core_distances[current_node];
    const double* current_point =
        points + static_cast<uint64_t>(current_node) * dimension;

    for (int candidate = 0; candidate < count; ++candidate) {
      if (in_tree[candidate]) continue;
      double distance =
          sqrt(squared_euclidean(
              current_point,
              points + static_cast<uint64_t>(candidate) * dimension,
              dimension));
      if (current_core > distance) distance = current_core;
      if (core_distances[candidate] > distance) {
        distance = core_distances[candidate];
      }
      if (distance < current_distances[candidate]) {
        current_distances[candidate] = distance;
        current_sources[candidate] = current_node;
      }
      if (current_distances[candidate] < next_distance) {
        next_distance = current_distances[candidate];
        next_source = current_sources[candidate];
        next_node = candidate;
      }
    }
    if (next_node < 0 || !isfinite(next_distance)) return -3;
    edges[edge_index] = {next_source, next_node, next_distance};
    current_node = next_node;
  }
  return 1;
}

int32_t boruvka_find(int32_t* parent, int32_t node) {
  int32_t root = node;
  while (parent[root] != root) root = parent[root];
  while (node != root) {
    const int32_t next = parent[node];
    parent[node] = root;
    node = next;
  }
  return root;
}

void boruvka_union(int32_t* parent, int32_t* rank, int32_t left,
                   int32_t right) {
  left = boruvka_find(parent, left);
  right = boruvka_find(parent, right);
  if (left == right) return;
  if (rank[left] < rank[right]) {
    parent[left] = right;
  } else if (rank[left] > rank[right]) {
    parent[right] = left;
  } else {
    parent[right] = left;
    rank[left] += 1;
  }
}

void update_kd_component_metadata(const KdTreeView& tree,
                                  const double* core_distances,
                                  const int32_t* point_components,
                                  double* minimum_core_squared,
                                  int32_t* homogeneous_component) {
  for (int node = tree.node_count - 1; node >= 0; --node) {
    const KdNode& current = tree.nodes[node];
    if (current.left < 0) {
      const int first_point = tree.indices[current.start];
      double minimum_core =
          core_distances[first_point] * core_distances[first_point];
      int32_t component = point_components[first_point];
      for (int cursor = current.start + 1; cursor < current.end; ++cursor) {
        const int point = tree.indices[cursor];
        const double core =
            core_distances[point] * core_distances[point];
        if (core < minimum_core) minimum_core = core;
        if (point_components[point] != component) component = -1;
      }
      minimum_core_squared[node] = minimum_core;
      homogeneous_component[node] = component;
      continue;
    }
    minimum_core_squared[node] =
        minimum_core_squared[current.left] <
                minimum_core_squared[current.right]
            ? minimum_core_squared[current.left]
            : minimum_core_squared[current.right];
    homogeneous_component[node] =
        homogeneous_component[current.left] ==
                homogeneous_component[current.right]
            ? homogeneous_component[current.left]
            : -1;
  }
}

double external_node_lower_bound(const KdTreeView& tree,
                                 const double* query, int node,
                                 double query_core_squared,
                                 const double* minimum_core_squared) {
  double result = point_box_distance_squared(query, tree, node);
  if (query_core_squared > result) result = query_core_squared;
  if (minimum_core_squared[node] > result) {
    result = minimum_core_squared[node];
  }
  return result;
}

void query_external_neighbor(
    const KdTreeView& tree, int query_index, int node, int32_t component,
    const double* core_distances, const int32_t* point_components,
    const double* minimum_core_squared,
    const int32_t* homogeneous_component, int32_t* candidate_source,
    int32_t* candidate_sink, double* candidate_distance_squared) {
  if (homogeneous_component[node] == component) return;
  const double* query =
      tree.points + static_cast<uint64_t>(query_index) * tree.dimension;
  const double query_core_squared =
      core_distances[query_index] * core_distances[query_index];
  const double lower_bound =
      external_node_lower_bound(tree, query, node, query_core_squared,
                                minimum_core_squared);
  if (lower_bound >= candidate_distance_squared[component]) return;

  const KdNode& current = tree.nodes[node];
  if (current.left < 0) {
    for (int cursor = current.start; cursor < current.end; ++cursor) {
      const int candidate = tree.indices[cursor];
      if (point_components[candidate] == component) continue;
      double mutual_reachability = squared_euclidean(
          query,
          tree.points +
              static_cast<uint64_t>(candidate) * tree.dimension,
          tree.dimension);
      if (query_core_squared > mutual_reachability) {
        mutual_reachability = query_core_squared;
      }
      const double candidate_core_squared =
          core_distances[candidate] * core_distances[candidate];
      if (candidate_core_squared > mutual_reachability) {
        mutual_reachability = candidate_core_squared;
      }
      if (mutual_reachability < candidate_distance_squared[component]) {
        candidate_distance_squared[component] = mutual_reachability;
        candidate_source[component] = query_index;
        candidate_sink[component] = candidate;
      }
    }
    return;
  }

  const double left_bound =
      external_node_lower_bound(tree, query, current.left,
                                query_core_squared, minimum_core_squared);
  const double right_bound =
      external_node_lower_bound(tree, query, current.right,
                                query_core_squared, minimum_core_squared);
  const int first = left_bound <= right_bound ? current.left : current.right;
  const int second = left_bound <= right_bound ? current.right : current.left;
  query_external_neighbor(
      tree, query_index, first, component, core_distances, point_components,
      minimum_core_squared, homogeneous_component, candidate_source,
      candidate_sink, candidate_distance_squared);
  query_external_neighbor(
      tree, query_index, second, component, core_distances, point_components,
      minimum_core_squared, homogeneous_component, candidate_source,
      candidate_sink, candidate_distance_squared);
}

int build_kd_boruvka_mst(
    const KdTreeView& tree, const double* core_distances, MstEdge* edges,
    double* minimum_core_squared, int32_t* homogeneous_component,
    int32_t* union_parent, int32_t* union_rank, int32_t* point_components,
    int32_t* candidate_source, int32_t* candidate_sink,
    double* candidate_distance_squared) {
  for (int point = 0; point < tree.count; ++point) {
    union_parent[point] = point;
    union_rank[point] = 0;
  }

  int edge_count = 0;
  while (edge_count < tree.count - 1) {
    const double infinity = std::numeric_limits<double>::infinity();
    for (int point = 0; point < tree.count; ++point) {
      const int32_t component = boruvka_find(union_parent, point);
      point_components[point] = component;
      candidate_source[point] = -1;
      candidate_sink[point] = -1;
      candidate_distance_squared[point] = infinity;
    }
    update_kd_component_metadata(tree, core_distances, point_components,
                                 minimum_core_squared,
                                 homogeneous_component);
    for (int point = 0; point < tree.count; ++point) {
      query_external_neighbor(
          tree, point, 0, point_components[point], core_distances,
          point_components, minimum_core_squared, homogeneous_component,
          candidate_source, candidate_sink, candidate_distance_squared);
    }

    const int previous_edge_count = edge_count;
    for (int component = 0; component < tree.count; ++component) {
      if (union_parent[component] != component ||
          candidate_source[component] < 0) {
        continue;
      }
      const int source = candidate_source[component];
      const int sink = candidate_sink[component];
      if (boruvka_find(union_parent, source) ==
          boruvka_find(union_parent, sink)) {
        continue;
      }
      if (edge_count >= tree.count - 1) return -3;
      edges[edge_count++] = {
          source, sink, sqrt(candidate_distance_squared[component])};
      boruvka_union(union_parent, union_rank, source, sink);
    }
    if (edge_count == previous_edge_count) return -3;
  }
  return 1;
}

void stable_sort_mst_edges(MstEdge* edges, MstEdge* temporary, int count) {
  MstEdge* source = edges;
  MstEdge* destination = temporary;
  for (int width = 1; width < count; width *= 2) {
    for (int start = 0; start < count; start += width * 2) {
      const int middle = start + width < count ? start + width : count;
      const int end = start + width * 2 < count ? start + width * 2 : count;
      int left = start;
      int right = middle;
      int output = start;
      while (left < middle && right < end) {
        // Taking the left edge on equality preserves the Prim output order.
        // This also matches NumPy's observed ordering for the tied
        // mutual-reachability plateaus in Senko's parity fixtures.
        if (source[left].distance <= source[right].distance) {
          destination[output++] = source[left++];
        } else {
          destination[output++] = source[right++];
        }
      }
      while (left < middle) destination[output++] = source[left++];
      while (right < end) destination[output++] = source[right++];
    }
    MstEdge* swap = source;
    source = destination;
    destination = swap;
  }
  if (source != edges) {
    memcpy(edges, source, static_cast<size_t>(count) * sizeof(MstEdge));
  }
}

int32_t linkage_find(int32_t* parent, int32_t node) {
  int32_t root = node;
  while (parent[root] != -1) root = parent[root];
  while (node != root) {
    const int32_t next = parent[node];
    parent[node] = root;
    node = next;
  }
  return root;
}

void build_single_linkage(MstEdge* edges, MstEdge* temporary_edges, int count,
                          int32_t* left, int32_t* right, int32_t* sizes,
                          double* distances, int32_t* union_parent,
                          int32_t* union_size) {
  stable_sort_mst_edges(edges, temporary_edges, count - 1);
  const int node_count = count * 2 - 1;
  for (int node = 0; node < node_count; ++node) {
    union_parent[node] = -1;
    union_size[node] = node < count ? 1 : 0;
  }
  for (int merge = 0; merge < count - 1; ++merge) {
    const int32_t left_root = linkage_find(union_parent, edges[merge].from);
    const int32_t right_root = linkage_find(union_parent, edges[merge].to);
    const int32_t new_node = count + merge;
    left[merge] = left_root;
    right[merge] = right_root;
    distances[merge] = edges[merge].distance;
    sizes[merge] = union_size[left_root] + union_size[right_root];
    union_parent[left_root] = new_node;
    union_parent[right_root] = new_node;
    union_size[new_node] = sizes[merge];
  }
}

int fill_hierarchy_bfs(int32_t root, int count, const int32_t* left,
                       const int32_t* right, int32_t* queue) {
  int head = 0;
  int tail = 1;
  queue[0] = root;
  while (head < tail) {
    const int32_t node = queue[head++];
    if (node < count) continue;
    const int merge = node - count;
    queue[tail++] = left[merge];
    queue[tail++] = right[merge];
  }
  return tail;
}

bool append_condensed_edge(CondensedEdge* result, int capacity, int* size,
                           int32_t parent, int32_t child, double lambda,
                           int32_t child_size) {
  if (*size >= capacity) return false;
  result[*size] = {parent, child, lambda, child_size};
  *size += 1;
  return true;
}

bool prune_subtree(int32_t root, int32_t parent_label, double lambda,
                   int count, const int32_t* left, const int32_t* right,
                   uint8_t* ignore, int32_t* queue, CondensedEdge* result,
                   int capacity, int* result_size) {
  const int subtree_size =
      fill_hierarchy_bfs(root, count, left, right, queue);
  for (int cursor = 0; cursor < subtree_size; ++cursor) {
    const int32_t node = queue[cursor];
    if (node < count &&
        !append_condensed_edge(result, capacity, result_size, parent_label,
                               node, lambda, 1)) {
      return false;
    }
    ignore[node] = 1;
  }
  return true;
}

int condense_tree(int count, int min_cluster_size, const int32_t* left,
                  const int32_t* right, const int32_t* sizes,
                  const double* distances, CondensedEdge* result,
                  int result_capacity, int32_t* main_queue,
                  int32_t* subtree_queue, uint8_t* ignore, int32_t* relabel) {
  const int32_t root = count * 2 - 2;
  const int node_count = count * 2 - 1;
  memset(ignore, 0, static_cast<size_t>(node_count) * sizeof(uint8_t));
  relabel[root] = count;
  int32_t next_label = count + 1;
  int result_size = 0;
  const int main_size =
      fill_hierarchy_bfs(root, count, left, right, main_queue);

  for (int cursor = 0; cursor < main_size; ++cursor) {
    const int32_t node = main_queue[cursor];
    if (ignore[node] || node < count) continue;
    const int merge = node - count;
    const int32_t left_node = left[merge];
    const int32_t right_node = right[merge];
    const int32_t left_count =
        left_node < count ? 1 : sizes[left_node - count];
    const int32_t right_count =
        right_node < count ? 1 : sizes[right_node - count];
    const double lambda =
        distances[merge] > 0.0
            ? 1.0 / distances[merge]
            : std::numeric_limits<double>::infinity();

    if (left_count >= min_cluster_size &&
        right_count >= min_cluster_size) {
      relabel[left_node] = next_label++;
      relabel[right_node] = next_label++;
      if (!append_condensed_edge(result, result_capacity, &result_size,
                                 relabel[node], relabel[left_node], lambda,
                                 left_count) ||
          !append_condensed_edge(result, result_capacity, &result_size,
                                 relabel[node], relabel[right_node], lambda,
                                 right_count)) {
        return -3;
      }
    } else if (left_count < min_cluster_size &&
               right_count < min_cluster_size) {
      if (!prune_subtree(left_node, relabel[node], lambda, count, left, right,
                         ignore, subtree_queue, result, result_capacity,
                         &result_size) ||
          !prune_subtree(right_node, relabel[node], lambda, count, left, right,
                         ignore, subtree_queue, result, result_capacity,
                         &result_size)) {
        return -3;
      }
    } else if (left_count < min_cluster_size) {
      relabel[right_node] = relabel[node];
      if (!prune_subtree(left_node, relabel[node], lambda, count, left, right,
                         ignore, subtree_queue, result, result_capacity,
                         &result_size)) {
        return -3;
      }
    } else {
      relabel[left_node] = relabel[node];
      if (!prune_subtree(right_node, relabel[node], lambda, count, left, right,
                         ignore, subtree_queue, result, result_capacity,
                         &result_size)) {
        return -3;
      }
    }
  }
  return result_size;
}

int32_t tree_union_find(int32_t* parent, int32_t node) {
  if (parent[node] != node) {
    parent[node] = tree_union_find(parent, parent[node]);
  }
  return parent[node];
}

void tree_union(int32_t* parent, int32_t* rank, int32_t left, int32_t right) {
  const int32_t left_root = tree_union_find(parent, left);
  const int32_t right_root = tree_union_find(parent, right);
  if (left_root == right_root) return;
  if (rank[left_root] < rank[right_root]) {
    parent[left_root] = right_root;
  } else if (rank[left_root] > rank[right_root]) {
    parent[right_root] = left_root;
  } else {
    parent[right_root] = left_root;
    rank[left_root] += 1;
  }
}

int select_and_label(const CondensedEdge* tree, int tree_size, int count,
                     int32_t* labels, double* births, double* stability,
                     int32_t* cluster_sizes, int32_t* first_child,
                     int32_t* second_child, uint8_t* is_cluster,
                     int32_t* stack, int32_t* label_map,
                     int32_t* union_parent, int32_t* union_rank) {
  if (tree_size <= 0) {
    for (int row = 0; row < count; ++row) labels[row] = -1;
    return 1;
  }
  const int capacity = count * 2 + 1;
  for (int node = 0; node < capacity; ++node) {
    births[node] = std::numeric_limits<double>::quiet_NaN();
    stability[node] = 0.0;
    cluster_sizes[node] = 0;
    first_child[node] = -1;
    second_child[node] = -1;
    is_cluster[node] = 0;
    label_map[node] = -1;
  }

  int32_t smallest_cluster = tree[0].parent;
  int32_t maximum_parent = tree[0].parent;
  for (int edge = 0; edge < tree_size; ++edge) {
    const CondensedEdge& record = tree[edge];
    if (record.parent < smallest_cluster) smallest_cluster = record.parent;
    if (record.parent > maximum_parent) maximum_parent = record.parent;
    if (record.parent < 0 || record.parent >= capacity || record.child < 0 ||
        record.child >= capacity) {
      return -3;
    }
    if (isnan(births[record.child]) ||
        record.lambda < births[record.child]) {
      births[record.child] = record.lambda;
    }
    if (record.child_size > 1) {
      cluster_sizes[record.child] = record.child_size;
      if (first_child[record.parent] < 0) {
        first_child[record.parent] = record.child;
      } else if (second_child[record.parent] < 0) {
        second_child[record.parent] = record.child;
      } else {
        return -3;
      }
    }
  }
  if (smallest_cluster != count) return -3;
  births[smallest_cluster] = 0.0;
  for (int edge = 0; edge < tree_size; ++edge) {
    const CondensedEdge& record = tree[edge];
    stability[record.parent] +=
        (record.lambda - births[record.parent]) * record.child_size;
  }

  // Upstream relies on monotonically assigned condensed cluster IDs as a
  // reverse topological order. The root is deliberately excluded because
  // allow_single_cluster is false in Senko.
  for (int32_t node = smallest_cluster + 1; node <= maximum_parent; ++node) {
    is_cluster[node] = 1;
  }
  for (int32_t node = maximum_parent; node > smallest_cluster; --node) {
    double subtree_stability = 0.0;
    if (first_child[node] >= 0) {
      subtree_stability += stability[first_child[node]];
    }
    if (second_child[node] >= 0) {
      subtree_stability += stability[second_child[node]];
    }
    if (subtree_stability > stability[node]) {
      is_cluster[node] = 0;
      stability[node] = subtree_stability;
      continue;
    }

    int stack_size = 0;
    if (first_child[node] >= 0) stack[stack_size++] = first_child[node];
    if (second_child[node] >= 0) stack[stack_size++] = second_child[node];
    while (stack_size > 0) {
      const int32_t descendant = stack[--stack_size];
      is_cluster[descendant] = 0;
      if (first_child[descendant] >= 0) {
        stack[stack_size++] = first_child[descendant];
      }
      if (second_child[descendant] >= 0) {
        stack[stack_size++] = second_child[descendant];
      }
      if (stack_size >= capacity) return -3;
    }
  }

  int32_t next_label = 0;
  for (int32_t node = smallest_cluster + 1; node <= maximum_parent; ++node) {
    if (is_cluster[node]) label_map[node] = next_label++;
  }

  for (int node = 0; node < capacity; ++node) {
    union_parent[node] = node;
    union_rank[node] = 0;
  }
  for (int edge = 0; edge < tree_size; ++edge) {
    const int32_t child = tree[edge].child;
    if (label_map[child] < 0) {
      tree_union(union_parent, union_rank, tree[edge].parent, child);
    }
  }
  for (int point = 0; point < count; ++point) {
    const int32_t cluster = tree_union_find(union_parent, point);
    labels[point] =
        cluster <= smallest_cluster || label_map[cluster] < 0
            ? -1
            : label_map[cluster];
  }
  return 1;
}

}  // namespace

uint32_t workspace_bytes(int count, int dimension, int min_samples,
                         int min_cluster_size) {
  const uint64_t bytes = calculate_workspace_bytes(
      count, dimension, min_samples, min_cluster_size);
  return bytes == 0 || bytes > UINT32_MAX ? 0 : static_cast<uint32_t>(bytes);
}

int run_f64_semantics_diagnostic(
    const float* projection, int count, int dimension, int min_samples,
    int min_cluster_size, int32_t* labels, double* diagnostic_core_distances,
    double* diagnostic_mst_rows, void* workspace_memory,
    uint32_t workspace_size) {
  if (!projection || !labels || !workspace_memory ||
      !valid_arguments(count, dimension, min_samples, min_cluster_size)) {
    return -1;
  }
  min_samples = min_samples < count ? min_samples : count - 1;
  const uint32_t required =
      workspace_bytes(count, dimension, min_samples, min_cluster_size);
  if (required == 0 || workspace_size < required) return -2;

  const uint64_t n = static_cast<uint64_t>(count);
  const uint64_t node_count = n * 2u - 1u;
  const uint64_t cluster_capacity = n * 2u + 1u;
  Workspace workspace(workspace_memory, workspace_size);

  double* points = workspace.allocate<double>(n * dimension);
  double* core_heaps = workspace.allocate<double>(n * min_samples);
  int32_t* core_heap_sizes = workspace.allocate<int32_t>(n);
  double* core_distances = workspace.allocate<double>(n);
  MstEdge* mst = workspace.allocate<MstEdge>(n - 1u);
  MstEdge* temporary_mst = workspace.allocate<MstEdge>(n - 1u);
  double* current_distances = workspace.allocate<double>(n);
  int32_t* current_sources = workspace.allocate<int32_t>(n);
  uint8_t* in_tree = workspace.allocate<uint8_t>(n);
  const uint64_t kd_node_count = kd_node_capacity(n);
  int32_t* kd_indices = workspace.allocate<int32_t>(n);
  KdNode* kd_nodes = workspace.allocate<KdNode>(kd_node_count);
  double* kd_bounds =
      workspace.allocate<double>(kd_node_count * dimension * 2u);
  double* kd_minimum_core_squared =
      workspace.allocate<double>(kd_node_count);
  int32_t* kd_homogeneous_component =
      workspace.allocate<int32_t>(kd_node_count);
  double* kd_query_heap = workspace.allocate<double>(min_samples);
  int32_t* boruvka_parent = workspace.allocate<int32_t>(n);
  int32_t* boruvka_rank = workspace.allocate<int32_t>(n);
  int32_t* point_components = workspace.allocate<int32_t>(n);
  int32_t* candidate_source = workspace.allocate<int32_t>(n);
  int32_t* candidate_sink = workspace.allocate<int32_t>(n);
  double* candidate_distance_squared = workspace.allocate<double>(n);
  int32_t* linkage_left = workspace.allocate<int32_t>(n - 1u);
  int32_t* linkage_right = workspace.allocate<int32_t>(n - 1u);
  int32_t* linkage_sizes = workspace.allocate<int32_t>(n - 1u);
  double* linkage_distances = workspace.allocate<double>(n - 1u);
  int32_t* linkage_union_parent = workspace.allocate<int32_t>(node_count);
  int32_t* linkage_union_size = workspace.allocate<int32_t>(node_count);
  CondensedEdge* condensed = workspace.allocate<CondensedEdge>(n * 2u);
  int32_t* main_queue = workspace.allocate<int32_t>(node_count);
  int32_t* subtree_queue = workspace.allocate<int32_t>(node_count);
  uint8_t* ignore = workspace.allocate<uint8_t>(node_count);
  int32_t* relabel = workspace.allocate<int32_t>(node_count);
  double* births = workspace.allocate<double>(cluster_capacity);
  double* stability = workspace.allocate<double>(cluster_capacity);
  int32_t* cluster_sizes = workspace.allocate<int32_t>(cluster_capacity);
  int32_t* first_child = workspace.allocate<int32_t>(cluster_capacity);
  int32_t* second_child = workspace.allocate<int32_t>(cluster_capacity);
  uint8_t* is_cluster = workspace.allocate<uint8_t>(cluster_capacity);
  int32_t* stack = workspace.allocate<int32_t>(cluster_capacity);
  int32_t* label_map = workspace.allocate<int32_t>(cluster_capacity);
  int32_t* label_union_parent =
      workspace.allocate<int32_t>(cluster_capacity);
  int32_t* label_union_rank = workspace.allocate<int32_t>(cluster_capacity);
  if (!points || !core_heaps || !core_heap_sizes || !core_distances || !mst ||
      !temporary_mst || !current_distances || !current_sources || !in_tree ||
      !kd_indices || !kd_nodes || !kd_bounds || !kd_minimum_core_squared ||
      !kd_homogeneous_component || !kd_query_heap || !boruvka_parent ||
      !boruvka_rank || !point_components || !candidate_source ||
      !candidate_sink || !candidate_distance_squared || !linkage_left ||
      !linkage_right || !linkage_sizes || !linkage_distances ||
      !linkage_union_parent || !linkage_union_size || !condensed ||
      !main_queue || !subtree_queue || !ignore || !relabel || !births ||
      !stability || !cluster_sizes || !first_child || !second_child ||
      !is_cluster || !stack || !label_map || !label_union_parent ||
      !label_union_rank) {
    return -2;
  }

  const uint64_t value_count = n * static_cast<uint64_t>(dimension);
  for (uint64_t value = 0; value < value_count; ++value) {
    points[value] = static_cast<double>(projection[value]);
  }
  int mst_status = 1;
  if (count >= kKdBoruvkaThreshold) {
    if (kd_node_count > INT32_MAX) return -2;
    KdTreeView kd_tree{};
    if (!build_kd_tree(points, count, dimension, kd_indices, kd_nodes,
                       kd_bounds, static_cast<int>(kd_node_count), &kd_tree) ||
        !calculate_core_distances_kd(kd_tree, min_samples, kd_query_heap,
                                     core_distances)) {
      return -3;
    }
    mst_status = build_kd_boruvka_mst(
        kd_tree, core_distances, mst, kd_minimum_core_squared,
        kd_homogeneous_component, boruvka_parent, boruvka_rank,
        point_components, candidate_source, candidate_sink,
        candidate_distance_squared);
  } else {
    calculate_core_distances(points, count, dimension, min_samples, core_heaps,
                             core_heap_sizes, core_distances);
    mst_status =
        build_exact_mst(points, count, dimension, core_distances, mst,
                        current_distances, current_sources, in_tree);
  }
  if (mst_status != 1) return mst_status;
  if (diagnostic_core_distances) {
    memcpy(diagnostic_core_distances, core_distances,
           static_cast<size_t>(count) * sizeof(double));
  }
  if (diagnostic_mst_rows) {
    for (int edge = 0; edge < count - 1; ++edge) {
      diagnostic_mst_rows[edge * 3] = static_cast<double>(mst[edge].from);
      diagnostic_mst_rows[edge * 3 + 1] = static_cast<double>(mst[edge].to);
      diagnostic_mst_rows[edge * 3 + 2] = mst[edge].distance;
    }
  }
  build_single_linkage(mst, temporary_mst, count, linkage_left, linkage_right,
                       linkage_sizes, linkage_distances, linkage_union_parent,
                       linkage_union_size);
  const int condensed_size =
      condense_tree(count, min_cluster_size, linkage_left, linkage_right,
                    linkage_sizes, linkage_distances, condensed,
                    static_cast<int>(n * 2u), main_queue, subtree_queue, ignore,
                    relabel);
  if (condensed_size < 0) return condensed_size;
  return select_and_label(
      condensed, condensed_size, count, labels, births, stability,
      cluster_sizes, first_child, second_child, is_cluster, stack, label_map,
      label_union_parent, label_union_rank);
}

int run_f64_semantics(const float* projection, int count, int dimension,
                      int min_samples, int min_cluster_size, int32_t* labels,
                      void* workspace_memory, uint32_t workspace_size) {
  return run_f64_semantics_diagnostic(
      projection, count, dimension, min_samples, min_cluster_size, labels,
      nullptr, nullptr, workspace_memory, workspace_size);
}

}  // namespace senko_hdbscan
