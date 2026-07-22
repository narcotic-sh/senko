#include <math.h>
#include <stdint.h>
#include <string.h>

#include <emscripten/emscripten.h>

namespace {

constexpr uint32_t kArenaBytes = 8u * 1024u * 1024u;
alignas(16) uint8_t arena[kArenaBytes];
uint32_t arena_cursor = 0;

uint32_t align_up(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

void* allocate_bytes(uint32_t bytes, uint32_t alignment = 16u) {
  const uint32_t start = align_up(arena_cursor, alignment);
  if (start > kArenaBytes || bytes > kArenaBytes - start) return nullptr;
  arena_cursor = start + bytes;
  return arena + start;
}

template <typename T>
T* allocate_items(uint32_t count) {
  if (count > UINT32_MAX / sizeof(T)) return nullptr;
  return static_cast<T*>(allocate_bytes(count * sizeof(T), alignof(T) < 16 ? 16 : alignof(T)));
}

double squared_euclidean_unrolled(const float* left, const float* right,
                                  int dim) {
  double result = 0.0;
  int column = 0;
  const int unrolled_end = dim - dim % 4;
  for (; column < unrolled_end; column += 4) {
    const double difference0 =
        static_cast<double>(left[column]) - right[column];
    const double difference1 =
        static_cast<double>(left[column + 1]) - right[column + 1];
    const double difference2 =
        static_cast<double>(left[column + 2]) - right[column + 2];
    const double difference3 =
        static_cast<double>(left[column + 3]) - right[column + 3];
    result += difference0 * difference0 + difference1 * difference1 +
              difference2 * difference2 + difference3 * difference3;
  }
  for (; column < dim; ++column) {
    const double difference =
        static_cast<double>(left[column]) - right[column];
    result += difference * difference;
  }
  return result;
}

double euclidean(const float* left, const float* right, int dim) {
  return sqrt(squared_euclidean_unrolled(left, right, dim));
}

double dot_rows(const float* left, const float* right, int dim) {
  double result = 0.0;
  int column = 0;
  const int unrolled_end = dim - dim % 4;
  for (; column < unrolled_end; column += 4) {
    result +=
        static_cast<double>(left[column]) * right[column] +
        static_cast<double>(left[column + 1]) * right[column + 1] +
        static_cast<double>(left[column + 2]) * right[column + 2] +
        static_cast<double>(left[column + 3]) * right[column + 3];
  }
  for (; column < dim; ++column) {
    result += static_cast<double>(left[column]) * right[column];
  }
  return result < -1.0 ? -1.0 : (result > 1.0 ? 1.0 : result);
}

void insert_neighbor(int32_t* indices, float* similarities, int offset,
                     int count, int candidate, double similarity) {
  int position = count - 1;
  const float last_similarity = similarities[offset + position];
  const int last_index = indices[offset + position];
  if (similarity < static_cast<double>(last_similarity) ||
      (similarity == static_cast<double>(last_similarity) && last_index >= 0 &&
       candidate > last_index)) {
    return;
  }
  while (position > 0) {
    const float previous_similarity = similarities[offset + position - 1];
    const int previous_index = indices[offset + position - 1];
    if (similarity < static_cast<double>(previous_similarity) ||
        (similarity == static_cast<double>(previous_similarity) &&
         candidate > previous_index)) {
      break;
    }
    similarities[offset + position] = previous_similarity;
    indices[offset + position] = previous_index;
    --position;
  }
  similarities[offset + position] = static_cast<float>(similarity);
  indices[offset + position] = candidate;
}

uint32_t xorshift32(uint32_t value) {
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

uint32_t unsigned_hash(int left, int right) {
  uint32_t value = static_cast<uint32_t>(left + 1) * 0x85ebca6bu ^
                   static_cast<uint32_t>(right + 1) * 0xc2b2ae35u;
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  return value;
}

struct DeterministicRandom {
  uint32_t state;

  double next() {
    state += 0x6d2b79f5u;
    uint32_t value = (state ^ (state >> 15)) * (1u | state);
    value = (value + (value ^ (value >> 7)) * (61u | value)) ^ value;
    return static_cast<double>(value ^ (value >> 14)) / 4294967296.0;
  }
};

int append_candidate(int candidate, int stamp, int32_t* seen,
                     int32_t* candidates, int candidate_count,
                     int candidate_capacity) {
  if (seen[candidate] == stamp || candidate_count == candidate_capacity) {
    return candidate_count;
  }
  seen[candidate] = stamp;
  candidates[candidate_count] = candidate;
  return candidate_count + 1;
}

int append_bucket(const int32_t* bucket, int bucket_length, int row, int salt,
                  int limit, int stamp, int32_t* seen, int32_t* candidates,
                  int candidate_count, int candidate_capacity) {
  if (bucket_length <= limit) {
    for (int i = 0; i < bucket_length && candidate_count < candidate_capacity;
         ++i) {
      candidate_count = append_candidate(bucket[i], stamp, seen, candidates,
                                           candidate_count, candidate_capacity);
    }
    return candidate_count;
  }

  const int phase = static_cast<int>(unsigned_hash(row, salt) % bucket_length);
  for (int sample = 0;
       sample < limit && candidate_count < candidate_capacity; ++sample) {
    const int index =
        (phase + static_cast<int>((static_cast<int64_t>(sample) * bucket_length) /
                                  limit)) %
        bucket_length;
    candidate_count = append_candidate(bucket[index], stamp, seen, candidates,
                                         candidate_count, candidate_capacity);
  }
  return candidate_count;
}

struct NeighborHeap {
  int32_t* indices;
  float* distances;
  uint8_t* is_new;
  int size;
};

int heap_push(NeighborHeap heap, int row, double distance, int index,
              uint8_t flag) {
  const int offset = row * heap.size;
  if (distance >= static_cast<double>(heap.distances[offset])) return 0;
  for (int position = 0; position < heap.size; ++position) {
    if (heap.indices[offset + position] == index) return 0;
  }

  int position = 0;
  while (true) {
    const int left = position * 2 + 1;
    if (left >= heap.size) break;
    const int right = left + 1;
    int swap = left;
    if (right < heap.size &&
        heap.distances[offset + right] > heap.distances[offset + left]) {
      swap = right;
    }
    if (distance >= static_cast<double>(heap.distances[offset + swap])) break;
    heap.distances[offset + position] = heap.distances[offset + swap];
    heap.indices[offset + position] = heap.indices[offset + swap];
    heap.is_new[offset + position] = heap.is_new[offset + swap];
    position = swap;
  }
  heap.distances[offset + position] = static_cast<float>(distance);
  heap.indices[offset + position] = index;
  heap.is_new[offset + position] = flag;
  return 1;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE void cluster_reset() { arena_cursor = 0; }

EMSCRIPTEN_KEEPALIVE uintptr_t cluster_heap_base() {
  return reinterpret_cast<uintptr_t>(arena);
}

EMSCRIPTEN_KEEPALIVE uint32_t cluster_heap_capacity() { return kArenaBytes; }

EMSCRIPTEN_KEEPALIVE uint32_t cluster_heap_used() { return arena_cursor; }

EMSCRIPTEN_KEEPALIVE uintptr_t cluster_alloc(uint32_t bytes,
                                             uint32_t alignment) {
  if (alignment == 0 || alignment > 64 || (alignment & (alignment - 1u)) != 0) {
    return 0;
  }
  return reinterpret_cast<uintptr_t>(allocate_bytes(bytes, alignment));
}

EMSCRIPTEN_KEEPALIVE int cluster_normalize_rows(const float* input,
                                                float* output, int count,
                                                int dim) {
  if (!input || !output || count < 0 || dim <= 0) return -1;
  for (int row = 0; row < count; ++row) {
    const float* source = input + row * dim;
    float* target = output + row * dim;
    double squared_norm = 0.0;
    for (int column = 0; column < dim; ++column) {
      const double value = source[column];
      squared_norm += value * value;
    }
    const double scale = squared_norm > 0.0 ? 1.0 / sqrt(squared_norm) : 0.0;
    for (int column = 0; column < dim; ++column) {
      target[column] = static_cast<float>(source[column] * scale);
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE int cluster_approximate_cosine_knn(
    const float* values, int count, int dim, int neighbor_count,
    int table_count, int bits, int bucket_sample_limit,
    int temporal_neighbor_radius, int32_t* output_indices,
    float* output_similarities) {
  if (!values || !output_indices || !output_similarities || count < 1 ||
      dim < 1 || neighbor_count < 1 || neighbor_count >= count ||
      table_count < 1 || bits < 1 || bits > 16 || bucket_sample_limit < 1 ||
      temporal_neighbor_radius < 0) {
    return -1;
  }
  const int bucket_count = 1 << bits;
  const int plane_count = table_count * bits;
  int8_t* planes = allocate_items<int8_t>(plane_count * dim);
  uint16_t* signatures = allocate_items<uint16_t>(count * table_count);
  int32_t* bucket_sizes = allocate_items<int32_t>(table_count * bucket_count);
  int32_t* bucket_offsets =
      allocate_items<int32_t>(table_count * (bucket_count + 1));
  int32_t* bucket_rows = allocate_items<int32_t>(table_count * count);
  int32_t* seen = allocate_items<int32_t>(count);
  const int candidate_capacity =
      table_count * bucket_sample_limit * 2 + temporal_neighbor_radius * 2 < count
          ? table_count * bucket_sample_limit * 2 + temporal_neighbor_radius * 2
          : count;
  int32_t* candidates = allocate_items<int32_t>(candidate_capacity);
  if (!planes || !signatures || !bucket_sizes || !bucket_offsets ||
      !bucket_rows || !seen || !candidates) {
    return -2;
  }

  uint32_t random_state = 0x9e3779b9u;
  for (int i = 0; i < plane_count * dim; ++i) {
    random_state = xorshift32(random_state);
    planes[i] = (random_state & 1u) == 0 ? -1 : 1;
  }
  for (int row = 0; row < count; ++row) {
    const float* source = values + row * dim;
    for (int table = 0; table < table_count; ++table) {
      int signature = 0;
      for (int bit = 0; bit < bits; ++bit) {
        const int8_t* plane = planes + (table * bits + bit) * dim;
        double projection = 0.0;
        for (int column = 0; column < dim; ++column) {
          projection += static_cast<double>(source[column]) * plane[column];
        }
        if (projection >= 0.0) signature |= 1 << bit;
      }
      signatures[row * table_count + table] =
          static_cast<uint16_t>(signature);
    }
  }

  memset(bucket_sizes, 0,
         table_count * bucket_count * sizeof(int32_t));
  for (int table = 0; table < table_count; ++table) {
    int32_t* sizes = bucket_sizes + table * bucket_count;
    int32_t* offsets = bucket_offsets + table * (bucket_count + 1);
    for (int row = 0; row < count; ++row) {
      ++sizes[signatures[row * table_count + table]];
    }
    offsets[0] = 0;
    for (int key = 0; key < bucket_count; ++key) {
      offsets[key + 1] = offsets[key] + sizes[key];
      sizes[key] = 0;
    }
    int32_t* rows = bucket_rows + table * count;
    for (int row = 0; row < count; ++row) {
      const int key = signatures[row * table_count + table];
      rows[offsets[key] + sizes[key]++] = row;
    }
  }

  const int output_length = count * neighbor_count;
  for (int i = 0; i < output_length; ++i) {
    output_indices[i] = -1;
    output_similarities[i] = -INFINITY;
  }
  memset(seen, 0, count * sizeof(int32_t));
  for (int row = 0; row < count; ++row) {
    const int stamp = row + 1;
    seen[row] = stamp;
    int candidate_count = 0;
    const int temporal_start =
        row - temporal_neighbor_radius > 0 ? row - temporal_neighbor_radius : 0;
    const int temporal_end =
        row + temporal_neighbor_radius + 1 < count
            ? row + temporal_neighbor_radius + 1
            : count;
    for (int candidate = temporal_start; candidate < temporal_end; ++candidate) {
      candidate_count = append_candidate(candidate, stamp, seen, candidates,
                                           candidate_count, candidate_capacity);
    }
    for (int table = 0; table < table_count; ++table) {
      const int signature = signatures[row * table_count + table];
      const int32_t* offsets = bucket_offsets + table * (bucket_count + 1);
      const int32_t* rows = bucket_rows + table * count;
      candidate_count = append_bucket(
          rows + offsets[signature], offsets[signature + 1] - offsets[signature],
          row, table, bucket_sample_limit, stamp, seen, candidates,
          candidate_count, candidate_capacity);
    }
    const int desired_candidates =
        neighbor_count * 3 > 64 ? neighbor_count * 3 : 64;
    const int desired = desired_candidates < count - 1 ? desired_candidates
                                                       : count - 1;
    if (candidate_count < desired) {
      bool complete = false;
      for (int table = 0; table < table_count && !complete; ++table) {
        const int signature = signatures[row * table_count + table];
        const int32_t* offsets = bucket_offsets + table * (bucket_count + 1);
        const int32_t* rows = bucket_rows + table * count;
        for (int bit = 0; bit < bits; ++bit) {
          const int key = signature ^ (1 << bit);
          candidate_count = append_bucket(
              rows + offsets[key], offsets[key + 1] - offsets[key], row,
              table + bit + 1, bucket_sample_limit, stamp, seen, candidates,
              candidate_count, candidate_capacity);
          if (candidate_count >= desired || candidate_count == candidate_capacity) {
            complete = true;
            break;
          }
        }
      }
    }
    for (int cursor = 0; cursor < candidate_count; ++cursor) {
      const int candidate = candidates[cursor];
      insert_neighbor(output_indices, output_similarities,
                      row * neighbor_count, neighbor_count, candidate,
                      dot_rows(values + row * dim, values + candidate * dim, dim));
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE int cluster_refine_euclidean_knn(
    const float* embeddings, int count, int dim, int neighbor_count,
    const int32_t* seed_indices, int seed_neighbor_count, uint32_t random_seed,
    int32_t* output_indices, float* output_distances,
    uint8_t* output_is_new) {
  if (!embeddings || !seed_indices || !output_indices || !output_distances ||
      !output_is_new || count < 1 || dim < 1 || neighbor_count < 2 ||
      seed_neighbor_count < 1) {
    return -1;
  }
  const int length = count * neighbor_count;
  for (int i = 0; i < length; ++i) {
    output_indices[i] = -1;
    output_distances[i] = INFINITY;
    output_is_new[i] = 0;
  }
  NeighborHeap heap{output_indices, output_distances, output_is_new,
                    neighbor_count};
  DeterministicRandom random{random_seed};
  for (int row = 0; row < count; ++row) {
    heap_push(heap, row, 0.0, row, 1);
    const int seed_offset = row * seed_neighbor_count;
    for (int rank = 0; rank < seed_neighbor_count; ++rank) {
      const int candidate = seed_indices[seed_offset + rank];
      if (candidate < 0) continue;
      const double distance = euclidean(embeddings + row * dim,
                                        embeddings + candidate * dim, dim);
      heap_push(heap, row, distance, candidate, 1);
      heap_push(heap, candidate, distance, row, 1);
    }
    for (int sample = 0; sample < 4; ++sample) {
      const int candidate = static_cast<int>(floor(random.next() * count));
      const double distance = euclidean(embeddings + row * dim,
                                        embeddings + candidate * dim, dim);
      heap_push(heap, row, distance, candidate, 1);
      heap_push(heap, candidate, distance, row, 1);
    }
  }

  int32_t* snapshot_indices = allocate_items<int32_t>(length);
  uint8_t* snapshot_flags = allocate_items<uint8_t>(length);
  if (!snapshot_indices || !snapshot_flags) return -2;
  const int convergence_limit =
      static_cast<int>(floor(0.001 * neighbor_count * count)) > 1
          ? static_cast<int>(floor(0.001 * neighbor_count * count))
          : 1;
  for (int iteration = 0; iteration < 6; ++iteration) {
    memcpy(snapshot_indices, output_indices, length * sizeof(int32_t));
    memcpy(snapshot_flags, output_is_new, length * sizeof(uint8_t));
    memset(output_is_new, 0, length * sizeof(uint8_t));
    int changes = 0;
    for (int row = 0; row < count; ++row) {
      const int row_offset = row * neighbor_count;
      for (int rank = 0; rank < neighbor_count; ++rank) {
        const int pivot_offset = row_offset + rank;
        const int pivot = snapshot_indices[pivot_offset];
        if (pivot < 0 || pivot == row) continue;
        const bool pivot_is_new = snapshot_flags[pivot_offset] != 0;
        const int neighbor_offset = pivot * neighbor_count;
        for (int candidate_rank = 0; candidate_rank < neighbor_count;
             ++candidate_rank) {
          const int candidate_offset = neighbor_offset + candidate_rank;
          const int candidate = snapshot_indices[candidate_offset];
          if (candidate < 0 || candidate == row ||
              (!pivot_is_new && snapshot_flags[candidate_offset] == 0)) {
            continue;
          }
          const double distance = euclidean(embeddings + row * dim,
                                            embeddings + candidate * dim, dim);
          changes += heap_push(heap, row, distance, candidate, 1);
          changes += heap_push(heap, candidate, distance, row, 1);
        }
      }
    }
    if (changes <= convergence_limit) break;
  }

  for (int row = 0; row < count; ++row) {
    const int offset = row * neighbor_count;
    for (int current = 1; current < neighbor_count; ++current) {
      const float distance = output_distances[offset + current];
      const int index = output_indices[offset + current];
      const uint8_t flag = output_is_new[offset + current];
      int position = current;
      while (position > 0 &&
             (distance < output_distances[offset + position - 1] ||
              (distance == output_distances[offset + position - 1] &&
               index < output_indices[offset + position - 1]))) {
        output_distances[offset + position] =
            output_distances[offset + position - 1];
        output_indices[offset + position] = output_indices[offset + position - 1];
        output_is_new[offset + position] = output_is_new[offset + position - 1];
        --position;
      }
      output_distances[offset + position] = distance;
      output_indices[offset + position] = index;
      output_is_new[offset + position] = flag;
    }
  }
  return 1;
}

EMSCRIPTEN_KEEPALIVE int cluster_exact_euclidean_knn(
    const float* values, int count, int dim, int neighbor_count,
    int32_t* output_indices, float* output_similarities) {
  if (!values || !output_indices || !output_similarities || count < 1 ||
      dim < 1 || neighbor_count < 1 || neighbor_count >= count) {
    return -1;
  }
  const int length = count * neighbor_count;
  for (int i = 0; i < length; ++i) {
    output_indices[i] = -1;
    output_similarities[i] = -INFINITY;
  }
  for (int left = 0; left < count; ++left) {
    for (int right = left + 1; right < count; ++right) {
      const double distance =
          euclidean(values + left * dim, values + right * dim, dim);
      const double similarity = 1.0 - distance;
      insert_neighbor(output_indices, output_similarities,
                      left * neighbor_count, neighbor_count, right, similarity);
      insert_neighbor(output_indices, output_similarities,
                      right * neighbor_count, neighbor_count, left, similarity);
    }
  }
  return 1;
}

}  // extern "C"
