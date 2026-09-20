#ifndef PREFETCHER_PATHFINDER_FILE_H
#define PREFETCHER_PATHFINDER_FILE_H

#include <cstdint>
#include <map>
#include <vector>

#include "address.h"
#include "modules.h"

// Consumes a prediction file generated OFFLINE by the original PATHFINDER
// Python/BindsNET implementation. This module performs NO learning or
// prediction of its own: it only replays pre-computed prefetch addresses
// associated with dynamic instruction IDs at LLC.
struct pathfinder_file : public champsim::modules::prefetcher {
  using key_type = uint64_t; // (instruction ID)
  using occurrence = std::vector<uint64_t>;       // 0, 1, or 2 predicted addresses for one occurrence

  std::map<key_type, occurrence> predictions;

  using prefetcher::prefetcher;
  void prefetcher_initialize();

  void prefetcher_cache_operate_with_id(champsim::address addr, champsim::address ip, bool cache_hit, access_type type, uint64_t instr_id,
                                        uint64_t cycle);
};

#endif
