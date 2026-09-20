#ifndef PREFETCHER_PATHFINDER_RECORDER_H
#define PREFETCHER_PATHFINDER_RECORDER_H

#include <cstdint>
#include <fstream>

#include "address.h"
#include "modules.h"

// Records successful dynamic LLC loads using the original trace.llc_pref format.
struct pathfinder_recorder : public champsim::modules::prefetcher {
  std::ofstream outfile;
  std::size_t row_id = 0;

  using prefetcher::prefetcher;

  void prefetcher_initialize();
  void prefetcher_cache_operate_with_id(champsim::address addr, champsim::address ip, bool cache_hit, access_type type, uint64_t instr_id,
                                        uint64_t cycle);
  void prefetcher_final_stats();
};

#endif
