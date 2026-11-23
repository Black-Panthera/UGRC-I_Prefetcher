#ifndef BINGO_H
#define BINGO_H

#include <cstdint>
#include <memory>

#include "address.h"
#include "champsim.h"
#include "modules.h"

namespace bingo_dpc3
{
class Bingo;
struct CacheIface;
} // namespace bingo_dpc3

// L2 wrapper around the authors' DPC-3 Bingo (bingo_dpc3.h)
struct bingo : public champsim::modules::prefetcher {
  // Bingo settings: DPC-3 values, except PHT_SIZE (DPC-3: 8K), which follows the Pythia repository's L2 bingo.ini
  constexpr static int REGION_SIZE = 2 * 1024; /* size of spatial region = 2KB */
  constexpr static int PC_WIDTH = 16;          /* number of PC bits used in PHT */
  constexpr static int MIN_ADDR_WIDTH = 5;     /* number of Address bits used for PC+Offset matching */
  constexpr static int MAX_ADDR_WIDTH = 16;    /* number of Address bits used for PC+Address matching */
  constexpr static int FT_SIZE = 64;           /* size of filter table */
  constexpr static int AT_SIZE = 128;          /* size of accumulation table */
  constexpr static int PHT_SIZE = 4 * 1024;    /* size of pattern history table (PHT) */
  constexpr static int PHT_WAYS = 16;          /* associativity of PHT */
  constexpr static int PF_STREAMER_SIZE = 128; /* size of prefetch streamer */

  std::shared_ptr<bingo_dpc3::Bingo> engine;
  std::shared_ptr<bingo_dpc3::CacheIface> iface;

  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_final_stats();
};

#endif
