#ifndef MLOP_H
#define MLOP_H

#include <cstdint>
#include <memory>

#include "address.h"
#include "champsim.h"
#include "modules.h"

namespace mlop_dpc3
{
class MLOP;
struct CacheIface;
} // namespace mlop_dpc3

// L2 wrapper around the authors' DPC-3 MLOP (mlop_dpc3.h)
struct mlop : public champsim::modules::prefetcher {
  /*=== MLOP Settings ===*/
  // DPC-3 formulas; degree, updates and thresholds are the Pythia repository's L2 mlop.ini values
  // (DPC-3 L1D values: L1D 0.40, L2C 0.30, LLC 2.00 = off)

  /* maximum possible prefetch degree (the actual prefetch degree is usually much smaller) */
  constexpr static int PREFETCH_DEGREE = 16;

  /* number of score updates before selecting prefetch offsets (thus, it is also the maximum possible score) */
  constexpr static int NUM_UPDATES = 500;

  /* prefetch offsets with `score >= LX_THRESH * NUM_UPDATES` into LX */
  constexpr static double L1D_THRESH = 2.00; /* off: an L2 prefetcher cannot fill the L1 */
  constexpr static double L2C_THRESH = 0.75;
  constexpr static double LLC_THRESH = 0.30;
  /*======================*/

  std::shared_ptr<mlop_dpc3::MLOP> engine;
  std::shared_ptr<mlop_dpc3::CacheIface> iface;

  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_final_stats();
};

#endif
