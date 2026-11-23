#ifndef LMOP3G_H
#define LMOP3G_H

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "address.h"
#include "champsim.h"
#include "modules.h"

// LMOP3g, the design in the UGRC-I report: the LMOP3 code below built with FIXED_POLICY = 1 (LMOP's fixed fill policy)
// and WINDOW = 0 (whole-map scoring), i.e. page gating + default next line. About 2 KB of state.
//
// LMOP3: LMOP2 (page access maps, global offset scores, set-dueled fill policy; see lmop2.h) with changes aimed at the
// irregular traces where LMOP prefetches far too much (Components, PageRank, mcf-184B):
//   0. Page gating (GATE_NEW). Offsets are learned only from pages already in the table (their second and later
//      accesses), mostly streams, but LMOP applies them on every access, including the first touch of a page that is
//      touched once (random vertex data in Ligra): on Components it read 2.5x the DRAM lines of TLS2 for the same useful
//      prefetches. With GATE_NEW a page's first access issues no offset prefetch (only the default next line).
//   1. Windowed scoring. A trigger at line x votes for offset x - i only for the lines i among the last WINDOW
//      accesses to its page, not for every line the page ever accessed. On a page that is accessed densely in random
//      order, the full map votes for nearly every offset (an offset "covers" x because x - d was touched long ago); the
//      window keeps only recent, i.e. causally close, accesses. WINDOW = 0 restores LMOP2's whole-map scoring.
//   2. Default next line (NL_DEFAULT). An access that issued no offset prefetch requests line x + 1, with its fill level
//      (L2 or LLC only) chosen by set dueling as in TLS2: pages hashing to group 0 always fill the L2, group 1 always
//      the LLC, the rest follow a 10-bit counter moved by demand misses in the two leader groups.
// Storage: LMOP2's 2.1 KB + per page a WINDOW x 6-bit FIFO and 4-bit head (64 x 100 bits = 0.8 KB) + 10-bit counter.
// Work per trigger: at most WINDOW score increments (LMOP2: up to 63).
// FIXED_POLICY >= 0: no policy dueling, every set uses that policy (1 is LMOP's: s >= 0.5 -> L2, s >= 0.3 -> LLC).
struct lmop3g : public champsim::modules::prefetcher {
  constexpr static int ENTRIES = 64;
  constexpr static int ROUND = 256;
  constexpr static int LINES = 64;
  constexpr static int POLICIES = 4;
  constexpr static int LEADER_STRIDE = 32;
  constexpr static int EPOCH_MISSES = 1024;
  constexpr static double MIN_SCORE = 0.15;
  constexpr static int MAX_OFFSETS = 32;
  constexpr static int FIXED_POLICY = 1;
  constexpr static int WINDOW = 0;
  constexpr static bool NL_DEFAULT = true;
  constexpr static bool GATE_NEW = true;
  constexpr static int PSEL_MAX = 1023;

  struct entry {
    uint64_t page = 0, acc = 0, req_l2 = 0, req_any = 0, lru = 0;
    std::array<int8_t, (WINDOW > 0 ? WINDOW : 1)> recent{};
    int n_recent = 0, head = 0;
    bool valid = false;
  };
  struct pick {
    int offset;
    double s;
    int rank;
  };

  std::array<entry, ENTRIES> table{};
  std::array<int, 2 * LINES - 1> score{};
  std::vector<pick> picks;
  uint64_t tick = 0;
  int updates = 0;
  int winner = 1;
  int psel = PSEL_MAX / 2;
  std::array<uint64_t, POLICIES> lat_sum{}, wins{};
  std::unordered_map<uint64_t, long long> pending;
  int done = 0;
  uint64_t issued_l2 = 0, issued_llc = 0, pq_full = 0, rounds = 0, nl_l2 = 0, nl_llc = 0;
  uint64_t picked_sum = 0;

  using champsim::modules::prefetcher::prefetcher;

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_final_stats();
  void end_round();
  long long now() const;
  int policy_of(uint64_t line) const;
  int decide(int policy, const pick& p) const; // 0 none, 1 L2, 2 LLC only
};

#endif
