#ifndef BOP_H
#define BOP_H

// Best-Offset prefetcher, Pierre Michaud's DPC-2 submission (http://comparch-conf.gatech.edu/dpc2/resource/dpc2_michaud.c),
// ported to this ChampSim. The algorithm, parameters and comments are the original's; only the simulator interface changed.

#include <cstdint>
#include <deque>
#include <unordered_set>

#include "address.h"
#include "champsim.h"
#include "modules.h"

struct bop : public champsim::modules::prefetcher {
  //######################################################################################
  //                             PREFETCHER PARAMETERS
  //######################################################################################

  // Because prefetch cannot cross 4KB-page boundaries, there is no need to consider offsets
  // greater than 63. However, with pages larger than 4KB, it would be beneficial to consider
  // larger offsets.

  constexpr static int NOFFSETS = 46;
  constexpr static int OFFSET[NOFFSETS] = {1,   -1,  2,   -2,  3,   -3,  4,   -4,  5,   -5,  6,   -6,  7,   -7,  8,   -8,
                                           9,   -9,  10,  -10, 11,  -11, 12,  -12, 13,  -13, 14,  -14, 15,  -15, 16,  -16,
                                           18,  -18, 20,  -20, 24,  -24, 30,  -30, 32,  -32, 36,  -36, 40,  -40};
  constexpr static int DEFAULT_OFFSET = 1;
  constexpr static int SCORE_MAX = 31;
  constexpr static int ROUND_MAX = 100;
  constexpr static int RRINDEX = 6;
  constexpr static int RRTAG = 12;
  constexpr static int DELAYQSIZE = 15;
  constexpr static int DELAY = 60;
  constexpr static int TIME_BITS = 12;
  constexpr static int LLC_RATE_MAX = 255;
  constexpr static int GAUGE_MAX = 8191;
  constexpr static int MSHR_THRESHOLD_MIN = 2;
  constexpr static int LOW_SCORE = 20;
  // DPC-2 knobs for its small-LLC and low-bandwidth configurations; both off for a normal machine
  constexpr static bool knob_small_llc = false;
  constexpr static bool knob_low_bandwidth = false;
  constexpr static int BAD_SCORE = knob_small_llc ? 10 : 1;
  constexpr static int BANDWIDTH = knob_low_bandwidth ? 64 : 16;
  int MSHR_THRESHOLD_MAX = 0; // (L2_MSHR_COUNT-4), set from the cache at initialization

  //######################################################################################
  //                               PREFETCHER STATE
  //######################################################################################

  using t_addr = long long;

  int prefetch_offset; // 7 bits (6-bit value + 1 sign bit)

  // Recent Requests (RR) table: 2 banks, 64 entries per bank, RRTAG bits per entry
  int recent_request[2][1 << RRINDEX]; // 2x64x12 = 1536 bits

  struct offsets_scores {
    int score[NOFFSETS]; // log2 SCORE_MAX = 5 bits per entry
    int max_score;       // log2 SCORE_MAX = 5 bits
    int best_offset;     // 7 bits (6-bit value + 1 sign bit)
    int round;           // log2 ROUND_MAX = 7 bits
    int p;               // log2 NOFFSETS = 6 bits
  } os;                  // 46x5+5+7+7+6 = 255 bits

  struct delay_queue {
    int lineaddr[DELAYQSIZE]; // RRINDEX+RTAG = 18 bits
    int cycle[DELAYQSIZE];    // TIME_BITS = 12 bits
    int valid[DELAYQSIZE];    // 1 bit
    int tail;                 // log2 DELAYQSIZE = 4 bits
    int head;                 // log2 DELAYQSIZE = 4 bits
  } dq;                       // 15x(18+12+1)+4+4 = 473 bits

  struct prefetch_throttle {
    int mshr_threshold; // log2 L2_MSHR_COUNT = 4 bits
    int prefetch_score; // log2 SCORE_MAX = 5 bits
    int llc_rate;       // log2 LLC_RATE_MAX = 8 bits
    int llc_rate_gauge; // log2 GAUGE_MAX = 13 bits
    int last_cycle;     // TIME_BITS = 12 bits
  } pt;                 // 4+5+8+13+12 = 42 bits

  // PORT: the DPC-2 simulator kept a fill's "prefetch" flag even when a demand merged into the in-flight prefetch,
  // and the original relies on it (see prefetcher_cache_fill). This ChampSim turns such fills into demand fills,
  // so the port remembers its own in-flight L2 prefetches, and keeps the 1 prefetch bit per L2 line itself.
  constexpr static std::size_t INFLIGHT_MAX = 256;
  std::deque<t_addr> inflight_order;
  std::unordered_set<t_addr> inflight;
  std::unordered_set<t_addr> prefetch_bit;

  //######################################################################################
  //                            SOME MACROS & DEFINITIONS
  //######################################################################################

  constexpr static int LOGLINE = 6;
  static bool SAMEPAGE(t_addr lineaddr1, t_addr lineaddr2) { return (((lineaddr1) ^ (lineaddr2)) >> 6) == 0; }
  static void INCREMENT(int& x, int n)
  {
    x++;
    if (x == (n))
      x = 0;
  }
  static int TRUNCATE(long long x, int nbits) { return (int)(((x) & ((1LL << (nbits)) - 1))); }

  long long get_current_cycle() const;

  void rr_init();
  int rr_tag(t_addr lineaddr);
  int rr_index_left(t_addr lineaddr);
  int rr_index_right(t_addr lineaddr);
  void rr_insert_left(t_addr lineaddr);
  void rr_insert_right(t_addr lineaddr);
  int rr_hit(t_addr lineaddr);
  void dq_init();
  void dq_push(t_addr lineaddr);
  int dq_ready();
  void dq_pop();
  void pt_init();
  void pt_update_mshr_threshold();
  void pt_llc_access();
  void os_reset();
  void os_learn_best_offset(t_addr lineaddr);
  int issue_prefetch(t_addr lineaddr, int offset);

  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
};

#endif
