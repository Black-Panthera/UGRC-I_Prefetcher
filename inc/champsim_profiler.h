#ifndef CHAMPSIM_PROFILER_H
#define CHAMPSIM_PROFILER_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include "access_type.h"
#include "address.h"
#include <unordered_set>
#include <limits>

namespace champsim
{

class channel;

enum class LineProvenance : uint8_t {
  DEMAND_FILLED   = 0,
  L2C_LOCAL_PF    = 1,
  L1D_UPSTREAM_PF = 2
};

// Per-access / per-prefetch event logs (CHAMPSIM_EVENT_LOG=<prefix> -> <prefix>.acc.bin, <prefix>.pf.bin).
// Detailed (post-warmup) phase only. Each file starts with a 16-byte header: 8-byte magic, record size, version.
enum class AccessOutcome : uint8_t {
  HIT            = 0, // ordinary hit
  PF_HIT_TIMELY  = 1, // first use of a line this cache prefetched
  PF_HIT_REUSE   = 2, // later use of a line this cache prefetched
  UPSTREAM_PF_HIT = 3,
  PF_HIT_LATE    = 4, // merged into this cache's in-flight prefetch
  MISS           = 5, // primary miss (allocated an MSHR)
  MISS_MERGED    = 6  // merged into an in-flight demand miss
};

enum class PrefetchEvent : uint8_t {
  REJECTED_PQ_FULL       = 0, // requested by the prefetcher, prefetch queue full
  FILTERED_RESIDENT      = 1, // line already in this cache
  MERGED_INFLIGHT_PF     = 2, // line already in flight as a prefetch
  MERGED_INFLIGHT_DEMAND = 3, // line already in flight as a demand miss
  DROPPED_MSHR_FULL      = 4,
  ISSUED_FILL_L2         = 5, // sent to the LLC, will fill this cache
  ISSUED_LLC_ONLY        = 6, // sent to the LLC without filling this cache
  EVICTED_UNUSED         = 7  // filled into this cache, evicted before any demand use
};

#pragma pack(push, 1)
struct AccessRecord {
  uint64_t instr_id;      // trace position of the (first) requesting instruction
  uint64_t pline;         // physical cache-line number
  uint64_t vline;         // virtual cache-line number
  uint64_t trigger_instr; // for prefetch hits: instr_id that triggered the covering prefetch, else UINT64_MAX
  int32_t timing;         // miss / late: cycles until the fill; timely prefetch hit: cycles the line waited since its fill; else 0
  uint8_t outcome;        // AccessOutcome
  uint8_t type;           // access_type (LOAD / RFO)
  uint8_t seen;           // 1 = this access was passed to the prefetcher
  uint8_t flags;          // bit0: covering prefetch was issued outside a prefetcher_cache_operate call (trigger approximate)
};
struct PrefetchRecord {
  uint64_t trigger_instr; // instr_id of the access being handled when the prefetch was requested
  uint64_t target_pline;  // physical cache-line number requested
  uint32_t cycle;         // cycle the prefetch was requested
  int32_t delta;          // target line - trigger line
  uint8_t event;          // PrefetchEvent
  uint8_t fill_this_level;
  uint8_t flags;          // bit0: requested outside prefetcher_cache_operate (trigger approximate)
  uint8_t pad;
  uint32_t aux;           // EVICTED_UNUSED: eviction cycle; ISSUED_*: 0
};
#pragma pack(pop)
static_assert(sizeof(AccessRecord) == 40, "AccessRecord must be exactly 40 bytes");
static_assert(sizeof(PrefetchRecord) == 32, "PrefetchRecord must be exactly 32 bytes");

struct PrefetchOriginRecord {
  uint64_t trigger_pc = 0;
  uint64_t trigger_addr = 0;
  int64_t predicted_delta = 0;
  uint64_t issue_cycle = 0;
  access_type trigger_type = access_type::LOAD;
  bool from_l1d = false;
  bool valid = false;
  LineProvenance provenance = LineProvenance::DEMAND_FILLED;
  bool first_use_credited = false;
  bool late_useful_merged = false;
  bool from_detailed_sim = false;
  uint64_t trigger_instr_id = std::numeric_limits<uint64_t>::max();
  uint64_t target_cl = 0;
  uint64_t fill_cycle = 0;
  bool fill_this_level = true;
  bool outside_operate = false;
};

class champsim_profiler {
public:
  bool enabled = false;
  std::string output_filename = "";
  std::string cache_name = "";

  uint32_t num_sets = 0;
  uint32_t num_ways = 0;

  // Channel pointers for source distinction
  const champsim::channel* l1d_channel = nullptr;
  const champsim::channel* l1i_channel = nullptr;

  // Active trigger context during prefetcher callbacks
  uint64_t current_trigger_ip = 0;
  uint64_t current_trigger_addr = 0;
  access_type current_trigger_type = access_type::LOAD;
  bool current_trigger_from_l1d = false;
  bool in_prefetcher_operate = false;
  uint64_t current_trigger_instr_id = std::numeric_limits<uint64_t>::max();

  // Global demand delta tracking
  uint64_t last_global_cl_addr = 0;
  bool has_last_global_cl = false;

  // 131-Bin Demand Spectrum [-64, +64]
  static constexpr int MIN_DELTA = -64;
  static constexpr int MAX_DELTA = 64;
  static constexpr int NUM_BINS = 131;

  static int delta_to_bin(int64_t delta) {
    if (delta < MIN_DELTA) return 0;
    if (delta > MAX_DELTA) return 130;
    return static_cast<int>(delta - MIN_DELTA + 1);
  }

  struct DemandBinEntry {
    uint64_t demands = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
  };
  std::array<DemandBinEntry, NUM_BINS> demand_spectrum{};

  struct PredBinEntry {
    uint64_t enqueued = 0;
    uint64_t timely_useful = 0;
    uint64_t late_useful = 0;
    uint64_t total_useful = 0;
  };
  std::array<PredBinEntry, NUM_BINS> pred_spectrum{};

  // Space-Saving Candidate PC Tracker (K = 256)
  static constexpr size_t SS_CAPACITY = 256;
  struct SpaceSavingEntry {
    uint64_t pc = 0;
    uint64_t count = 0; // estimated misses
    uint64_t error = 0; // error bound
    uint64_t aggregate_wait_cycles = 0;
    uint64_t last_cl_addr = 0;
    bool has_last_addr = false;
    uint64_t total_accesses = 0;
    std::unordered_map<int64_t, uint64_t> stride_hist{};
  };
  std::vector<SpaceSavingEntry> ss_entries;
  uint64_t ss_total_misses = 0;

  // Admission Origin Table: keyed by block number, holds records for prefetches pending tag check
  std::unordered_map<uint64_t, std::deque<PrefetchOriginRecord>> admission_origins{};

  // In-flight MSHR Origin Table: keyed by block number
  std::unordered_map<uint64_t, PrefetchOriginRecord> inflight_mshr_origins{};

  // Cache-Line Origin Table: [set][way]
  std::vector<std::vector<PrefetchOriginRecord>> line_pf_origins{};

  // Accounting Counters
  uint64_t l2c_data_demands = 0;
  uint64_t l2c_data_loads = 0;
  uint64_t l2c_data_rfos = 0;
  uint64_t l2c_data_demand_hits = 0;
  uint64_t l2c_data_demand_misses = 0;
  uint64_t l2c_instruction_demands = 0;
  uint64_t l2c_translation_demands = 0;

  uint64_t callbacks_on_l1d_load = 0;
  uint64_t callbacks_on_l1i_load = 0;
  uint64_t callbacks_on_prefetch = 0;
  uint64_t callbacks_on_rfo = 0;
  uint64_t total_callbacks = 0;

  uint64_t pf_requested = 0;
  uint64_t rejected_pq_full = 0;
  uint64_t enqueued_pf_issued = 0;

  uint64_t terminal_filtered_resident = 0;
  uint64_t terminal_merged_inflight_prefetch = 0;
  uint64_t terminal_merged_inflight_demand = 0;
  uint64_t terminal_dropped_mshr_full = 0;
  uint64_t terminal_forwarded_lower = 0;

  uint64_t fills_completed = 0;
  uint64_t late_useful_demand_fills = 0;
  uint64_t prefetches_filled_to_cache = 0;

  uint64_t timely_useful_hits = 0;
  uint64_t late_useful_hits = 0;
  uint64_t total_useful_prefetches = 0;
  uint64_t unused_prefetch_evictions = 0;

  uint64_t detailed_timely_useful_hits = 0;
  uint64_t warmup_timely_useful_hits = 0;
  uint64_t detailed_unused_prefetch_evictions = 0;
  uint64_t warmup_unused_prefetch_evictions = 0;
  bool in_detailed_simulation = false;

  uint64_t useful_l1d_loads = 0;
  uint64_t useful_l1d_rfos = 0;
  uint64_t useful_l1i_loads = 0;
  uint64_t useful_translations = 0;
  uint64_t upper_level_prefetches_consuming_l2c_lines = 0;
  uint64_t detailed_upper_level_prefetches_consuming_l2c_lines = 0;
  uint64_t warmup_upper_level_prefetches_consuming_l2c_lines = 0;

  uint64_t multi_level_l1d_prefetches_arriving = 0;
  uint64_t multi_level_l1d_prefetches_filtered = 0;
  uint64_t multi_level_l1d_prefetches_merged_inflight = 0;
  uint64_t multi_level_l1d_prefetches_forwarded = 0;

  // Shadow Tag Table for potential pollution candidate tracking (1024 entries FIFO)
  static constexpr size_t SHADOW_TAG_CAPACITY = 1024;
  std::deque<uint64_t> shadow_tag_table{};
  uint64_t potential_pollution_miss_candidates = 0;

  static constexpr size_t EVENT_BUFFER_SIZE = 65536;
  uint64_t mshr_demand_merged_late_cl = 0;

  // Event logs
  bool event_logging_enabled = false;
  std::unique_ptr<std::ofstream> access_log_file = nullptr;
  std::unique_ptr<std::ofstream> prefetch_log_file = nullptr;
  std::vector<AccessRecord> access_buffer{};
  std::vector<PrefetchRecord> prefetch_buffer{};
  struct PendingMiss {
    AccessRecord rec;
    uint64_t arrival_cycle;
  };
  std::unordered_map<uint64_t, std::vector<PendingMiss>> pending_misses{}; // keyed by physical line, emitted when the fill completes
  bool last_miss_merged_into_demand = false;
  bool logging_active() const { return event_logging_enabled && in_detailed_simulation; }
  void log_access(const AccessRecord& rec);
  void log_prefetch_event(const PrefetchOriginRecord& origin, PrefetchEvent event, uint32_t aux = 0);
  void flush_event_logs();
  void close_event_logs();

  champsim_profiler() = default;
  ~champsim_profiler();
  champsim_profiler(champsim_profiler&&) noexcept = default;
  champsim_profiler& operator=(champsim_profiler&&) noexcept = default;
  champsim_profiler(const champsim_profiler&) = delete;
  champsim_profiler& operator=(const champsim_profiler&) = delete;

  void init(std::string name, uint32_t sets, uint32_t ways, const champsim::channel* l1d_ch, const champsim::channel* l1i_ch);
  void on_begin_detailed_phase();

  // Hook 1: Prefetch admission
  void on_prefetch_requested();
  void on_prefetch_rejected_pq_full(uint64_t pf_addr, bool fill_this_level, uint64_t current_cycle);
  void on_prefetch_enqueued(uint64_t pf_addr, uint64_t current_cycle, bool fill_this_level);
  void on_prefetch_forwarded_llc_only(uint64_t addr);

  // Hook 2: Tag lookup & Hits
  void on_tag_check_hit(uint64_t addr, uint64_t ip, access_type type, const champsim::channel* src_ch,
                        long set, long way, bool useful_prefetch, uint64_t current_cycle,
                        uint64_t instr_id = 0, uint64_t v_addr = 0, bool seen_by_prefetcher = false);
  void on_tag_check_miss(uint64_t addr, uint64_t ip, access_type type, const champsim::channel* src_ch, uint64_t current_cycle,
                         uint64_t instr_id = 0, uint64_t v_addr = 0, bool seen_by_prefetcher = false);

  // Hook 3: MSHR Merges & Terminal Outcomes
  void on_mshr_merge_prefetch_into_demand(uint64_t addr);
  void on_mshr_merge_prefetch_into_prefetch(uint64_t addr);
  void on_mshr_merge_demand_into_prefetch(uint64_t addr, access_type type, const champsim::channel* src_ch, uint64_t current_cycle);
  void on_mshr_dropped_full(uint64_t addr);
  void on_mshr_allocated_forwarded_lower(uint64_t addr, uint64_t current_cycle);
  void on_l1d_prefetch_forwarded_lower();
  void on_l1d_prefetch_merged_inflight();

  // Hook 4: Fills & Evictions
  void on_handle_fill(uint64_t addr, long set, long way, bool prefetch_fill, uint64_t evicted_addr, bool evicted_valid, uint64_t current_cycle = 0);
  void on_eviction(long set, long way, bool was_prefetch, uint64_t current_cycle = 0);
  void on_fill_complete(uint64_t addr, uint64_t current_cycle);

  // Hook 5: Prefetcher Operate Callback tracking
  void on_prefetcher_operate_start(uint64_t ip, uint64_t addr, access_type type, const champsim::channel* src_ch,
                                   uint64_t instr_id = std::numeric_limits<uint64_t>::max());
  void on_prefetcher_operate_end();

  // Final Report & Verification
  void dump_json(uint64_t sim_insts, uint64_t sim_cycles, uint64_t official_pf_useful,
                 size_t pending_internal_pq, size_t pending_inflight_tag, size_t pending_trans_stash,
                 size_t pending_mshr, size_t pending_inflight_fills, size_t resident_prefetches_at_sim_end);
};

} // namespace champsim

#endif // CHAMPSIM_PROFILER_H
