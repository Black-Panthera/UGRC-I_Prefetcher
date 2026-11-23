#include "champsim_profiler.h"
#include <iomanip>
#include <nlohmann/json.hpp>
#include <fmt/core.h>

namespace champsim
{

void champsim_profiler::init(std::string name, uint32_t sets, uint32_t ways, const champsim::channel* l1d_ch, const champsim::channel* l1i_ch)
{
  cache_name = name;
  num_sets = sets;
  num_ways = ways;
  l1d_channel = l1d_ch;
  l1i_channel = l1i_ch;

  const char* env_p = std::getenv("CHAMPSIM_PROFILER_OUTPUT");

  bool wants_profiler = (env_p != nullptr && name.find("L2C") != std::string::npos);
  const char* event_log = std::getenv("CHAMPSIM_EVENT_LOG");
  bool wants_events = (event_log != nullptr && name.find("L2C") != std::string::npos);

  if (wants_profiler || wants_events) {
    enabled = true;
    if (wants_events) {
      event_logging_enabled = true;
      auto open_log = [prefix = std::string(event_log)](const char* suffix, const char* magic, uint32_t rec_size) {
        auto f = std::make_unique<std::ofstream>(prefix + suffix, std::ios::binary | std::ios::out | std::ios::trunc);
        const uint32_t version = 1;
        f->write(magic, 8);
        f->write(reinterpret_cast<const char*>(&rec_size), sizeof(rec_size));
        f->write(reinterpret_cast<const char*>(&version), sizeof(version));
        return f;
      };
      access_log_file = open_log(".acc.bin", "CSACCLOG", sizeof(AccessRecord));
      prefetch_log_file = open_log(".pf.bin", "CSPFLOG\0", sizeof(PrefetchRecord));
      access_buffer.reserve(EVENT_BUFFER_SIZE);
      prefetch_buffer.reserve(EVENT_BUFFER_SIZE);
    }
    if (wants_profiler) {
      output_filename = env_p;
    }
    line_pf_origins.assign(num_sets, std::vector<PrefetchOriginRecord>(num_ways));
    ss_entries.reserve(SS_CAPACITY);
  } else {
    enabled = false;
  }
}

champsim_profiler::~champsim_profiler()
{
  close_event_logs();
}

void champsim_profiler::log_access(const AccessRecord& rec)
{
  access_buffer.push_back(rec);
  if (access_buffer.size() >= EVENT_BUFFER_SIZE) {
    flush_event_logs();
  }
}

void champsim_profiler::log_prefetch_event(const PrefetchOriginRecord& origin, PrefetchEvent event, uint32_t aux)
{
  if (!logging_active()) return;
  PrefetchRecord rec{};
  rec.trigger_instr = origin.trigger_instr_id;
  rec.target_pline = origin.target_cl;
  rec.cycle = static_cast<uint32_t>(origin.issue_cycle);
  rec.delta = static_cast<int32_t>(std::clamp<int64_t>(origin.predicted_delta, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
  rec.event = static_cast<uint8_t>(event);
  rec.fill_this_level = origin.fill_this_level ? 1 : 0;
  rec.flags = origin.outside_operate ? 1 : 0;
  rec.aux = aux;
  prefetch_buffer.push_back(rec);
  if (prefetch_buffer.size() >= EVENT_BUFFER_SIZE) {
    flush_event_logs();
  }
}

void champsim_profiler::flush_event_logs()
{
  if (access_log_file && access_log_file->is_open() && !access_buffer.empty()) {
    access_log_file->write(reinterpret_cast<const char*>(access_buffer.data()), static_cast<std::streamsize>(access_buffer.size() * sizeof(AccessRecord)));
  }
  access_buffer.clear();
  if (prefetch_log_file && prefetch_log_file->is_open() && !prefetch_buffer.empty()) {
    prefetch_log_file->write(reinterpret_cast<const char*>(prefetch_buffer.data()),
                             static_cast<std::streamsize>(prefetch_buffer.size() * sizeof(PrefetchRecord)));
  }
  prefetch_buffer.clear();
}

void champsim_profiler::close_event_logs()
{
  if (!event_logging_enabled) return;
  // Misses still waiting for their fill when the simulation ended
  for (auto& [cl, pending] : pending_misses) {
    for (auto& p : pending) {
      p.rec.timing = -1;
      access_buffer.push_back(p.rec);
    }
  }
  pending_misses.clear();
  flush_event_logs();
  if (access_log_file && access_log_file->is_open()) access_log_file->close();
  if (prefetch_log_file && prefetch_log_file->is_open()) prefetch_log_file->close();
  event_logging_enabled = false;
}

void champsim_profiler::on_begin_detailed_phase()
{
  if (!enabled) return;
  in_detailed_simulation = true;

  l2c_data_demands = 0;
  l2c_data_loads = 0;
  l2c_data_rfos = 0;
  l2c_data_demand_hits = 0;
  l2c_data_demand_misses = 0;
  l2c_instruction_demands = 0;
  l2c_translation_demands = 0;

  callbacks_on_l1d_load = 0;
  callbacks_on_l1i_load = 0;
  callbacks_on_prefetch = 0;
  callbacks_on_rfo = 0;
  total_callbacks = 0;

  pf_requested = 0;
  rejected_pq_full = 0;
  enqueued_pf_issued = 0;

  terminal_filtered_resident = 0;
  terminal_merged_inflight_prefetch = 0;
  terminal_merged_inflight_demand = 0;
  terminal_dropped_mshr_full = 0;
  terminal_forwarded_lower = 0;

  fills_completed = 0;
  late_useful_demand_fills = 0;
  prefetches_filled_to_cache = 0;

  timely_useful_hits = 0;
  late_useful_hits = 0;
  total_useful_prefetches = 0;
  unused_prefetch_evictions = 0;

  detailed_timely_useful_hits = 0;
  warmup_timely_useful_hits = 0;
  detailed_unused_prefetch_evictions = 0;
  warmup_unused_prefetch_evictions = 0;

  useful_l1d_loads = 0;
  useful_l1d_rfos = 0;
  useful_l1i_loads = 0;
  useful_translations = 0;
  upper_level_prefetches_consuming_l2c_lines = 0;
  detailed_upper_level_prefetches_consuming_l2c_lines = 0;
  warmup_upper_level_prefetches_consuming_l2c_lines = 0;

  multi_level_l1d_prefetches_arriving = 0;
  multi_level_l1d_prefetches_filtered = 0;
  multi_level_l1d_prefetches_merged_inflight = 0;
  multi_level_l1d_prefetches_forwarded = 0;

  potential_pollution_miss_candidates = 0;

  demand_spectrum.fill({});
  pred_spectrum.fill({});

  ss_entries.clear();
  ss_total_misses = 0;

  has_last_global_cl = false;

  for (auto& [cl, deq] : admission_origins) {
    for (auto& rec : deq) {
      rec.from_detailed_sim = false;
    }
  }
  for (auto& [cl, rec] : inflight_mshr_origins) {
    rec.from_detailed_sim = false;
  }
  for (auto& set_vec : line_pf_origins) {
    for (auto& rec : set_vec) {
      rec.from_detailed_sim = false;
    }
  }
  shadow_tag_table.clear();
  pending_misses.clear();
  last_miss_merged_into_demand = false;
}

void champsim_profiler::on_prefetch_requested()
{
  if (!enabled) return;
  pf_requested++;
}

void champsim_profiler::on_prefetch_rejected_pq_full(uint64_t pf_addr, bool fill_this_level, uint64_t current_cycle)
{
  if (!enabled) return;
  rejected_pq_full++;

  if (logging_active()) {
    PrefetchOriginRecord rec;
    rec.trigger_instr_id = current_trigger_instr_id;
    rec.target_cl = pf_addr >> 6;
    rec.predicted_delta = static_cast<int64_t>(pf_addr >> 6) - static_cast<int64_t>(current_trigger_addr >> 6);
    rec.issue_cycle = current_cycle;
    rec.fill_this_level = fill_this_level;
    rec.outside_operate = !in_prefetcher_operate;
    log_prefetch_event(rec, PrefetchEvent::REJECTED_PQ_FULL);
  }
}

void champsim_profiler::on_prefetch_forwarded_llc_only(uint64_t addr)
{
  if (!enabled) return;
  uint64_t cl_addr = addr >> 6;
  auto it = admission_origins.find(cl_addr);
  if (it != admission_origins.end() && !it->second.empty()) {
    log_prefetch_event(it->second.front(), PrefetchEvent::ISSUED_LLC_ONLY);
    it->second.pop_front();
    if (it->second.empty()) admission_origins.erase(it);
  }
}

void champsim_profiler::on_prefetch_enqueued(uint64_t pf_addr, uint64_t current_cycle, bool fill_this_level)
{
  if (!enabled) return;
  enqueued_pf_issued++;

  uint64_t pf_cl = pf_addr >> 6;
  uint64_t trigger_cl = current_trigger_addr >> 6;
  int64_t pred_delta = static_cast<int64_t>(pf_cl) - static_cast<int64_t>(trigger_cl);
  int bin = delta_to_bin(pred_delta);
  pred_spectrum[bin].enqueued++;

  PrefetchOriginRecord rec;
  rec.trigger_pc = current_trigger_ip;
  rec.trigger_addr = current_trigger_addr;
  rec.predicted_delta = pred_delta;
  rec.issue_cycle = current_cycle;
  rec.trigger_type = current_trigger_type;
  rec.from_l1d = current_trigger_from_l1d;
  rec.valid = true;
  rec.late_useful_merged = false;
  rec.from_detailed_sim = in_detailed_simulation;
  rec.trigger_instr_id = current_trigger_instr_id;
  rec.target_cl = pf_cl;
  rec.fill_this_level = fill_this_level;
  rec.outside_operate = !in_prefetcher_operate;

  admission_origins[pf_cl].push_back(rec);
}

void champsim_profiler::on_tag_check_hit(uint64_t addr, uint64_t ip, access_type type, const champsim::channel* src_ch,
                                         long set, long way, bool useful_prefetch, uint64_t current_cycle,
                                         uint64_t instr_id, uint64_t v_addr, bool seen_by_prefetcher)
{
  if (!enabled) return;

  uint64_t cl_addr = addr >> 6;

  if (type == access_type::PREFETCH) {
    if (src_ch == nullptr) {
      // Local prefetch issued by this cache hitting resident line -> filtered
      terminal_filtered_resident++;
      auto it = admission_origins.find(cl_addr);
      if (it != admission_origins.end() && !it->second.empty()) {
        log_prefetch_event(it->second.front(), PrefetchEvent::FILTERED_RESIDENT);
        it->second.pop_front();
        if (it->second.empty()) admission_origins.erase(it);
      }
    } else {
      // Upper level prefetch (e.g. from L1D) arriving at L2C
      multi_level_l1d_prefetches_arriving++;
      multi_level_l1d_prefetches_filtered++;
      if (useful_prefetch) {
        upper_level_prefetches_consuming_l2c_lines++;
        if (set >= 0 && set < static_cast<long>(num_sets) &&
            way >= 0 && way < static_cast<long>(num_ways)) {
          auto& rec = line_pf_origins[set][way];
          rec.first_use_credited = true;
          if (rec.valid && rec.from_detailed_sim) {
            detailed_upper_level_prefetches_consuming_l2c_lines++;
          } else {
            warmup_upper_level_prefetches_consuming_l2c_lines++;
          }
          rec.valid = false;
        } else {
          warmup_upper_level_prefetches_consuming_l2c_lines++;
        }
      }
    }
    return;
  }

  // DEMAND hit (LOAD, RFO, TRANSLATION, etc.)
  bool is_l1d = (src_ch == l1d_channel);
  bool is_l1i = (src_ch == l1i_channel);
  bool is_trans = (type == access_type::TRANSLATION);

  if (is_l1d) {
    l2c_data_demands++;
    l2c_data_demand_hits++;
    if (type == access_type::LOAD) {
      l2c_data_loads++;
    } else if (type == access_type::RFO) {
      l2c_data_rfos++;
    }
  } else if (is_l1i) {
    l2c_instruction_demands++;
  } else if (is_trans) {
    l2c_translation_demands++;
  }

  // Global Demand Spectrum
  if (has_last_global_cl) {
    int64_t delta = static_cast<int64_t>(cl_addr) - static_cast<int64_t>(last_global_cl_addr);
    int bin = delta_to_bin(delta);
    demand_spectrum[bin].demands++;
    demand_spectrum[bin].hits++;
  } else {
    int bin = delta_to_bin(0);
    demand_spectrum[bin].demands++;
    demand_spectrum[bin].hits++;
  }
  last_global_cl_addr = cl_addr;
  has_last_global_cl = true;

  // Space-Saving tracker: update PC private last_cl_addr on demand hit (Section 2.5)
  if (ip != 0 && is_l1d) {
    auto ss_it = std::find_if(ss_entries.begin(), ss_entries.end(), [ip](const auto& e) { return e.pc == ip; });
    if (ss_it != ss_entries.end()) {
      if (ss_it->has_last_addr) {
        int64_t stride = static_cast<int64_t>(cl_addr) - static_cast<int64_t>(ss_it->last_cl_addr);
        ss_it->stride_hist[stride]++;
      }
      ss_it->last_cl_addr = cl_addr;
      ss_it->has_last_addr = true;
      ss_it->total_accesses++;
    }
  }

  // Timely Useful Prefetch attribution
  if (useful_prefetch) {
    timely_useful_hits++;
    total_useful_prefetches++;

    if (is_l1d) {
      if (type == access_type::LOAD) useful_l1d_loads++;
      else if (type == access_type::RFO) useful_l1d_rfos++;
    } else if (is_l1i) {
      useful_l1i_loads++;
    } else if (is_trans) {
      useful_translations++;
    }

    if (set >= 0 && set < static_cast<long>(num_sets) &&
        way >= 0 && way < static_cast<long>(num_ways)) {
      auto& rec = line_pf_origins[set][way];
      rec.first_use_credited = true;
      if (rec.valid) {
        if (rec.from_detailed_sim) {
          detailed_timely_useful_hits++;
          int bin = delta_to_bin(rec.predicted_delta);
          pred_spectrum[bin].timely_useful++;
          pred_spectrum[bin].total_useful++;
        } else {
          warmup_timely_useful_hits++;
        }
        rec.valid = false;
      } else {
        warmup_timely_useful_hits++;
      }
    } else {
      warmup_timely_useful_hits++;
    }
  }

  if (is_l1d && (type == access_type::LOAD || type == access_type::RFO) && logging_active()) {
    AccessRecord rec{};
    rec.instr_id = instr_id;
    rec.pline = cl_addr;
    rec.vline = v_addr >> 6;
    rec.trigger_instr = std::numeric_limits<uint64_t>::max();
    rec.outcome = static_cast<uint8_t>(AccessOutcome::HIT);
    rec.type = static_cast<uint8_t>(type);
    rec.seen = seen_by_prefetcher ? 1 : 0;
    if (set >= 0 && set < static_cast<long>(num_sets) && way >= 0 && way < static_cast<long>(num_ways)) {
      const auto& origin = line_pf_origins[set][way];
      if (useful_prefetch) {
        rec.outcome = static_cast<uint8_t>(AccessOutcome::PF_HIT_TIMELY);
        rec.trigger_instr = origin.trigger_instr_id;
        rec.timing = (origin.fill_cycle > 0 && current_cycle >= origin.fill_cycle) ? static_cast<int32_t>(current_cycle - origin.fill_cycle) : -1;
        rec.flags = origin.outside_operate ? 1 : 0;
      } else if (origin.provenance != LineProvenance::DEMAND_FILLED) {
        // L1D_UPSTREAM_PF here means an L2 prefetch triggered by an L1D demand (no L1 prefetcher is configured)
        rec.outcome = static_cast<uint8_t>(AccessOutcome::PF_HIT_REUSE);
        rec.trigger_instr = origin.trigger_instr_id;
      }
    }
    log_access(rec);
  }
}

void champsim_profiler::on_tag_check_miss(uint64_t addr, uint64_t ip, access_type type, const champsim::channel* src_ch, uint64_t current_cycle,
                                          uint64_t instr_id, uint64_t v_addr, bool seen_by_prefetcher)
{
  if (!enabled) return;
  const bool merged_into_demand = last_miss_merged_into_demand;
  last_miss_merged_into_demand = false;

  if (type == access_type::PREFETCH) {
    if (src_ch != nullptr) {
      multi_level_l1d_prefetches_arriving++;
    }
    return;
  }

  uint64_t cl_addr = addr >> 6;
  bool is_l1d = (src_ch == l1d_channel);
  bool is_l1i = (src_ch == l1i_channel);
  bool is_trans = (type == access_type::TRANSLATION);

  if (is_l1d) {
    l2c_data_demands++;
    l2c_data_demand_misses++;
    if (type == access_type::LOAD) {
      l2c_data_loads++;
    } else if (type == access_type::RFO) {
      l2c_data_rfos++;
    }
  } else if (is_l1i) {
    l2c_instruction_demands++;
  } else if (is_trans) {
    l2c_translation_demands++;
  }

  // Global Demand Spectrum
  if (has_last_global_cl) {
    int64_t delta = static_cast<int64_t>(cl_addr) - static_cast<int64_t>(last_global_cl_addr);
    int bin = delta_to_bin(delta);
    demand_spectrum[bin].demands++;
    demand_spectrum[bin].misses++;
  } else {
    int bin = delta_to_bin(0);
    demand_spectrum[bin].demands++;
    demand_spectrum[bin].misses++;
  }
  last_global_cl_addr = cl_addr;
  has_last_global_cl = true;

  // Space-Saving Miss Tracking
  if (is_l1d) {
    ss_total_misses++;
    if (ip != 0) {
      auto ss_it = std::find_if(ss_entries.begin(), ss_entries.end(), [ip](const auto& e) { return e.pc == ip; });
      if (ss_it != ss_entries.end()) {
        ss_it->count++;
        if (ss_it->has_last_addr) {
          int64_t stride = static_cast<int64_t>(cl_addr) - static_cast<int64_t>(ss_it->last_cl_addr);
          ss_it->stride_hist[stride]++;
        }
        ss_it->last_cl_addr = cl_addr;
        ss_it->has_last_addr = true;
        ss_it->total_accesses++;
      } else {
        if (ss_entries.size() < SS_CAPACITY) {
          SpaceSavingEntry entry;
          entry.pc = ip;
          entry.count = 1;
          entry.error = 0;
          entry.last_cl_addr = cl_addr;
          entry.has_last_addr = true;
          entry.total_accesses = 1;
          ss_entries.push_back(std::move(entry));
        } else {
          auto min_it = std::min_element(ss_entries.begin(), ss_entries.end(),
                                         [](const auto& a, const auto& b) { return a.count < b.count; });
          min_it->pc = ip;
          min_it->error = min_it->count;
          min_it->count = min_it->count + 1;
          min_it->last_cl_addr = cl_addr;
          min_it->has_last_addr = true;
          min_it->total_accesses = 1;
          min_it->stride_hist.clear();
          min_it->aggregate_wait_cycles = 0;
        }
      }
    }
  }

  // Shadow table check for potential pollution
  auto shadow_it = std::find(shadow_tag_table.begin(), shadow_tag_table.end(), cl_addr);
  if (shadow_it != shadow_tag_table.end()) {
    potential_pollution_miss_candidates++;
    shadow_tag_table.erase(shadow_it);
  }

  const bool was_late = (mshr_demand_merged_late_cl == cl_addr);
  mshr_demand_merged_late_cl = 0;

  if (is_l1d && (type == access_type::LOAD || type == access_type::RFO) && logging_active()) {
    AccessRecord rec{};
    rec.instr_id = instr_id;
    rec.pline = cl_addr;
    rec.vline = v_addr >> 6;
    rec.trigger_instr = std::numeric_limits<uint64_t>::max();
    rec.type = static_cast<uint8_t>(type);
    rec.seen = seen_by_prefetcher ? 1 : 0;
    if (was_late) {
      rec.outcome = static_cast<uint8_t>(AccessOutcome::PF_HIT_LATE);
      if (auto it = inflight_mshr_origins.find(cl_addr); it != inflight_mshr_origins.end()) {
        rec.trigger_instr = it->second.trigger_instr_id;
        rec.flags = it->second.outside_operate ? 1 : 0;
      }
    } else {
      rec.outcome = static_cast<uint8_t>(merged_into_demand ? AccessOutcome::MISS_MERGED : AccessOutcome::MISS);
    }
    // Emitted with its latency once the fill completes (on_fill_complete)
    pending_misses[cl_addr].push_back(PendingMiss{rec, current_cycle});
  }
}

void champsim_profiler::on_mshr_merge_prefetch_into_demand(uint64_t addr)
{
  if (!enabled) return;
  terminal_merged_inflight_demand++;
  uint64_t cl_addr = addr >> 6;
  auto it = admission_origins.find(cl_addr);
  if (it != admission_origins.end() && !it->second.empty()) {
    log_prefetch_event(it->second.front(), PrefetchEvent::MERGED_INFLIGHT_DEMAND);
    it->second.pop_front();
    if (it->second.empty()) admission_origins.erase(it);
  }
}

void champsim_profiler::on_mshr_merge_prefetch_into_prefetch(uint64_t addr)
{
  if (!enabled) return;
  terminal_merged_inflight_prefetch++;
  uint64_t cl_addr = addr >> 6;
  auto it = admission_origins.find(cl_addr);
  if (it != admission_origins.end() && !it->second.empty()) {
    log_prefetch_event(it->second.front(), PrefetchEvent::MERGED_INFLIGHT_PF);
    it->second.pop_front();
    if (it->second.empty()) admission_origins.erase(it);
  }
}

void champsim_profiler::on_mshr_merge_demand_into_prefetch(uint64_t addr, access_type type, const champsim::channel* src_ch, uint64_t current_cycle)
{
  if (!enabled) return;
  (void)current_cycle;
  late_useful_hits++;
  total_useful_prefetches++;

  bool is_l1d = (src_ch == l1d_channel);
  bool is_l1i = (src_ch == l1i_channel);
  bool is_trans = (type == access_type::TRANSLATION);

  if (is_l1d) {
    if (type == access_type::LOAD) useful_l1d_loads++;
    else if (type == access_type::RFO) useful_l1d_rfos++;
  } else if (is_l1i) {
    useful_l1i_loads++;
  } else if (is_trans) {
    useful_translations++;
  }

  uint64_t cl_addr = addr >> 6;
  mshr_demand_merged_late_cl = cl_addr;
  auto it = inflight_mshr_origins.find(cl_addr);
  if (it != inflight_mshr_origins.end()) {
    if (it->second.from_detailed_sim) {
      int bin = delta_to_bin(it->second.predicted_delta);
      pred_spectrum[bin].late_useful++;
      pred_spectrum[bin].total_useful++;
    }
    it->second.late_useful_merged = true;
  }
}

void champsim_profiler::on_mshr_dropped_full(uint64_t addr)
{
  if (!enabled) return;
  terminal_dropped_mshr_full++;
  uint64_t cl_addr = addr >> 6;
  auto it = admission_origins.find(cl_addr);
  if (it != admission_origins.end() && !it->second.empty()) {
    log_prefetch_event(it->second.front(), PrefetchEvent::DROPPED_MSHR_FULL);
    it->second.pop_front();
    if (it->second.empty()) admission_origins.erase(it);
  }
}

void champsim_profiler::on_mshr_allocated_forwarded_lower(uint64_t addr, uint64_t current_cycle)
{
  if (!enabled) return;
  (void)current_cycle;
  terminal_forwarded_lower++;
  uint64_t cl_addr = addr >> 6;
  auto it = admission_origins.find(cl_addr);
  if (it != admission_origins.end() && !it->second.empty()) {
    log_prefetch_event(it->second.front(), PrefetchEvent::ISSUED_FILL_L2);
    inflight_mshr_origins[cl_addr] = it->second.front();
    it->second.pop_front();
    if (it->second.empty()) admission_origins.erase(it);
  } else {
    PrefetchOriginRecord rec;
    rec.valid = true;
    inflight_mshr_origins[cl_addr] = rec;
  }
}

void champsim_profiler::on_l1d_prefetch_forwarded_lower()
{
  if (!enabled) return;
  multi_level_l1d_prefetches_forwarded++;
}

void champsim_profiler::on_l1d_prefetch_merged_inflight()
{
  if (!enabled) return;
  multi_level_l1d_prefetches_merged_inflight++;
}

void champsim_profiler::on_handle_fill(uint64_t addr, long set, long way, bool prefetch_fill, uint64_t evicted_addr, bool evicted_valid, uint64_t current_cycle)
{
  if (!enabled) return;
  uint64_t cl_addr = addr >> 6;

  // Potential pollution tracking
  if (prefetch_fill && evicted_valid && evicted_addr != 0) {
    uint64_t evicted_cl = evicted_addr >> 6;
    if (shadow_tag_table.size() >= SHADOW_TAG_CAPACITY) {
      shadow_tag_table.pop_front();
    }
    shadow_tag_table.push_back(evicted_cl);
  }

  auto it = inflight_mshr_origins.find(cl_addr);
  if (it != inflight_mshr_origins.end()) {
    if (it->second.from_detailed_sim) {
      fills_completed++;
      if (it->second.late_useful_merged) {
        late_useful_demand_fills++;
        if (set >= 0 && set < static_cast<long>(num_sets) &&
            way >= 0 && way < static_cast<long>(num_ways)) {
          line_pf_origins[set][way] = it->second;
          line_pf_origins[set][way].provenance = it->second.from_l1d ? LineProvenance::L1D_UPSTREAM_PF : LineProvenance::L2C_LOCAL_PF;
          line_pf_origins[set][way].first_use_credited = true;
          line_pf_origins[set][way].valid = false;
        }
      } else {
        prefetches_filled_to_cache++;
        if (set >= 0 && set < static_cast<long>(num_sets) &&
            way >= 0 && way < static_cast<long>(num_ways)) {
          line_pf_origins[set][way] = it->second;
          line_pf_origins[set][way].provenance = it->second.from_l1d ? LineProvenance::L1D_UPSTREAM_PF : LineProvenance::L2C_LOCAL_PF;
          line_pf_origins[set][way].first_use_credited = false;
          line_pf_origins[set][way].valid = true;
          line_pf_origins[set][way].fill_cycle = current_cycle;
        }
      }
    } else {
      // Warmup fill
      if (set >= 0 && set < static_cast<long>(num_sets) &&
          way >= 0 && way < static_cast<long>(num_ways)) {
        line_pf_origins[set][way] = it->second;
        line_pf_origins[set][way].provenance = it->second.from_l1d ? LineProvenance::L1D_UPSTREAM_PF : LineProvenance::L2C_LOCAL_PF;
        line_pf_origins[set][way].from_detailed_sim = false;
        line_pf_origins[set][way].fill_cycle = current_cycle;
        if (!it->second.late_useful_merged) {
          line_pf_origins[set][way].valid = true;
          line_pf_origins[set][way].first_use_credited = false;
        } else {
          line_pf_origins[set][way].valid = false;
          line_pf_origins[set][way].first_use_credited = true;
        }
      }
    }
    inflight_mshr_origins.erase(it);
  } else if (prefetch_fill) {
    if (in_detailed_simulation) {
      fills_completed++;
      prefetches_filled_to_cache++;
    }
    if (set >= 0 && set < static_cast<long>(num_sets) &&
        way >= 0 && way < static_cast<long>(num_ways)) {
      line_pf_origins[set][way].provenance = LineProvenance::L2C_LOCAL_PF;
      line_pf_origins[set][way].valid = true;
      line_pf_origins[set][way].first_use_credited = false;
      line_pf_origins[set][way].trigger_instr_id = std::numeric_limits<uint64_t>::max();
      line_pf_origins[set][way].target_cl = cl_addr;
      line_pf_origins[set][way].fill_cycle = current_cycle;
    }
  } else {
    // Demand fill: clears prefetch provenance
    if (set >= 0 && set < static_cast<long>(num_sets) &&
        way >= 0 && way < static_cast<long>(num_ways)) {
      line_pf_origins[set][way].provenance = LineProvenance::DEMAND_FILLED;
      line_pf_origins[set][way].valid = false;
      line_pf_origins[set][way].first_use_credited = false;
    }
  }
}

void champsim_profiler::on_eviction(long set, long way, bool was_prefetch, uint64_t current_cycle)
{
  if (!enabled) return;
  if (was_prefetch && set >= 0 && set < static_cast<long>(num_sets) && way >= 0 && way < static_cast<long>(num_ways)) {
    const auto& origin = line_pf_origins[set][way];
    if (origin.valid && origin.provenance != LineProvenance::DEMAND_FILLED) {
      log_prefetch_event(origin, PrefetchEvent::EVICTED_UNUSED, static_cast<uint32_t>(current_cycle));
    }
  }
  if (was_prefetch) {
    unused_prefetch_evictions++;
    if (set >= 0 && set < static_cast<long>(num_sets) &&
        way >= 0 && way < static_cast<long>(num_ways)) {
      if (line_pf_origins[set][way].valid && line_pf_origins[set][way].from_detailed_sim) {
        detailed_unused_prefetch_evictions++;
      } else {
        warmup_unused_prefetch_evictions++;
      }
    } else {
      warmup_unused_prefetch_evictions++;
    }
  }
  if (set >= 0 && set < static_cast<long>(num_sets) &&
      way >= 0 && way < static_cast<long>(num_ways)) {
    line_pf_origins[set][way].provenance = LineProvenance::DEMAND_FILLED;
    line_pf_origins[set][way].valid = false;
    line_pf_origins[set][way].first_use_credited = false;
  }
}

void champsim_profiler::on_fill_complete(uint64_t addr, uint64_t current_cycle)
{
  if (!enabled || pending_misses.empty()) return;
  auto it = pending_misses.find(addr >> 6);
  if (it == pending_misses.end()) return;
  for (auto& p : it->second) {
    p.rec.timing = static_cast<int32_t>(std::min<uint64_t>(current_cycle - p.arrival_cycle, std::numeric_limits<int32_t>::max()));
    log_access(p.rec);
  }
  pending_misses.erase(it);
}

void champsim_profiler::on_prefetcher_operate_start(uint64_t ip, uint64_t addr, access_type type, const champsim::channel* src_ch, uint64_t instr_id)
{
  if (!enabled) return;
  current_trigger_instr_id = instr_id;
  current_trigger_ip = ip;
  current_trigger_addr = addr;
  current_trigger_type = type;
  current_trigger_from_l1d = (src_ch == l1d_channel);
  in_prefetcher_operate = true;

  total_callbacks++;
  if (src_ch == l1d_channel && type == access_type::LOAD) {
    callbacks_on_l1d_load++;
  } else if (src_ch == l1i_channel && type == access_type::LOAD) {
    callbacks_on_l1i_load++;
  } else if (type == access_type::PREFETCH) {
    callbacks_on_prefetch++;
  } else if (type == access_type::RFO) {
    callbacks_on_rfo++;
  }
}

void champsim_profiler::on_prefetcher_operate_end()
{
  if (!enabled) return;
  in_prefetcher_operate = false;
}

void champsim_profiler::dump_json(uint64_t sim_insts, uint64_t sim_cycles, uint64_t official_pf_useful,
                                  size_t pending_internal_pq, size_t pending_inflight_tag, size_t pending_trans_stash,
                                  size_t pending_mshr, size_t pending_inflight_fills, size_t resident_prefetches_at_sim_end)
{
  close_event_logs();

  if (!enabled || output_filename.empty()) return;
  (void)resident_prefetches_at_sim_end;

  uint64_t demand_useful_sum = useful_l1d_loads + useful_l1d_rfos + useful_l1i_loads + useful_translations;
  uint64_t total_useful_sum = demand_useful_sum + upper_level_prefetches_consuming_l2c_lines;

  bool demand_invariant_verified = (l2c_data_demands == (l2c_data_demand_hits + l2c_data_demand_misses));
  bool admission_invariant_verified = (pf_requested == (rejected_pq_full + enqueued_pf_issued));
  bool useful_invariant_verified = (total_useful_prefetches == (timely_useful_hits + late_useful_hits));
  bool reconciliation_verified = (total_useful_sum == official_pf_useful);

  size_t total_pending_tag_check = pending_internal_pq + pending_inflight_tag + pending_trans_stash;
  uint64_t stage1_sum = terminal_filtered_resident + terminal_merged_inflight_prefetch +
                        terminal_merged_inflight_demand + terminal_dropped_mshr_full +
                        terminal_forwarded_lower + total_pending_tag_check;
  bool stage1_enqueued_sum_verified = (stage1_sum == enqueued_pf_issued);

  size_t total_pending_lower = pending_mshr + pending_inflight_fills;
  uint64_t stage2_sum = fills_completed + total_pending_lower;
  bool stage2_forwarded_sum_verified = (stage2_sum == terminal_forwarded_lower);

  size_t detailed_resident_prefetches = 0;
  size_t warmup_resident_prefetches = 0;
  for (size_t s = 0; s < num_sets; ++s) {
    for (size_t w = 0; w < num_ways; ++w) {
      if (s < line_pf_origins.size() && w < line_pf_origins[s].size()) {
        if (line_pf_origins[s][w].valid) {
          if (line_pf_origins[s][w].from_detailed_sim) {
            detailed_resident_prefetches++;
          } else {
            warmup_resident_prefetches++;
          }
        }
      }
    }
  }
  size_t total_resident_prefetches = detailed_resident_prefetches + warmup_resident_prefetches;

  uint64_t detailed_consumed = detailed_timely_useful_hits + detailed_upper_level_prefetches_consuming_l2c_lines;
  uint64_t resident_sum = detailed_consumed + detailed_unused_prefetch_evictions + detailed_resident_prefetches;
  bool resident_sum_verified = (resident_sum == prefetches_filled_to_cache);
  uint64_t stage3_sum = late_useful_demand_fills + prefetches_filled_to_cache;
  bool stage3_fills_sum_verified = (stage3_sum == fills_completed);

  double demand_accuracy_pct = enqueued_pf_issued > 0 ? (100.0 * static_cast<double>(demand_useful_sum) / static_cast<double>(enqueued_pf_issued)) : 0.0;
  double official_accuracy_pct = enqueued_pf_issued > 0 ? (100.0 * static_cast<double>(official_pf_useful) / static_cast<double>(enqueued_pf_issued)) : 0.0;

  // Space-Saving sorting and extraction
  std::sort(ss_entries.begin(), ss_entries.end(), [](const auto& a, const auto& b) {
    return a.count > b.count;
  });

  uint64_t estimated_captured_misses = 0;
  for (const auto& e : ss_entries) {
    estimated_captured_misses += e.count;
  }
  double captured_miss_ratio_estimate = ss_total_misses > 0 ? static_cast<double>(estimated_captured_misses) / static_cast<double>(ss_total_misses) : 0.0;

  nlohmann::json j;

  j["simulation_metadata"] = {
    {"instructions", sim_insts},
    {"cycles", sim_cycles},
    {"profiler_target", "L2C"},
    {"filtered_access_population", "l1d_originating_requests"}
  };

  j["accounting_validation"] = {
    {"l2c_data_demands", l2c_data_demands},
    {"l2c_data_loads", l2c_data_loads},
    {"l2c_data_rfos", l2c_data_rfos},
    {"l2c_data_demand_hits", l2c_data_demand_hits},
    {"l2c_data_demand_misses", l2c_data_demand_misses},
    {"demand_invariant_verified", demand_invariant_verified},
    {"prefetcher_callbacks", {
      {"callbacks_on_l1d_load", callbacks_on_l1d_load},
      {"callbacks_on_l1i_load", callbacks_on_l1i_load},
      {"callbacks_on_prefetch", callbacks_on_prefetch},
      {"callbacks_on_rfo", callbacks_on_rfo},
      {"total_callbacks", total_callbacks}
    }},
    {"pf_requested", pf_requested},
    {"rejected_pq_full", rejected_pq_full},
    {"enqueued_pf_issued", enqueued_pf_issued},
    {"admission_invariant_verified", admission_invariant_verified},
    {"official_sim_stats_pf_useful", official_pf_useful},
    {"tracked_timely_useful_prefetches", timely_useful_hits},
    {"tracked_late_useful_prefetches", late_useful_hits},
    {"tracked_total_useful_prefetches", total_useful_prefetches},
    {"useful_invariant_verified", useful_invariant_verified},
    {"useful_by_access_origin", {
      {"l1d_loads", useful_l1d_loads},
      {"l1d_rfos", useful_l1d_rfos},
      {"l1i_loads", useful_l1i_loads},
      {"translations", useful_translations},
      {"upper_level_prefetches_consuming_l2c_lines", upper_level_prefetches_consuming_l2c_lines},
      {"demand_useful_sum", demand_useful_sum},
      {"total_useful_sum", total_useful_sum},
      {"reconciliation_verified", reconciliation_verified}
    }}
  };

  j["prefetches_lifecycle_breakdown"] = {
    {"enqueued_requests", enqueued_pf_issued},
    {"terminal_filtered_resident", terminal_filtered_resident},
    {"terminal_merged_inflight_prefetch", terminal_merged_inflight_prefetch},
    {"terminal_merged_inflight_demand", terminal_merged_inflight_demand},
    {"terminal_dropped_mshr_full", terminal_dropped_mshr_full},
    {"terminal_forwarded_lower", terminal_forwarded_lower},
    {"pending_tag_check_at_sim_end", {
      {"internal_PQ", pending_internal_pq},
      {"inflight_tag_check", pending_inflight_tag},
      {"translation_stash", pending_trans_stash},
      {"total_pending", total_pending_tag_check}
    }},
    {"stage1_enqueued_sum_verified", stage1_enqueued_sum_verified},
    {"forwarded_lower_decomposition", {
      {"fills_completed", fills_completed},
      {"lower_level_pending_at_sim_end", {
        {"mshr_inflight", pending_mshr},
        {"inflight_fills", pending_inflight_fills},
        {"total_pending", total_pending_lower}
      }},
      {"stage2_forwarded_sum_verified", stage2_forwarded_sum_verified}
    }},
    {"fills_completed_decomposition", {
      {"late_useful_demand_fills", late_useful_demand_fills},
      {"prefetches_filled_to_cache", prefetches_filled_to_cache},
      {"prefetches_filled_to_cache_breakdown", {
        {"detailed_timely_useful_hits", detailed_timely_useful_hits},
        {"detailed_upper_level_prefetches_consuming_l2c_lines", detailed_upper_level_prefetches_consuming_l2c_lines},
        {"detailed_unused_prefetch_evictions", detailed_unused_prefetch_evictions},
        {"detailed_resident_prefetches_at_sim_end", detailed_resident_prefetches},
        {"warmup_timely_useful_hits", warmup_timely_useful_hits},
        {"warmup_upper_level_prefetches_consuming_l2c_lines", warmup_upper_level_prefetches_consuming_l2c_lines},
        {"warmup_unused_prefetch_evictions", warmup_unused_prefetch_evictions},
        {"warmup_resident_prefetches_at_sim_end", warmup_resident_prefetches},
        {"total_timely_useful_hits", timely_useful_hits},
        {"total_upper_level_prefetches_consuming_l2c_lines", upper_level_prefetches_consuming_l2c_lines},
        {"total_unused_prefetch_evictions", unused_prefetch_evictions},
        {"total_resident_prefetches_at_sim_end", total_resident_prefetches},
        {"resident_sum_verified", resident_sum_verified}
      }},
      {"stage3_fills_sum_verified", stage3_fills_sum_verified}
    }}
  };

  uint64_t multi_level_arriving_sum = multi_level_l1d_prefetches_filtered + multi_level_l1d_prefetches_merged_inflight + multi_level_l1d_prefetches_forwarded;
  bool multi_level_arriving_sum_verified = (multi_level_arriving_sum == multi_level_l1d_prefetches_arriving);

  j["multi_level_hierarchy_traffic"] = {
    {"l1d_prefetches_arriving_at_l2c", multi_level_l1d_prefetches_arriving},
    {"l1d_prefetches_filtered_resident_at_l2c", multi_level_l1d_prefetches_filtered},
    {"l1d_prefetches_merged_inflight_at_l2c", multi_level_l1d_prefetches_merged_inflight},
    {"l1d_prefetches_forwarded_to_lower_level", multi_level_l1d_prefetches_forwarded},
    {"multi_level_arriving_sum_verified", multi_level_arriving_sum_verified},
    {"l1d_prefetches_consumed_l2c_resident_prefetches", upper_level_prefetches_consuming_l2c_lines}
  };

  // Demand Spectrum (131 Bins)
  nlohmann::json in_range_demands = nlohmann::json::object();
  for (int d = MIN_DELTA; d <= MAX_DELTA; ++d) {
    int bin = delta_to_bin(d);
    if (demand_spectrum[bin].demands > 0) {
      std::string key = fmt::format("delta_{:+d}", d);
      in_range_demands[key] = {
        {"demands", demand_spectrum[bin].demands},
        {"hits", demand_spectrum[bin].hits}
      };
    }
  }
  j["global_interleaved_demand_spectrum"] = {
    {"in_range_bins_129", in_range_demands},
    {"underflow_below_-64", {
      {"demands", demand_spectrum[0].demands},
      {"hits", demand_spectrum[0].hits}
    }},
    {"overflow_above_+64", {
      {"demands", demand_spectrum[130].demands},
      {"hits", demand_spectrum[130].hits}
    }}
  };

  // Prediction Spectrum (131 Bins)
  nlohmann::json in_range_preds = nlohmann::json::object();
  for (int d = MIN_DELTA; d <= MAX_DELTA; ++d) {
    int bin = delta_to_bin(d);
    if (pred_spectrum[bin].enqueued > 0 || pred_spectrum[bin].total_useful > 0) {
      std::string key = fmt::format("pred_disp_{:+d}", d);
      in_range_preds[key] = {
        {"enqueued", pred_spectrum[bin].enqueued},
        {"timely_useful", pred_spectrum[bin].timely_useful},
        {"late_useful", pred_spectrum[bin].late_useful},
        {"total_useful", pred_spectrum[bin].total_useful}
      };
    }
  }
  j["prediction_attribution_spectrum"] = {
    {"in_range_bins_129", in_range_preds},
    {"underflow_below_-64", {
      {"enqueued", pred_spectrum[0].enqueued},
      {"timely_useful", pred_spectrum[0].timely_useful},
      {"late_useful", pred_spectrum[0].late_useful},
      {"total_useful", pred_spectrum[0].total_useful}
    }},
    {"overflow_above_+64", {
      {"enqueued", pred_spectrum[130].enqueued},
      {"timely_useful", pred_spectrum[130].timely_useful},
      {"late_useful", pred_spectrum[130].late_useful},
      {"total_useful", pred_spectrum[130].total_useful}
    }}
  };

  // Space-Saving entries
  nlohmann::json ss_list = nlohmann::json::array();
  for (const auto& e : ss_entries) {
    std::vector<std::pair<int64_t, uint64_t>> sorted_strides(e.stride_hist.begin(), e.stride_hist.end());
    std::sort(sorted_strides.begin(), sorted_strides.end(), [](const auto& a, const auto& b) {
      return a.second > b.second;
    });

    nlohmann::json dom_strides = nlohmann::json::array();
    uint64_t top_stride_freq = 0;
    uint64_t total_strides = 0;
    for (size_t i = 0; i < sorted_strides.size(); ++i) {
      total_strides += sorted_strides[i].second;
      if (i < 3) {
        dom_strides.push_back({
          {"stride", sorted_strides[i].first},
          {"frequency", sorted_strides[i].second}
        });
        if (i == 0) top_stride_freq = sorted_strides[0].second;
      }
    }

    double regularity = total_strides > 0 ? static_cast<double>(top_stride_freq) / static_cast<double>(total_strides) : 0.0;

    ss_list.push_back({
      {"pc", fmt::format("{:#x}", e.pc)},
      {"estimated_misses", e.count},
      {"max_error_bound", e.error},
      {"aggregate_demand_wait_cycles", e.aggregate_wait_cycles},
      {"private_dominant_strides", dom_strides},
      {"stride_regularity_score", regularity}
    });
  }

  j["space_saving_top_miss_pcs"] = {
    {"tracked_entries", ss_entries.size()},
    {"total_observed_misses", ss_total_misses},
    {"estimated_captured_misses", estimated_captured_misses},
    {"captured_miss_ratio_estimate", captured_miss_ratio_estimate},
    {"entries", ss_list}
  };

  j["system_efficiency_and_diagnostics"] = {
    {"demand_prefetch_accuracy_pct", demand_accuracy_pct},
    {"demand_accuracy_definition", "demand_useful_sum / enqueued_pf_issued"},
    {"official_reconciliation_accuracy_pct", official_accuracy_pct},
    {"official_accuracy_definition", "sim_stats.pf_useful / enqueued_pf_issued"},
    {"potential_pollution_miss_candidates", potential_pollution_miss_candidates},
    {"net_demand_miss_increase_vs_no_prefetch", 0},
    {"dram_bandwidth_traffic_increase_pct", 0.0}
  };

  std::ofstream out(output_filename);
  if (out.is_open()) {
    out << std::setw(2) << j << std::endl;
    out.close();
  }
}

} // namespace champsim
