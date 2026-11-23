#include "bop.h"

#include "cache.h"

// Function bodies are Michaud's DPC-2 code; lines marked PORT replace a DPC-2 simulator call.

long long bop::get_current_cycle() const { return intern_->current_time.time_since_epoch() / intern_->clock_period; } // PORT

//######################################################################################
//                            RECENT REQUESTS TABLE (RR)
//######################################################################################

void bop::rr_init()
{
  int i;
  for (i = 0; i < (1 << RRINDEX); i++) {
    recent_request[0][i] = 0;
    recent_request[1][i] = 0;
  }
}

int bop::rr_tag(t_addr lineaddr) { return TRUNCATE(lineaddr >> RRINDEX, RRTAG); }

int bop::rr_index_left(t_addr lineaddr) { return TRUNCATE(lineaddr ^ (lineaddr >> RRINDEX), RRINDEX); }

int bop::rr_index_right(t_addr lineaddr) { return TRUNCATE(lineaddr ^ (lineaddr >> (2 * RRINDEX)), RRINDEX); }

void bop::rr_insert_left(t_addr lineaddr)
{
  int i = rr_index_left(lineaddr);
  recent_request[0][i] = rr_tag(lineaddr);
}

void bop::rr_insert_right(t_addr lineaddr)
{
  int i = rr_index_right(lineaddr);
  recent_request[1][i] = rr_tag(lineaddr);
}

int bop::rr_hit(t_addr lineaddr)
{
  int i = rr_index_left(lineaddr);
  int j = rr_index_right(lineaddr);
  int tag = rr_tag(lineaddr);
  return (recent_request[0][i] == tag) || (recent_request[1][j] == tag);
}

//######################################################################################
//                               DELAY QUEUE (DQ)
//######################################################################################

// Without the delay queue, the prefetcher would always try to select an offset value
// large enough for having timely prefetches. However, sometimes, a small offset yields
// late prefetches but greater prefetch accuracy and better performance. The delay queue
// is an imperfect solution to this problem.

// This implementation of the delay queue is specific to the DPC2 simulator, as the DPC2
// prefetcher can act only at certain clock cycles. In a real processor, the delay queue
// implementation can be simpler.

void bop::dq_init()
{
  int i;
  for (i = 0; i < DELAYQSIZE; i++) {
    dq.lineaddr[i] = 0;
    dq.cycle[i] = 0;
    dq.valid[i] = 0;
  }
  dq.tail = 0;
  dq.head = 0;
}

void bop::dq_push(t_addr lineaddr)
{
  // enqueue one line address
  if (dq.valid[dq.tail]) {
    // delay queue is full
    // dequeue the oldest entry and write the "left" bank of the RR table
    rr_insert_left(dq.lineaddr[dq.head]);
    INCREMENT(dq.head, DELAYQSIZE);
  }
  dq.lineaddr[dq.tail] = TRUNCATE(lineaddr, RRINDEX + RRTAG);
  dq.cycle[dq.tail] = TRUNCATE(get_current_cycle(), TIME_BITS);
  dq.valid[dq.tail] = 1;
  INCREMENT(dq.tail, DELAYQSIZE);
}

int bop::dq_ready()
{
  // tells whether or not the oldest entry is ready to be dequeued
  if (!dq.valid[dq.head]) {
    // delay queue is empty
    return 0;
  }
  int cycle = TRUNCATE(get_current_cycle(), TIME_BITS);
  int issuecycle = dq.cycle[dq.head];
  int readycycle = TRUNCATE(issuecycle + DELAY, TIME_BITS);
  if (readycycle >= issuecycle) {
    return (cycle < issuecycle) || (cycle >= readycycle);
  } else {
    return (cycle < issuecycle) && (cycle >= readycycle);
  }
}

void bop::dq_pop()
{
  // dequeue the entries that are ready to be dequeued,
  // and do a write in the "left" bank of the RR table for each of them
  int i;
  for (i = 0; i < DELAYQSIZE; i++) {
    if (!dq_ready()) {
      break;
    }
    rr_insert_left(dq.lineaddr[dq.head]);
    dq.valid[dq.head] = 0;
    INCREMENT(dq.head, DELAYQSIZE);
  }
}

//######################################################################################
//                               PREFETCH THROTTLE (PT)
//######################################################################################

// The following prefetch throttling method is specific to the DPC2 simulator, as other
// parts of the microarchitecture (requests schedulers, cache replacement policy,
// LLC hit/miss information,...) can be neither modified nor observed. Consequently,
// we ignore hardware implementation considerations here.

void bop::pt_init()
{
  pt.mshr_threshold = MSHR_THRESHOLD_MAX;
  pt.prefetch_score = SCORE_MAX;
  pt.llc_rate = 0;
  pt.llc_rate_gauge = GAUGE_MAX / 2;
  pt.last_cycle = 0;
}

// The pt_update_mshr_threshold function is for adjusting the MSHR threshold
// (a prefetch request is dropped when the MSHR occupancy exceeds the threshold)

void bop::pt_update_mshr_threshold()
{
  if ((pt.prefetch_score > LOW_SCORE) || (pt.llc_rate > (2 * BANDWIDTH))) {
    // prefetch accuracy not too bad, or low bandwidth requirement
    // ==> maximum prefetch aggressiveness
    pt.mshr_threshold = MSHR_THRESHOLD_MAX;
  } else if (pt.llc_rate < BANDWIDTH) {
    // LLC access rate exceeds memory bandwidth, implying that there are some LLC hits.
    // If there are more LLC misses than hits, perhaps memory bandwidth saturates.
    // If there are more LLC hits than misses, the MSHR is probably not stressed.
    // So we set the MSHR threshold low.
    pt.mshr_threshold = MSHR_THRESHOLD_MIN;
  } else {
    // in-between situation: we set the MSHR threshold proportionally to the (inverse) LLC rate
    pt.mshr_threshold = MSHR_THRESHOLD_MIN + (MSHR_THRESHOLD_MAX - MSHR_THRESHOLD_MIN) * (double)(pt.llc_rate - BANDWIDTH) / BANDWIDTH;
  }
}

// The pt_llc_access function estimates the average time between consecutive LLC accesses.
// It is called on every LLC access.

void bop::pt_llc_access()
{
  // update the gauge
  int cycle = TRUNCATE(get_current_cycle(), TIME_BITS);
  int dt = TRUNCATE(cycle - pt.last_cycle, TIME_BITS);
  pt.last_cycle = cycle;
  pt.llc_rate_gauge += dt - pt.llc_rate;

  // if the gauge reaches its upper limit, increment the rate counter
  // if the gauge reaches its lower limit, decrement the rate counter
  // otherwise leave the rate counter unchanged
  if (pt.llc_rate_gauge > GAUGE_MAX) {
    pt.llc_rate_gauge = GAUGE_MAX;
    if (pt.llc_rate < LLC_RATE_MAX) {
      pt.llc_rate++;
      pt_update_mshr_threshold();
    }
  } else if (pt.llc_rate_gauge < 0) {
    pt.llc_rate_gauge = 0;
    if (pt.llc_rate > 0) {
      pt.llc_rate--;
      pt_update_mshr_threshold();
    }
  }
}

//######################################################################################
//                               OFFSETS SCORES (OS)
//######################################################################################

// A method for determining the best offset value

void bop::os_reset()
{
  int i;
  for (i = 0; i < NOFFSETS; i++) {
    os.score[i] = 0;
  }
  os.max_score = 0;
  os.best_offset = 0;
  os.round = 0;
  os.p = 0;
}

// The os_learn_best_offset function tests one offset at a time, trying to determine
// if the current line would have been successfully prefetched with that offset

void bop::os_learn_best_offset(t_addr lineaddr)
{
  int testoffset = OFFSET[os.p];
  t_addr testlineaddr = lineaddr - testoffset;

  if (SAMEPAGE(lineaddr, testlineaddr) && rr_hit(testlineaddr)) {
    // the current line would likely have been prefetched successfully with that offset
    // ==> increment the score
    os.score[os.p]++;
    if (os.score[os.p] >= os.max_score) {
      os.max_score = os.score[os.p];
      os.best_offset = testoffset;
    }
  }

  if (os.p == (NOFFSETS - 1)) {
    // one round finished
    os.round++;

    if ((os.max_score == SCORE_MAX) || (os.round == ROUND_MAX)) {
      // learning phase is finished, update the prefetch offset
      prefetch_offset = (os.best_offset != 0) ? os.best_offset : DEFAULT_OFFSET;
      pt.prefetch_score = os.max_score;
      pt_update_mshr_threshold();

      if (os.max_score <= BAD_SCORE) {
        // prefetch accuracy is likely to be very low ==> turn the prefetch off
        prefetch_offset = 0;
      }
      // new learning phase starts
      os_reset();
      return;
    }
  }
  INCREMENT(os.p, NOFFSETS); // prepare to test the next offset
}

//######################################################################################
//                               OFFSET PREFETCHER
//######################################################################################

// Issue at most one prefetch request. The prefetch line address is obtained by adding
// the prefetch offset to the current line address

int bop::issue_prefetch(t_addr lineaddr, int offset)
{
  if (offset == 0) {
    // The prefetcher is currently turned off.
    // Just push the line address into the delay queue for best-offset learning.
    dq_push(lineaddr);
    return 0;
  }
  if (!SAMEPAGE(lineaddr, lineaddr + offset)) {
    // crossing the page boundary, no prefetch request issued
    return 0;
  }
  if ((int)intern_->get_mshr_occupancy() < pt.mshr_threshold) { // PORT
    // prefetch into L2
    dq_push(lineaddr);
    bool issued = prefetch_line(champsim::address{(uint64_t)(lineaddr + offset) << LOGLINE}, true, 0); // PORT
    if (issued) {
      // PORT: remember the in-flight L2 prefetch (see bop.h)
      if (inflight.insert(lineaddr + offset).second) {
        inflight_order.push_back(lineaddr + offset);
        if (inflight_order.size() > INFLIGHT_MAX) {
          inflight.erase(inflight_order.front());
          inflight_order.pop_front();
        }
      }
    }
    return issued;
  }
  // could not prefetch into L2
  // try to prefetch into LLC if prefetch accuracy not too bad
  if (pt.prefetch_score > LOW_SCORE) {
    return prefetch_line(champsim::address{(uint64_t)(lineaddr + offset) << LOGLINE}, false, 0); // PORT
  }
  return 0;
}

//######################################################################################
//                               CHAMPSIM INTERFACE (DPC2 INTERFACE in the original)
//######################################################################################

void bop::prefetcher_initialize()
{
  MSHR_THRESHOLD_MAX = (int)intern_->get_mshr_size() - 4; // PORT: (L2_MSHR_COUNT-4)
  prefetch_offset = DEFAULT_OFFSET;
  rr_init();
  os_reset();
  dq_init();
  pt_init();
  prefetch_bit.clear();
}

uint32_t bop::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                       uint32_t metadata_in)
{
  t_addr lineaddr = (t_addr)(addr.to<uint64_t>() >> LOGLINE);

  int l2_hit = cache_hit; // PORT: DPC-2 looked the line up with l2_get_set / l2_get_way
  int prefetched = 0;

  if (l2_hit) {
    // read the prefetch bit, and reset it
    prefetched = (int)prefetch_bit.erase(lineaddr);
  } else {
    pt_llc_access();
  }

  dq_pop();

  int prefetch_issued = 0;

  if (!l2_hit || prefetched) {
    os_learn_best_offset(lineaddr);
    prefetch_issued = issue_prefetch(lineaddr, prefetch_offset);
    if (prefetch_issued) {
      // assume the prefetch request is a L2 miss (we don't know actually)
      pt_llc_access();
    }
  }
  return metadata_in;
}

uint32_t bop::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  // In this version of the DPC2 simulator, the "prefetch" boolean passed
  // as input here is not reset whenever a demand request hits in the L2
  // MSHR on an in-flight prefetch request. Fortunately, this is the information
  // we need for updating the RR table for best-offset learning.
  // However, the prefetch bit stored in the L2 is not completely accurate
  // (though hopefully this does not impact performance too much).
  // In a real hardware implementation of the BO prefetcher, we would distinguish
  // "prefetched" and "demand-requested", which are independent informations.

  t_addr lineaddr = (t_addr)(addr.to<uint64_t>() >> LOGLINE);

  // PORT: recover the DPC-2 meaning of "prefetch" (this fill was started by our prefetch)
  bool was_prefetch = prefetch || inflight.erase(lineaddr);

  // write the prefetch bit
  if (evicted_addr != champsim::address{})
    prefetch_bit.erase((t_addr)(evicted_addr.to<uint64_t>() >> LOGLINE)); // PORT: the victim's bit leaves with it
  if (was_prefetch)
    prefetch_bit.insert(lineaddr);
  else
    prefetch_bit.erase(lineaddr);

  // write the "right" bank of the RR table
  t_addr baselineaddr;
  if (was_prefetch || (prefetch_offset == 0)) {
    baselineaddr = lineaddr - prefetch_offset;
    if (SAMEPAGE(lineaddr, baselineaddr)) {
      rr_insert_right(baselineaddr);
    }
  }
  return metadata_in;
}
