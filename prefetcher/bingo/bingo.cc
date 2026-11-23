#include "bingo.h"

#include "bingo_dpc3.h"
#include "cache.h"

void bingo::prefetcher_initialize()
{
  // PrefetchStreamer's view of this cache (DPC-3 read CACHE::PQ / CACHE::MSHR directly)
  iface = std::make_shared<bingo_dpc3::CacheIface>();
  iface->pq_occupancy = [this] { return intern_->get_pq_occupancy().back(); };
  iface->pq_size = [this] { return intern_->get_pq_size().back(); };
  iface->mshr_occupancy = [this] { return intern_->get_mshr_occupancy(); };
  iface->mshr_size = [this] { return intern_->get_mshr_size(); };
  // From the L2, FILL_L2 fills this level and FILL_LLC fills only the next one
  iface->prefetch_line = [this](uint64_t pf_address, int fill_level) {
    return prefetch_line(champsim::address{pf_address}, fill_level != bingo_dpc3::FILL_LLC, 0);
  };

  engine = std::make_shared<bingo_dpc3::Bingo>(REGION_SIZE >> LOG2_BLOCK_SIZE, MIN_ADDR_WIDTH, MAX_ADDR_WIDTH, PC_WIDTH, FT_SIZE, AT_SIZE, PHT_SIZE,
                                               PHT_WAYS, PF_STREAMER_SIZE, bingo_dpc3::DEBUG_LEVEL);
}

uint32_t bingo::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                         uint32_t metadata_in)
{
  if (type != access_type::LOAD)
    return metadata_in;

  uint64_t block_number = addr.to<uint64_t>() >> LOG2_BLOCK_SIZE;

  /* update BINGO with most recent LOAD access */
  engine->access(block_number, ip.to<uint64_t>());

  /* issue prefetches */
  engine->prefetch(iface.get(), block_number);

  return metadata_in;
}

uint32_t bingo::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  // ChampSim passes a zero evicted address when the way was invalid (DPC-3 checked block[set][way].valid)
  if (evicted_addr == champsim::address{})
    return metadata_in; /* no eviction */

  engine->eviction(evicted_addr.to<uint64_t>() >> LOG2_BLOCK_SIZE);
  return metadata_in;
}

void bingo::prefetcher_final_stats() {}
