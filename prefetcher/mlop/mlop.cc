#include "mlop.h"

#include "cache.h"
#include "mlop_dpc3.h"

void mlop::prefetcher_initialize()
{
  // MLOP::prefetch's view of this cache (DPC-3 read CACHE::PQ / CACHE::MSHR directly)
  iface = std::make_shared<mlop_dpc3::CacheIface>();
  iface->pq_occupancy = [this] { return intern_->get_pq_occupancy().back(); };
  iface->pq_size = [this] { return intern_->get_pq_size().back(); };
  iface->mshr_occupancy = [this] { return intern_->get_mshr_occupancy(); };
  iface->mshr_size = [this] { return intern_->get_mshr_size(); };
  // From the L2, FILL_L2 fills this level and FILL_LLC fills only the next one
  iface->prefetch_line = [this](uint64_t pf_address, int fill_level) {
    return prefetch_line(champsim::address{pf_address}, fill_level != mlop_dpc3::FILL_LLC, 0);
  };

  const int BLOCKS_IN_CACHE = (int)(intern_->NUM_SET * intern_->NUM_WAY);
  const int BLOCKS_IN_ZONE = PAGE_SIZE / BLOCK_SIZE;
  const int AMT_SIZE = 32 * BLOCKS_IN_CACHE / BLOCKS_IN_ZONE; /* size of access map table */

  engine = std::make_shared<mlop_dpc3::MLOP>(BLOCKS_IN_ZONE, AMT_SIZE, PREFETCH_DEGREE, NUM_UPDATES, L1D_THRESH, L2C_THRESH, LLC_THRESH,
                                             mlop_dpc3::DEBUG_LEVEL);
}

uint32_t mlop::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                        uint32_t metadata_in)
{
  if (type != access_type::LOAD)
    return metadata_in;

  uint64_t block_number = addr.to<uint64_t>() >> LOG2_BLOCK_SIZE;

  /* check prefetch hit */
  // DPC-3 read block[set][way].prefetch; useful_prefetch is that bit on a demand hit (cleared after this call, as in DPC-3)
  bool prefetch_hit = (cache_hit == 1) && useful_prefetch;

  /* check trigger access */
  bool trigger_access = false;
  if (cache_hit == 0 || prefetch_hit)
    trigger_access = true;

  if (trigger_access)
    /* update MLOP with most recent trigger access */
    engine->access(block_number);

  /* issue prefetches */
  engine->mark(block_number, mlop_dpc3::State::ACCESS);
  engine->prefetch(iface.get(), block_number);

  /* stats */
  engine->track(block_number);

  return metadata_in;
}

uint32_t mlop::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  // ChampSim passes a zero evicted address when the way was invalid (DPC-3 checked block[set][way].valid)
  if (evicted_addr == champsim::address{})
    return metadata_in; /* no eviction */

  uint64_t evicted_block_number = evicted_addr.to<uint64_t>() >> LOG2_BLOCK_SIZE;
  engine->mark(evicted_block_number, mlop_dpc3::State::INIT);

  /* stats */
  engine->track(evicted_block_number);
  return metadata_in;
}

void mlop::prefetcher_final_stats()
{
  std::cout << "=== " << intern_->NAME << " MLOP Prefetcher Statistics ===" << std::endl;
  engine->print_stats();
}
