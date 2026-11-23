#include "lmop3g.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "cache.h"

namespace
{
int nl_group(uint64_t page) { return static_cast<int>(((page * 0x9E3779B97F4A7C15ull) >> 58) % 64); }
} // namespace

long long lmop3g::now() const { return intern_->current_time.time_since_epoch() / intern_->clock_period; }

int lmop3g::policy_of(uint64_t line) const
{
  if constexpr (FIXED_POLICY >= 0)
    return FIXED_POLICY;
  const int g = static_cast<int>((line % intern_->NUM_SET) % LEADER_STRIDE);
  return g < POLICIES ? g : winner;
}

int lmop3g::decide(int policy, const pick& p) const
{
  switch (policy) {
  case 0:
    return p.s >= 0.50 ? 1 : 0;
  case 1:
    return p.s >= 0.50 ? 1 : (p.s >= 0.30 && p.rank < 16 ? 2 : 0);
  case 2:
    return p.s >= 0.50 ? 1 : (p.s >= 0.15 ? 2 : 0);
  default:
    return p.s >= 0.30 && p.rank < 16 ? 1 : 0;
  }
}

void lmop3g::end_round()
{
  std::vector<int> order;
  for (int o = -(LINES - 1); o <= LINES - 1; ++o)
    if (o != 0 && score[o + LINES - 1] >= MIN_SCORE * ROUND)
      order.push_back(o);
  std::sort(order.begin(), order.end(), [this](int a, int b) { return score[a + LINES - 1] > score[b + LINES - 1]; });
  if (static_cast<int>(order.size()) > MAX_OFFSETS)
    order.resize(MAX_OFFSETS);
  picks.clear();
  for (int r = 0; r < static_cast<int>(order.size()); ++r)
    picks.push_back({order[r], double(score[order[r] + LINES - 1]) / ROUND, r});
  std::sort(picks.begin(), picks.end(), [](const pick& a, const pick& b) { return std::abs(a.offset) < std::abs(b.offset); });
  picked_sum += picks.size();
  score.fill(0);
  updates = 0;
  ++rounds;
}

uint32_t lmop3g::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                         uint32_t metadata_in)
{
  if (type != access_type::LOAD && type != access_type::RFO)
    return metadata_in;

  const uint64_t block = addr.to<uint64_t>() >> LOG2_BLOCK_SIZE;
  const uint64_t page = block / LINES;
  const int x = static_cast<int>(block % LINES);
  const uint64_t bit = uint64_t{1} << x;
  ++tick;

  if (FIXED_POLICY < 0 && !cache_hit && static_cast<int>((block % intern_->NUM_SET) % LEADER_STRIDE) < POLICIES)
    pending.emplace(block, now());

  // Next-line fill level dueling (TLS2): a demand miss in a leader page counts against that leader's fill level
  const int group = nl_group(page);
  if (NL_DEFAULT && !cache_hit) {
    if (group == 0)
      psel = std::min(psel + 1, PSEL_MAX);
    else if (group == 1)
      psel = std::max(psel - 1, 0);
  }

  entry* e = nullptr;
  for (auto& t : table)
    if (t.valid && t.page == page)
      e = &t;

  const bool fresh = e == nullptr;
  if (e == nullptr) {
    e = &table[0];
    for (auto& t : table)
      if (!t.valid || t.lru < e->lru)
        e = &t;
    *e = entry{};
    e->valid = true;
    e->page = page;
  } else if ((!cache_hit || useful_prefetch) && !(e->acc & bit)) {
    if constexpr (WINDOW > 0) {
      uint64_t seen = bit;
      for (int k = 0; k < e->n_recent; ++k) {
        const int i = e->recent[k];
        if (seen & (uint64_t{1} << i))
          continue;
        seen |= uint64_t{1} << i;
        ++score[x - i + LINES - 1];
      }
    } else {
      for (uint64_t m = e->acc; m; m &= m - 1)
        ++score[x - __builtin_ctzll(m) + LINES - 1];
    }
    if (++updates == ROUND)
      end_round();
  }

  e->lru = tick;
  e->acc |= bit;
  e->req_l2 |= bit;
  e->req_any |= bit;
  if constexpr (WINDOW > 0) {
    if (e->n_recent == 0 || e->recent[(e->head + WINDOW - 1) % WINDOW] != x) {
      e->recent[e->head] = static_cast<int8_t>(x);
      e->head = (e->head + 1) % WINDOW;
      e->n_recent = std::min(e->n_recent + 1, WINDOW);
    }
  }

  auto issue = [&](int target, bool to_l2) {
    if (!prefetch_line(champsim::address{(page * LINES + static_cast<uint64_t>(target)) << LOG2_BLOCK_SIZE}, to_l2, 0)) {
      ++pq_full;
      return false;
    }
    const uint64_t tb = uint64_t{1} << target;
    e->req_any |= tb;
    if (to_l2)
      e->req_l2 |= tb;
    return true;
  };

  int issued_now = 0;
  bool blocked = false;
  for (const auto& p : picks) {
    if (GATE_NEW && fresh)
      break;
    const int target = x + p.offset;
    if (target < 0 || target >= LINES)
      continue;
    const int how = decide(policy_of(page * LINES + static_cast<uint64_t>(target)), p);
    if (how == 0)
      continue;
    const bool to_l2 = how == 1;
    if ((to_l2 ? e->req_l2 : e->req_any) & (uint64_t{1} << target))
      continue;
    if (!issue(target, to_l2)) {
      blocked = true;
      break;
    }
    ++issued_now;
    ++(to_l2 ? issued_l2 : issued_llc);
  }

  if (NL_DEFAULT && issued_now == 0 && !blocked && x + 1 < LINES) {
    const bool to_l2 = group == 0 || (group != 1 && psel <= PSEL_MAX / 2);
    if (!((to_l2 ? e->req_l2 : e->req_any) & (bit << 1)) && issue(x + 1, to_l2))
      ++(to_l2 ? nl_l2 : nl_llc);
  }
  return metadata_in;
}

uint32_t lmop3g::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  if constexpr (FIXED_POLICY < 0) {
    const uint64_t filled = addr.to<uint64_t>() >> LOG2_BLOCK_SIZE;
    if (auto it = pending.find(filled); it != pending.end()) {
      const int g = static_cast<int>((filled % intern_->NUM_SET) % LEADER_STRIDE);
      lat_sum[g] += static_cast<uint64_t>(std::max(0LL, now() - it->second));
      pending.erase(it);
      if (++done == EPOCH_MISSES) {
        winner = static_cast<int>(std::min_element(lat_sum.begin(), lat_sum.end()) - lat_sum.begin());
        ++wins[winner];
        lat_sum.fill(0);
        done = 0;
      }
    }
    if (pending.size() > 4096)
      pending.clear();
  }

  if (evicted_addr != champsim::address{}) {
    const uint64_t block = evicted_addr.to<uint64_t>() >> LOG2_BLOCK_SIZE;
    for (auto& t : table)
      if (t.valid && t.page == block / LINES) {
        const uint64_t b = ~(uint64_t{1} << (block % LINES));
        t.acc &= b;
        t.req_l2 &= b;
        t.req_any &= b;
      }
  }
  return metadata_in;
}

void lmop3g::prefetcher_final_stats()
{
  std::cout << "LMOP3 issued L2 " << issued_l2 << " LLC-only " << issued_llc << " PQ-full " << pq_full << " rounds " << rounds << " mean picks "
            << (rounds ? double(picked_sum) / rounds : 0.0) << " next-line L2 " << nl_l2 << " LLC " << nl_llc << " psel " << psel
            << " policy wins";
  for (auto w : wins)
    std::cout << " " << w;
  std::cout << " final " << winner << std::endl;
}
