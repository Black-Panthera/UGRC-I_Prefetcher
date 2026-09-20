#include "pathfinder_file.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace
{
// Path to the translated PATHFINDER prediction file. Kept configurable via an
// environment variable so no ChampSim core/config-schema changes are needed.
std::string predictions_file_path()
{
  if (const char* env = std::getenv("PATHFINDER_PREDICTIONS_FILE"))
    return std::string{env};
  return "pathfinder_predictions.txt";
}
} // namespace

void pathfinder_file::prefetcher_initialize()
{
  const auto path = predictions_file_path();
  std::ifstream infile{path};
  if (!infile.is_open()) {
    std::cerr << "[pathfinder_file] WARNING: could not open predictions file \"" << path
              << "\" (set PATHFINDER_PREDICTIONS_FILE). No prefetches will be issued." << std::endl;
    return;
  }

  std::string line;
  std::size_t loaded_predictions = 0;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss{line};
    uint64_t instr_id;
    std::string pf_hex;
    if (!(iss >> instr_id >> pf_hex))
      continue;

    predictions[instr_id].push_back(std::stoull(pf_hex, nullptr, 16));
    ++loaded_predictions;
  }

  std::cout << "[pathfinder_file] loaded " << loaded_predictions << " predictions for " << predictions.size() << " instr_ids from \"" << path << "\""
            << std::endl;
}

void pathfinder_file::prefetcher_cache_operate_with_id(champsim::address addr, champsim::address ip, bool cache_hit, access_type type, uint64_t instr_id,
                                                       uint64_t cycle)
{
  static const bool debug = std::getenv("PATHFINDER_DEBUG") != nullptr;
  if (type != access_type::LOAD || instr_id == 0)
    return;

  auto it = predictions.find(instr_id);
  if (it == predictions.end())
    return;

  if (debug)
    std::cerr << "[pathfinder_file] instr_id=" << instr_id << " -> " << it->second.size() << " prefetch(es):";
  for (const auto pf_addr : it->second) {
    if (debug)
      std::cerr << " " << std::hex << pf_addr;
    prefetch_line(champsim::address{pf_addr}, true, 0);
  }
  if (debug)
    std::cerr << std::dec << std::endl;
}
