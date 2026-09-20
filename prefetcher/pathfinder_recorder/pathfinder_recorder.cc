#include "pathfinder_recorder.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
std::string record_file_path()
{
  if (const char* env = std::getenv("PATHFINDER_RECORD_FILE"))
    return std::string{env};
  return "pathfinder_record.txt";
}
} // namespace

void pathfinder_recorder::prefetcher_initialize()
{
  const auto path = record_file_path();
  outfile.open(path);
  if (!outfile.is_open())
    std::cerr << "[pathfinder_recorder] WARNING: could not open \"" << path << "\" for writing (set PATHFINDER_RECORD_FILE)" << std::endl;
  else
    std::cout << "[pathfinder_recorder] recording demand loads to \"" << path << "\"" << std::endl;
}

void pathfinder_recorder::prefetcher_cache_operate_with_id(champsim::address addr, champsim::address ip, bool cache_hit, access_type type,
                                                            uint64_t instr_id, uint64_t cycle)
{
  if (type != access_type::LOAD || instr_id == 0)
    return;

  if (outfile.is_open())
    outfile << instr_id << ", " << cycle << ", " << std::hex << addr.to<uint64_t>() << ", " << ip.to<uint64_t>() << std::dec << ", "
            << static_cast<int>(cache_hit) << "\n";
  ++row_id;
}

void pathfinder_recorder::prefetcher_final_stats()
{
  if (outfile.is_open())
    outfile.flush();
  std::cout << "[pathfinder_recorder] recorded " << row_id << " rows" << std::endl;
}
