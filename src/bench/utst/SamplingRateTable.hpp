#ifndef VERNIER_BENCH_UTST_SAMPLINGRATETABLE_HPP
#define VERNIER_BENCH_UTST_SAMPLINGRATETABLE_HPP
/**
 * @file SamplingRateTable.hpp
 * @brief Reads the sampling-rate table both languages are tested against
 * (tools/rust/tests/fixtures/profile_frequency.tsv, which the Rust tools
 * read in place): the rates a request may give, the variable it is applied
 * through, and how a request and the environment agree.
 *
 * Test support, private to the bench unit tests. The table's own header
 * comment states its kinds and fields.
 */

#include <fstream>
#include <string>
#include <vector>

#ifndef VERNIER_SHARED_FIXTURE_DIR
#error "VERNIER_SHARED_FIXTURE_DIR must name tools/rust/tests/fixtures"
#endif

namespace vernier {
namespace bench {
namespace test {

/**
 * @brief The rows of @p kind in the table, each split at every tab, the kind
 * first; a '-' field is returned as it is. Empty when the table cannot be
 * read, which a caller checks by the number of rows it expects.
 */
inline std::vector<std::vector<std::string>> samplingRateRows(const std::string& kind) {
  std::ifstream in(std::string{VERNIER_SHARED_FIXTURE_DIR} + "/profile_frequency.tsv");
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (std::size_t tab = line.find('\t'); tab != std::string::npos;
         start = tab + 1, tab = line.find('\t', start)) {
      fields.push_back(line.substr(start, tab - start));
    }
    fields.push_back(line.substr(start));
    if (fields[0] == kind) {
      rows.push_back(fields);
    }
  }
  return rows;
}

/** @brief A field as the text it stands for: '-' is the empty text. */
inline std::string tableText(const std::string& field) { return field == "-" ? "" : field; }

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_BENCH_UTST_SAMPLINGRATETABLE_HPP
