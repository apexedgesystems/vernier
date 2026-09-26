/**
 * @file CuptiCollectorDecision_uTest.cpp
 * @brief The CUPTI collector applies the shared decision before it registers:
 *        a setting that stands it down, or that is rejected, registers nothing.
 *
 * Built against the counting stand-in in fake_cupti/ instead of CUPTI (first on
 * this target's include path), so it runs on any machine: it compiles the real
 * CuptiCollector.cu into this test and counts cuptiActivityRegisterCallbacks.
 * Each test sets or clears the variables it depends on and restores them.
 */

#include "src/bench/src/CuptiCollector.cu" // the real source, on the counting stand-in

#include "src/bench/utst/ScopedEnv.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>

using vernier::bench::CuptiCollector;
using vernier::bench::test::ScopedEnv;

namespace {

/// Every value the common grammar reads as true, false, or neither.
const char* const TRUE_SPELLINGS[] = {"1", "true", "TRUE", "True", "yes", "YES", "on", "On"};
const char* const FALSE_SPELLINGS[] = {"0", "false", "FALSE", "no", "No", "off", "OFF", ""};
const char* const INVALID_VALUES[] = {"2", "-1", "maybe", "disable", "truee", "y", " 1", "1 "};

} // namespace

/** @brief No session, no override, and no registration counted yet. */
class CuptiCollectorDecision : public ::testing::Test {
protected:
  void SetUp() override {
    for (const char* name : {"VERNIER_DISABLE_CUPTI", "VERNIER_EXTERNAL_WRAP",
                             "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE"}) {
      scrub_[count_++].emplace(name, nullptr);
    }
    fake_cupti::calls() = {};
  }

  /** @brief Registrations made while building one collector for @p value, or -1 when it threw. */
  static int registrationsFor(const char* value, bool forceDisabled = false) {
    const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", value);
    fake_cupti::calls() = {};
    try {
      const CuptiCollector COLLECTOR(forceDisabled);
      return fake_cupti::calls().registrations;
    } catch (const std::invalid_argument&) {
      return fake_cupti::calls().registrations == 0 ? -1 : -2;
    }
  }

private:
  std::optional<ScopedEnv> scrub_[4];
  int count_ = 0;
};

/** @test Without a session or an override the collector registers once, and is available. */
TEST_F(CuptiCollectorDecision, RegistersWithoutASessionOrOverride) {
  const CuptiCollector COLLECTOR;
  EXPECT_TRUE(COLLECTOR.isAvailable());
  EXPECT_EQ(fake_cupti::calls().registrations, 1);
}

/** @test Every false spelling, in any case, and the empty value leave it registering. */
TEST_F(CuptiCollectorDecision, EveryFalseSpellingKeepsItRegistering) {
  for (const char* value : FALSE_SPELLINGS) {
    EXPECT_EQ(registrationsFor(value), 1) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test Every true spelling, in any case, stands it down before it registers. */
TEST_F(CuptiCollectorDecision, EveryTrueSpellingStandsItDownFirst) {
  for (const char* value : TRUE_SPELLINGS) {
    EXPECT_EQ(registrationsFor(value), 0) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test Inside a session a false value does not keep it on: nothing registers. */
TEST_F(CuptiCollectorDecision, ASessionWinsOverFalse) {
  const ScopedEnv SESSION("NSYS_PROFILING_SESSION_ID", "1017521");
  for (const char* value : FALSE_SPELLINGS) {
    EXPECT_EQ(registrationsFor(value), 0) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test Any other value is a configuration error, raised before anything registers. */
TEST_F(CuptiCollectorDecision, AnInvalidValueIsRejectedBeforeRegistering) {
  for (const char* value : INVALID_VALUES) {
    EXPECT_EQ(registrationsFor(value), -1)
        << "VERNIER_DISABLE_CUPTI='" << value << "' was not rejected before registering";
  }
  const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", "maybe");
  try {
    const CuptiCollector COLLECTOR;
    FAIL() << "an invalid value was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(std::string(e.what()).rfind("configuration: VERNIER_DISABLE_CUPTI='maybe' is not a "
                                          "boolean. Use 1, true, yes or on",
                                          0),
              0U)
        << e.what();
  }
}

/** @test An invalid value is still rejected inside a session: the error is never silent. */
TEST_F(CuptiCollectorDecision, AnInvalidValueIsRejectedInsideASession) {
  const ScopedEnv SESSION("NV_NSIGHT_INJECTION_PORT_BASE", "49152");
  EXPECT_EQ(registrationsFor("maybe"), -1);
}

/** @test An explicit forceDisabled wins without reading the setting: no error, no registration. */
TEST_F(CuptiCollectorDecision, ForceDisabledWinsWithoutReadingTheSetting) {
  EXPECT_EQ(registrationsFor("maybe", /*forceDisabled=*/true), 0);
  EXPECT_EQ(registrationsFor(nullptr, /*forceDisabled=*/true), 0);
}
