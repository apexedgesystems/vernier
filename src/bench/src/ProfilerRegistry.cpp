/**
 * @file ProfilerRegistry.cpp
 * @brief ProfilerRegistry singleton, readiness routing and dispatch.
 */

#include "src/bench/inc/ProfilerRegistry.hpp"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <exception>
#include <mutex>
#include <set>
#include <utility>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

namespace {

/** @brief The request @p cfg states, for the registered backend @p name. */
ReadinessRequest requestFor(const std::string& name, const PerfConfig& cfg, ReadinessScope scope,
                            const ReadinessContext& ctx) {
  ReadinessRequest request = readinessRequestFor(cfg, scope, ctx);
  request.backend = name;
  const auto WRAP = ctx.get("VERNIER_EXTERNAL_WRAP");
  request.launch = (WRAP && ProfilerRegistry::canonicalName(*WRAP) == name)
                       ? LaunchContext::RUNNER_WRAPPED
                       : LaunchContext::IN_PROCESS;
  return request;
}

/** @brief The doctor's inventory request: one backend's default mode. */
ReadinessRequest inventoryRequest(const std::string& name) {
  ReadinessRequest request;
  request.backend = name;
  request.scope = ReadinessScope::DEFAULT_INVENTORY;
  return request;
}

/** @brief Length-prefixed field, so no value can imitate a separator. */
void appendField(std::string& key, const std::string& value) {
  key += std::to_string(value.size());
  key += ':';
  key += value;
  key += ';';
}

/**
 * @brief Memo key: every field of the request (its identity), the decision
 * inputs of its context, and the registrations.
 */
std::string memoKey(const ReadinessRequest& request, const std::string& fingerprint,
                    std::uint64_t generation) {
  std::string key = request.identity();
  appendField(key, fingerprint);
  appendField(key, std::to_string(generation));
  return key;
}

bool hasText(const std::string& text) {
  for (const char CH : text) {
    if (std::isspace(static_cast<unsigned char>(CH)) == 0) {
      return true;
    }
  }
  return false;
}

const char* tagFor(EnvReport::Status status) {
  switch (status) {
  case EnvReport::Status::Ok:
    return "[OK]  ";
  case EnvReport::Status::Warning:
    return "[WARN]";
  case EnvReport::Status::Error:
    break;
  }
  return "[FAIL]";
}

/** @brief The status a recorded failure gives the run, as its reports state it. */
std::string failingStatus() {
  return "(exit status " + std::to_string(BENCH_PROFILE_FAILED_EXIT_CODE) + " if the tests pass)";
}

/** @brief One report to stderr: status, who, message, remedy and consequence. */
void printReport(const std::string& who, const EnvReport& report, const std::string& consequence) {
  std::fprintf(stderr, "\n%s Profiler %s: %s\n", tagFor(report.status), who.c_str(),
               report.message.c_str());
  if (!report.hint.empty()) {
    std::fprintf(stderr, "   %s\n", report.hint.c_str());
  }
  if (!consequence.empty()) {
    std::fprintf(stderr, "   %s\n", consequence.c_str());
  }
  std::fprintf(stderr, "\n");
}

/** @brief A run's readiness report, printed once per decision to stderr. */
void printNotice(const std::string& name, const ReadinessResult& result) {
  std::string consequence;
  if (!result.collectionReady()) {
    consequence =
        "Nothing is collected for this request; the run will fail " + failingStatus() + ".";
  } else if (result.report.status == EnvReport::Status::Error) {
    consequence = "Collection proceeds and keeps the raw capture; the analysis is skipped and the "
                  "run will fail " +
                  failingStatus() + ".";
  }
  printReport("'" + name + "'", result.report, consequence);
}

/** @brief A request as the command line states it. */
std::string describeRequest(const ReadinessRequest& request) {
  std::string text = "--profile " + request.backend;
  if (!request.profileArgs.empty()) {
    text += " --profile-args '" + request.profileArgs + "'";
  }
  if (!request.bpfScripts.empty()) {
    text += " --bpf ";
    for (std::size_t i = 0; i < request.bpfScripts.size(); ++i) {
      text += (i == 0 ? "" : ",") + request.bpfScripts[i];
    }
  }
  if (request.analyze) {
    text += " --profile-analyze";
  }
  return text;
}

/** @brief One doctor row. */
void printRow(const std::string& name, const EnvReport& report) {
  std::fprintf(stdout, "  %s %-10s %s\n", tagFor(report.status), name.c_str(),
               report.message.c_str());
  if (!report.hint.empty()) {
    std::fprintf(stdout, "             %s\n", report.hint.c_str());
  }
}

/** @brief Who a failure concerns: the backend, and the case when there is one. */
std::string failureSubject(const std::string& backend, const std::string& test) {
  return test.empty() ? backend : backend + " (" + test + ")";
}

} // namespace

/* ------------------------------- Outcome ------------------------------- */

struct ProfilerRegistry::Outcome {
  std::mutex mutex;
  std::vector<ProfileFailure> failures; ///< In the order they were first reported.
  std::set<std::string> recorded;       ///< Backend, test, stage and message of each.
  std::atomic<bool> created{false};     ///< A backend built a profiler during the run.
};

/* ------------------------------- API ------------------------------- */

ProfilerRegistry::ProfilerRegistry() : outcome_(std::make_unique<Outcome>()) {}

ProfilerRegistry::~ProfilerRegistry() = default;

ProfilerRegistry& ProfilerRegistry::instance() {
  static ProfilerRegistry s_instance;
  return s_instance;
}

void ProfilerRegistry::registerBackend(std::string name, Factory factory, EnvCheck check,
                                       std::string unavailableHint) {
  Entry entry;
  entry.factory = std::move(factory);
  entry.check = std::move(check);
  entry.unavailableHint = std::move(unavailableHint);
  backends_[std::move(name)] = std::move(entry);
  ++generation_;
}

void ProfilerRegistry::registerReadinessBackend(std::string name, ReadinessCheck check,
                                                PlannedFactory factory, std::string unavailableHint,
                                                std::vector<std::string> contextKeys) {
  Entry entry;
  entry.readiness = std::move(check);
  entry.planned = std::move(factory);
  entry.unavailableHint = std::move(unavailableHint);
  entry.contextKeys = std::move(contextKeys);
  backends_[std::move(name)] = std::move(entry);
  ++generation_;
}

bool ProfilerRegistry::registerReadinessBackendIfAbsent(std::string name, ReadinessCheck check,
                                                        PlannedFactory factory,
                                                        std::string unavailableHint,
                                                        std::vector<std::string> contextKeys) {
  if (backends_.count(name) != 0) {
    return false;
  }
  registerReadinessBackend(std::move(name), std::move(check), std::move(factory),
                           std::move(unavailableHint), std::move(contextKeys));
  return true;
}

bool ProfilerRegistry::unregisterBackend(const std::string& name) {
  const bool REMOVED = backends_.erase(name) > 0;
  if (REMOVED) {
    ++generation_;
  }
  return REMOVED;
}

std::string ProfilerRegistry::canonicalName(const std::string& name) {
  // Friendly aliases for the names users actually type. `nsys` is what
  // the tool is called everywhere outside this registry; the backend
  // registered itself as `nsight` (it wraps nsys and pairs with ncu).
  if (name == "nsys") {
    return "nsight";
  }
  return name;
}

ReadinessResult ProfilerRegistry::decide(const std::string& name, const Entry& entry,
                                         const ReadinessRequest& request,
                                         const ReadinessContext& ctx) const {
  ReadinessResult result;
  try {
    if (entry.readiness) {
      result = entry.readiness(request, ctx);
    } else if (request.scope == ReadinessScope::RUNTIME &&
               request.launch == LaunchContext::RUNNER_WRAPPED) {
      // A zero-argument check may start helpers of its own, which would run
      // under the wrap; the wrapped run's completion evidence decides instead.
      result = readinessResult(
          ReadinessCause::UNVERIFIED,
          "collection is owned by the " + name + " wrap; completion is checked at exit", "");
    } else if (!entry.check) {
      result = readinessResult(ReadinessCause::UNVERIFIED,
                               "no environment check is registered for " + name, "");
    } else {
      const EnvReport LEGACY = entry.check();
      if (request.scope != ReadinessScope::DEFAULT_INVENTORY &&
          LEGACY.status == EnvReport::Status::Ok && hasText(request.profileArgs)) {
        // The check verified the default mode, not the requested one.
        result = readinessResult(ReadinessCause::UNVERIFIED,
                                 name + " checks its default mode only; '" + request.profileArgs +
                                     "' was not checked",
                                 "");
      } else {
        result.report = LEGACY;
        result.cause =
            LEGACY.status == EnvReport::Status::Ok
                ? ReadinessCause::READY
                : (LEGACY.status == EnvReport::Status::Warning ? ReadinessCause::CAVEAT
                                                               : ReadinessCause::UNUSABLE);
      }
    }
  } catch (const std::exception& e) {
    result = readinessResult(ReadinessCause::INTERNAL,
                             "the " + name + " check failed: " + std::string{e.what()}, "");
  } catch (...) {
    result = readinessResult(ReadinessCause::INTERNAL, "the " + name + " check failed", "");
  }
  if (!result.context) {
    result.context = std::make_shared<const ReadinessContext>(ctx);
  }
  return result;
}

ReadinessResult ProfilerRegistry::checkRequest(const ReadinessRequest& request) const {
  return checkRequest(request, ReadinessContext::capture());
}

ReadinessResult ProfilerRegistry::checkRequest(const ReadinessRequest& requestIn,
                                               const ReadinessContext& ctx) const {
  ReadinessRequest request = requestIn;
  request.backend = canonicalName(request.backend);
  const auto IT = backends_.find(request.backend);
  if (IT == backends_.end()) {
    return unknownResult(request.backend, ctx);
  }
  return decide(request.backend, IT->second, request, ctx);
}

ReadinessResult ProfilerRegistry::unknownResult(const std::string& name,
                                                const ReadinessContext& ctx) const {
  std::string available;
  for (const auto& [n, _] : backends_) {
    available += (available.empty() ? "" : ", ") + n;
  }
  ReadinessResult unknown;
  unknown.report = EnvReport{EnvReport::Status::Error, "unknown profiler '" + name + "'",
                             "Available: " + available + "."};
  unknown.cause = ReadinessCause::MISSING;
  unknown.context = std::make_shared<const ReadinessContext>(ctx);
  return unknown;
}

void ProfilerRegistry::resetReadiness() { memo_.reset(); }

bool ProfilerRegistry::recordFailure(const std::string& backend, const std::string& test,
                                     const ReadinessResult& failure) const {
  if (failure.report.status != EnvReport::Status::Error) {
    return false;
  }
  std::string key;
  appendField(key, backend);
  appendField(key, test);
  appendField(key, std::to_string(static_cast<int>(failure.stage)));
  appendField(key, failure.report.message);
  const std::lock_guard<std::mutex> LOCK(outcome_->mutex);
  if (!outcome_->recorded.insert(key).second) {
    return false;
  }
  outcome_->failures.push_back(ProfileFailure{backend, test, failure});
  return true;
}

void ProfilerRegistry::reportFailure(const std::string& backend, const std::string& test,
                                     const ReadinessResult& failure) const noexcept {
  try {
    if (recordFailure(backend, test, failure)) {
      const std::string WHO = "'" + backend + "'" + (test.empty() ? "" : " (" + test + ")");
      printReport(WHO, failure.report, "The run will fail " + failingStatus() + ".");
    }
  } catch (...) {
    // Reporting must not end the run: a failure that cannot be recorded
    // (out of memory) is dropped rather than thrown into a profiler hook.
  }
}

void ProfilerRegistry::announce(const std::string& backend, const ReadinessResult& decision,
                                const std::string& noticeKey) const {
  if (decision.report.status != EnvReport::Status::Ok && memo_.claimNotice(noticeKey)) {
    printNotice(backend, decision);
  }
  (void)recordFailure(backend, "", decision);
}

bool ProfilerRegistry::reportDecision(const std::string& backend,
                                      const ReadinessResult& decision) const noexcept {
  try {
    announce(backend, decision, "direct;" + backend + ";" + decision.report.message);
  } catch (...) {
    // As reportFailure(): a report that cannot be made does not end the run.
  }
  return decision.collectionReady();
}

std::vector<ProfileFailure> ProfilerRegistry::failures() const {
  const std::lock_guard<std::mutex> LOCK(outcome_->mutex);
  return outcome_->failures;
}

void ProfilerRegistry::resetFailures() {
  const std::lock_guard<std::mutex> LOCK(outcome_->mutex);
  outcome_->failures.clear();
  outcome_->recorded.clear();
  outcome_->created.store(false);
}

int ProfilerRegistry::finishRun(const PerfConfig& cfg, int testStatus, int testsRun) const {
  const std::string NAME = canonicalName(cfg.profileTool);
  const ReadinessContext CTX = ReadinessContext::capture();
  if (!NAME.empty() && NAME != "cupti" && backends_.find(NAME) == backends_.end()) {
    // An unknown name fails the run even when no case asked for a profiler.
    const ReadinessResult UNKNOWN = unknownResult(NAME, CTX);
    if (memo_.claimNotice("unknown;" + NAME)) {
      printNotice(NAME, UNKNOWN);
    }
    (void)recordFailure(NAME, "", UNKNOWN);
  }

  const std::vector<ProfileFailure> FAILED = failures();
  const std::string REQUEST =
      NAME.empty() ? std::string{"a profile"}
                   : describeRequest(requestFor(NAME, cfg, ReadinessScope::RUNTIME, CTX));
  int status = testStatus;
  if (!FAILED.empty()) {
    if (status == 0) {
      status = BENCH_PROFILE_FAILED_EXIT_CODE;
      std::fprintf(stderr, "\n[profile] %s failed; the run exits with status %d:\n",
                   REQUEST.c_str(), status);
    } else {
      std::fprintf(stderr, "\n[profile] %s failed; the run exits with the tests' status %d:\n",
                   REQUEST.c_str(), status);
    }
    for (const ProfileFailure& failure : FAILED) {
      std::fprintf(stderr, "[profile]   %s: %s\n",
                   failureSubject(failure.backend, failure.test).c_str(),
                   failure.result.report.message.c_str());
    }
    std::fprintf(stderr, "\n");
  } else if (!NAME.empty() && NAME != "cupti" && testsRun > 0 && !outcome_->created.load()) {
    const bool WRAPPED =
        requestFor(NAME, cfg, ReadinessScope::RUNTIME, CTX).launch == LaunchContext::RUNNER_WRAPPED;
    if (WRAPPED) {
      std::fprintf(stderr,
                   "\n[profile] %s: no case that ran was built with the profiler guard; the %s "
                   "wrap still recorded the whole process.\n\n",
                   REQUEST.c_str(), NAME.c_str());
    } else {
      std::fprintf(stderr,
                   "\n[profile] %s: no case that ran was built with the profiler guard, so "
                   "nothing was profiled.\n\n",
                   REQUEST.c_str());
    }
  }
  std::fflush(stderr);
  return status;
}

std::unique_ptr<Profiler> ProfilerRegistry::make(const std::string& rawName, const PerfConfig& cfg,
                                                 const std::string& testName) const {
  return make(rawName, cfg, testName, ReadinessContext::capture());
}

std::unique_ptr<Profiler> ProfilerRegistry::make(const std::string& rawName, const PerfConfig& cfg,
                                                 const std::string& testName,
                                                 const ReadinessContext& ctx) const {
  const std::string name = canonicalName(rawName);
  if (name == "cupti") {
    // Not a wrap: per-kernel CUPTI columns are always collected in GPU
    // builds. Say so instead of "unknown profiler".
    std::fprintf(stderr, "\n[INFO] 'cupti' needs no --profile: per-kernel columns "
                         "(kernelTimeUs, cuptiKernelLaunches) are always collected in GPU "
                         "builds. Proceeding unprofiled.\n\n");
    return std::make_unique<detail::NoOpProfiler>(name, "");
  }
  const auto it = backends_.find(name);
  if (it == backends_.end()) {
    const ReadinessResult UNKNOWN = unknownResult(name, ctx);
    if (memo_.claimNotice("unknown;" + name)) {
      printNotice(name, UNKNOWN);
    }
    (void)recordFailure(name, "", UNKNOWN);
    return std::make_unique<detail::NoOpProfiler>();
  }
  const Entry& entry = it->second;

  // One decision per request and context, shared by every case of the run
  // and identical to the doctor's selected row. An Error at either stage is
  // the run's failure.
  const ReadinessRequest REQUEST = requestFor(name, cfg, ReadinessScope::RUNTIME, ctx);
  const std::string KEY = memoKey(REQUEST, ctx.fingerprint(entry.contextKeys), generation_);
  const ReadinessResult RESULT =
      memo_.getOrCompute(KEY, [&] { return decide(name, entry, REQUEST, ctx); });
  announce(name, RESULT, KEY);
  if (!RESULT.collectionReady()) {
    return std::make_unique<detail::NoOpProfiler>();
  }

  std::unique_ptr<Profiler> p;
  if (entry.planned) {
    p = entry.planned(cfg, testName, RESULT);
  } else if (entry.factory) {
    p = entry.factory(cfg, testName);
  }
  if (p) {
    outcome_->created.store(true);
    return p;
  }

  // A build or platform guard in the factory: nothing can collect here.
  const ReadinessResult UNAVAILABLE = readinessResult(
      ReadinessCause::UNUSABLE, name + " is unavailable on this platform", entry.unavailableHint);
  if (memo_.claimNotice("unavailable;" + KEY)) {
    printNotice(name, UNAVAILABLE);
  }
  (void)recordFailure(name, "", UNAVAILABLE);
  return std::make_unique<detail::NoOpProfiler>();
}

bool ProfilerRegistry::hasBackend(const std::string& name) const noexcept {
  return backends_.find(canonicalName(name)) != backends_.end();
}

std::vector<std::string> ProfilerRegistry::backendNames() const {
  std::vector<std::string> names;
  names.reserve(backends_.size());
  for (const auto& [n, _] : backends_) {
    names.push_back(n);
  }
  return names;
}

EnvReport ProfilerRegistry::runCheck(const std::string& rawName) const {
  return checkRequest(inventoryRequest(canonicalName(rawName))).report;
}

std::vector<std::pair<std::string, EnvReport>> ProfilerRegistry::runAllChecks() const {
  const ReadinessContext CTX = ReadinessContext::capture();
  std::vector<std::pair<std::string, EnvReport>> out;
  out.reserve(backends_.size());
  for (const auto& [n, e] : backends_) {
    out.emplace_back(n, decide(n, e, inventoryRequest(n), CTX).report);
  }
  return out;
}

int ProfilerRegistry::printDoctor() const { return printDoctor(PerfConfig{}); }

int ProfilerRegistry::printDoctor(const PerfConfig& cfg) const {
  const ReadinessContext CTX = ReadinessContext::capture();
  std::fprintf(stdout, "\n=== Profiler Backend Doctor (default mode of each backend) ===\n\n");
  int fails = 0;
  for (const auto& [NAME, ENTRY] : backends_) {
    const EnvReport REPORT = decide(NAME, ENTRY, inventoryRequest(NAME), CTX).report;
    if (REPORT.status == EnvReport::Status::Error) {
      ++fails;
    }
    printRow(NAME, REPORT);
  }
  std::fprintf(stdout, "\n  %zu backend(s), %d fail.\n", backends_.size(), fails);
  if (!cfg.profileTool.empty()) {
    const std::string NAME = canonicalName(cfg.profileTool);
    const ReadinessRequest REQUEST = requestFor(NAME, cfg, ReadinessScope::PREFLIGHT, CTX);
    std::fprintf(stdout, "\n  Selected request: %s\n", describeRequest(REQUEST).c_str());
    printRow(NAME, checkRequest(REQUEST, CTX).report);
  }
  std::fprintf(stdout,
               "\n  Each row checks one backend's default mode for this user and environment.\n"
               "  A run checks its own --profile request when its profiler is created, and\n"
               "  only cases built with the profiler guard create one. Add --profile <name>\n"
               "  [--profile-args <args>] to --profile-check to check one request.\n"
               "  --require accepts only [OK]; a run may proceed with a [WARN] caveat.\n\n");
  return fails;
}

} // namespace bench
} // namespace vernier
