/**
 * @file MonitorQueue_uTest.cpp
 * @brief Unit tests for vernier::monitor::MonitorQueue.
 *
 * Notes:
 *  - Tests are platform-agnostic: assert invariants, not exact values.
 */

#include "src/monitor/inc/MonitorQueue.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using vernier::monitor::MonitorQueue;
using vernier::monitor::MonitorTag;
using vernier::monitor::Sample;
using vernier::monitor::SampleKind;

/* ----------------------------- Default Construction ----------------------------- */

/** @test Queue capacity is rounded up to power of two */
TEST(MonitorQueueTest, CapacityRoundedUp) {
  const MonitorQueue Q1(100);
  EXPECT_EQ(Q1.capacity(), 128u);

  const MonitorQueue Q2(128);
  EXPECT_EQ(Q2.capacity(), 128u);

  const MonitorQueue Q3(2);
  EXPECT_EQ(Q3.capacity(), 2u);
}

/* ----------------------------- MonitorQueue Method Tests ----------------------------- */

/** @test Push and pop a single sample */
TEST(MonitorQueueTest, SinglePushPop) {
  MonitorQueue q(4);

  Sample in;
  in.timestampNs = 12345;
  in.tag = MonitorTag("test", 1);
  std::strncpy(in.scope, "work", sizeof(in.scope));
  in.kind = SampleKind::SCOPE;
  in.durationNs = 1000;

  EXPECT_TRUE(q.tryPush(in));

  Sample out;
  EXPECT_TRUE(q.tryPop(out));
  EXPECT_EQ(out.timestampNs, 12345u);
  EXPECT_STREQ(out.tag.name, "test");
  EXPECT_EQ(out.tag.id, 1);
  EXPECT_EQ(out.durationNs, 1000u);
}

/** @test Pop from empty queue returns false */
TEST(MonitorQueueTest, PopEmpty) {
  MonitorQueue q(4);
  Sample out;
  EXPECT_FALSE(q.tryPop(out));
}

/** @test Queue overflow increments dropped counter */
TEST(MonitorQueueTest, Overflow) {
  MonitorQueue q(4); // capacity = 4

  Sample s;
  s.timestampNs = 1;

  // Fill the queue
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(q.tryPush(s));
  }

  // This should fail (queue full)
  EXPECT_FALSE(q.tryPush(s));
  EXPECT_EQ(q.droppedCount(), 1u);

  // Another overflow
  EXPECT_FALSE(q.tryPush(s));
  EXPECT_EQ(q.droppedCount(), 2u);
}

/** @test FIFO ordering preserved */
TEST(MonitorQueueTest, FifoOrder) {
  MonitorQueue q(8);

  for (uint64_t i = 0; i < 5; ++i) {
    Sample s;
    s.timestampNs = i;
    EXPECT_TRUE(q.tryPush(s));
  }

  for (uint64_t i = 0; i < 5; ++i) {
    Sample out;
    EXPECT_TRUE(q.tryPop(out));
    EXPECT_EQ(out.timestampNs, i);
  }
}

/** @test Wraps a capacity-4 queue three times in one thread: in order, full at four, empty after */
TEST(MonitorQueueTest, SingleThreadWrapsThreeLaps) {
  MonitorQueue q(4);
  std::uint64_t next = 0;
  for (int lap = 0; lap < 3; ++lap) {
    for (std::uint64_t i = 0; i < 4; ++i) {
      Sample s;
      s.timestampNs = next + i;
      ASSERT_TRUE(q.tryPush(s)) << "lap " << lap << ", sample " << i;
    }
    EXPECT_FALSE(q.tryPush(Sample{})) << "lap " << lap << ": a full queue took a fifth sample";
    for (std::uint64_t i = 0; i < 4; ++i) {
      Sample out;
      ASSERT_TRUE(q.tryPop(out)) << "lap " << lap << ", sample " << i;
      EXPECT_EQ(out.timestampNs, next + i);
    }
    Sample out;
    EXPECT_FALSE(q.tryPop(out)) << "lap " << lap << ": an empty queue gave a sample";
    next += 4;
  }
  EXPECT_EQ(q.droppedCount(), 3u); // the one refused push of each lap
}

/* ----------------------------- Determinism Tests ----------------------------- */

namespace {

/// What runQueueStress() saw.
struct QueueStress {
  bool finished{false};     ///< every sample arrived before the deadline
  long missing{0};          ///< samples that never arrived
  long duplicates{0};       ///< samples that arrived more than once
  long outOfOrder{0};       ///< a producer's sample a consumer saw after a later one
  long torn{0};             ///< samples whose two fields do not belong together
  long refused{0};          ///< pushes that returned false
  std::uint64_t dropped{0}; ///< the queue's droppedCount()
};

/**
 * @brief Push and pop through a capacity-64 queue from several threads.
 *
 * Each of @p producers pushes @p perProducer samples carrying its id and a
 * sequence number, and a second field derived from both against torn copies,
 * retrying a refused push; @p consumers pop until every sample has arrived.
 * The samples outnumber the slots, so the queue wraps and runs full. A
 * deadline far above the normal run time ends a queue that stops accepting or
 * delivering, which then reports unfinished instead of hanging.
 */
QueueStress runQueueStress(int producers, int consumers, long perProducer) {
  MonitorQueue q(64);
  const long TOTAL = producers * perProducer;
  std::vector<std::atomic<unsigned char>> seen(static_cast<std::size_t>(TOTAL));
  std::atomic<long> popped{0};
  std::atomic<long> refused{0};
  std::atomic<long> outOfOrder{0};
  std::atomic<long> torn{0};
  std::atomic<long> duplicates{0};
  std::atomic<bool> expired{false};
  const auto DEADLINE = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  const auto pastDeadline = [&] {
    if (std::chrono::steady_clock::now() > DEADLINE) {
      expired.store(true);
    }
    return expired.load();
  };

  std::vector<std::thread> threads;
  for (int p = 0; p < producers; ++p) {
    threads.emplace_back([&, p] {
      Sample s;
      s.tag.id = static_cast<std::uint16_t>(p);
      for (long i = 0; i < perProducer; ++i) {
        s.timestampNs = static_cast<std::uint64_t>(i);
        s.durationNs = static_cast<std::uint64_t>(i) * 3 + static_cast<std::uint64_t>(p);
        while (!q.tryPush(s)) {
          refused.fetch_add(1);
          if (pastDeadline()) {
            return;
          }
          std::this_thread::yield();
        }
      }
    });
  }
  for (int c = 0; c < consumers; ++c) {
    threads.emplace_back([&] {
      std::vector<long> last(static_cast<std::size_t>(producers), -1);
      Sample out;
      while (popped.load() < TOTAL) {
        if (!q.tryPop(out)) {
          if (pastDeadline()) {
            return;
          }
          std::this_thread::yield();
          continue;
        }
        popped.fetch_add(1);
        const long P = out.tag.id;
        const long I = static_cast<long>(out.timestampNs);
        if (P >= producers || I < 0 || I >= perProducer ||
            out.durationNs != static_cast<std::uint64_t>(I) * 3 + static_cast<std::uint64_t>(P)) {
          torn.fetch_add(1);
          continue;
        }
        if (I <= last[static_cast<std::size_t>(P)]) {
          outOfOrder.fetch_add(1);
        }
        last[static_cast<std::size_t>(P)] = I;
        if (seen[static_cast<std::size_t>(P * perProducer + I)].fetch_add(1) != 0) {
          duplicates.fetch_add(1);
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  QueueStress result;
  result.finished = !expired.load() && popped.load() == TOTAL;
  for (const auto& count : seen) {
    result.missing += count.load() == 0 ? 1 : 0;
  }
  result.duplicates = duplicates.load();
  result.outOfOrder = outOfOrder.load();
  result.torn = torn.load();
  result.refused = refused.load();
  result.dropped = q.droppedCount();
  return result;
}

} // namespace

/** @test Delivers every sample of four producers once, whole and in order, through one consumer */
TEST(MonitorQueueTest, ConcurrentPushPopOneConsumer) {
  const QueueStress RESULT = runQueueStress(4, 1, 20000);

  EXPECT_TRUE(RESULT.finished) << "the deadline passed with samples undelivered";
  EXPECT_EQ(RESULT.missing, 0);
  EXPECT_EQ(RESULT.duplicates, 0);
  EXPECT_EQ(RESULT.torn, 0);
  EXPECT_EQ(RESULT.outOfOrder, 0);
  EXPECT_EQ(RESULT.dropped, static_cast<std::uint64_t>(RESULT.refused));
}

/** @test Delivers every sample of four producers once, whole and in order, through two consumers */
TEST(MonitorQueueTest, ConcurrentPushPopTwoConsumers) {
  const QueueStress RESULT = runQueueStress(4, 2, 20000);

  EXPECT_TRUE(RESULT.finished) << "the deadline passed with samples undelivered";
  EXPECT_EQ(RESULT.missing, 0);
  EXPECT_EQ(RESULT.duplicates, 0);
  EXPECT_EQ(RESULT.torn, 0);
  EXPECT_EQ(RESULT.outOfOrder, 0);
  EXPECT_EQ(RESULT.dropped, static_cast<std::uint64_t>(RESULT.refused));
}
