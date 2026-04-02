/**
 * @file MonitorQueue_uTest.cpp
 * @brief Unit tests for vernier::monitor::MonitorQueue.
 *
 * Notes:
 *  - Tests are platform-agnostic: assert invariants, not exact values.
 */

#include "src/monitor/inc/MonitorQueue.hpp"

#include <gtest/gtest.h>

#include <cstring>

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

/* ----------------------------- Determinism Tests ----------------------------- */

/** @test Multi-threaded concurrent push/pop */
TEST(MonitorQueueTest, ConcurrentPushPop) {
  constexpr int NUM_PRODUCERS = 4;
  constexpr int SAMPLES_PER_PRODUCER = 1000;
  MonitorQueue q(4096);

  std::vector<std::thread> producers;
  for (int p = 0; p < NUM_PRODUCERS; ++p) {
    producers.emplace_back([&q, p] {
      for (int i = 0; i < SAMPLES_PER_PRODUCER; ++i) {
        Sample s;
        s.timestampNs = static_cast<uint64_t>(p * 10000 + i);
        s.tag = MonitorTag("prod", static_cast<uint16_t>(p));
        while (!q.tryPush(s)) {
          std::this_thread::yield();
        }
      }
    });
  }

  int consumed = 0;
  std::thread consumer([&q, &consumed] {
    const int TARGET = NUM_PRODUCERS * SAMPLES_PER_PRODUCER;
    Sample out;
    while (consumed < TARGET) {
      if (q.tryPop(out)) {
        consumed++;
      } else {
        std::this_thread::yield();
      }
    }
  });

  for (auto& t : producers)
    t.join();
  consumer.join();

  EXPECT_EQ(consumed, NUM_PRODUCERS * SAMPLES_PER_PRODUCER);
  EXPECT_EQ(q.droppedCount(), 0u);
}
