// Copyright (C) 2024 Bodo Inc. All rights reserved.

// Tests for SingleThreadedCpuThreadPool. Based heavily on Arrow's test suite:
// https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/util/thread_pool_test.cc

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../libs/_io_cpu_thread_pool.h"
#include "./test.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "arrow/status.h"

// Copied from Arrow
// (https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/testing/gtest_util.cc#L637)
void AssertChildExit(int child_pid, int expected_exit_status = 0) {
#if !defined(_WIN32)
    bodo::tests::check(child_pid > 0);
    int child_status;
    int got_pid = waitpid(child_pid, &child_status, 0);
    bodo::tests::check(got_pid == child_pid);
    if (WIFSIGNALED(child_status)) {
        std::cerr << "Child terminated by signal " << WTERMSIG(child_status);
    }
    if (!WIFEXITED(child_status)) {
        std::cerr << "Child didn't terminate normally?? Child status = "
                  << child_status;
    }
    bodo::tests::check(WEXITSTATUS(child_status) == expected_exit_status);
#endif
}

// Copied from
// https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/testing/gtest_util.cc#L768
void SleepFor(double seconds) {
    std::this_thread::sleep_for(
        std::chrono::nanoseconds(static_cast<int64_t>(seconds * 1e9)));
}

// Copied from
// https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/util/thread_pool_test.cc#L50

template <typename T>
static void task_add(T x, T y, T* out) {
    *out = x + y;
}

template <typename T>
struct task_slow_add {
    void operator()(T x, T y, T* out) {
        SleepFor(seconds_);
        *out = x + y;
    }

    const double seconds_;
};

using AddTaskFunc = std::function<void(int, int, int*)>;

template <typename T>
static T add(T x, T y) {
    return x + y;
}

template <typename T>
static T slow_add(double seconds, T x, T y) {
    SleepFor(seconds);
    return x + y;
}

//

// A class to spawn "add" tasks to a pool and check the results when done.
// Based on
// https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/util/thread_pool_test.cc#L85

class AddTester {
   public:
    explicit AddTester(int nadds, arrow::StopToken stop_token =
                                      arrow::StopToken::Unstoppable())
        : nadds_(nadds),
          stop_token_(std::move(stop_token)),
          xs_(nadds),
          ys_(nadds),
          outs_(nadds, -1) {
        int x = 0, y = 0;
        std::ranges::generate(xs_, [&] {
            ++x;
            return x;
        });
        std::ranges::generate(ys_, [&] {
            y += 10;
            return y;
        });
    }

    AddTester(AddTester&&) = default;

    void SpawnTasks(bodo::SingleThreadedCpuThreadPool* pool,
                    AddTaskFunc add_func) {
        for (int i = 0; i < nadds_; ++i) {
            bodo::tests::check(
                pool->Spawn([this, add_func,
                             i] { add_func(xs_[i], ys_[i], &outs_[i]); },
                            stop_token_)
                    .ok());
        }
    }

    void CheckResults() {
        for (int i = 0; i < nadds_; ++i) {
            bodo::tests::check(outs_[i] == (i + 1) * 11);
        }
    }

    bool AllCompleted() {
        for (int i = 0; i < nadds_; i++) {
            if (outs_[i] == -1) {
                return false;
            }
        }
        return true;
    }

    bool AnyCompleted() {
        for (int i = 0; i < nadds_; i++) {
            if (outs_[i] != -1) {
                return true;
            }
        }
        return false;
    }

    void CheckNotAllComputed() {
        for (int i = 0; i < nadds_; ++i) {
            if (outs_[i] == -1) {
                return;
            }
        }
        bodo::tests::check(false);
    }

   private:
    ARROW_DISALLOW_COPY_AND_ASSIGN(AddTester);

    int nadds_;
    arrow::StopToken stop_token_;
    std::vector<int> xs_;
    std::vector<int> ys_;
    std::vector<int> outs_;
};

// Test suite
static bodo::tests::suite tests([] {
    // Test that creating a new pool works as expected.
    bodo::tests::test("test_construct_deconstruct", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
    });
    // Test the OwnsThisThread functionality.
    bodo::tests::test("test_owns_current_thread", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
        std::atomic<bool> one_failed{false};

        for (int i = 0; i < 1000; ++i) {
            bodo::tests::check(pool->Spawn([&] {
                                       if (pool->OwnsThisThread())
                                           return;

                                       one_failed = true;
                                   })
                                   .ok());
        }
        ARROW_UNUSED(pool->Shutdown());
        bodo::tests::check(!pool->OwnsThisThread());
        bodo::tests::check(!one_failed);
    });

    // Test main submit/spawn functionality
    bodo::tests::test("test_spawn_basic", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
        // Create 50 tasks.
        AddTester add_tester(50, arrow::StopToken::Unstoppable());
        add_tester.SpawnTasks(pool.get(), task_add<int>);
        // The thread pool should be able to pick up and execute the tasks.
        int tries = 0;
        do {
            pool->ResumeExecutingTasks();
            // If we immediately call WaitForIdle, the worker thread will miss
            // the RUNNING state and not execute anything. So, we sleep for
            // 0.1s.
            SleepFor(0.1);
            bool all_done = add_tester.AllCompleted();
            pool->WaitForIdle();
            if (all_done) {
                break;
            }
            tries++;
        } while (tries < 10);
        bodo::tests::check(add_tester.AllCompleted());
    });

    // Submit tasks from multiple threads
    bodo::tests::test("test_spawn_from_multiple_threads", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
        // Submit 10 tasks each from 10 threads.
        int nthreads = 10;
        std::vector<AddTester> add_testers;
        std::vector<std::thread> threads;
        for (int i = 0; i < nthreads; ++i) {
            add_testers.emplace_back(10, arrow::StopToken::Unstoppable());
        }
        for (auto& add_tester : add_testers) {
            threads.emplace_back([&] {
                add_tester.SpawnTasks(pool.get(), task_slow_add<int>{0.02});
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        // Verify that the worker executes them all.
        int tries = 0;
        do {
            pool->ResumeExecutingTasks();
            SleepFor(0.1);
            bool all_done = true;
            for (auto& add_tester : add_testers) {
                all_done &= add_tester.AllCompleted();
            }
            pool->WaitForIdle();
            if (all_done) {
                break;
            }
            tries++;
        } while (tries < 10);
        bool all_done = true;
        for (auto& add_tester : add_testers) {
            all_done &= add_tester.AllCompleted();
        }
        bodo::tests::check(all_done);
    });

    // Submit tasks while worker is processing tasks
    bodo::tests::test("test_spawn_while_running", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
        // Submit 10 tasks each from 10 threads.
        int nthreads = 10;
        std::vector<AddTester> add_testers;
        std::vector<std::thread> threads;
        for (int i = 0; i < 2 * nthreads; ++i) {
            add_testers.emplace_back(10, arrow::StopToken::Unstoppable());
        }
        for (int i = 0; i < nthreads; i++) {
            auto& add_tester = add_testers[i];
            threads.emplace_back([&] {
                add_tester.SpawnTasks(pool.get(), task_slow_add<int>{0.02});
            });
        }
        for (int i = 0; i < nthreads; i++) {
            auto& thread = threads[i];
            thread.join();
        }

        // Start the worker
        pool->ResumeExecutingTasks();

        // Submit more tasks
        for (int i = nthreads; i < 2 * nthreads; i++) {
            auto& add_tester = add_testers[i];
            threads.emplace_back([&] {
                add_tester.SpawnTasks(pool.get(), task_slow_add<int>{0.02});
            });
        }
        for (int i = nthreads; i < 2 * nthreads; i++) {
            auto& thread = threads[i];
            thread.join();
        }

        // Wait for all tasks to be done.
        int tries = 0;
        do {
            SleepFor(0.1);
            bool all_done = true;
            for (auto& add_tester : add_testers) {
                all_done &= add_tester.AllCompleted();
            }
            if (all_done) {
                break;
            }
            tries++;
        } while (tries < 10);

        // Ask the worker to pause and verify that all tasks are indeed done.
        pool->WaitForIdle();
        bool all_done = true;
        for (auto& add_tester : add_testers) {
            all_done &= add_tester.AllCompleted();
        }
        bodo::tests::check(all_done);
    });

    // Test that no task is processed until the worker is explicitly resumed.
    bodo::tests::test("test_no_work_while_paused", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();

        // Spawn the tasks
        AddTester add_tester(50, arrow::StopToken::Unstoppable());
        add_tester.SpawnTasks(pool.get(), task_add<int>);

        // Verify that no work is done
        int tries = 0;
        do {
            SleepFor(0.01);
            bool any_done = add_tester.AnyCompleted();
            if (any_done) {
                break;
            }
            tries++;
        } while (tries < 10);
        bodo::tests::check(!add_tester.AnyCompleted());

        // Now resume and check that all tasks are done
        pool->ResumeExecutingTasks();
        tries = 0;
        do {
            SleepFor(0.1);
            bool all_done = add_tester.AllCompleted();
            if (all_done) {
                break;
            }
            tries++;
        } while (tries < 10);
        bodo::tests::check(add_tester.AllCompleted());
        pool->WaitForIdle();
    });

    // Test that a resumed worker is able to pick up tasks as they come.
    bodo::tests::test("test_worker_pick_up_tasks", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();

        // Resume the worker when there is no work
        pool->ResumeExecutingTasks();

        // Let the worker go into waiting mode for tasks.
        SleepFor(0.1);

        // Submit tasks
        AddTester add_tester(50, arrow::StopToken::Unstoppable());
        add_tester.SpawnTasks(pool.get(), task_add<int>);

        // Verify that all tasks were picked up and completed.
        int tries = 0;
        do {
            SleepFor(0.1);
            bool all_done = add_tester.AllCompleted();
            if (all_done) {
                break;
            }
            tries++;
        } while (tries < 10);
        bodo::tests::check(add_tester.AllCompleted());

        // Pause the worker.
        pool->WaitForIdle();
    });

    // Tests that repeated resume-pause (no tasks) doesn't cause any hangs or
    // side-effects.
    bodo::tests::test("test_repeated_resume_pause", [] {
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
        for (int i = 0; i < 10; i++) {
            pool->ResumeExecutingTasks();
            pool->WaitForIdle();
        }
        ARROW_UNUSED(pool->Shutdown());
    });

    // Test fork safety on Unix

#if !(defined(_WIN32) || defined(ARROW_VALGRIND) || \
      defined(ADDRESS_SANITIZER) || defined(THREAD_SANITIZER))

    bodo::tests::test("test_basic_fork_safety_after_task_submission", [] {
        // Fork after task submission
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
        auto fut = pool->Submit(add<int>, 4, 5).ValueOrDie();
        int i = 0;
        do {
            pool->ResumeExecutingTasks();
            bodo::tests::check(fut.Wait(1));
            pool->WaitForIdle();
            i++;
        } while (i < 5);
        bodo::tests::check(fut.result() == 9);

        auto child_pid = fork();
        if (child_pid == 0) {
            // Child: thread pool should be usable
            auto fut = pool->Submit(add<int>, 3, 4).ValueOrDie();
            int i = 0;
            do {
                pool->ResumeExecutingTasks();
                bodo::tests::check(fut.Wait(1));
                pool->WaitForIdle();
                i++;
            } while (i < 5);
            bodo::tests::check(fut.result() == 7);
            // Shutting down shouldn't hang or fail
            arrow::Status st = pool->Shutdown();
            std::exit(st.ok() ? 0 : 2);
        } else {
            // Parent
            AssertChildExit(child_pid);
            bodo::tests::check(pool->Shutdown().ok());
        }
    });

    bodo::tests::test("test_basic_fork_safety_after_shutdown", [] {
        // Fork after shutdown
        auto pool =
            bodo::SingleThreadedCpuThreadPool::MakeEternal().ValueOrDie();
        bodo::tests::check(pool->Shutdown().ok());
        auto child_pid = fork();
        if (child_pid == 0) {
            // Child
            // Spawning a task should return with error (pool was shutdown)
            arrow::Status st = pool->Spawn([] {});
            if (!st.IsInvalid()) {
                std::exit(1);
            }
            // Trigger destructor
            pool.reset();
            std::exit(0);
        } else {
            // Parent
            AssertChildExit(child_pid);
        }
    });
#endif
});
