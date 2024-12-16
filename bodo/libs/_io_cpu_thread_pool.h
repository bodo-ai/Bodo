
#pragma once

#include <arrow/result.h>
#include <arrow/util/thread_pool.h>
#include <memory>

namespace bodo {

/**
 * @brief A resumable single threaded thread pool for executing CPU tasks during
 * Arrow IO. See https://bodo.atlassian.net/l/cp/nJby4eaK for more details on
 * why this is required.
 *
 */
class SingleThreadedCpuThreadPool : public arrow::internal::Executor {
   public:
    // Construct a thread pool instance.
    static ::arrow::Result<std::shared_ptr<SingleThreadedCpuThreadPool>> Make();
    // Like Make(), but takes care that the returned ThreadPool is compatible
    // with destruction late at process exit.
    static ::arrow::Result<std::shared_ptr<SingleThreadedCpuThreadPool>>
    MakeEternal();
    // Destroy thread pool; the pool will first be shut down
    ~SingleThreadedCpuThreadPool() override;
    // We always have 1 worker thread.
    int GetCapacity() override { return 1; }
    // Shutdown the pool. Once the pool starts shutting down, new tasks
    // cannot be submitted anymore. Since we cannot force a thread to stop, we
    // must wait for it to finish processing its current task. We clear the task
    // queue to ensure it doesn't pick up any more tasks.
    ::arrow::Status Shutdown();

    /**
     * @brief This function sets the desired state to RUNNING and wakes up the
     * worker thread so that it can start processing tasks in the task queue.
     *
     */
    void ResumeExecutingTasks();

    /**
     * @brief This function sets the desired state to WAITING and then waits for
     * the worker thread to go into idle state. The function returns when the
     * worker thread has paused. Even after issuing this new desired state, the
     * worker will only pause once it runs out of tasks in the task queue.
     *
     */
    void WaitForIdle();

    bool OwnsThisThread() override;

    // Store the resource in the state so that it's kept alive for the duration
    // of this thread pool.
    void KeepAlive(std::shared_ptr<Executor::Resource> resource) override;

    struct State;

    /**
     * @brief Default Singleton Pool Object. This is what will be used
     * everywhere. The dynamically created pools (using make_shared) are
     * primarily for unit-testing purposes. Follows the same pattern as Arrow's
     * global CPU and IO thread pools.
     *
     * @return std::shared_ptr<SingleThreadedCpuThreadPool>
     */
    static std::shared_ptr<SingleThreadedCpuThreadPool> Default() {
        static std::shared_ptr<SingleThreadedCpuThreadPool> pool_ =
            SingleThreadedCpuThreadPool::MakeThreadPool();
        return pool_;
    }

    /**
     * @brief Simple wrapper for getting a pointer to the
     * SingleThreadedCpuThreadPool singleton
     *
     * @return SingleThreadedCpuThreadPool*
     */
    static SingleThreadedCpuThreadPool* DefaultPtr() {
        static SingleThreadedCpuThreadPool* pool =
            SingleThreadedCpuThreadPool::Default().get();
        return pool;
    }

   protected:
    SingleThreadedCpuThreadPool();

    // The underlying API used for submitting tasks to the thread pool.
    ::arrow::Status SpawnReal(arrow::internal::TaskHints hints,
                              arrow::internal::FnOnce<void()> task,
                              arrow::StopToken stop_token,
                              StopCallback&& stop_callback) override;

    // Wrapper around MakeEternal with additional error handling.
    static std::shared_ptr<SingleThreadedCpuThreadPool> MakeThreadPool();
    // Shared pointer to the state.
    std::shared_ptr<State> sp_state_;
    // Raw pointer to the state.
    State* state_;

   private:
    /// @brief Helper function to create and initialize the worker thread for
    /// the pool.
    void create_worker_thread();
};

}  // namespace bodo
