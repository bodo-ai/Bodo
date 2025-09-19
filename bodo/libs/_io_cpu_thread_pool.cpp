
#include "_io_cpu_thread_pool.h"
#include <arrow/util/io_util.h>
#include <arrow/util/logging.h>
#include <fmt/format.h>
#include <algorithm>
#include <any>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <thread>

// Set this to 1 and re-compile to enable logging.
#define ENABLE_THREAD_POOL_LOGGING 0

namespace bodo {

// This namespace contains structs/functions copied from Arrow since they are
// not exposed directly in the headers.
namespace arrow_compat {

// Copied from
// https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/util/thread_pool.cc#L49
struct Task {
    ::arrow::internal::FnOnce<void()> callable;
    ::arrow::StopToken stop_token;
    ::arrow::internal::Executor::StopCallback stop_callback;
};

// Copied from
// https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/util/atfork_internal.h#L30
struct AtForkHandler {
    using CallbackBefore = std::function<std::any()>;
    using CallbackAfter = std::function<void(std::any)>;

    // The before-fork callback can return an arbitrary token (wrapped in
    // std::any) that will passed as-is to after-fork callbacks.  This can
    // ensure that any resource necessary for after-fork handling is kept alive.
    CallbackBefore before;
    CallbackAfter parent_after;
    CallbackAfter child_after;

    AtForkHandler() = default;

    explicit AtForkHandler(CallbackAfter child_after)
        : child_after(std::move(child_after)) {}

    AtForkHandler(CallbackBefore before, CallbackAfter parent_after,
                  CallbackAfter child_after)
        : before(std::move(before)),
          parent_after(std::move(parent_after)),
          child_after(std::move(child_after)) {}
};

namespace {

// Singleton state for at-fork management.
// Copied from
// https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/util/atfork_internal.cc#L41
struct AtForkState {
    struct RunningHandler {
        // A temporary owning copy of a handler, to make sure that a handler
        // that runs before fork can still run after fork.
        std::shared_ptr<AtForkHandler> handler;
        // The token returned by the before-fork handler, to pass to after-fork
        // handlers.
        std::any token;

        explicit RunningHandler(std::shared_ptr<AtForkHandler> handler)
            : handler(std::move(handler)) {}
    };

    void MaintainHandlersUnlocked() {
        auto it = std::ranges::remove_if(
            handlers_, [](const std::weak_ptr<AtForkHandler>& ptr) {
                return ptr.expired();
            });
        handlers_.erase(it.begin(), handlers_.end());
    }

    void BeforeFork() {
        // Lock the mutex and keep it locked until the end of AfterForkParent(),
        // to avoid multiple concurrent forks and atforks.
        mutex_.lock();

        ARROW_DCHECK(
            handlers_while_forking_.empty());  // AfterForkParent clears it

        for (const auto& weak_handler : handlers_) {
            if (auto handler = weak_handler.lock()) {
                handlers_while_forking_.emplace_back(std::move(handler));
            }
        }

        // XXX can the handler call RegisterAtFork()?
        for (auto&& handler : handlers_while_forking_) {
            if (handler.handler->before) {
                handler.token = handler.handler->before();
            }
        }
    }

    void AfterForkParent() {
        // The mutex was locked by BeforeFork()
        auto handlers = std::move(handlers_while_forking_);
        handlers_while_forking_.clear();

        // Execute handlers in reverse order
        for (auto& handler : std::ranges::reverse_view(handlers)) {
            if (handler.handler->parent_after) {
                handler.handler->parent_after(std::move(handler.token));
            }
        }

        mutex_.unlock();
        // handlers will be destroyed here without the mutex locked, so that
        // any action taken by destructors might call RegisterAtFork
    }

    void AfterForkChild() {
        // Need to reinitialize the mutex as it is probably invalid.  Also, the
        // old mutex destructor may fail.
        // Fortunately, we are a single thread in the child process by now, so
        // no additional synchronization is needed.
        new (&mutex_) std::mutex;

        auto handlers = std::move(handlers_while_forking_);
        handlers_while_forking_.clear();

        // Execute handlers in reverse order
        for (auto& handler : std::ranges::reverse_view(handlers)) {
            if (handler.handler->child_after) {
                handler.handler->child_after(std::move(handler.token));
            }
        }
    }

    void RegisterAtFork(std::weak_ptr<AtForkHandler> weak_handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        // This is O(n) for each at-fork registration. We assume that n remains
        // typically low and calls to this function are not
        // performance-critical.
        MaintainHandlersUnlocked();
        handlers_.push_back(std::move(weak_handler));
    }

    std::mutex mutex_;
    std::vector<std::weak_ptr<AtForkHandler>> handlers_;
    std::vector<RunningHandler> handlers_while_forking_;
};

// Copied from
// https://github.com/apache/arrow/blob/apache-arrow-18.1.0/cpp/src/arrow/util/atfork_internal.cc#L131
AtForkState* GetAtForkState() {
    static std::unique_ptr<AtForkState> state = []() {
        auto state = std::make_unique<AtForkState>();
#ifndef _WIN32
        int r = pthread_atfork(
            /*prepare=*/[] { GetAtForkState()->BeforeFork(); },
            /*parent=*/[] { GetAtForkState()->AfterForkParent(); },
            /*child=*/[] { GetAtForkState()->AfterForkChild(); });
        if (r != 0) {
            arrow::internal::IOErrorFromErrno(
                r, "Error when calling pthread_atfork: ")
                .Abort();
        }
#endif
        return state;
    }();
    return state.get();
}

};  // namespace

// Register the given at-fork handlers. Their intended lifetime should be
// tracked by calling code using an owning shared_ptr.
// Copied from
// https://github.com/apache/arrow/blob/apache-arrow-18.1.0/cpp/src/arrow/util/atfork_internal.cc#L149
void RegisterAtFork(std::weak_ptr<AtForkHandler> weak_handler) {
    GetAtForkState()->RegisterAtFork(std::move(weak_handler));
}

}  // namespace arrow_compat

/**
 * @brief State for the thread pool.
 *
 */
struct SingleThreadedCpuThreadPool::State {
    enum DesiredStateEnum {
        RUNNING = 0,
        SHUTTING_DOWN = 1,
        WAITING = 2,
    };

    State() = default;

    // State lock.
    std::mutex mutex_;
    // Desired state of the worker. Should only be set by the main thread.
    DesiredStateEnum desired_state = DesiredStateEnum::WAITING;

    // There's a single worker.
    std::optional<std::thread> worker_;
    // Queue of tasks
    std::deque<bodo::arrow_compat::Task> pending_tasks_;

    // Main thread uses this to notify the worker thread when
    // the desired state is modified. The worker waits on this in the outer loop
    // when the desired state is WAITING.
    std::condition_variable cv_outer_loop_;
    // In addition to modifications to desired state, this is notified when a
    // new task is added to the queue. The worker waits on this in the inner
    // loop when there are no more tasks in the queue to execute but the desired
    // state is RUNNING.
    std::condition_variable cv_;

    // These are only used when initializing the worker thread. The worker
    // thread notifies the main thread of its successful initialization using
    // this cv.
    bool done_init = false;
    std::condition_variable cv_init_;

    // This is used by the worker thread to notify the main thread that its done
    // executing the tasks in the queue. The main thread waits on this during
    // WaitForIdle.
    bool done_with_tasks = false;
    std::condition_variable cv_tasks_done_;

    // This is used during shutdown. The worker thread will notify the main
    // thread that it acknowledges the shut down request and is exiting. The
    // main thread will wait on this to get the notification before calling
    // thread.join, deleting the thread and finalizing shut down.
    bool done_shutting_down = false;
    std::condition_variable cv_shutdown_;

    // List of objects to keep alive.
    std::vector<std::shared_ptr<Resource>> kept_alive_resources_;

    // At-fork machinery
    void BeforeFork() { mutex_.lock(); }
    void ParentAfterFork() { mutex_.unlock(); }
    void ChildAfterFork() {
        DesiredStateEnum desired_state_ = this->desired_state;
        new (this)
            State;  // force-reinitialize, including synchronization primitives
        this->desired_state = desired_state_;
    }
    std::shared_ptr<bodo::arrow_compat::AtForkHandler> atfork_handler_;
};

thread_local SingleThreadedCpuThreadPool* current_thread_pool_ = nullptr;

bool SingleThreadedCpuThreadPool::OwnsThisThread() {
    return current_thread_pool_ == this;
}

// Macro for detailed logging. We use a compile time constant so that when
// disabled (e.g. in production builds), the compiler can completely optimize
// out the prints.
#define LOG(msg)                       \
    if (ENABLE_THREAD_POOL_LOGGING) {  \
        std::cerr << msg << std::endl; \
    }

#define LOG_LOOP(msg) \
    LOG(fmt::format("[SingleThreadedCpuThreadPool::WorkerLoop] {}", msg))

/**
 * @brief The loop that the worker thread will execute. Heavily modelled after
 * Arrow's WorkerLoop
 * (https://github.com/apache/arrow/blob/apache-arrow-17.0.0/cpp/src/arrow/util/thread_pool.cc#L424)
 *
 * @param state ThreadPool state.
 */
void WorkerLoop(std::shared_ptr<SingleThreadedCpuThreadPool::State> state) {
    using StateEnum = SingleThreadedCpuThreadPool::State::DesiredStateEnum;
    LOG_LOOP("Entered, acquiring lock.")
    std::unique_lock<std::mutex> lock(state->mutex_);
    LOG_LOOP("Acquired lock, notifying main thread about successful init.")
    state->done_init = true;
    state->cv_init_.notify_one();
    LOG_LOOP("Notified main thread about init, entering outer loop.")
    while (true) {
        if (state->desired_state == StateEnum::RUNNING) {
            LOG_LOOP(
                "Desired state: RUNNING. Setting done_with_tasks=false and "
                "entering work loop.")
            // If there is a pending task --> pop it, unlock, execute task,
            // re-acquire the lock.
            // If no task:
            // - If state is not running --> break
            // - Else --> wait on cv_ which is notified when more work is
            // available or when desired state is changed
            state->done_with_tasks = false;
            while (true) {
                if (!state->pending_tasks_.empty()) {
                    bodo::arrow_compat::Task task =
                        std::move(state->pending_tasks_.front());
                    state->pending_tasks_.pop_front();
                    arrow::StopToken* stop_token = &task.stop_token;
                    LOG_LOOP("Popped task, releasing lock and executing task.")
                    lock.unlock();
                    if (!stop_token->IsStopRequested()) {
                        std::move(task.callable)();
                    } else {
                        if (task.stop_callback) {
                            std::move(task.stop_callback)(stop_token->Poll());
                        }
                    }
                    // Release resources before waiting for lock
                    ARROW_UNUSED(std::move(task));
                    LOG_LOOP("Executed task, Reacquiring lock.")
                    lock.lock();
                    LOG_LOOP("Reacquired lock.")
                } else if (state->desired_state != StateEnum::RUNNING) {
                    LOG_LOOP("Desired State != RUNNING, exiting work loop.")
                    break;
                } else {
                    LOG_LOOP("Waiting on cv_.")
                    state->cv_.wait(lock);
                    LOG_LOOP("Exited wait on cv_.")
                }
            }
            LOG_LOOP(
                "Setting done_with_tasks=true and notifying main "
                "thread using cv_tasks_done_.")
            state->done_with_tasks = true;
            state->cv_tasks_done_.notify_one();
        } else if (state->desired_state == StateEnum::SHUTTING_DOWN) {
            LOG_LOOP(
                "Desired state: SHUTTING_DOWN. Setting done_with_tasks=true, "
                "done_shutting_down=true and notifying on cv_tasks_done_ and  "
                "cv_shutdown_.")
            // Notifying cv_tasks_done_ is likely not required, but added it to
            // be safe.
            state->done_with_tasks = true;
            state->cv_tasks_done_.notify_one();
            state->done_shutting_down = true;
            state->cv_shutdown_.notify_one();
            LOG_LOOP("Exiting...")
            break;
        } else if (state->desired_state == StateEnum::WAITING) {
            LOG_LOOP(
                "Desired state: WAITING. Setting done_with_tasks=true and "
                "notifying main thread using cv_tasks_done_.")
            // This is required since both Resume and WaitForIdle may get called
            // before the worker thread is woken up.
            state->done_with_tasks = true;
            state->cv_tasks_done_.notify_one();
            LOG_LOOP("Waiting on cv_outer_loop_...")
            state->cv_outer_loop_.wait(lock);
            LOG_LOOP("Exited wait on cv_outer_loop_...")
        } else {
            // Unreachable code, but adding it to be safe.
            LOG_LOOP(
                fmt::format("Unsupported desired state ({}))! Throwing runtime "
                            "error and exiting.",
                            std::to_string(state->desired_state)))
            throw std::runtime_error(
                fmt::format("[SingleThreadedCpuThreadPool::WorkerLoop] "
                            "Unsupported desired state ({})!",
                            std::to_string(state->desired_state)));
        }
    }
}
#undef LOG_LOOP

#define LOG_CTOR(msg) LOG(fmt::format("[SingleThreadedCpuThreadPool] {}", msg))

SingleThreadedCpuThreadPool::SingleThreadedCpuThreadPool()
    : sp_state_(std::make_shared<SingleThreadedCpuThreadPool::State>()),
      state_(sp_state_.get()) {
    // Eternal thread pools would produce false leak reports in the vector of
    // atfork handlers.
#if !(defined(_WIN32) || defined(ADDRESS_SANITIZER) || defined(ARROW_VALGRIND))
    state_->atfork_handler_ =
        std::make_shared<bodo::arrow_compat::AtForkHandler>(
            /*before=*/
            [weak_state = std::weak_ptr<SingleThreadedCpuThreadPool::State>(
                 sp_state_)]() {
                auto state = weak_state.lock();
                if (state) {
                    state->BeforeFork();
                }
                return state;  // passed to after-forkers
            },
            /*parent_after=*/
            [](std::any token) {
                auto state = std::any_cast<
                    std::shared_ptr<SingleThreadedCpuThreadPool::State>>(token);
                if (state) {
                    state->ParentAfterFork();
                }
            },
            /*child_after=*/
            [this](std::any token) {
                auto state = std::any_cast<
                    std::shared_ptr<SingleThreadedCpuThreadPool::State>>(token);
                if (state) {
                    state->ChildAfterFork();
                }
                this->create_worker_thread();
            });
    RegisterAtFork(state_->atfork_handler_);
#endif

    // Create the worker thread and wait for it to initialize.
    this->create_worker_thread();
}

#undef LOG_CTOR

#define LOG_CW(msg)                                                           \
    LOG(fmt::format("[SingleThreadedCpuThreadPool::create_worker_thread] {}", \
                    msg))

void SingleThreadedCpuThreadPool::create_worker_thread() {
    std::unique_lock<std::mutex> lock(this->state_->mutex_);
    if (this->state_->worker_.has_value()) {
        throw std::runtime_error(
            "[SingleThreadedCpuThreadPool::create_worker_thread] Worker "
            "thread already exists!");
    }
    if (this->state_->done_init) {
        throw std::runtime_error(
            "[SingleThreadedCpuThreadPool::create_worker_thread] Worker "
            "already initialized!");
    }
    std::shared_ptr<State> state = this->sp_state_;
    this->state_->worker_.emplace([this, state]() -> void {
        current_thread_pool_ = this;
        WorkerLoop(state);
    });
    LOG_CW("Created worker thread, waiting for it to init.")
    // MPIEXEC/SLURM automatically pin the worker thread to the same
    // CPU core as the main thread, so we don't need to do explicitly.
    this->state_->cv_init_.wait(
        lock, [this]() -> bool { return this->state_->done_init; });
    LOG_CW("Received init confirmation from worker.")
}

#undef LOG_CW

arrow::Result<std::shared_ptr<SingleThreadedCpuThreadPool>>
SingleThreadedCpuThreadPool::Make() {
    auto pool = std::shared_ptr<SingleThreadedCpuThreadPool>(
        new SingleThreadedCpuThreadPool());
    return pool;
}

arrow::Result<std::shared_ptr<SingleThreadedCpuThreadPool>>
SingleThreadedCpuThreadPool::MakeEternal() {
    ARROW_ASSIGN_OR_RAISE(auto pool, Make());
    return pool;
}

SingleThreadedCpuThreadPool::~SingleThreadedCpuThreadPool() {
    ARROW_UNUSED(this->Shutdown());
}

#define LOG_SD(msg) \
    LOG(fmt::format("[SingleThreadedCpuThreadPool::Shutdown] {}", msg))

::arrow::Status SingleThreadedCpuThreadPool::Shutdown() {
    LOG_SD("Acquiring lock.")
    std::unique_lock<std::mutex> lock(this->state_->mutex_);
    LOG_SD("Acquired lock.")
    if (this->state_->desired_state == State::SHUTTING_DOWN) {
        LOG_SD("SHUTDOWN has already been called!")
        return ::arrow::Status::Invalid("Shutdown() already called");
    }
    LOG_SD("Setting desired state to SHUTTING_DOWN and clearing queue.")
    this->state_->desired_state = State::SHUTTING_DOWN;
    this->state_->pending_tasks_.clear();

    // Make sure the worker thread has exited.
    if (this->state_->worker_.has_value()) {
        LOG_SD("Notifying the worker.")
        this->state_->done_shutting_down = false;
        this->state_->cv_outer_loop_.notify_one();
        this->state_->cv_.notify_one();
        LOG_SD("Waiting on cv_shutdown_.")
        this->state_->cv_shutdown_.wait(lock, [this]() -> bool {
            return this->state_->done_shutting_down;
        });
        LOG_SD("Done waiting on cv_shutdown_. Deleting worker.")
        this->state_->worker_.value().join();
        this->state_->worker_.reset();
        LOG_SD("Deleted worker.")
    }

    return ::arrow::Status::OK();
}

#undef LOG_SD

std::shared_ptr<SingleThreadedCpuThreadPool>
SingleThreadedCpuThreadPool::MakeThreadPool() {
    auto maybe_pool = SingleThreadedCpuThreadPool::MakeEternal();
    if (!maybe_pool.ok()) {
        maybe_pool.status().Abort("Failed to create Bodo CPU thread pool");
    }
    return *std::move(maybe_pool);
}

#define LOG_SPAWN(msg) \
    LOG(fmt::format("[SingleThreadedCpuThreadPool::SpawnReal] {}", msg))

::arrow::Status SingleThreadedCpuThreadPool::SpawnReal(
    arrow::internal::TaskHints hints, arrow::internal::FnOnce<void()> task,
    arrow::StopToken stop_token, StopCallback&& stop_callback) {
    {
        LOG_SPAWN("Acquiring lock.")
        std::lock_guard<std::mutex> lock(this->state_->mutex_);
        LOG_SPAWN("Acquired lock, adding to queue.")
        if (this->state_->desired_state == State::SHUTTING_DOWN) {
            LOG_SPAWN("Cannot add task since pool is shutting down!")
            return ::arrow::Status::Invalid(
                "Operation forbidden during or after shutdown!");
        }
        this->state_->pending_tasks_.push_back(
            {std::move(task), std::move(stop_token), std::move(stop_callback)});
        LOG_SPAWN(
            "Added to queue, releasing lock and notifying worker thread "
            "through cv_.")
    }
    this->state_->cv_.notify_one();
    LOG_SPAWN("Notified worker thread, exiting.")
    return arrow::Status::OK();
}

#undef LOG_SPAWN

#define LOG_RESUME(msg)                                                       \
    LOG(fmt::format("[SingleThreadedCpuThreadPool::ResumeExecutingTasks] {}", \
                    msg))

void SingleThreadedCpuThreadPool::ResumeExecutingTasks() {
    LOG_RESUME("Acquiring lock.")
    std::unique_lock<std::mutex> lock(this->state_->mutex_);
    LOG_RESUME("Acquired lock.")
    if (this->state_->desired_state == State::SHUTTING_DOWN) {
        LOG_RESUME("SHUTDOWN has already been called!")
        throw std::runtime_error(
            "[SingleThreadedCpuThreadPool::ResumeExecutingTasks] Shutdown() "
            "already called!");
    }
    LOG_RESUME("Setting state and notifying worker.")
    this->state_->done_with_tasks = false;
    this->state_->desired_state = State::RUNNING;
    this->state_->cv_outer_loop_.notify_one();
    this->state_->cv_.notify_one();
    LOG_RESUME("Notified, exiting...")
    // XXX TODO Wait for worker to acknowledge that it's running?
}

#undef LOG_RESUME

#define LOG_WAIT(msg) \
    LOG(fmt::format("[SingleThreadedCpuThreadPool::WaitForIdle] {}", msg))

void SingleThreadedCpuThreadPool::WaitForIdle() {
    LOG_WAIT("Acquiring lock.")
    std::unique_lock<std::mutex> lock(this->state_->mutex_);
    LOG_WAIT("Acquired lock.")
    std::shared_ptr<State> state = this->sp_state_;
    if (state->desired_state == State::SHUTTING_DOWN) {
        LOG_WAIT("SHUTDOWN has already been called!")
        throw std::runtime_error(
            "[SingleThreadedCpuThreadPool::WaitForIdle] Shutdown() already "
            "called!");
    }
    LOG_WAIT(
        "Setting desired state to WAITING and notifying worker using cv_ and "
        "cv_outer_loop_...")
    state->desired_state = State::WAITING;
    state->cv_outer_loop_.notify_one();
    state->cv_.notify_one();
    LOG_WAIT(
        "Notified worker using cv_ and cv_outer_loop_, starting wait on "
        "cv_tasks_done_...")
    // Wait for all tasks to be done.
    state->done_with_tasks = false;
    state->cv_tasks_done_.wait(
        lock, [this] { return this->state_->done_with_tasks; });
    LOG_WAIT("Done waiting on cv_tasks_done_, exiting.")
}

#undef LOG_WAIT

#define LOG_KA(msg) \
    LOG(fmt::format("[SingleThreadedCpuThreadPool::KeepAlive] {}", msg))

void SingleThreadedCpuThreadPool::KeepAlive(
    std::shared_ptr<Executor::Resource> resource) {
    // Seems unlikely but we might as well guard against concurrent calls to
    // KeepAlive
    LOG_KA("Acquiring lock.")
    std::lock_guard<std::mutex> lk(this->state_->mutex_);
    LOG_KA("Acquired lock. Adding resource to list and exiting.")
    this->state_->kept_alive_resources_.push_back(std::move(resource));
}

#undef LOG_KA
#undef LOG
#undef ENABLE_THREAD_POOL_LOGGING

}  // namespace bodo
