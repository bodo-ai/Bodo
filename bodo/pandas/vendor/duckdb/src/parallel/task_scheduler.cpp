#include "duckdb/parallel/task_scheduler.hpp"

#include "duckdb/common/chrono.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

#include "concurrentqueue.h"
#include "duckdb/common/thread.hpp"
#include "lightweightsemaphore.h"

#include <thread>
#include <queue>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__GNUC__)
#include <sched.h>
#include <unistd.h>
#endif

namespace duckdb {

struct SchedulerThread {
};

struct ConcurrentQueue {
	reference_map_t<QueueProducerToken, std::queue<shared_ptr<Task>>> q;
	mutex qlock;

	void Enqueue(ProducerToken &token, shared_ptr<Task> task);
	bool DequeueFromProducer(ProducerToken &token, shared_ptr<Task> &task);
};

void ConcurrentQueue::Enqueue(ProducerToken &token, shared_ptr<Task> task) {
	lock_guard<mutex> lock(qlock);
	q[std::ref(*token.token)].push(std::move(task));
}

bool ConcurrentQueue::DequeueFromProducer(ProducerToken &token, shared_ptr<Task> &task) {
	lock_guard<mutex> lock(qlock);
	D_ASSERT(!q.empty());

	const auto it = q.find(std::ref(*token.token));
	if (it == q.end() || it->second.empty()) {
		return false;
	}

	task = std::move(it->second.front());
	it->second.pop();

	return true;
}

struct QueueProducerToken {
	explicit QueueProducerToken(ConcurrentQueue &queue) : queue(&queue) {
	}

	~QueueProducerToken() {
		lock_guard<mutex> lock(queue->qlock);
		queue->q.erase(*this);
	}

private:
	ConcurrentQueue *queue;
};


ProducerToken::ProducerToken(TaskScheduler &scheduler, unique_ptr<QueueProducerToken> token)
    : scheduler(scheduler), token(std::move(token)) {
}

ProducerToken::~ProducerToken() {
}

TaskScheduler::TaskScheduler(DatabaseInstance &db)
    : db(db), queue(make_uniq<ConcurrentQueue>()),
      allocator_flush_threshold(db.config.options.allocator_flush_threshold),
      allocator_background_threads(db.config.options.allocator_background_threads), requested_thread_count(0),
      current_thread_count(1) {
	SetAllocatorBackgroundThreads(db.config.options.allocator_background_threads);
}

TaskScheduler::~TaskScheduler() {
}

TaskScheduler &TaskScheduler::GetScheduler(ClientContext &context) {
	return TaskScheduler::GetScheduler(DatabaseInstance::GetDatabase(context));
}

TaskScheduler &TaskScheduler::GetScheduler(DatabaseInstance &db) {
	return db.GetScheduler();
}

unique_ptr<ProducerToken> TaskScheduler::CreateProducer() {
	auto token = make_uniq<QueueProducerToken>(*queue);
	return make_uniq<ProducerToken>(*this, std::move(token));
}

void TaskScheduler::ScheduleTask(ProducerToken &token, shared_ptr<Task> task) {
	// Enqueue a task for the given producer token and signal any sleeping threads
	queue->Enqueue(token, std::move(task));
}

bool TaskScheduler::GetTaskFromProducer(ProducerToken &token, shared_ptr<Task> &task) {
	return queue->DequeueFromProducer(token, task);
}

void TaskScheduler::ExecuteForever(atomic<bool> *marker) {
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
}

idx_t TaskScheduler::ExecuteTasks(atomic<bool> *marker, idx_t max_tasks) {
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
}

void TaskScheduler::ExecuteTasks(idx_t max_tasks) {
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
}

int32_t TaskScheduler::NumberOfThreads() {
	return current_thread_count.load();
}

idx_t TaskScheduler::GetNumberOfTasks() const {
	idx_t task_count = 0;
	for (auto &producer : queue->q) {
		task_count += producer.second.size();
	}
	return task_count;
}

idx_t TaskScheduler::GetProducerCount() const {
	return queue->q.size();
}

idx_t TaskScheduler::GetTaskCountForProducer(ProducerToken &token) const {
	const auto it = queue->q.find(std::ref(*token.token));
	if (it == queue->q.end()) {
		return 0;
	}
	return it->second.size();
}

void TaskScheduler::SetThreads(idx_t total_threads, idx_t external_threads) {
	if (total_threads == 0) {
		throw SyntaxException("Number of threads must be positive!");
	}
	if (total_threads != external_threads) {
		throw NotImplementedException(
		    "DuckDB was compiled without threads! Setting total_threads != external_threads is not allowed.");
	}
	requested_thread_count = NumericCast<int32_t>(total_threads - external_threads);
}

void TaskScheduler::SetAllocatorFlushTreshold(idx_t threshold) {
	allocator_flush_threshold = threshold;
}

void TaskScheduler::SetAllocatorBackgroundThreads(bool enable) {
	allocator_background_threads = enable;
	Allocator::SetBackgroundThreads(enable);
}

void TaskScheduler::Signal(idx_t n) {}

void TaskScheduler::YieldThread() {}

idx_t TaskScheduler::GetEstimatedCPUId() {
#if defined(EMSCRIPTEN)
	// FIXME: Wasm + multithreads can likely be implemented as
	//   return return (idx_t)std::hash<std::thread::id>()(std::this_thread::get_id());
	return 0;
#else
	// this code comes from jemalloc
#if defined(_WIN32)
	return (idx_t)GetCurrentProcessorNumber();
#elif defined(_GNU_SOURCE)
	auto cpu = sched_getcpu();
	if (cpu < 0) {
#ifndef DUCKDB_NO_THREADS
		// fallback to thread id
		return (idx_t)std::hash<std::thread::id>()(std::this_thread::get_id());
#else

		return 0;
#endif
	}
	return (idx_t)cpu;
#elif defined(__aarch64__) && defined(__APPLE__)
	/* Other oses most likely use tpidr_el0 instead */
	uintptr_t c;
	asm volatile("mrs %x0, tpidrro_el0" : "=r"(c)::"memory");
	return (idx_t)(c & (1 << 3) - 1);
#else
#ifndef DUCKDB_NO_THREADS
	// fallback to thread id
	return (idx_t)std::hash<std::thread::id>()(std::this_thread::get_id());
#else
	return 0;
#endif
#endif
#endif
}

void TaskScheduler::RelaunchThreads() {
	lock_guard<mutex> t(thread_lock);
	auto n = requested_thread_count.load();
	RelaunchThreadsInternal(n);
}

void TaskScheduler::RelaunchThreadsInternal(int32_t n) {
}

} // namespace duckdb
