#pragma once

#include <algorithm>
#include <deque>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cudf/concatenate.hpp>  // cudf::concatenate
#include <cudf/copying.hpp>      // cudf::slice
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

class TableFifoBatcher {
   public:
    using table_ptr = std::unique_ptr<cudf::table>;
    using size_type = cudf::size_type;

    TableFifoBatcher(const size_type batch_size_,
                     const double low_thresh_ = 1.0,
                     const double high_thresh_ = 1.0)
        : batch_size(batch_size_),
          low_thresh(low_thresh_),
          high_thresh(high_thresh_) {
        if (low_thresh > high_thresh) {
            throw std::invalid_argument(
                "low_threshold must be less than high_threshold");
        }
    }

    ~TableFifoBatcher() = default;

    // Non-copyable
    TableFifoBatcher(const TableFifoBatcher &) = delete;
    TableFifoBatcher &operator=(const TableFifoBatcher &) = delete;

    // Movable
    TableFifoBatcher(TableFifoBatcher &&) = default;
    TableFifoBatcher &operator=(TableFifoBatcher &&) = default;

    bool empty() const { return fifo_.empty(); }

    // Push a table to the back of the FIFO
    void push_table(table_ptr tbl) {
        if (!tbl)
            return;
        fifo_.push_back(std::move(tbl));
    }

    // Return total rows available across all queued tables
    size_type size() const {
        size_type total = 0;
        for (auto const &t : fifo_)
            total += t->num_rows();
        return total;
    }

    // Add this inside your TableFifoBatcher class (adjust includes/namespaces
    // as needed)
    void keep_first_n_rows(size_type n) {
        // If n == 0: remove everything
        if (n == 0) {
            fifo_.clear();
            return;
        }

        // Nothing to do if FIFO empty
        if (fifo_.empty())
            return;

        size_type accumulated = 0;
        size_t idx = 0;  // index of the current element we are examining

        // Walk the FIFO from the front until we've accounted for n rows or run
        // out of tables
        while (idx < fifo_.size() && accumulated < n) {
            auto &tbl_ptr = fifo_[idx];  // unique_ptr<cudf::table>&
            size_type rows = tbl_ptr->num_rows();

            // If taking the whole current table still keeps us <= n, keep it
            // whole
            if (accumulated + rows <= n) {
                accumulated += rows;
                ++idx;
                continue;
            }

            // We need only part of this table: split it into head (keep) and
            // tail (drop)
            size_type need = static_cast<size_type>(n - accumulated);
            cudf::table_view tv = tbl_ptr->view();
            std::vector<cudf::size_type> offsets = {
                0, static_cast<cudf::size_type>(need)};
            auto slices = cudf::slice(tv, offsets);

            // Update current fifo entry to the split head.
            table_ptr head = std::make_unique<cudf::table>(
                slices[0]);  // first 'need' rows (kept)
            // Replace current element with the head slice (no structural change
            // to deque)
            tbl_ptr = std::move(head);
            accumulated += need;
            ++idx;  // we keep this element (now reduced to 'need' rows)
            break;  // we've reached n rows; stop scanning
        }

        // If we consumed all tables but still have fewer than n rows, do
        // nothing (keep all)
        if (accumulated < n && idx >= fifo_.size()) {
            // Not enough rows available: keep everything (no removal)
            return;
        }

        // Erase everything after index idx-1 (we want to keep elements [0 ..
        // idx-1]) If idx == fifo_.size() then nothing to erase.
        if (idx < fifo_.size()) {
            fifo_.erase(fifo_.begin() + static_cast<std::ptrdiff_t>(idx),
                        fifo_.end());
        }
    }

    // Get a batch of approximately batch_size rows following the rules
    // described. If no tables are available, returns nullptr.
    table_ptr get_batch() {
        if (batch_size <= 0) {
            throw std::invalid_argument("batch_size must be > 0");
        }

        if (fifo_.empty())
            return nullptr;

        // Otherwise first < 80%: accumulate as many whole entries as possible,
        // preferring whole entries and applying the 80%-120% preference for
        // combined totals.
        size_type accumulated = 0;
        std::vector<table_ptr>
            tables_to_concat;  // will hold moved whole tables to concatenate

        while (accumulated < batch_size && !fifo_.empty()) {
            size_type next_rows = fifo_.front()->num_rows();
            size_type combined = accumulated + next_rows;
            double combined_ratio =
                static_cast<double>(combined) / static_cast<double>(batch_size);

            // If taking the whole next entry yields combined in [80%,120%],
            // prefer whole and stop
            if (combined_ratio >= low_thresh && combined_ratio <= high_thresh) {
                // take whole next
                tables_to_concat.push_back(std::move(fifo_.front()));
                fifo_.pop_front();
                accumulated = combined;
                break;
            }

            if (combined <= batch_size) {
                tables_to_concat.push_back(std::move(fifo_.front()));
                fifo_.pop_front();
                accumulated = combined;
                continue;
            }

            // At this point combined_ratio > high_thresh.
            // We must split the next entry to take exactly (batch_size -
            // accumulated) rows.
            size_type need = batch_size - accumulated;
            assert(need != 0);
            // Split the front table: take 'need' rows from it, put remainder
            // back as front.
            table_ptr head = split_front_take(
                need);  // this mutates fifo_.front() to be remainder
            // concatenate tables_to_concat + head and return
            std::vector<cudf::table_view> views;
            views.reserve(tables_to_concat.size() + 1);
            for (auto &t : tables_to_concat)
                views.push_back(t->view());
            views.push_back(head->view());
            return cudf::concatenate(views);
        }

        assert(!table_to_concat.empty());
        std::vector<cudf::table_view> views;
        views.reserve(tables_to_concat.size());
        for (auto &t : tables_to_concat)
            views.push_back(t->view());
        return cudf::concatenate(views);
    }

   private:
    std::deque<table_ptr> fifo_;
    const size_type batch_size;
    double low_thresh;
    double high_thresh;

    // Pop front table and return it
    table_ptr pop_front_table() {
        if (fifo_.empty())
            return nullptr;
        table_ptr t = std::move(fifo_.front());
        fifo_.pop_front();
        return t;
    }

    // Split the front table: take first `take_rows` rows and put remainder back
    // to front. Returns the taken portion as a new table.
    table_ptr split_front_take(size_type take_rows) {
        if (fifo_.empty())
            return nullptr;
        table_ptr &front = fifo_.front();
        size_type front_rows = front->num_rows();
        if (take_rows <= 0 || take_rows > front_rows) {
            throw std::invalid_argument("split_front_take: invalid take_rows");
        }

        cudf::table_view tv = front->view();
        std::vector<cudf::size_type> offsets = {
            0, static_cast<cudf::size_type>(take_rows),
            static_cast<cudf::size_type>(front_rows)};
        std::vector<cudf::table_view> slices = cudf::slice(tv, offsets);

        table_ptr head = std::make_unique<cudf::table>(slices[0]);
        table_ptr tail = std::make_unique<cudf::table>(slices[1]);

        // replace front with tail (remainder)
        front = std::move(tail);

        return head;
    }
};
