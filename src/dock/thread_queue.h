#pragma once
// Lean bounded FIFO queue for CPU multi-thread + GPU feeder.
// No priority, no stealing - simple mutex+cv, move semantics.
#include <deque>
#include <mutex>
#include <condition_variable>
#include <utility>

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity = 0) : cap_(capacity), closed_(false) {}
    void set_capacity(size_t c) { cap_ = c; }
    size_t capacity() const { return cap_; }

    // push - blocks if bounded and full, unless closed
    // Use copy (not move) for DOCKMol safety - DOCKMol has shallow move
    bool push(const T& v) {
        std::unique_lock<std::mutex> lk(m_);
        if (closed_) return false;
        if (cap_ > 0) {
            cv_not_full_.wait(lk, [&]{ return closed_ || q_.size() < cap_; });
            if (closed_) return false;
        }
        q_.push_back(v);
        lk.unlock();
        cv_not_empty_.notify_one();
        return true;
    }

    // wait_pop - blocks until item or closed - copy for DOCKMol safety
    bool wait_pop(T &out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_not_empty_.wait(lk, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return false; // closed and empty
        out = q_.front();
        q_.pop_front();
        lk.unlock();
        cv_not_full_.notify_one();
        return true;
    }

    bool try_pop(T &out) {
        std::unique_lock<std::mutex> lk(m_);
        if (q_.empty()) return false;
        out = q_.front();
        q_.pop_front();
        lk.unlock();
        cv_not_full_.notify_one();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size();
    }
    bool empty() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.empty();
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }
    bool is_closed() const {
        std::lock_guard<std::mutex> lk(m_);
        return closed_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::deque<T> q_;
    size_t cap_;
    bool closed_;
};
