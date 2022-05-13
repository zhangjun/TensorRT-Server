#include <queue>
#include <mutex>
#include <memory>
#include <condition_variable>

template<typename T>
class ConsumerProducerQueue {
public:
  ConsumerProducerQueue(int max_size, bool drop): max_size_(max_size), drop_(drop) {}
  void add(T request) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (drop && isFull()) {
        lock.unlock();
        return;
    } else {
        cond_.wait(lock, [this]() {
            return !isFull();
        });
        queue_.push(request);
        cond_.notify_all();
    }
  }
  void consume(T& request) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]() {
        return !isEmpty();
    });
    request = queue_.front();
    queue_.pop();
    cond_notify_all();
  }
  bool isFull() const {
      return queue_.size() > max_size_;
  }
  bool isEmpty() const {
      return queue_.size() == 0;
  }
  int size() {
      queue_.size();
  }
  void clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!isEmpty()) {
        queue_.pop();
    }
    lock.unlock();
    cond_.notify_all();
  }
private:
  std::condition_variable cond_;
  std::mutex mutex_;
  std::queue<T> queue_;
  int max_size_;
  bool drop_;
};