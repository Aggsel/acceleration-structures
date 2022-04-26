#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
//https://stackoverflow.com/questions/36762248/why-is-stdqueue-not-thread-safe
template <typename T>
class SharedQueue{
    public:
        SharedQueue();
        ~SharedQueue();

        T& front();
        T pop_front();

        void push(const T& item);
        void push(T&& item);

        int size();
        bool empty();

    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_;
    }; 

template <typename T>
SharedQueue<T>::SharedQueue(){}

template <typename T>
SharedQueue<T>::~SharedQueue(){}

template <typename T>
T& SharedQueue<T>::front(){
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
        cond_.wait(mlock);
    }
    return queue_.front();
}

template <typename T>
T SharedQueue<T>::pop_front(){
    std::unique_lock<std::mutex> lock(mutex_);
    while(queue_.empty()){
      cond_.wait(lock);
    }
    T val = queue_.front();
    queue_.pop();
    return val;
}

template <typename T>
bool SharedQueue<T>::empty(){
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.empty();
}

template <typename T>
void SharedQueue<T>::push(const T& item){
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();     // unlock before notificiation to minimize mutex con
    cond_.notify_one(); // notify one waiting thread
}

template <typename T>
void SharedQueue<T>::push(T&& item){
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(std::move(item));
    mlock.unlock();     // unlock before notificiation to minimize mutex con
    cond_.notify_one(); // notify one waiting thread

}

template <typename T>
int SharedQueue<T>::size(){
    std::unique_lock<std::mutex> mlock(mutex_);
    int size = queue_.size();
    mlock.unlock();
    return size;
}