#include <iostream>

class Thread {
public:
    void* operator new(size_t size) {
        // Bug: Memory leak, no corresponding delete
        return malloc(size);
    }

    void operator delete(void* ptr) {
        // Bug: Incorrect deallocation
        // free(ptr);
    }

    void start() {
        // Bug: Dereferencing a null pointer
        int* ptr = nullptr;
        *ptr = 10;
    }
};

int main() {
    Thread* t = new Thread();
    t->start();
    // Bug: Memory leak, t is not deleted
    // delete t;
    return 0;
}
