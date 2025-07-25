#include <iostream>
#include <vector>

class GC {
public:
    void collect() {
        // Bug: Use-after-free
        std::vector<int>* vec = new std::vector<int>();
        vec->push_back(1);
        delete vec;
        std::cout << (*vec)[0] << std::endl;
    }
};

int main() {
    GC gc;
    gc.collect();
}
