/*
 * Test mmap implementation for Windows
 */

#include "../libs/_bodo_common.h"
#include "../libs/_memory.h"
#include "../libs/_mman.h"
#include "./test.hpp"

void test_mmap_helper(bool reserve) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    if (reserve) {
        flags |= SEC_RESERVE;
    }

    int prot = PROT_READ | PROT_WRITE;
    size_t size = 16484;

    void* addr = mmap(nullptr, size, prot, flags, -1, 0);

    if (addr == MAP_FAILED) {
        printf("uh oh\n");
        return;
    }

    memset(addr, 0xCB, size);
}

static bodo::tests::suite tests([] {
    bodo::tests::test("test_mmap", [] {
#if defined(WIN32)
        // commit (actually allocates and zeros out memory)
        test_mmap_helper(false);

        // Reserve don't commit
        // test_mmap_helper(true);
#endif
    });
});
