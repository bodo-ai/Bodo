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
    size_t virt_size = 68719476736;

    void* addr = VirtualAlloc(0, virt_size, MEM_RESERVE, PAGE_NOACCESS);

    size_t size = 16384;

    VirtualAlloc(addr, size, MEM_COMMIT, PAGE_READWRITE);

    memset(addr, 0xCB, size);
}

static bodo::tests::suite tests([] {
    bodo::tests::test("test_mmap", [] {
#if defined(WIN32)
        // commit (actually allocates and zeros out memory)
        test_mmap_helper(false);

        // Reserve don't commit
        test_mmap_helper(true);
#endif
    });
});
