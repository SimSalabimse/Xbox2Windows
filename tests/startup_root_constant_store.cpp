// Experimental (docs/startup-runtime.md, #870): the root constants a command list keeps for compute
// (list slots 37 and 39). Plain standard C++; synthetic values, no title data.
#include "../src/startup_stubs/root_constant_store.h"

#include <cstdio>
#include <string>

namespace {
using x2w_root_constants::Result;
using x2w_root_constants::Store;
int failures;

void expect(bool ok, const std::string& what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}
}  // namespace

int main()
{
    Store store;
    const std::uint32_t data[4] = {11, 22, 33, 44};

    expect(store.set(3, 4, data, 2) == Result::stored, "four words at offset 2 are stored");
    expect(store.word(3, 2) == 11 && store.word(3, 5) == 44, "stored words read back at their offsets");
    expect(store.word(3, 1) == 0 && store.word(3, 6) == 0, "neighbouring words stay zero");

    expect(store.set_one(5, 77, 7) == Result::stored, "a single word at the last offset is stored");
    expect(store.word(5, 7) == 77, "the single word reads back");

    expect(store.set(6, 4, data, 6) == Result::truncated, "a write past the end of the parameter is cut off");
    expect(store.word(6, 6) == 11 && store.word(6, 7) == 22, "the words that fit are kept");
    expect(store.word(7, 0) == 0, "a cut-off write does not spill into the next parameter");

    expect(store.set(16, 1, data, 0) == Result::dropped, "parameter 16 is refused");
    expect(store.set(0, 1, data, 8) == Result::dropped, "offset 8 is refused");
    expect(store.set(0, 0, data, 0) == Result::dropped, "an empty write is refused");
    expect(store.set(0, 1, nullptr, 0) == Result::dropped, "a null source is refused");
    expect(store.set_one(15, 5, 0) == Result::stored && store.word(15, 0) == 5, "parameter 15 is the last one kept");
    expect(store.word(16, 0) == 0 && store.word(-1, 0) == 0 && store.word(0, 8) == 0, "reads outside the range give 0");

    expect(store.is_bound(3, 2, 4) && store.is_bound(3, 3, 1), "written words are marked set");
    expect(!store.is_bound(3, 1, 1) && !store.is_bound(3, 1, 2) && !store.is_bound(3, 6, 1),
           "words never written are not set, and a range with one unset word is not set");
    expect(store.is_bound(6, 6, 2) && !store.is_bound(6, 6, 3), "a cut-off write marks only the words that fit");
    expect(!store.is_bound(0, 0, 1), "a dropped write marks nothing");
    expect(!store.is_bound(16, 0, 1) && !store.is_bound(0, 8, 1) && !store.is_bound(0, 0, 0) && !store.is_bound(0, 7, 2),
           "ranges outside the kept words are never set");
    expect(store.set_one(1, 0, 0) == Result::stored && store.is_bound(1, 0, 1) && store.word(1, 0) == 0,
           "writing 0 is set, unlike never written");

    store.clear();
    expect(store.word(3, 2) == 0 && store.word(15, 0) == 0, "clear zeroes every word");
    expect(!store.is_bound(3, 2, 4) && !store.is_bound(1, 0, 1), "clear forgets which words were set");

    if (failures) return 1;
    std::puts("root constant store ok");
    return 0;
}
