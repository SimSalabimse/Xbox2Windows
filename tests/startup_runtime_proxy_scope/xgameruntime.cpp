// Source-built stand-in for xgameruntime.dll used only by startup_runtime_proxy_scope.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
LONG calls;
void* fake_slots[8];
struct FakeObject {
    void** vtable;
};
FakeObject fake_object{fake_slots};
}

extern "C" __declspec(dllexport) LONG WINAPI query_calls() { return InterlockedCompareExchange(&calls, 0, 0); }

extern "C" __declspec(dllexport) void* WINAPI query_object() { return &fake_object; }

// Keep a conventional MSVC x64 prologue (mov [rsp+…], … / push / sub rsp) so the export trampoline
// can size whole instructions. An optimised body often starts with rip-relative lock xadd, which
// the trampoline refuses to relocate.
#if defined(_MSC_VER)
#pragma optimize("", off)
#endif
extern "C" __declspec(dllexport) HRESULT WINAPI QueryApiImpl(const GUID* provider, const GUID*, void** out)
{
    InterlockedIncrement(&calls);
    if (!out) return E_INVALIDARG;
    // Readable provider GUID → hand back a real object so the proxy can wrap it (provider … proxied).
    if (provider) {
        *out = &fake_object;
        return S_OK;
    }
    *out = nullptr;
    return E_FAIL;
}
#if defined(_MSC_VER)
#pragma optimize("", on)
#endif
