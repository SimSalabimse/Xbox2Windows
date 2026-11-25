// Windows-only: EXE + late-loaded package helper through the real loader.
//
// Loads a carrier DLL that hosts runtime_proxy (X2W_RUNTIME_PROXY=1), a fake xgameruntime.dll, an
// unrelated decoy that also exports QueryApiImpl, then a helper beside the EXE. The helper's
// GetProcAddress IAT is rewritten after linking (outside LdrDllNotification). Looking up
// QueryApiImpl on xgameruntime must return the proxy; the same name on the decoy must still be the
// decoy's export. A QueryApiImpl pointer cached before the proxy attaches must forward through the
// export trampoline (proxied object). The decoy export body is not trampolined. No game files.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "startup_stubs/runtime_gpa_patch.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures;

void check(const char* label, bool ok)
{
    std::printf("%s: %s\n", label, ok ? "ok" : "MISMATCH");
    if (!ok) ++failures;
}

using ResolveFn = FARPROC(WINAPI*)(HMODULE);
using QueryFn = HRESULT(WINAPI*)(const GUID*, const GUID*, void**);
using ObjectFn = void*(WINAPI*)();

std::wstring dir_of_self()
{
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring full(path);
    const auto slash = full.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : full.substr(0, slash);
}

HMODULE load_beside(const std::wstring& dir, const wchar_t* name)
{
    const std::wstring path = dir + L"\\" + name;
    return LoadLibraryW(path.c_str());
}

bool wait_for_helper_hook(ResolveFn resolve, HMODULE xgr, FARPROC expect_proxy, int ms)
{
    const DWORD start = GetTickCount();
    for (;;) {
        const FARPROC got = resolve(xgr);
        if (got == expect_proxy) return true;
        if (GetTickCount() - start > static_cast<DWORD>(ms)) return false;
        Sleep(10);
    }
}

bool wait_for_export_trampoline(FARPROC export_site, FARPROC proxy, int ms)
{
    if (!export_site || !proxy) return false;
    const auto* site = reinterpret_cast<const unsigned char*>(export_site);
    const DWORD start = GetTickCount();
    for (;;) {
        if (x2w_runtime_patch::is_abs_jmp_to(site, proxy)) return true;
        if (GetTickCount() - start > static_cast<DWORD>(ms)) return false;
        Sleep(10);
    }
}
}

int main()
{
    SetEnvironmentVariableW(L"X2W_RUNTIME_PROXY", L"1");
    const std::wstring dir = dir_of_self();
    SetDllDirectoryW(dir.c_str());

    const HMODULE decoy = load_beside(dir, L"startup_runtime_proxy_decoy.dll");
    check("load decoy", decoy != nullptr);
    const FARPROC decoy_query = decoy ? GetProcAddress(decoy, "QueryApiImpl") : nullptr;
    check("decoy exports QueryApiImpl", decoy_query != nullptr);
    unsigned char decoy_prologue[16] = {};
    if (decoy_query) std::memcpy(decoy_prologue, reinterpret_cast<const void*>(decoy_query), sizeof(decoy_prologue));

    const HMODULE xgr = load_beside(dir, L"xgameruntime.dll");
    check("load fake xgameruntime", xgr != nullptr);
    const FARPROC xgr_query = xgr ? GetProcAddress(xgr, "QueryApiImpl") : nullptr;
    check("xgameruntime exports QueryApiImpl", xgr_query != nullptr);
    check("decoy and xgameruntime exports differ", decoy_query && xgr_query && decoy_query != xgr_query);

    // Pre-attach cache: resolve before the proxy carrier loads. After attach the export trampoline
    // must make this same pointer enter the proxy (IAT/GPA alone leave it as the real export).
    const FARPROC cached_before_proxy = xgr_query;

    const HMODULE carrier = load_beside(dir, L"startup_runtime_proxy_carrier.dll");
    check("load proxy carrier", carrier != nullptr);
    if (!carrier) return 1;
    check("release the caller's carrier reference", FreeLibrary(carrier) != FALSE);
    check("enabled proxy stays live through process exit",
          GetModuleHandleW(L"startup_runtime_proxy_carrier.dll") == carrier);
    if (failures) return 1;

    // Host GetProcAddress is rewritten on attach; xgameruntime lookups become the proxy.
    const FARPROC host_xgr = xgr ? GetProcAddress(xgr, "QueryApiImpl") : nullptr;
    check("host GetProcAddress(xgameruntime) is proxied", host_xgr && host_xgr != xgr_query);
    check("export trampoline installed at pre-attach site",
          wait_for_export_trampoline(cached_before_proxy, host_xgr, 2000));
    check("release the caller's runtime reference", xgr && FreeLibrary(xgr) != FALSE);
    check("the cached provider stays live", GetModuleHandleW(L"xgameruntime.dll") == xgr);
    if (failures) return 1;

    const FARPROC host_decoy = decoy ? GetProcAddress(decoy, "QueryApiImpl") : nullptr;
    check("host GetProcAddress(decoy) stays original", host_decoy == decoy_query);
    check("decoy export not trampolined",
          decoy_query && std::memcmp(decoy_prologue, reinterpret_cast<const void*>(decoy_query), sizeof(decoy_prologue)) == 0);

    const HMODULE helper = load_beside(dir, L"startup_runtime_proxy_helper.dll");
    check("late-load package helper", helper != nullptr);
    const auto resolve = helper ? reinterpret_cast<ResolveFn>(GetProcAddress(helper, "resolve_query")) : nullptr;
    check("helper exports resolve_query", resolve != nullptr);

    check("helper sees proxied xgameruntime QueryApiImpl",
          resolve && wait_for_helper_hook(resolve, xgr, host_xgr, 2000));
    const FARPROC helper_decoy = resolve ? resolve(decoy) : nullptr;
    check("helper GetProcAddress(decoy) stays original", helper_decoy == decoy_query);

    // Intended call: invoke the proxied entry; the fake xgameruntime must run.
    using SeenFn = LONG(WINAPI*)();
    const auto xgr_seen = xgr ? reinterpret_cast<SeenFn>(GetProcAddress(xgr, "query_calls")) : nullptr;
    const auto xgr_object = xgr ? reinterpret_cast<ObjectFn>(GetProcAddress(xgr, "query_object")) : nullptr;
    const auto decoy_seen = decoy ? reinterpret_cast<SeenFn>(GetProcAddress(decoy, "query_calls")) : nullptr;
    const LONG xgr_before = xgr_seen ? xgr_seen() : -1;
    const LONG decoy_before = decoy_seen ? decoy_seen() : -1;
    if (host_xgr) reinterpret_cast<QueryFn>(host_xgr)(nullptr, nullptr, nullptr);
    if (helper_decoy) reinterpret_cast<QueryFn>(helper_decoy)(nullptr, nullptr, nullptr);
    check("proxied call reached xgameruntime", xgr_seen && xgr_seen() == xgr_before + 1);
    check("decoy call reached decoy only", decoy_seen && decoy_seen() == decoy_before + 1);

    // Pre-attach cache: same pointer value, now trampolined into the proxy; readable GUIDs yield a
    // proxied object distinct from the fake export's raw object.
    GUID provider{};
    provider.Data1 = 0x11111111;
    void* out = nullptr;
    const HRESULT cached_hr =
        cached_before_proxy ? reinterpret_cast<QueryFn>(cached_before_proxy)(&provider, &provider, &out) : E_FAIL;
    check("pre-attach cache call succeeded", cached_hr == S_OK);
    check("pre-attach cache reached xgameruntime", xgr_seen && xgr_seen() == xgr_before + 2);
    const void* raw = xgr_object ? xgr_object() : nullptr;
    check("pre-attach cache returns proxied object", out && raw && out != raw);

    // Negative: decoy still answers with its own HRESULT and is not the gaming-runtime proxy.
    void* decoy_out = reinterpret_cast<void*>(0x1);
    const HRESULT decoy_hr =
        decoy_query ? reinterpret_cast<QueryFn>(decoy_query)(&provider, &provider, &decoy_out) : S_OK;
    check("decoy QueryApiImpl stays decoy", decoy_hr == static_cast<HRESULT>(0xA11C0001));

    if (failures) {
        std::printf("startup_runtime_proxy_scope: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_runtime_proxy_scope: ok");
    return 0;
}
