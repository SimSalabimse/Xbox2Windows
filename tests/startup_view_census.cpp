// X2W_D3D12_VIEW_CENSUS=1 (docs/startup-runtime.md#d3d12-view-census): the stand-in counts new distinct view records
// by slot and logs the vector capacity and hash bucket count before an insertion that grows either. The fixture compares
// a stream of one repeated descriptor with a stream of distinct descriptors, both through the real device view slots
// and both under a job commit limit set above what the process holds after setup.
//
//   startup_view_census <warp|hardware> [off|vary]
//     (X2W_D3D12_BUDGET=1 X2W_D3D12_PERMISSIVE=1 X2W_D3D12_RENDER=1 X2W_D3D12_VIEW_CENSUS=1 X2W_TRAP_LOG=<file>)
//     "off" runs without X2W_D3D12_VIEW_CENSUS: nothing may be reported or logged. "vary" is the negative control:
//     the stream meant to repeat one descriptor changes it every call, which the fixture must report as a failure.
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <psapi.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_view_census_line(char*, std::size_t);

namespace {

constexpr unsigned long long headroom = 256ull << 20;  // job commit limit above the commit held after setup
constexpr UINT repeated_calls = 200000;
constexpr UINT distinct_buffers = 200000;  // slot 17
constexpr UINT distinct_views = 3000;      // slot 18
constexpr UINT distinct_samplers = 100;    // slot 22

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    return SUCCEEDED(hr);
}

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

unsigned long long process_commit()
{
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)))
        return 0;
    return counters.PrivateUsage;
}

// Puts this process in its own job with a commit limit. The job lives until the process ends.
bool limit_commit(unsigned long long limit)
{
    const HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
    info.JobMemoryLimit = static_cast<SIZE_T>(limit);
    return job && SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info)) &&
           AssignProcessToJobObject(job, GetCurrentProcess());
}

struct Census {
    unsigned long long cbv, srv, uav, rtv, dsv, sampler, records, capacity, ids, buckets, grows_records, grows_ids, missed;
};

bool read_census(Census& c, const char* stage)
{
    char line[1024]{};
    if (!x2w_d3d12_view_census_line(line, sizeof(line))) {
        std::printf("FAIL: %s: the census line is empty although the switch is on\n", stage);
        return false;
    }
    const int fields = std::sscanf(line,
        "x2w-view-census distinct by slot: cbv=%llu srv=%llu uav=%llu rtv=%llu dsv=%llu sampler=%llu records=%llu "
        "capacity=%llu ids=%llu buckets=%llu grows-records=%llu grows-ids=%llu missed=%llu",
        &c.cbv, &c.srv, &c.uav, &c.rtv, &c.dsv, &c.sampler, &c.records, &c.capacity, &c.ids, &c.buckets,
        &c.grows_records, &c.grows_ids, &c.missed);
    std::printf("%s: %s\n", stage, line);
    if (fields != 13) std::printf("FAIL: %s: unreadable census line (%d fields)\n", stage, fields);
    return fields == 13;
}

bool check(bool ok, const char* what)
{
    if (!ok) std::printf("FAIL: %s\n", what);
    return ok;
}

std::string read_log()
{
    wchar_t path[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"X2W_TRAP_LOG", path, MAX_PATH);
    std::string text;
    if (!length || length >= MAX_PATH) return text;
    const HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return text;
    char chunk[4096];
    DWORD got = 0;
    while (ReadFile(file, chunk, sizeof(chunk), &got, nullptr) && got) text.append(chunk, got);
    CloseHandle(file);
    return text;
}

unsigned long long occurrences(const std::string& text, const char* needle)
{
    unsigned long long count = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++count;
    return count;
}

struct Streams {
    IUnknown* guest;
    UINT64 view_handle, sampler_handle;

    void constant_buffer(UINT index) const
    {
        const D3D12_CONSTANT_BUFFER_VIEW_DESC desc{0x100000ull + 256ull * index, 256};
        invoke<void>(guest, 17, &desc, view_handle);
    }
    void buffer_view(UINT index) const
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        std::memset(&desc, 0, sizeof(desc));  // padding is part of the record's content
        desc.Format = DXGI_FORMAT_R32_FLOAT;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.FirstElement = index;
        desc.Buffer.NumElements = 1;
        invoke<void>(guest, 18, static_cast<void*>(nullptr), &desc, view_handle);
    }
    void sampler(UINT index) const
    {
        D3D12_SAMPLER_DESC desc{};
        desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        desc.AddressU = desc.AddressV = desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        desc.MipLODBias = static_cast<float>(index) / 16.0f;
        desc.MaxAnisotropy = 1;
        desc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        desc.MaxLOD = D3D12_FLOAT32_MAX;
        invoke<void>(guest, 22, &desc, sampler_handle);
    }
};

bool scenario(const Streams& s, bool vary)
{
    // One of each first, so the repeated stream starts from records that already exist.
    s.constant_buffer(0);
    s.buffer_view(0);
    s.sampler(0);
    Census start{}, repeated{}, distinct{};
    if (!read_census(start, "start")) return false;

    const unsigned long long before_repeated = process_commit();
    for (UINT i = 0; i < repeated_calls; ++i) s.constant_buffer(vary ? i : 0);
    for (UINT i = 0; i < distinct_views; ++i) s.buffer_view(0);
    const unsigned long long after_repeated = process_commit();
    if (!read_census(repeated, "repeated")) return false;
    if (!check(repeated.records == start.records && repeated.cbv == start.cbv && repeated.srv == start.srv &&
                   repeated.sampler == start.sampler,
               "the repeated stream added records"))
        return false;
    if (!check(repeated.capacity == start.capacity && repeated.buckets == start.buckets &&
                   repeated.grows_records == start.grows_records && repeated.grows_ids == start.grows_ids,
               "the repeated stream grew a container"))
        return false;

    for (UINT i = 1; i <= distinct_buffers; ++i) s.constant_buffer(i);
    for (UINT i = 1; i <= distinct_views; ++i) s.buffer_view(i);
    for (UINT i = 1; i <= distinct_samplers; ++i) s.sampler(i);
    const unsigned long long after_distinct = process_commit();
    if (!read_census(distinct, "distinct")) return false;
    std::printf("commit bytes: repeated stream %+lld, distinct stream %+lld\n",
                static_cast<long long>(after_repeated - before_repeated),
                static_cast<long long>(after_distinct - after_repeated));
    if (!check(distinct.cbv - repeated.cbv == distinct_buffers, "slot 17 count") ||
        !check(distinct.srv - repeated.srv == distinct_views, "slot 18 count") ||
        !check(distinct.sampler - repeated.sampler == distinct_samplers, "slot 22 count") ||
        !check(!distinct.uav && !distinct.rtv && !distinct.dsv, "a slot without calls has records") ||
        !check(distinct.records - repeated.records == distinct_buffers + distinct_views + distinct_samplers,
               "record total") ||
        !check(distinct.ids == distinct.records, "ids and records differ") ||
        !check(distinct.capacity >= distinct.records, "capacity below size") ||
        !check(distinct.grows_records > repeated.grows_records, "no vector growth announced") ||
        !check(distinct.grows_ids > repeated.grows_ids, "no bucket growth announced") ||
        !check(!distinct.missed, "a container grew without an announcement"))
        return false;

    // Every announced growth has its log line.
    const std::string log = read_log();
    const unsigned long long both = occurrences(log, "x2w-view-census before growth of records and ids:");
    const unsigned long long records = occurrences(log, "x2w-view-census before growth of records:") + both;
    const unsigned long long ids = occurrences(log, "x2w-view-census before growth of ids:") + both;
    std::printf("growth lines in the log: records %llu, ids %llu\n", records, ids);
    return check(records == distinct.grows_records && ids == distinct.grows_ids, "growth lines and counters differ");
}

// With the switch off the same calls report nothing.
bool nothing_reported(const Streams& s)
{
    for (UINT i = 0; i < 1000; ++i) s.constant_buffer(i);
    char line[256];
    std::memset(line, 'x', sizeof(line));
    const std::size_t length = x2w_d3d12_view_census_line(line, sizeof(line));
    return check(!length && !line[0], "the census reported without its switch") &&
           check(!occurrences(read_log(), "x2w-view-census"), "the census logged without its switch");
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "off") && std::strcmp(argv[2], "vary")))
        return 2;
    const bool off = argc == 3 && !std::strcmp(argv[2], "off");
    const bool vary = argc == 3 && !std::strcmp(argv[2], "vary");
    char value[2]{};
    const bool census_on = GetEnvironmentVariableA("X2W_D3D12_VIEW_CENSUS", value, sizeof(value)) == 1 && value[0] == '1';
    wchar_t log_path[MAX_PATH];
    const DWORD log_length = GetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path, MAX_PATH);
    if (census_on == off || !log_length || log_length >= MAX_PATH) {
        std::puts("FAIL: the environment does not match the requested mode");
        return 2;
    }
    DeleteFileW(log_path);  // before the stand-in writes its first line

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    ComPtr<IUnknown> guest, views, samplers;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                       reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device"))
        return 3;
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } view_heap{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1}, 0, 0}, sampler_heap{{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1}, 0, 0};
    if (!require(invoke(guest.Get(), 14, &view_heap, &__uuidof(ID3D12DescriptorHeap), views.GetAddressOf()), "view heap") ||
        !require(invoke(guest.Get(), 14, &sampler_heap, &__uuidof(ID3D12DescriptorHeap), samplers.GetAddressOf()), "sampler heap"))
        return 3;
    const Streams streams{guest.Get(), invoke<UINT64>(views.Get(), 9), invoke<UINT64>(samplers.Get(), 9)};

    const unsigned long long held = process_commit();
    if (!held || !limit_commit(held + headroom)) {
        std::printf("FAIL: could not set the job commit limit (error %lu)\n", GetLastError());
        return 4;
    }
    // The limit is real: a commit larger than the room under it is refused.
    if (void* too_much = VirtualAlloc(nullptr, static_cast<SIZE_T>(headroom + held), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
        VirtualFree(too_much, 0, MEM_RELEASE);
        std::puts("FAIL: the job commit limit did not refuse an allocation above it");
        return 4;
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (SUCCEEDED(adapter->GetDesc1(&adapter_desc))) std::printf("adapter=%ls\n", adapter_desc.Description);
    std::printf("mode=%s, commit after setup %llu bytes, job limit %llu bytes\n", off ? "off" : vary ? "vary" : "on", held,
                held + headroom);
    if (!(off ? nothing_reported(streams) : scenario(streams, vary))) return 5;
    std::puts(off ? "view-census off checks passed" : "view-census checks passed");
    return 0;
}
