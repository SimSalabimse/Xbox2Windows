// Command-list slot 14 (Dispatch) driven through the d3d12_xs stand-in on a desktop device. Slots 86 (compute
// pipeline stream), 25 (pipeline), 33 (compute root), 43 (compute root SRV in the recording's upload pages), 148
// (upload allocation), 9 (Close) and 139 (Reset) set up each case. A refused call must log exactly one
// "x2w-render Dispatch(x, y, z) refused: <reason>" line to %X2W_TRAP_LOG% and record nothing. A forwarded call must
// log no line, and the desktop Dispatch is observed through a pipeline-statistics query that the fixture wraps
// around the slot 14 call on the underlying desktop list: CSInvocations is groups * 4 for a forwarded call and 0
// for a refused one. The root SRV roots carry no UAV, so the shader writes nothing; the invocation count is the
// result. The recordings are executed to a fence, and the debug layer, when present, must report no error.
//
// `skip-srv` is the negative control: the first forwarded case omits slot 43, so the fixture must fail on it.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Microsoft::WRL::ComPtr;

void check(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) {
        std::fprintf(stderr, "%s failed (HRESULT 0x%08lx)\n", stage, static_cast<unsigned long>(hr));
        throw std::runtime_error(stage);
    }
}

template <typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

using Bind = void(WINAPI*)(IUnknown*, IUnknown*);
using Address = void(WINAPI*)(IUnknown*, UINT, UINT64);
using Dispatch = void(WINAPI*)(IUnknown*, UINT, UINT, UINT);
using Upload = void*(WINAPI*)(IUnknown*, UINT64, UINT64);
using Reset = HRESULT(WINAPI*)(IUnknown*, IUnknown*, void*, void*);
using Close = HRESULT(WINAPI*)(IUnknown*);

constexpr UINT threads_per_group = 4;
constexpr char shader[] = R"(
ByteAddressBuffer input : register(t0);
[numthreads(4, 1, 1)]
void main() {}
)";

template <typename T>
ComPtr<T> native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* object)
{
    // SetPrivateDataInterface unwraps a guest argument before the call reaches the shared desktop device.
    constexpr GUID key{0x10a62244, 0x1615, 0x4953, {0xa2, 0x52, 0xbe, 0x42, 0x35, 0x27, 0xa0, 0x61}};
    check(guest->SetPrivateDataInterface(key, object), "publish underlying object");
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "clear underlying object");
    check(read, "retrieve underlying object");
    ComPtr<T> result;
    check(stored.As(&result), "query underlying object");
    return result;
}

ComPtr<ID3D12RootSignature> guest_root(ID3D12Device* guest, const D3D12_ROOT_SIGNATURE_DESC& desc, const char* name)
{
    ComPtr<ID3DBlob> serialized, errors;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors), name);
    ComPtr<ID3D12RootSignature> result;
    check(guest->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&result)),
          name);
    return result;
}

// A compute pipeline created through device slot 86 (pipeline state stream) against a guest root.
ComPtr<IUnknown> guest_pipeline(ID3D12Device* guest, ID3D12RootSignature* root, ID3DBlob* code)
{
    struct {
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_type;
        ID3D12RootSignature* root;
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE cs_type;
        D3D12_SHADER_BYTECODE code;
    } stream{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root,
             D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, {code->GetBufferPointer(), code->GetBufferSize()}};
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    using Create = HRESULT(WINAPI*)(IUnknown*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    ComPtr<IUnknown> result;
    check(slot<Create>(guest, 86)(guest, &desc, __uuidof(ID3D12PipelineState), reinterpret_cast<void**>(result.GetAddressOf())),
          "create guest compute pipeline");
    return result;
}

std::wstring log_path;
std::size_t lines_seen = 0;

// The Dispatch lines logged since the previous call.
std::vector<std::string> new_dispatch_lines()
{
    std::vector<std::string> all;
    std::ifstream file(log_path, std::ios::binary);
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.rfind("x2w-render Dispatch(", 0) == 0) all.push_back(line);
    }
    std::vector<std::string> fresh;
    if (all.size() > lines_seen) fresh.assign(all.begin() + static_cast<std::ptrdiff_t>(lines_seen), all.end());
    lines_seen = all.size();
    return fresh;
}

struct Case {
    const char* label;
    UINT x, y, z;
    const char* reason;  // null: the call is forwarded
    int query;           // pipeline-statistics query index, or -1 on a closed list
};

class Fixture {
public:
    Fixture(ID3D12Device* device, ID3D12Device* guest, bool skip_srv) : device_(device), guest_(guest), skip_srv_(skip_srv)
    {
        ComPtr<ID3DBlob> errors;
        const HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0,
                                      &code_, &errors);
        if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        check(hr, "compile compute shader");

        // root_a: one root SRV (t0). root_b: two root SRVs (t0, t1), so its serialized form and desktop object differ
        // from root_a's. root_table: one descriptor table. root_sampler: no parameters, one static sampler.
        D3D12_ROOT_PARAMETER srvs[2]{};
        for (UINT i = 0; i < 2; ++i) {
            srvs[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            srvs[i].Descriptor = {i, 0};
            srvs[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        root_a_ = guest_root(guest, {1, srvs, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE}, "create root with one root SRV");
        root_b_ = guest_root(guest, {2, srvs, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE}, "create root with two root SRVs");
        const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER table{};
        table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table.DescriptorTable = {1, &range};
        table.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_table_ = guest_root(guest, {1, &table, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE}, "create root with a table");
        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        root_sampler_ = guest_root(guest, {0, nullptr, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE},
                                   "create root with a static sampler");
        pso_a_ = guest_pipeline(guest, root_a_.Get(), code_.Get());
        pso_b_ = guest_pipeline(guest, root_b_.Get(), code_.Get());
        pso_table_ = guest_pipeline(guest, root_table_.Get(), code_.Get());
        pso_sampler_ = guest_pipeline(guest, root_sampler_.Get(), code_.Get());

        const D3D12_QUERY_HEAP_DESC heap_desc{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, max_queries, 0};
        check(device->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&queries_)), "create pipeline-statistics query heap");
        D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = max_queries * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS);
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        check(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                              nullptr, IID_PPV_ARGS(&readback_)), "create query readback buffer");
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_)), "create desktop queue");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "create completion fence");

        check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)), "create guest allocator");
        using CreateList = HRESULT(WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
        const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
        check(slot<CreateList>(guest, 73)(guest, list_desc, allocator_.Get(), nullptr, __uuidof(ID3D12GraphicsCommandList),
                                         reinterpret_cast<void**>(list_.GetAddressOf())), "create guest command list");
        native_ = native_object<ID3D12GraphicsCommandList>(device, guest, list_.Get());
    }

    bool run()
    {
        // Recording 1: the fully bound call alone, executed and counted before any refused case runs.
        reset();
        bool ok = expect({"bound", 2, 3, 1, nullptr, 0}, [&] {
            bind(pso_a_.Get(), root_a_.Get());
            if (!skip_srv_) srv(upload_);
        });
        if (!ok) return false;
        resolve(1);
        check(slot<Close>(list_.Get(), 9)(list_.Get()), "close first recording");
        if (!execute(0, 1)) return false;

        // Recording 2: each refusal after a fresh Close, so no binding is left over from recording 1.
        reset();
        ok = expect({"root SRV never set", 3, 1, 1, "a compute root SRV is not bound", 0},
                    [&] { bind(pso_a_.Get(), root_a_.Get()); });
        ok = ok && expect({"rejected replacement after a valid root SRV", 4, 1, 1, "a compute root SRV is not bound", 1}, [&] {
            bind(pso_a_.Get(), root_a_.Get());
            srv(upload_);
            srv(reinterpret_cast<UINT64>(&foreign_));  // not in this recording's upload pages
        });
        ok = ok && expect({"misaligned replacement after a valid root SRV", 5, 1, 1, "a compute root SRV is not bound", 2}, [&] {
            bind(pso_a_.Get(), root_a_.Get());
            srv(upload_);
            srv(upload_ + 2);  // in the upload pages, not 4-byte aligned
        });
        ok = ok && expect({"root changed and changed back", 6, 1, 1, "a compute root SRV is not bound", 3}, [&] {
            bind(pso_a_.Get(), root_a_.Get());
            srv(upload_);
            root(root_b_.Get());
            root(root_a_.Get());
        });
        ok = ok && expect({"pipeline built against another root", 7, 1, 1,
                           "compute pipeline was built against another root signature", 4}, [&] {
            bind(pso_b_.Get(), root_a_.Get());
            srv(upload_);
        });
        ok = ok && expect({"root with a descriptor table", 8, 1, 1, "compute root arguments are not forwarded", 5},
                          [&] { bind(pso_table_.Get(), root_table_.Get()); });
        ok = ok && expect({"root with a static sampler", 9, 1, 1, "compute root arguments are not forwarded", 6},
                          [&] { bind(pso_sampler_.Get(), root_sampler_.Get()); });
        ok = ok && expect({"root SRV never set, zero groups", 0, 10, 1, "a compute root SRV is not bound", 7},
                          [&] { bind(pso_a_.Get(), root_a_.Get()); });
        ok = ok && expect({"bound, X over the limit", 65536, 1, 1, "group count is over 65535", 8}, [&] {
            bind(pso_a_.Get(), root_a_.Get());
            srv(upload_);
        });
        ok = ok && expect({"bound, zero groups", 0, 1, 1, nullptr, 9}, [] {});
        ok = ok && expect({"valid root SRV after a rejected one", 1, 2, 1, nullptr, 10}, [&] {
            srv(reinterpret_cast<UINT64>(&foreign_));
            srv(upload_);
        });
        if (!ok) return false;
        resolve(11);
        check(slot<Close>(list_.Get(), 9)(list_.Get()), "close second recording");
        // Close drops the recording's bindings; a Dispatch on the closed list is refused before any of them.
        if (!expect({"closed list", 11, 1, 1, "list is closed", -1}, [] {})) return false;
        if (!execute(0, 11)) return false;

        // Recording 3: Reset without rebinding, then rebound.
        reset();
        ok = expect({"after Reset without rebinding", 12, 1, 1, "requires a desktop compute pipeline and root", 0}, [] {});
        ok = ok && expect({"rebound after Reset", 2, 2, 1, nullptr, 1}, [&] {
            bind(pso_a_.Get(), root_a_.Get());
            srv(upload_);
        });
        if (!ok) return false;
        resolve(2);
        check(slot<Close>(list_.Get(), 9)(list_.Get()), "close third recording");
        return execute(0, 2);
    }

private:
    static constexpr UINT max_queries = 16;

    void reset()
    {
        check(slot<Reset>(list_.Get(), 139)(list_.Get(), allocator_.Get(), nullptr, nullptr), "reset guest command list");
        upload_ = reinterpret_cast<UINT64>(slot<Upload>(list_.Get(), 148)(list_.Get(), 256, 256));
        if (!upload_) throw std::runtime_error("allocate upload pages through slot 148");
        expected_.clear();
    }
    void bind(IUnknown* pipeline, IUnknown* root_signature)
    {
        slot<Bind>(list_.Get(), 25)(list_.Get(), pipeline);
        root(root_signature);
    }
    void root(IUnknown* root_signature) { slot<Bind>(list_.Get(), 33)(list_.Get(), root_signature); }
    void srv(UINT64 address) { slot<Address>(list_.Get(), 43)(list_.Get(), 0, address); }

    template <typename Setup>
    bool expect(const Case& c, Setup setup)
    {
        setup();
        if (c.query >= 0) native_->BeginQuery(queries_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS, c.query);
        slot<Dispatch>(list_.Get(), 14)(list_.Get(), c.x, c.y, c.z);
        if (c.query >= 0) native_->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS, c.query);
        char wanted[256] = "";
        if (c.reason) std::snprintf(wanted, sizeof(wanted), "x2w-render Dispatch(%u, %u, %u) refused: %s", c.x, c.y, c.z, c.reason);
        const auto lines = new_dispatch_lines();
        const bool matched = c.reason ? lines.size() == 1 && lines[0] == wanted : lines.empty();
        if (!matched) {
            std::fprintf(stderr, "%s: expected %s%s, got %zu Dispatch line(s)\n", c.label, c.reason ? "exactly " : "no Dispatch line",
                         wanted, lines.size());
            for (const auto& line : lines) std::fprintf(stderr, "  %s\n", line.c_str());
            return false;
        }
        std::printf("%s: Dispatch(%u, %u, %u) %s%s\n", c.label, c.x, c.y, c.z, c.reason ? "refused: " : "forwarded",
                    c.reason ? c.reason : "");
        if (c.query >= 0)
            expected_.push_back({c.label, c.reason ? 0 : UINT64{c.x} * c.y * c.z * threads_per_group});
        return true;
    }

    void resolve(UINT count)
    {
        native_->ResolveQueryData(queries_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, count, readback_.Get(), 0);
    }

    bool execute(UINT first, UINT count)
    {
        ID3D12CommandList* lists[]{native_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        check(queue_->Signal(fence_.Get(), ++fence_value_), "signal completion fence");
        check(fence_->SetEventOnCompletion(fence_value_, nullptr), "wait for completion");
        check(device_->GetDeviceRemovedReason(), "GPU completion");
        void* mapped = nullptr;
        const D3D12_RANGE read{0, count * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS)};
        check(readback_->Map(0, &read, &mapped), "map query readback");
        std::vector<D3D12_QUERY_DATA_PIPELINE_STATISTICS> stats(count);
        std::memcpy(stats.data(), mapped, read.End);
        const D3D12_RANGE untouched{0, 0};
        readback_->Unmap(0, &untouched);
        if (expected_.size() != count) {
            std::fprintf(stderr, "internal: %zu queries recorded, %u resolved\n", expected_.size(), count);
            return false;
        }
        bool ok = true;
        for (UINT i = first; i < count; ++i) {
            const UINT64 got = stats[i].CSInvocations;
            std::printf("  query %u (%s): CSInvocations %llu (expected %llu)\n", i, expected_[i].label,
                        static_cast<unsigned long long>(got), static_cast<unsigned long long>(expected_[i].invocations));
            if (got != expected_[i].invocations) {
                std::fprintf(stderr, "%s: CSInvocations %llu, expected %llu\n", expected_[i].label,
                             static_cast<unsigned long long>(got), static_cast<unsigned long long>(expected_[i].invocations));
                ok = false;
            }
        }
        return ok;
    }

    struct Expected {
        const char* label;
        UINT64 invocations;
    };
    ID3D12Device* device_;
    ID3D12Device* guest_;
    bool skip_srv_;
    ComPtr<ID3DBlob> code_;
    ComPtr<ID3D12RootSignature> root_a_, root_b_, root_table_, root_sampler_;
    ComPtr<IUnknown> pso_a_, pso_b_, pso_table_, pso_sampler_;
    ComPtr<ID3D12QueryHeap> queries_;
    ComPtr<ID3D12Resource> readback_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    UINT64 fence_value_ = 0;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<IUnknown> list_;
    ComPtr<ID3D12GraphicsCommandList> native_;
    UINT64 upload_ = 0;
    std::uint32_t foreign_ = 0;
    std::vector<Expected> expected_;
};
}  // namespace

int main(int argc, char** argv) try
{
    const bool warp = argc >= 2 && std::strcmp(argv[1], "warp") == 0;
    if (!warp && (argc < 2 || std::strcmp(argv[1], "hardware") != 0)) return 2;
    const bool skip_srv = argc == 3 && std::strcmp(argv[2], "skip-srv") == 0;
    if (argc > 3 || (argc == 3 && !skip_srv)) return 2;
    ComPtr<ID3D12Debug> debug;
    const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
    if (debug_hr != DXGI_ERROR_SDK_COMPONENT_MISSING) check(debug_hr, "get debug layer");
    if (debug) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "find WARP");
    } else {
        bool found = false;
        for (UINT index = 0; factory->EnumAdapters1(index, adapter.ReleaseAndGetAddressOf()) == S_OK; ++index) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                found = true;
                break;
            }
        }
        if (!found) return 77;
    }
    wchar_t directory[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, directory)) return 4;
    log_path = std::wstring(directory) + L"x2w_list_dispatch_" + std::to_wstring(GetCurrentProcessId()) + L".log";
    DeleteFileW(log_path.c_str());
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    SetEnvironmentVariableA("X2W_D3D12_DEBUG", "0");  // keep the debug messages for the check below
    ComPtr<ID3D12Device> device;
    check(D3D12XboxCreateDevice(adapter.Get(), nullptr, IID_PPV_ARGS(&device)), "create desktop device through d3d12_xs");
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1");
    SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");
    const std::array<unsigned char, 64> parameters{};
    ComPtr<ID3D12Device> guest;
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
    DXGI_ADAPTER_DESC1 adapter_desc{};
    check(adapter->GetDesc1(&adapter_desc), "get adapter description");
    std::printf("adapter: %ls, debug layer %s\n", adapter_desc.Description, debug ? "enabled" : "unavailable");
    LARGE_INTEGER driver{};
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::printf("driver: %u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart), HIWORD(driver.LowPart),
                    LOWORD(driver.LowPart));
    ComPtr<ID3D12InfoQueue> info;
    if (debug) check(device.As(&info), "get debug message queue");

    bool ok = false;
    {
        Fixture fixture(device.Get(), guest.Get(), skip_srv);
        ok = fixture.run();
    }
    if (info) {
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T bytes = 0;
            check(info->GetMessage(i, nullptr, &bytes), "get debug message size");
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(info->GetMessage(i, message, &bytes), "get debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::fprintf(stderr, "D3D12 debug message %u: %s\n", static_cast<unsigned>(message->ID), message->pDescription);
                ok = false;
            }
        }
    }
    DeleteFileW(log_path.c_str());
    if (ok) std::printf("all slot 14 cases matched\n");
    return ok ? 0 : 1;
}
catch (const std::exception&) {
    return 3;
}
