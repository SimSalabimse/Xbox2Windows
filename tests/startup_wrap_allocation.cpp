// #2653: when the D3D12 stand-in cannot allocate an object wrapper, the slot that creates the object returns
// E_OUTOFMEMORY with a null output instead of writing through a null pointer. The fixture-only export
// x2w_d3d12_fail_wraps makes the next wrapper allocations fail; each slot is then called once armed and once
// normally.
//
//   startup_wrap_allocation [control]   (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_DEBUG=0)
//     "control" never arms the failure, so every armed expectation must fail and the fixture must print FAIL.
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_fail_wraps(LONG count);

namespace {

bool control;
int failures;

template<class... Args>
HRESULT invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = HRESULT (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

void arm()
{
    if (!control) x2w_d3d12_fail_wraps(1);
}

// One creation: armed it must return E_OUTOFMEMORY and leave the output null; then it must succeed.
template<class Create>
void check(const char* name, Create create)
{
    ComPtr<IUnknown> object;
    arm();
    HRESULT hr = create(object.ReleaseAndGetAddressOf());
    x2w_d3d12_fail_wraps(0);
    const bool refused = hr == E_OUTOFMEMORY && !object;
    if (!refused) {
        std::printf("FAIL: %s with the wrapper allocation failing returned 0x%08lx and %s output\n", name,
                    static_cast<unsigned long>(hr), object ? "a non-null" : "a null");
        ++failures;
    }
    hr = create(object.ReleaseAndGetAddressOf());
    if (FAILED(hr) || !object) {
        std::printf("FAIL: %s afterwards returned 0x%08lx\n", name, static_cast<unsigned long>(hr));
        ++failures;
        return;
    }
    if (refused) std::printf("%s: E_OUTOFMEMORY with a null output, then S_OK\n", name);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "control"))) return 2;
    control = argc == 2;

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))))
        return 77;
    std::array<unsigned char, 64> parameters{};

    // Device creation itself: the desktop device reference is released and the output stays null.
    ComPtr<IUnknown> device;
    check("D3D12XboxCreateDevice", [&](IUnknown** out) {
        return D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), reinterpret_cast<void**>(out));
    });
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                     reinterpret_cast<void**>(device.GetAddressOf()))) || !device) {
        std::puts("FAIL: no device to continue with");
        return 3;
    }

    // Slot 55, derived pipeline: the placeholder path of the reported stop.
    check("Device::slot55 derived pipeline", [&](IUnknown** out) {
        return invoke(device.Get(), 55, static_cast<void*>(nullptr), 0u, static_cast<const void*>(nullptr),
                      &__uuidof(ID3D12PipelineState), reinterpret_cast<void**>(out));
    });

    // Slot 8, command queue: a desktop queue is created before the wrapper.
    const D3D12_COMMAND_QUEUE_DESC queue{D3D12_COMMAND_LIST_TYPE_DIRECT};
    check("Device::CreateCommandQueue", [&](IUnknown** out) {
        return invoke(device.Get(), 8, &queue, &__uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(out));
    });

    // A failed device creation must leave the previously published device visible to its children.
    if (!control) {
        ComPtr<ID3D12CommandQueue> existing_queue;
        HRESULT hr = invoke(device.Get(), 8, &queue, &__uuidof(ID3D12CommandQueue),
                            reinterpret_cast<void**>(existing_queue.GetAddressOf()));
        if (FAILED(hr) || !existing_queue) return 3;

        ComPtr<IUnknown> second_device;
        arm();
        hr = D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                  reinterpret_cast<void**>(second_device.GetAddressOf()));
        x2w_d3d12_fail_wraps(0);
        if (hr != E_OUTOFMEMORY || second_device) {
            std::puts("FAIL: second device allocation did not return E_OUTOFMEMORY and a null output");
            ++failures;
        }
        ComPtr<IUnknown> parent;
        hr = existing_queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(parent.GetAddressOf()));
        if (FAILED(hr) || parent.Get() != device.Get()) {
            std::puts("FAIL: failed device creation changed the existing queue's device");
            ++failures;
        }
    }

    // Slot 16, root signature: a desktop root signature is created before the wrapper.
    ComPtr<ID3DBlob> blob;
    const D3D12_ROOT_SIGNATURE_DESC empty{};
    if (FAILED(D3D12SerializeRootSignature(&empty, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr))) return 3;
    check("Device::CreateRootSignature", [&](IUnknown** out) {
        return invoke(device.Get(), 16, 0u, blob->GetBufferPointer(), blob->GetBufferSize(), &__uuidof(ID3D12RootSignature),
                      reinterpret_cast<void**>(out));
    });

    // Slot 14, descriptor heap: memory of the stand-in's own may back the heap before the wrapper.
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4}, 0, 0};
    check("Device::CreateDescriptorHeap", [&](IUnknown** out) {
        return invoke(device.Get(), 14, &heap, &__uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(out));
    });

    // Slot 9, command allocator: the shared creation path (create_slot) wraps what desktop returned.
    check("Device::CreateCommandAllocator", [&](IUnknown** out) {
        return invoke(device.Get(), 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                      reinterpret_cast<void**>(out));
    });

    // Slot 73, command list on that allocator: a desktop list is created before the 64 KiB list wrapper.
    ComPtr<IUnknown> allocator;
    if (FAILED(invoke(device.Get(), 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                      reinterpret_cast<void**>(allocator.GetAddressOf()))))
        return 3;
    const std::array<UINT, 4> list_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    check("Device::slot73 command list", [&](IUnknown** out) {
        return invoke(device.Get(), 73, static_cast<const void*>(list_desc.data()), static_cast<void*>(allocator.Get()),
                      static_cast<void*>(nullptr), &__uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(out));
    });

    // Slot 127, committed resource: a desktop buffer is created before the wrapper.
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 65536;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    check("Device::slot127 committed resource", [&](IUnknown** out) {
        return invoke(device.Get(), 127, &props, 0u, &buffer, 0u, static_cast<const D3D12_CLEAR_VALUE*>(nullptr),
                      static_cast<void*>(nullptr), 0u, static_cast<const void*>(nullptr), &__uuidof(ID3D12Resource),
                      reinterpret_cast<void**>(out));
    });

    if (failures) {
        std::printf("FAIL: %d check(s)\n", failures);
        return 1;
    }
    std::puts("PASS");
    return 0;
}
