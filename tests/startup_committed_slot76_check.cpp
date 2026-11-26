// Device check for committed resources made through the startup device wrapper's slot 76 (Xbox
// order, +0x260): the seven arguments of desktop CreateCommittedResource (heap properties, heap
// flags, desc, initial state, clear value, riid, ppv). It shares the desktop creation of slot 127
// (tests/startup_committed_heap_state_check.cpp). A 2D render target with a clear value and buffers on
// the default, upload and readback heaps, passed as a title passes them (initial state 0xffffffff),
// must all be created and the device must stay alive.
// usage: startup_committed_slot76_check warp|hardware [--control]
// --control also requires a 2D upload-heap texture to return E_INVALIDARG with no output resource.
#include "xmem_test_device.h"

#include <cstdio>
#include <cstring>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
HRESULT create_wrapper(IUnknown* adapter, REFIID iid, void** out)
{
    unsigned char parameters[64]{};
    return D3D12XboxCreateDevice(adapter, parameters, iid, out);
}

using Committed76 = HRESULT (*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT,
                                const D3D12_CLEAR_VALUE*, const GUID*, void**);
constexpr UINT state_undefined = 0xffffffff;

Committed76 slot76(ID3D12Device* device)
{
    return reinterpret_cast<Committed76>((*reinterpret_cast<void***>(device))[76]);
}

D3D12_RESOURCE_DESC buffer(UINT64 bytes)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_DESC texture(UINT width, UINT height, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

const char* heap_name(D3D12_HEAP_TYPE type)
{
    return type == D3D12_HEAP_TYPE_DEFAULT ? "default" : type == D3D12_HEAP_TYPE_UPLOAD ? "upload" : "readback";
}

// Returns 0 only when the requested creation or rejection is observed and the device stays alive.
int check(ID3D12Device* wrapper, D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC& desc,
          const D3D12_CLEAR_VALUE* clear = nullptr, bool reject = false)
{
    const D3D12_HEAP_PROPERTIES props{type};
    const GUID iid = __uuidof(ID3D12Resource);
    void* created = nullptr;
    const HRESULT hr = slot76(wrapper)(wrapper, &props, 0, &desc, state_undefined, clear, &iid, &created);
    const HRESULT removed = wrapper->GetDeviceRemovedReason();
    const char* what = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? "buffer" : "2D texture";
    std::printf("%s heap, %s of 0x%llx%s: slot 76 -> 0x%08lx\n", heap_name(type), what,
                static_cast<unsigned long long>(desc.Width), clear ? " with clear value" : "", static_cast<unsigned long>(hr));
    const bool expected = reject ? hr == E_INVALIDARG && !created : SUCCEEDED(hr) && created;
    const int result = expected && SUCCEEDED(removed) ? 0 : 1;
    if (reject && !result) std::puts("slot 76 rejected the upload-heap texture with E_INVALIDARG and no resource");
    if (result)
        std::printf("FAIL: %s heap %s: hr 0x%08lx object %p device 0x%08lx\n", heap_name(type), what, static_cast<unsigned long>(hr),
                    created, static_cast<unsigned long>(removed));
    if (created) static_cast<IUnknown*>(created)->Release();
    return result;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && !(argc == 3 && std::strcmp(argv[2], "--control") == 0)) return 1;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    const bool control = argc == 3;
    bool unmatched = false;
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");
    auto wrapper = create_device(warp, unmatched, create_wrapper);
    if (!wrapper) {
        if (warp || unmatched) {
            std::puts("FAIL: no wrapped device");
            return 1;
        }
        return 77;
    }
    int failures = 0;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    failures += check(wrapper.Get(), D3D12_HEAP_TYPE_DEFAULT, texture(128, 64, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), &clear);
    failures += check(wrapper.Get(), D3D12_HEAP_TYPE_DEFAULT, buffer(0x10000));
    failures += check(wrapper.Get(), D3D12_HEAP_TYPE_UPLOAD, buffer(0x10000));
    failures += check(wrapper.Get(), D3D12_HEAP_TYPE_READBACK, buffer(768));
    if (control) failures += check(wrapper.Get(), D3D12_HEAP_TYPE_UPLOAD, texture(64, 64), nullptr, true);
    if (failures) {
        std::printf("startup_committed_slot76: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_committed_slot76: PASS (render target, default, upload and readback resources created through slot 76)");
    return 0;
}
