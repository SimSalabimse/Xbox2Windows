#pragma once

#include <windows.h>
#include <cstdio>
#include <cwchar>
#include <iterator>

// Hardware fixtures match XBOX2WINDOWS_ADAPTER against the DXGI description.
class AdapterNameFilter {
public:
    bool read()
    {
        length = GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
        if (length < std::size(name)) return true;
        std::fputs("XBOX2WINDOWS_ADAPTER is longer than 127 characters\n", stderr);
        return false;
    }

    bool matches(const wchar_t* description) const
    {
        if (!length) return true;
        for (const wchar_t* p = description; *p; ++p)
            if (_wcsnicmp(p, name, length) == 0) return true;
        return false;
    }

    int no_match() const
    {
        if (!length) return 77;
        std::fprintf(stderr, "No D3D12 hardware adapter matches XBOX2WINDOWS_ADAPTER=%ls\n", name);
        return 4;
    }

private:
    wchar_t name[128]{};
    DWORD length = 0;
};
