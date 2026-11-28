# Direct C++ compute: GLOBAL DWORD stores through a root UAV.
# encodings checks the source-built words. hardware compiles them and reads the buffer back.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_global_store_encodings COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_global_store_check.py" encodings)
endif()
if(EXISTS "${INTAKE_PYTHON}" AND TARGET compile_dxil AND TARGET native_store_replay)
    add_test(NAME native_global_store_hardware COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_global_store_check.py" "$<TARGET_FILE:compile_dxil>"
        "$<TARGET_FILE:native_store_replay>" "${DXC_BIN_DIR}")
    set_tests_properties(native_global_store_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
