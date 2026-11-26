# compile_lds_request.py writes a replay for a synthetic compute request. No GPU.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME native_lds_request_cli COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_lds_request_cli_check.py")
    set_tests_properties(native_lds_request_cli PROPERTIES TIMEOUT 60)
endif()
