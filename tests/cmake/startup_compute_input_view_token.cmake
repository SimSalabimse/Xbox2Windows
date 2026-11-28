# Compute input view tokens. reference and controls are pure Python. The readback needs a
# desktop device and otherwise reports a skip (77).
if(EXISTS "${INTAKE_PYTHON}")
    foreach(mode IN ITEMS reference controls)
        add_test(NAME startup_compute_input_view_token_${mode} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_compute_input_view_token_check.py" ${mode})
        set_tests_properties(startup_compute_input_view_token_${mode} PROPERTIES TIMEOUT 60)
    endforeach()
endif()
if(TARGET startup_stream_consumer AND EXISTS "${INTAKE_PYTHON}" AND EXISTS "${DXC_EXECUTABLE}")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_compute_input_view_token_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_compute_input_view_token_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" "${DXC_EXECUTABLE}" ${adapter})
        set_tests_properties(startup_compute_input_view_token_${adapter} PROPERTIES
            TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
