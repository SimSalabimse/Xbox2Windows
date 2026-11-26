; coverage: record cpu_aot_bmi2_bit
; Source-built MASM fixture for the four BMI2 bit-manipulation forms from
; docs/coverage/x64_zen2.tsv group BMI/ADX: BZHI, MULX, PDEP, and PEXT in 32-bit
; and 64-bit register and flat zero-displacement [rcx|rdx|r8] memory forms, plus
; BZHI CF/ZF/SF/OF flag checks, CMP -> PDEP/PEXT/MULX -> SETcc flag preservation
; (CF, ZF, SF, OF, PF), MULX op0==op1 aliasing, 32-bit zero-extension, [r8]
; source-then-store, and discarded faulting m64 read.
.code

PACK_CF_ZF_SF_OF MACRO
    setb r9b
    sete r10b
    sets r11b
    seto byte ptr [r8]
    movzx r9d, r9b
    movzx r10d, r10b
    mov eax, 2
    pdep r10d, r10d, eax
    or r9d, r10d
    movzx r11d, r11b
    mov eax, 4
    pdep r11d, r11d, eax
    or r9d, r11d
    movzx r10d, byte ptr [r8]
    mov eax, 8
    pdep r10d, r10d, eax
    or r9d, r10d
    mov eax, r9d
    mov dword ptr [r8], eax
ENDM

; --- Register forms (mov rax,-1 / r9,-1 / r10,-1 before 32-bit ops checks [63:32] zero-ext) ---

op_bzhi32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_BIT
    pext eax, ecx, edx
ELSE
    bzhi eax, ecx, edx
ENDIF
    ret
op_bzhi32 ENDP

op_bzhi64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_BIT
    pext rax, rcx, rdx
ELSE
    bzhi rax, rcx, rdx
ENDIF
    ret
op_bzhi64 ENDP

op_mulx32 PROC FRAME
    .endprolog
    mov r9, -1
    mov r10, -1
IFDEF ALTER_BMI2_BIT
    mulx r10d, r9d, ecx
ELSE
    mulx r9d, r10d, ecx
ENDIF
    mov qword ptr [r8], r10
    mov rax, r9
    ret
op_mulx32 ENDP

op_mulx64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_BIT
    mulx r10, r9, rcx
ELSE
    mulx r9, r10, rcx
ENDIF
    mov qword ptr [r8], r10
    mov rax, r9
    ret
op_mulx64 ENDP

op_mulx32_alias PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_BIT
    mulx r9d, eax, ecx
ELSE
    mulx eax, eax, ecx
ENDIF
    ret
op_mulx32_alias ENDP

op_mulx64_alias PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_BIT
    mulx r9, rax, rcx
ELSE
    mulx rax, rax, rcx
ENDIF
    ret
op_mulx64_alias ENDP

op_pdep32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_BIT
    pext eax, ecx, edx
ELSE
    pdep eax, ecx, edx
ENDIF
    ret
op_pdep32 ENDP

op_pdep64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_BIT
    pext rax, rcx, rdx
ELSE
    pdep rax, rcx, rdx
ENDIF
    ret
op_pdep64 ENDP

op_pext32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI2_BIT
    pdep eax, ecx, edx
ELSE
    pext eax, ecx, edx
ENDIF
    ret
op_pext32 ENDP

op_pext64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI2_BIT
    pdep rax, rcx, rdx
ELSE
    pext rax, rcx, rdx
ENDIF
    ret
op_pext64 ENDP

; --- Flat zero-displacement [rcx|rdx] m32 / m64 source forms ---

op_bzhi32_mem PROC FRAME
    .endprolog
    mov rax, -1
    mov r9d, dword ptr [rdx]
    bzhi eax, dword ptr [rcx], r9d
    ret
op_bzhi32_mem ENDP

op_bzhi64_mem PROC FRAME
    .endprolog
    mov r9, qword ptr [rdx]
    bzhi rax, qword ptr [rcx], r9
    ret
op_bzhi64_mem ENDP

op_mulx32_mem PROC FRAME
    .endprolog
    mov r9, -1
    mov r10, -1
    mulx r9d, r10d, dword ptr [rcx]
    mov qword ptr [r8], r10
    mov rax, r9
    ret
op_mulx32_mem ENDP

op_mulx64_mem PROC FRAME
    .endprolog
    mulx r9, r10, qword ptr [rcx]
    mov qword ptr [r8], r10
    mov rax, r9
    ret
op_mulx64_mem ENDP

op_pdep32_mem PROC FRAME
    .endprolog
    mov rax, -1
    pdep eax, ecx, dword ptr [rdx]
    ret
op_pdep32_mem ENDP

op_pdep64_mem PROC FRAME
    .endprolog
    pdep rax, rcx, qword ptr [rdx]
    ret
op_pdep64_mem ENDP

op_pext32_mem PROC FRAME
    .endprolog
    mov rax, -1
    pext eax, ecx, dword ptr [rdx]
    ret
op_pext32_mem ENDP

op_pext64_mem PROC FRAME
    .endprolog
    pext rax, rcx, qword ptr [rdx]
    ret
op_pext64_mem ENDP

; --- [r8] source-then-store and program order ---

op_mem_r8_source PROC FRAME
    .endprolog
    mov r9, qword ptr [rcx]
    pdep r10, r9, qword ptr [r8]
    pext r11, r10, qword ptr [rdx]
    mulx r9, rax, qword ptr [r8]
    xor rax, r11
    mov qword ptr [r8], r10
    ret
op_mem_r8_source ENDP

; --- BZHI flags: (CF << 0) | (ZF << 1) | (SF << 2) | (OF << 3) ---

op_bzhi32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    mov rax, -1
    bzhi eax, ecx, edx
    PACK_CF_ZF_SF_OF
    ret
op_bzhi32_flags ENDP

op_bzhi64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    bzhi rax, rcx, rdx
    PACK_CF_ZF_SF_OF
    ret
op_bzhi64_flags ENDP

; --- Flag preservation across CMP -> PDEP/PEXT/MULX -> SETcc (CF/ZF/SF/OF and PF) ---

op_flags_preserve32 PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    cmp ecx, edx
    pdep r9d, ecx, edx
    pext r10d, r9d, edx
    mulx r11d, eax, ecx
    PACK_CF_ZF_SF_OF
    ret
op_flags_preserve32 ENDP

op_flags_preserve64 PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    cmp rcx, rdx
    pdep r9, rcx, rdx
    pext r10, r9, rdx
    mulx r11, rax, rcx
    PACK_CF_ZF_SF_OF
    ret
op_flags_preserve64 ENDP

op_flags_preserve_pf PROC FRAME
    .endprolog
    cmp rcx, rdx
    pdep r9, rcx, rdx
    pext r10, r9, rdx
    mulx r11, rax, rcx
    setp byte ptr [r8]
    ret
op_flags_preserve_pf ENDP

op_flags_preserve_mem PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    mov r9, qword ptr [rcx]
    mov r10, qword ptr [rdx]
    cmp r9, r10
    pdep r11, r9, qword ptr [rdx]
    pext r9, r11, qword ptr [rcx]
    mulx r10, rax, qword ptr [rcx]
    PACK_CF_ZF_SF_OF
    ret
op_flags_preserve_mem ENDP

; --- Discarded memory read must still fault on NULL rdx even under /O2 ---

op_bmi2_bit_discarded_read PROC FRAME
    .endprolog
    pext r9, rcx, qword ptr [rdx]
    mov rax, rcx
    ret
op_bmi2_bit_discarded_read ENDP

END
