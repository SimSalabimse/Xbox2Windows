; coverage: record cpu_aot_avx_horiz_fp_dedicated
; Source-owned MASM fixture for VEX.128 and VEX.256 AVX horizontal and alternating
; floating-point add/sub ops (vhaddps, vhaddpd, vhsubps, vhsubpd, vaddsubps, vaddsubpd).
; ABI: void fn(const void* rcx, const void* rdx, void* r8).
.code

op_vhaddps_xmm_reg PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
    vhaddps xmm0, xmm1, xmm2
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddps_xmm_reg ENDP

op_vhaddpd_xmm_reg PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_AVX_HORIZ_FP
    vhaddpd ymm0, ymm1, ymm2
ELSE
    vhaddpd xmm0, xmm1, xmm2
ENDIF
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddpd_xmm_reg ENDP

op_vhsubps_xmm_reg PROC FRAME
    .endprolog
    vmovaps ymm0, YMMWORD PTR [r8]
    vmovaps ymm1, YMMWORD PTR [rcx]
    vmovaps ymm2, YMMWORD PTR [rdx]
    vhsubps xmm0, xmm1, xmm2
    vmovaps YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubps_xmm_reg ENDP

op_vhsubpd_xmm_reg PROC FRAME
    .endprolog
    vmovapd ymm0, YMMWORD PTR [r8]
    vmovapd ymm1, YMMWORD PTR [rcx]
    vmovapd ymm2, YMMWORD PTR [rdx]
    vhsubpd xmm0, xmm1, xmm2
    vmovapd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubpd_xmm_reg ENDP

op_vaddsubps_xmm_reg PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
    vaddsubps xmm0, xmm1, xmm2
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubps_xmm_reg ENDP

op_vaddsubpd_xmm_reg PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
    vaddsubpd xmm0, xmm1, xmm2
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubpd_xmm_reg ENDP

op_vhaddps_xmm_mem PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vhaddps xmm0, xmm1, XMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddps_xmm_mem ENDP

op_vhaddpd_xmm_mem PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vhaddpd xmm0, xmm1, XMMWORD PTR [rdx]
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddpd_xmm_mem ENDP

op_vhsubps_xmm_mem PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vhsubps xmm0, xmm1, XMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubps_xmm_mem ENDP

op_vhsubpd_xmm_mem PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vhsubpd xmm0, xmm1, XMMWORD PTR [rdx]
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubpd_xmm_mem ENDP

op_vaddsubps_xmm_mem PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vaddsubps xmm0, xmm1, XMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubps_xmm_mem ENDP

op_vaddsubpd_xmm_mem PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vaddsubpd xmm0, xmm1, XMMWORD PTR [rdx]
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubpd_xmm_mem ENDP

op_vhaddps_ymm_reg PROC FRAME
    .endprolog
    vmovaps ymm1, YMMWORD PTR [rcx]
    vmovaps ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_AVX_HORIZ_FP
    vhsubps ymm0, ymm1, ymm2
ELSE
    vhaddps ymm0, ymm1, ymm2
ENDIF
    vmovaps YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddps_ymm_reg ENDP

op_vhaddpd_ymm_reg PROC FRAME
    .endprolog
    vmovapd ymm1, YMMWORD PTR [rcx]
    vmovapd ymm2, YMMWORD PTR [rdx]
    vhaddpd ymm0, ymm1, ymm2
    vmovapd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddpd_ymm_reg ENDP

op_vhsubps_ymm_reg PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
    vhsubps ymm0, ymm1, ymm2
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubps_ymm_reg ENDP

op_vhsubpd_ymm_reg PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
    vhsubpd ymm0, ymm1, ymm2
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubpd_ymm_reg ENDP

op_vaddsubps_ymm_reg PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_AVX_HORIZ_FP
    vhaddps ymm0, ymm1, ymm2
ELSE
    vaddsubps ymm0, ymm1, ymm2
ENDIF
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubps_ymm_reg ENDP

op_vaddsubpd_ymm_reg PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
    vaddsubpd ymm0, ymm1, ymm2
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubpd_ymm_reg ENDP

op_vhaddps_ymm_mem PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vhaddps ymm0, ymm1, YMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddps_ymm_mem ENDP

op_vhaddpd_ymm_mem PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vhaddpd ymm0, ymm1, YMMWORD PTR [rdx]
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddpd_ymm_mem ENDP

op_vhsubps_ymm_mem PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vhsubps ymm0, ymm1, YMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubps_ymm_mem ENDP

op_vhsubpd_ymm_mem PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vhsubpd ymm0, ymm1, YMMWORD PTR [rdx]
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubpd_ymm_mem ENDP

op_vaddsubps_ymm_mem PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vaddsubps ymm0, ymm1, YMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubps_ymm_mem ENDP

op_vaddsubpd_ymm_mem PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_AVX_HORIZ_FP
    vhsubpd ymm0, ymm1, YMMWORD PTR [rdx]
ELSE
    vaddsubpd ymm0, ymm1, YMMWORD PTR [rdx]
ENDIF
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubpd_ymm_mem ENDP

op_vhaddps_ymm_mem_rcx PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rdx]
    vhaddps ymm0, ymm1, YMMWORD PTR [rcx]
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddps_ymm_mem_rcx ENDP

op_vaddsubpd_ymm_mem_r8 PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vaddsubpd ymm0, ymm1, YMMWORD PTR [r8]
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubpd_ymm_mem_r8 ENDP

op_vhaddps_ymm_alias_dst_src1 PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups ymm1, YMMWORD PTR [rdx]
    vhaddps ymm0, ymm0, ymm1
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddps_ymm_alias_dst_src1 ENDP

op_vhaddpd_ymm_alias_dst_src2 PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm0, YMMWORD PTR [rdx]
    vhaddpd ymm0, ymm1, ymm0
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhaddpd_ymm_alias_dst_src2 ENDP

op_vhsubps_ymm_alias_src1_src2 PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vhsubps ymm0, ymm1, ymm1
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vhsubps_ymm_alias_src1_src2 ENDP

op_vaddsubps_ymm_alias_all PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vaddsubps ymm0, ymm0, ymm0
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubps_ymm_alias_all ENDP

op_vaddsubpd_xmm_alias_dst_src2 PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm0, YMMWORD PTR [rdx]
    vaddsubpd xmm0, xmm1, xmm0
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vaddsubpd_xmm_alias_dst_src2 ENDP

END
