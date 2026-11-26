; coverage: source cpu_aot_avx_int_ymm_widen_unpack
; Source-owned differential fixture for 256-bit (VEX.L=1, AVX2) packed-integer
; cross-lane extend moves (VPMOVSX*/VPMOVZX*) and in-lane unpacks (VPUNPCK{L,H}{DQ,QDQ}).
;   RCX = const void* input_a (32 bytes)
;   RDX = const void* input_b (32 bytes)
;   R8  = void*       output  (32 bytes)
.code

op_vpmovsxbw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovzxbw ymm0, xmm1
ELSE
    vpmovsxbw ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxbw_ymm ENDP

op_vpmovsxbd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovzxbd ymm0, xmm1
ELSE
    vpmovsxbd ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxbd_ymm ENDP

op_vpmovsxbq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovzxbq ymm0, xmm1
ELSE
    vpmovsxbq ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxbq_ymm ENDP

op_vpmovsxwd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovzxwd ymm0, xmm1
ELSE
    vpmovsxwd ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxwd_ymm ENDP

op_vpmovsxwq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovzxwq ymm0, xmm1
ELSE
    vpmovsxwq ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxwq_ymm ENDP

op_vpmovsxdq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovzxdq ymm0, xmm1
ELSE
    vpmovsxdq ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxdq_ymm ENDP

op_vpmovzxbw_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovsxbw ymm0, xmm1
ELSE
    vpmovzxbw ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovzxbw_ymm ENDP

op_vpmovzxbd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovsxbd ymm0, xmm1
ELSE
    vpmovzxbd ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovzxbd_ymm ENDP

op_vpmovzxbq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovsxbq ymm0, xmm1
ELSE
    vpmovzxbq ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovzxbq_ymm ENDP

op_vpmovzxwd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovsxwd ymm0, xmm1
ELSE
    vpmovzxwd ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovzxwd_ymm ENDP

op_vpmovzxwq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovsxwq ymm0, xmm1
ELSE
    vpmovzxwq ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovzxwq_ymm ENDP

op_vpmovzxdq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_YMM
    vpmovsxdq ymm0, xmm1
ELSE
    vpmovzxdq ymm0, xmm1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovzxdq_ymm ENDP

op_vpunpckldq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpunpckhdq ymm0, ymm1, ymm2
ELSE
    vpunpckldq ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpunpckldq_ymm ENDP

op_vpunpckhdq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpunpckldq ymm0, ymm1, ymm2
ELSE
    vpunpckhdq ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpunpckhdq_ymm ENDP

op_vpunpcklqdq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpunpckhqdq ymm0, ymm1, ymm2
ELSE
    vpunpcklqdq ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpunpcklqdq_ymm ENDP

op_vpunpckhqdq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_YMM
    vpunpcklqdq ymm0, ymm1, ymm2
ELSE
    vpunpckhqdq ymm0, ymm1, ymm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpunpckhqdq_ymm ENDP

op_vpmovsxbd_ymm_stale_dst PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpmovsxbd ymm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxbd_ymm_stale_dst ENDP

op_vpmovsxbw_ymm_alias PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpmovsxbw ymm1, xmm1
    vmovdqu YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_vpmovsxbw_ymm_alias ENDP

op_vpmovzxdq_ymm_alias PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vpmovzxdq ymm2, xmm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vpmovzxdq_ymm_alias ENDP

op_vpunpckldq_ymm_dst_src1 PROC FRAME
    .endprolog
    vmovdqu ymm3, YMMWORD PTR [rcx]
    vmovdqu ymm4, YMMWORD PTR [rdx]
    vpunpckldq ymm3, ymm3, ymm4
    vmovdqu YMMWORD PTR [r8], ymm3
    vzeroupper
    ret
op_vpunpckldq_ymm_dst_src1 ENDP

op_vpunpckhqdq_ymm_dst_src2 PROC FRAME
    .endprolog
    vmovdqu ymm4, YMMWORD PTR [rcx]
    vmovdqu ymm5, YMMWORD PTR [rdx]
    vpunpckhqdq ymm5, ymm4, ymm5
    vmovdqu YMMWORD PTR [r8], ymm5
    vzeroupper
    ret
op_vpunpckhqdq_ymm_dst_src2 ENDP

op_vpunpckhdq_ymm_all PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vpunpckhdq ymm2, ymm2, ymm2
    vmovdqu YMMWORD PTR [r8], ymm2
    vzeroupper
    ret
op_vpunpckhdq_ymm_all ENDP

op_vpmovsxwd_ymm_then_xmm_vpunpcklqdq PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpmovsxwd ymm0, xmm1
    vpunpcklqdq xmm0, xmm0, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vpmovsxwd_ymm_then_xmm_vpunpcklqdq ENDP

op_chain_ymm_widen_unpack PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpmovsxwd ymm3, xmm1
    vpmovzxbd ymm4, xmm2
    vpunpckldq ymm0, ymm3, ymm4
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_chain_ymm_widen_unpack ENDP

END
