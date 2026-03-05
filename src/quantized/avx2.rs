use core::arch::asm;

use super::pack::KCB;

/// AVX2 uint8×int8→int32 GEMM micro-kernel for mc=12 rows × NR=8 columns.
///
/// Processes 12 rows of A against one NR-column block of B over kc K-elements.
/// A must be packed with KCB=512 stride per row. B is row-interleaved.
/// C is accumulated (C += A*B). ldc_bytes is C stride in bytes.
///
/// Register allocation:
///   ymm0-ymm11: C accumulators (12 rows × 1 column block)
///   ymm12: temp result
///   ymm13: 16-bit ones vector (for vpmaddwd)
///   ymm14: B register
///   ymm15: A broadcast register
#[inline(never)]
pub unsafe fn avx2_i8_kernel_12(
    buffer_a: *const u8,
    buffer_b: *const i8,
    buffer_c: *mut i32,
    kc: usize,        // must be > 0 and multiple of 4
    ldc_bytes: usize,
) {
    debug_assert!(kc > 0 && kc % 4 == 0);
    asm!(
        // Init ones: ymm13 = all 16-bit 1s
        "vpcmpeqw ymm13, ymm13, ymm13",
        "vpsrlw ymm13, ymm13, 15",

        // Zero accumulators
        "vpxor ymm0, ymm0, ymm0",
        "vpxor ymm1, ymm1, ymm1",
        "vpxor ymm2, ymm2, ymm2",
        "vpxor ymm3, ymm3, ymm3",
        "vpxor ymm4, ymm4, ymm4",
        "vpxor ymm5, ymm5, ymm5",
        "vpxor ymm6, ymm6, ymm6",
        "vpxor ymm7, ymm7, ymm7",
        "vpxor ymm8, ymm8, ymm8",
        "vpxor ymm9, ymm9, ymm9",
        "vpxor ymm10, ymm10, ymm10",
        "vpxor ymm11, ymm11, ymm11",

        // K loop (each iteration processes ROW_INTERLEAVE=4 k-elements)
        "2:",
        "vmovdqu ymm14, ymmword ptr [{b}]",

        "vpbroadcastd ymm15, dword ptr [{a}]",
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm0, ymm0, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(512), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm1, ymm1, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(1024), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm2, ymm2, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(1536), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm3, ymm3, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(2048), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm4, ymm4, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(2560), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm5, ymm5, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(3072), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm6, ymm6, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(3584), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm7, ymm7, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(4096), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm8, ymm8, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(4608), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm9, ymm9, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(5120), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm10, ymm10, ymm12",

        concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!(5632), "]"),
        "vpmaddubsw ymm12, ymm15, ymm14",
        "vpmaddwd ymm12, ymm13, ymm12",
        "vpaddd ymm11, ymm11, ymm12",

        // Advance pointers
        "add {a}, 4",
        "add {b}, 32",
        "sub {kc}, 4",
        "jnz 2b",

        // Store: C[row] += accumulator[row]
        "mov rax, {c}",
        "vpaddd ymm0, ymm0, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm0",
        "add rax, {ldc}",
        "vpaddd ymm1, ymm1, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm1",
        "add rax, {ldc}",
        "vpaddd ymm2, ymm2, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm2",
        "add rax, {ldc}",
        "vpaddd ymm3, ymm3, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm3",
        "add rax, {ldc}",
        "vpaddd ymm4, ymm4, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm4",
        "add rax, {ldc}",
        "vpaddd ymm5, ymm5, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm5",
        "add rax, {ldc}",
        "vpaddd ymm6, ymm6, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm6",
        "add rax, {ldc}",
        "vpaddd ymm7, ymm7, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm7",
        "add rax, {ldc}",
        "vpaddd ymm8, ymm8, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm8",
        "add rax, {ldc}",
        "vpaddd ymm9, ymm9, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm9",
        "add rax, {ldc}",
        "vpaddd ymm10, ymm10, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm10",
        "add rax, {ldc}",
        "vpaddd ymm11, ymm11, ymmword ptr [rax]",
        "vmovdqu ymmword ptr [rax], ymm11",

        "vzeroupper",

        a = inout(reg) buffer_a => _,
        b = inout(reg) buffer_b => _,
        c = in(reg) buffer_c,
        kc = inout(reg) kc => _,
        ldc = in(reg) ldc_bytes,
        out("rax") _,
        out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
        out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
        out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
        options(nostack),
    );
}

/// Pack A tile: copy mc rows × kc columns from quantized A (K stride) into
/// a KCB-strided scratchpad for the SIMD kernel.
pub fn pack_a_tile(
    a_u8: &[u8],
    m_start: usize,
    k_start: usize,
    mc: usize,
    kc: usize,
    k: usize,
    dst: &mut [u8],
) {
    debug_assert!(dst.len() >= mc * KCB);
    for i in 0..mc {
        let src_offset = (m_start + i) * k + k_start;
        let dst_offset = i * KCB;
        dst[dst_offset..dst_offset + kc]
            .copy_from_slice(&a_u8[src_offset..src_offset + kc]);
        // Zero-pad remainder (dst was zero-initialized by caller)
    }
}
