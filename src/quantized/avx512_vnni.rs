use core::arch::asm;

/// Dispatch to the AVX512-VNNI uint8×int8→int32 micro-kernel for the given row count.
///
/// This uses the same packed A/B ABI as the AVX2 path, but replaces the
/// vpmaddubsw + vpmaddwd pair with the VNNI vpdpbusd dot-product instruction.
///
/// # Safety
/// - CPU must support AVX512VNNI and AVX512VL
/// - `kc` must be > 0 and a multiple of 4
/// - `buffer_a` must have KCB=512 byte stride per row, with `mc` rows packed
/// - `buffer_b` must be in row-interleaved NR=8 format
/// - `buffer_c` must point to valid i32 storage with `ldc_bytes` byte stride
pub unsafe fn dispatch_i8_kernel(
    mc: usize,
    buffer_a: *const u8,
    buffer_b: *const i8,
    buffer_c: *mut i32,
    kc: usize,
    ldc_bytes: usize,
) {
    debug_assert!(mc >= 1 && mc <= 12);
    match mc {
        1 => avx512_vnni_i8_kernel_1(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        2 => avx512_vnni_i8_kernel_2(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        3 => avx512_vnni_i8_kernel_3(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        4 => avx512_vnni_i8_kernel_4(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        5 => avx512_vnni_i8_kernel_5(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        6 => avx512_vnni_i8_kernel_6(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        7 => avx512_vnni_i8_kernel_7(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        8 => avx512_vnni_i8_kernel_8(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        9 => avx512_vnni_i8_kernel_9(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        10 => avx512_vnni_i8_kernel_10(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        11 => avx512_vnni_i8_kernel_11(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        12 => avx512_vnni_i8_kernel_12(buffer_a, buffer_b, buffer_c, kc, ldc_bytes),
        _ => unreachable!("mc must be 1..=12"),
    }
}

macro_rules! define_i8_kernel {
    ($name:ident, [$(($row:literal, $offset:literal)),+ $(,)?]) => {
        #[inline(never)]
        unsafe fn $name(
            buffer_a: *const u8,
            buffer_b: *const i8,
            buffer_c: *mut i32,
            kc: usize,
            ldc_bytes: usize,
        ) {
            debug_assert!(kc > 0 && kc % 4 == 0);
            asm!(
                // Zero accumulators.
                $(concat!(
                    "vpxor ymm", stringify!($row),
                    ", ymm", stringify!($row),
                    ", ymm", stringify!($row),
                ),)+

                // K loop (each iteration processes ROW_INTERLEAVE=4 k-elements).
                "2:",
                "vmovdqu ymm14, ymmword ptr [{b}]",
                $(
                    concat!("vpbroadcastd ymm15, dword ptr [{a} + ", stringify!($offset), "]"),
                    concat!(
                        "vpdpbusd ymm", stringify!($row),
                        ", ymm15, ymm14",
                    ),
                )+

                "add {a}, 4",
                "add {b}, 32",
                "sub {kc}, 4",
                "jnz 2b",

                // Store: C[row] += accumulator[row].
                "mov rax, {c}",
                $(
                    concat!(
                        "vpaddd ymm", stringify!($row),
                        ", ymm", stringify!($row),
                        ", ymmword ptr [rax]",
                    ),
                    concat!("vmovdqu ymmword ptr [rax], ymm", stringify!($row)),
                    "add rax, {ldc}",
                )+

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
                out("ymm14") _, out("ymm15") _,
                options(nostack),
            );
        }
    };
}

define_i8_kernel!(avx512_vnni_i8_kernel_1, [(0, 0)]);
define_i8_kernel!(avx512_vnni_i8_kernel_2, [(0, 0), (1, 512)]);
define_i8_kernel!(avx512_vnni_i8_kernel_3, [(0, 0), (1, 512), (2, 1024)]);
define_i8_kernel!(
    avx512_vnni_i8_kernel_4,
    [(0, 0), (1, 512), (2, 1024), (3, 1536)]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_5,
    [(0, 0), (1, 512), (2, 1024), (3, 1536), (4, 2048)]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_6,
    [(0, 0), (1, 512), (2, 1024), (3, 1536), (4, 2048), (5, 2560)]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_7,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072)
    ]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_8,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584)
    ]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_9,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096)
    ]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_10,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096),
        (9, 4608)
    ]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_11,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096),
        (9, 4608),
        (10, 5120)
    ]
);
define_i8_kernel!(
    avx512_vnni_i8_kernel_12,
    [
        (0, 0),
        (1, 512),
        (2, 1024),
        (3, 1536),
        (4, 2048),
        (5, 2560),
        (6, 3072),
        (7, 3584),
        (8, 4096),
        (9, 4608),
        (10, 5120),
        (11, 5632)
    ]
);
