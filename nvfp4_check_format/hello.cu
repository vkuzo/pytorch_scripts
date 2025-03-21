#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda_fp4.h>
#include <stdint.h>

int main() {
    
    // 1.0f in fp32 is encoded as 0b0010 in fp4_e2m1
    float val1 = 1.0f;
    // -6.0f in fp32 is encoded as 0b1111 in fp4_e2m1
    float val2 = -6.0f;

    printf("Original %f %f\n", val1, val2);

    __nv_fp4x2_e2m1 packed_fp4 = __nv_fp4x2_e2m1{float2{val1, val2}};

    // get the raw representation
    union {
        __nv_fp4x2_storage_t fp4_val;
        uint8_t byte;  // __nv_fp4x2_storage_t is 1 byte (two 4-bit values)
    } converter;
    converter.fp4_val = packed_fp4.__x;

    // just to double check, round trip back to float with NVIDIA APIs
    __nv_fp4_interpretation_t interp = __NV_E2M1;
    __half2_raw packed_half = __nv_cvt_fp4x2_to_halfraw2(packed_fp4.__x, interp);
    __half2 unpacked_half = *(__half2*)&packed_half;
    __half high = __high2half(unpacked_half);
    __half low = __low2half(unpacked_half);
    float high_f = (float)high;
    float low_f = (float)low;

    // prints 242, which is 0b11110010, i.e. the encoding is val2:val1, from MSB to LSB
    printf("%u:%2.1f|%2.1f\n", converter.byte, high_f, low_f);

    return 0;
}
