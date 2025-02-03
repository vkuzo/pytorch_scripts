#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include <bitset>
#include <cmath>
#include <iostream>
#include <string>

using namespace llvm;

// testing out LLVM's e8m0
// PR adding this functionality, for context: https://github.com/llvm/llvm-project/pull/107127
// this isn't in released LLVM versions, so need to build from source to test it out

// access bits of a float
union float_or_uint32_t {
  float f;
  uint32_t i;
};


void int_exponent_to_e8m0(int unbiased_exponent, const std::string& description) {
  std::cout << "description: " << description << ", unbiased exponent: " << unbiased_exponent << std::endl;
  // create an e8m0 value
  APFloat::Semantics Sem = APFloat::S_Float8E8M0FNU;
  const llvm::fltSemantics &S = APFloat::EnumToSemantics(Sem);
  const auto e8m0_bias = 127;
  const auto biased_exponent = unbiased_exponent + e8m0_bias;
  std::cout << "  biased exponent: " << biased_exponent << std::endl;
  APFloat e8m0_val(S, APInt(8, biased_exponent));

  // print the raw bits
  uint64_t raw_bits = e8m0_val.bitcastToAPInt().getZExtValue();
  std::cout << "  e8m0 bits: " << std::bitset<8>(raw_bits) << "\n";

  // convert it to float 
  const float e8m0_val_fp32 = e8m0_val.convertToFloat();
  std::cout << "  e8m0 -> float32 cast result: " << e8m0_val_fp32 << std::endl;
  // note: below is wrong for nan, but not worth fixing here
  std::cout << "  expected cast result: " << pow(2, unbiased_exponent) << std::endl;
  std::cout << std::endl;
}

float float32_val_to_e8m0(
  float val, 
  const std::string& description, 
  bool debug=false
) {

  APFloat val_float32(val);  // 32-bit float
  if (debug) {
    std::cout << "description: " << description << ", float32 val: " << val_float32.convertToFloat() << std::endl;
  }

  APFloat::Semantics Sem = APFloat::S_Float8E8M0FNU;
  const llvm::fltSemantics &S = APFloat::EnumToSemantics(Sem);

  uint64_t raw_bits_fp32 = val_float32.bitcastToAPInt().getZExtValue();
  if (debug) {
    std::cout << "  fp32_org bits: " << std::bitset<32>(raw_bits_fp32) << "\n";
  }

  bool losesInfo;
  val_float32.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);

  if (debug) {
    // print the raw bits
    uint64_t raw_bits_e8m0 = val_float32.bitcastToAPInt().getZExtValue();
    std::cout << "  e8m0 bits:      " << std::bitset<8>(raw_bits_e8m0) << "\n";
  }

  float val_float32_e8m0_float32 = val_float32.convertToFloat();

  if (debug) {
    // access bits of a float
    union float_or_uint32_t u;
    u.f = val_float32_e8m0_float32;
    std::cout << "  fp32_new bits: " << std::bitset<32>(u.i) << "\n";

    std::cout << "  e8m0 -> float32 cast result: " << val_float32_e8m0_float32 << std::endl;
    std::cout << std::endl;
  }

  return val_float32_e8m0_float32;
}

void fp32_e8m0_fp32_check_all_grs(const uint8_t exponent) {
  union float_or_uint32_t u;
  // iterate through the 8 possible combinations of GRS
  for (uint8_t grs = 0; grs < 8; grs++) {
    // create a float32 number via bit manipulation
    u.i = (0b0 << 31) | (exponent << 23) | (grs << (23 - 3));
    // TODO NaNs
    bool expect_round_up;
    uint8_t g = grs >> 2;
    uint8_t r = (grs & 0b010) >> 1;
    uint8_t s = grs & 0b1;

    if (g == 0) {
      expect_round_up = false;
    } else {
      if ((r == 1) | (s == 1)) {
        expect_round_up = true;
      } else {
        if (exponent > 0) {
          // normal, round up
          expect_round_up = true;
        } else {
          // denormal, round down
          expect_round_up = false;
        }
      }
    }

    const auto description = expect_round_up ? "round up" : "truncate";
    const auto float32_e8m0_float32 = float32_val_to_e8m0(u.f, description);
    float expected_value = expect_round_up ? pow(2, exponent + 1 - 127) : pow(2, exponent - 127);
    if (float32_e8m0_float32 != expected_value) {
      std::cout << "MISMATCH: expected " << expected_value << ", got " << float32_e8m0_float32 << std::endl;
      std::cout << "exponent " << static_cast<int>(exponent) << 
        " grs " << static_cast<int>(grs) << 
        " res " << float32_e8m0_float32 << 
        " pow2 " << pow(2, exponent - 127) << 
        " pow2+1 " << pow(2, exponent + 1 - 127) << 
        " round_up " << expect_round_up << std::endl << std::endl;
    }
  }
}

int main() {
  std::cout << "start e8m0 test" << std::endl;
  std::cout << std::endl;

  // test directly instantiang e8m0 from an integer, and the subsequent cast to float32
  std::cout << "=== test int -> e8m0 -> float32 ===" << std::endl << std::endl;

  int_exponent_to_e8m0(-127, "min_representable");
  int_exponent_to_e8m0(-2, "neg_two");
  int_exponent_to_e8m0(-1, "neg_one");
  int_exponent_to_e8m0(0, "zero");
  int_exponent_to_e8m0(1, "one");
  int_exponent_to_e8m0(2, "two");
  int_exponent_to_e8m0(127, "max_representable");
  // note: below deviates from MX spec
  //   in MX spec, e8m0 11111111 means nan, so I'd expect 2**nan to be nan
  //   in LLVM, the result is inf
  int_exponent_to_e8m0(128, "nan");
  std::cout << std::endl;

  // test casting float32 to e8m0 and then back to float32
  // the cast back is just for convenience of interpretation
  std::cout << "=== test float32 -> e8m0 -> float32 ===" << std::endl << std::endl;
  std::cout << "===== manual test cases =====" << std::endl << std::endl;

  // min representable value with e8m0 exponent
  // 2 ** -127 = 5.877472e-39
  float32_val_to_e8m0(5.877472e-39, "2**-127", true);

  // max denormal value
  float32_val_to_e8m0(1.1754942e-38, "max_denormal", true);

  // min normal
  float32_val_to_e8m0(1.17549435e-38, "min_normal", true);

  // basic cases
  float32_val_to_e8m0(0.25, "0.25", true);
  float32_val_to_e8m0(0.5, "0.5", true);
  float32_val_to_e8m0(1.0, "1.0", true);
  float32_val_to_e8m0(2.0, "2.0", true);
  float32_val_to_e8m0(4.0, "4.0", true);

  // max normal
  float32_val_to_e8m0(3.4028235e38, "max_normal", true);

  // test rounding (asking for RNE in the test case)
  // this seems to always round up to the larger power of two at the midpoint
  float32_val_to_e8m0(6.0, "", true);
  float32_val_to_e8m0(3.0, "", true);
  float32_val_to_e8m0(1.5, "", true);
  float32_val_to_e8m0(0.75, "", true);
  float32_val_to_e8m0(0.375, "", true);

  std::cout << "===== sweep =====" << std::endl << std::endl;

  // rules of RNE for general floating point rounding:
  // LSB - last bit we'll keep
  // GRS - guard, round, and sticky bits
  //
  // if G == 0:
  //   round down (truncate)
  // else: // G == 1
  //   if (R == 1) or (S == 1):
  //     round up
  //   else:
  //     if LSB == 1:
  //       round up (to make LSB even)
  //     else:
  //       round down
  //
  // for e8m0, LSB is the implied mantissa bit, so it equals to 0 for denormals, and to 1 for normals
  //
  // if G == 0:
  //   round down (truncate)
  // else: // G == 1
  //   if (R == 1) or (S == 1):
  //     round up
  //   else:
  //     if normal_number:
  //       round up (to next power of two)
  //     else:  // denormal number
  //       round down (to zero)

  // Now, let's test if the LLVM e8m0 semantics match the logic above
  // for (uint8_t exponent = 0; exponent < 256; exponent++) {
  for (uint16_t exponent_large = 0; exponent_large <= 255; exponent_large++) {
    uint8_t exponent_small = exponent_large;
    fp32_e8m0_fp32_check_all_grs(exponent_small);
  }
  
  std::cout << "end e8m0 test" << std::endl;
  return 0;
}
