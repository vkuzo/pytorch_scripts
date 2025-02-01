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

void float32_val_to_e8m0(float val, const std::string& description) {
  APFloat val_float32(val);  // 32-bit float
  std::cout << "description: " << description << ", float32 val: " << val_float32.convertToFloat() << std::endl;

  APFloat::Semantics Sem = APFloat::S_Float8E8M0FNU;
  const llvm::fltSemantics &S = APFloat::EnumToSemantics(Sem);

  uint64_t raw_bits_fp32 = val_float32.bitcastToAPInt().getZExtValue();
  std::cout << "  fp32_org bits: " << std::bitset<32>(raw_bits_fp32) << "\n";

  bool losesInfo;
  val_float32.convert(S, APFloat::rmNearestTiesToEven, &losesInfo);

  // print the raw bits
  uint64_t raw_bits_e8m0 = val_float32.bitcastToAPInt().getZExtValue();
  std::cout << "  e8m0 bits:      " << std::bitset<8>(raw_bits_e8m0) << "\n";

  float val_float32_e8m0_float32 = val_float32.convertToFloat();

  // access bits of a float
  union {
    float f;
    uint32_t i;
  } u;
  u.f = val_float32_e8m0_float32;
  // uint64_t raw_bits_e8m0_fp32 = val_float32_e8m0_float32.bitcastToAPInt().getZExtValue();
  std::cout << "  fp32_new bits: " << std::bitset<32>(u.i) << "\n";

  std::cout << "  e8m0 -> float32 cast result: " << val_float32_e8m0_float32 << std::endl;
  std::cout << std::endl;

}

int main() {
  std::cout << "start e8m0 test" << std::endl;
  std::cout << std::endl;

  // test directly instantiang e8m0 from an integer, and the subsequent cast to float32
  std::cout << "=== int -> e8m0 -> float32 ===" << std::endl << std::endl;

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
  std::cout << "=== float32 -> e8m0 -> float32 ===" << std::endl << std::endl;

  // min representable value with e8m0 exponent
  // 2 ** -127 = 5.877472e-39
  float32_val_to_e8m0(5.877472e-39, "2**-127");

  // max denormal value
  float32_val_to_e8m0(1.1754942e-38, "max_denormal");

  // min normal
  float32_val_to_e8m0(1.17549435e-38, "min_normal");

  // basic cases
  float32_val_to_e8m0(0.25, "0.25");
  float32_val_to_e8m0(0.5, "0.5");
  float32_val_to_e8m0(1.0, "1.0");
  float32_val_to_e8m0(2.0, "2.0");
  float32_val_to_e8m0(4.0, "4.0");

  // max normal
  float32_val_to_e8m0(3.4028235e38, "max_normal");

  // test rounding (asking for RNE in the test case)
  // this seems to always round up to the larger power of two at the midpoint
  float32_val_to_e8m0(6.0, "");
  float32_val_to_e8m0(3.0, "");
  float32_val_to_e8m0(1.5, "");
  float32_val_to_e8m0(0.75, "");
  float32_val_to_e8m0(0.375, "");
  
  std::cout << "end e8m0 test" << std::endl;
  return 0;
}
