#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <sstream>
#include <bitset>
#include <utility>

using namespace std;

std::vector<float>
encode_ins_to_bit_array(const int in1, const int in2, bool verbose = false) {
  std::string binary1 = std::bitset<8>(in1).to_string(); //to binary
  std::string binary2 = std::bitset<8>(in2).to_string(); //to binary

  if (verbose) {
    std::cout << "in1 =" << binary1 << "\n";
    std::cout << "in2 =" << binary2 << "\n";
  }

  std::vector<float> input;

  for (auto ch : binary1) {
    if (ch == '0') {
      input.push_back(-1.0);
    } else {
      input.push_back(1.0);
    }
  }

  for (auto ch : binary2) {
    if (ch == '0') {
      input.push_back(-1.0);
    } else {
      input.push_back(1.0);
    }
  }

  return input;
}

std::pair<int, int>
float_array_to_outs(const vector<float>& nn_output, bool verbose = false) {
  std::string binary_nn_output;

  for (float v : nn_output) {
    binary_nn_output += (v > 0.5 ? '1' : '0');
  }

  if (verbose)
    std::cout << "binary output:" << binary_nn_output << "\n";

  int out1 = std::bitset<8>(binary_nn_output.substr(0, 8)).to_ulong();
  int out2 = std::bitset<8>(binary_nn_output.substr(8, 16)).to_ulong();
  return std::make_pair(out1, out2);
}

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: model_tester <path-to-exported-script-module> <input {(in1, in2) / all}>\n";
    return -1;
  }

  std::stringstream input_stream( std::string(argv[2]) + " " +
                                  std::string(argv[3]));
  if (input_stream.str() == "all") {
    //
  }
  int in1 = 0, in2 = 0;
  input_stream >> in1 >> in2;

  std::cout << "in1: " << in1 << " in2: " << in2 << "\n";

  if (in1 > 1 || in1 < 0) {
    std::cerr << "invalid in1 should be either 0 or 1:\n"
    "model_tester <path-to-exported-module> <input (0, 222) >\n";
    return -1;
  }

  if (in2 >= 256 || in2 < 0) {
    std::cerr << "invalid in2 input, should be in interval [0, 255]:\n"
    "model_tester <path-to-exported-module> <input (0, 222) >\n";
    return -1;
  }
  std::vector<float> input = encode_ins_to_bit_array(in1, in2);
  std::cout << "NN input: (" << in1 << " " << in2 << ") -> [" << input << "]\n";

  torch::Tensor nn_input = torch::from_blob(&input[0], {1, 1, 16});

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(nn_input);

    // Execute the model and turn its output into a tensor.
    at::Tensor nn_output = module.forward(inputs).toTensor();
    float *out = nn_output.data<float>();
    std::vector<float> output(out, out + 2);
    std::cout << "NN[(" << in1 << ", " << in2 << ")]";
    std::cout << "-> (" << output << ")\n";
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model: " << e.what() << "\n";
    return -1;
  }
}
