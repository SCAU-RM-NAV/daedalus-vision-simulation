#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <openvino/openvino.hpp>

namespace
{
nlohmann::json port_json(const ov::Output<const ov::Node> & port)
{
  nlohmann::json result = {
    {"shape", port.get_partial_shape().to_string()},
    {"element_type", port.get_element_type().get_type_name()},
    {"names", port.get_names()}};
  try {
    result["any_name"] = port.get_any_name();
  } catch (const std::exception &) {
    result["any_name"] = "";
  }
  return result;
}
}  // namespace

int main(int argc, char * argv[])
{
  if (argc != 2) {
    std::cerr << "usage: sim_model_probe <model.xml|model.onnx>\n";
    return 2;
  }

  nlohmann::json report = {{"model_path", argv[1]}};
  try {
    ov::Core core;
    report["available_devices"] = core.get_available_devices();
    const auto model = core.read_model(argv[1]);
    report["inputs"] = nlohmann::json::array();
    report["outputs"] = nlohmann::json::array();
    for (const auto & input : model->inputs()) report["inputs"].push_back(port_json(input));
    for (const auto & output : model->outputs()) report["outputs"].push_back(port_json(output));
    for (const std::string device : {"CPU", "GPU"}) {
      try {
        const auto compiled = core.compile_model(model, device);
        report["compile"][device] = {
          {"ok", true}, {"optimal_requests", compiled.get_property(ov::optimal_number_of_infer_requests)}};
      } catch (const std::exception & error) {
        report["compile"][device] = {{"ok", false}, {"error", error.what()}};
      }
    }
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << report.dump(2) << '\n';
  return 0;
}
