#include "timesel.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include <pybind11/numpy.h>

PYBIND11_MODULE(helper_timesel, m) {
  m.doc() = "pybind11 timesel interface"; // optional module docstring

  m.def("select_dynProg_max", &select_dynProg_max, "base time step selection");
}
