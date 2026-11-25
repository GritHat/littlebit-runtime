// shared between the parts of the library; not installed
#pragma once

#include <string>

namespace littlebit_internal {
void init();                             // backends and logging, once
void set_error(const std::string & what); // littlebit_last_error() of this thread
}
