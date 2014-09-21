/*
  utilities.cpp
  utility functions.
*/

#include <string>
#include <sstream>
#include <vector>
using namespace std;

#include "utilities.hpp"

namespace utilities {
  /* String-splitting and iterable-joining functions */
  std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    // splits a string by a delimiter into a vector.  
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
      elems.push_back(item);
    }
    return elems;
  }

  std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
  }
}