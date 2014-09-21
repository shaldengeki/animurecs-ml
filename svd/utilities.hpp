/*
  utilities.hpp
  utility functions.
*/

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <string>
#include <sstream>
#include <vector>
using namespace std;

namespace utilities {
  /* String-splitting and iterable-joining functions */
  std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

  std::vector<std::string> split(const std::string &s, char delim);

  template <class T>
  inline std::string to_string(const T& t) {
    // casts a thing into a std::string.
    std::stringstream ss;
    ss << t;
    return ss.str();
  }

  template <typename Iter>
  std::string join(Iter begin, Iter end, std::string const& separator) {
    // joins an iterable of strings by a separator. returns the resultant string.
    std::ostringstream result;
    if (begin != end)
      result << *begin++;
    while (begin != end)
      result << separator << *begin++;
    return result.str();
  }
}
#endif