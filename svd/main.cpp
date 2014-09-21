/*
  main.cpp
  Takes two tables of weights data - one with baseline data, the other with users/items for which we want features to be outputted (users table)
  Loads weights from the two tables, using itemIDs and translating users table user IDs to avoid overlap
  Performs SVD on the resultant weights matrix and outputs the features and item predictions for the desired users into an output table.
  For up-to-date help: svd --help
*/

#include <boost/program_options.hpp>
namespace options = boost::program_options;
using namespace std;

#include <mysql++.h>
using namespace mysqlpp;
using namespace std;

#include <string>

#include "svd.hpp"

int main(int argc, char* argv[]) {
  // Declare the supported options.
  options::options_description generalOptions("General options");
  generalOptions.add_options()
    ("help", "produce this help message")
    ("host", options::value<std::string>()->default_value("localhost"), "MySQL host")
    ("username", options::value<std::string>()->default_value("root"), "MySQL username")
    ("password", options::value<std::string>()->default_value(""), "MySQL password")
    ("database", options::value<std::string>()->default_value("animurecs"), "MySQL database")
    ("baseline_table", options::value<std::string>(), "table with baseline training data")
    ("weights_table", options::value<std::string>(), "weights table with entities for whom you want features to be outputted")
    ("means_table", options::value<std::string>()->default_value("entity_stats"), "table with entity means")
    ("test_table", options::value<std::string>(), "table with test data")
    ("min_weight", options::value<float>()->default_value(0.1), "lower floor on weight values")
    ("max_weight", options::value<float>()->default_value(1), "upper ceiling on weight values")
    ("type", options::value<std::string>(), "entity type to calculate features for")
  ;
  options::options_description svdOptions("SVD options");
  svdOptions.add_options()
    ("global_table", options::value<std::string>()->default_value("global_stats"), "table to store global values (e.g. global weight mean)")
    ("feature_table", options::value<std::string>()->default_value("entity_features"), "output table for features")
    ("features", options::value<int>()->default_value(50), "number of features")
    ("num_priors", options::value<int>()->default_value(25), "weight of priors in regularizing means")
    ("f_init", options::value<float>()->default_value(0.01), "initial value of features")
    ("min_epochs", options::value<int>()->default_value(50), "minimal number of epochs to train")
    ("max_epochs", options::value<int>()->default_value(200), "maximal number of epochs to train")
    ("min_improvement", options::value<float>()->default_value(0.0001), "minimal rmse improvement per epoch required to continue training a feature")
    ("min_weights", options::value<int>()->default_value(20), "minimal number of weights to have features calculated")
    ("l_rate", options::value<float>()->default_value(0.001), "learning rate")
    ("tikhonov", options::value<float>()->default_value(0.02), "tikhonov regularization parameter")
  ;

  options::options_description allOptions("Allowed options");
  allOptions.add(generalOptions).add(svdOptions);
  options::variables_map vm;
  options::store(options::parse_command_line(argc, argv, allOptions), vm);
  options::notify(vm);

  if (vm.count("help")) {
    cout << allOptions << endl;
    return 1;
  }

  try {
    Connection database(vm["database"].as<std::string>().c_str(), vm["host"].as<std::string>().c_str(), vm["username"].as<std::string>().c_str(), vm["password"].as<std::string>().c_str());
    // perform SVD on weights matrix.
    SVD* svd = new SVD(database, vm["type"].as<std::string>(), vm["features"].as<int>(), vm["min_epochs"].as<int>(), vm["max_epochs"].as<int>(), vm["min_improvement"].as<float>(), vm["l_rate"].as<float>(), vm["tikhonov"].as<float>(), vm["f_init"].as<float>(), vm["num_priors"].as<int>(), vm["min_weights"].as<int>(), vm["min_weight"].as<float>(), vm["max_weight"].as<float>());
    if (vm.count("baseline_table")) {
      svd->LoadBaseline(vm["baseline_table"].as<std::string>());
    }
    if (vm.count("weights_table")) {
      svd->LoadWeights(vm["weights_table"].as<std::string>());
    }
    if (vm.count("test_table")) {
      svd->LoadTests(vm["test_table"].as<std::string>());
    }
    svd->CalcMetrics();
    //svd->NormalizeWeights();
    svd->CalcFeatures();
    if (vm.count("test_table")) {
      svd->RunTest();
    }
    svd->SaveModel(vm["global_table"].as<std::string>(), vm["means_table"].as<std::string>(), vm["feature_table"].as<std::string>());
    svd->PrintTimings();
    svd->PrintBenchmarks();
  } catch (ConnectionFailed er) {
    cerr << "Could not connect to MySQL database." << endl;
    throw er;
  }
  return 0;
}