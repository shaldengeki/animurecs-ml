/*
  main.cpp
  Takes two tables of weights data - one with baseline data, the other with users/items for which we want features to be outputted (users table)
  Loads weights from the two tables, using item_ids and translating users table user IDs to avoid overlap
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

#include "insert_queue.hpp"
#include "svd.hpp"

void loadBaseline(SVD* svd, Connection& db, std::string baseline_table, std::string item_id_col) {
  // Loads all of the weights in the baseline table.

  cout << "Loading baseline data." << endl;
  unsigned int compact_id = 0, user_id = 0, item_id = 0, i = 0, baseline_count = 0;
  float weight = 0;
  UseQueryResult weights_itr;
  Row weight_row;

  std::string item_type = svd->ItemType();
  float min_weight = svd->MinWeight(), max_weight = svd->MaxWeight();

  // load the training data row by row.
  try {
    Query weights_query = db.query();
    weights_query << "SELECT COUNT(*) FROM " << baseline_table << " WHERE type = " << quote << item_type;
    baseline_count = (unsigned int) atoi(weights_query.store()[0]["COUNT(*)"]);
    cout << "Found " << baseline_count << " weights." << endl;
    weights_query.reset();
    weights_query << "SELECT user_id, " << item_id_col << ", score FROM " << baseline_table << " WHERE type = " << quote << item_type << " ORDER BY user_id ASC, " << item_id_col << " ASC";
    weights_itr = weights_query.use();
  } catch (BadQuery er) {
    cerr << "Error loading baseline weights." << endl;
    throw er;
  }

  cout << "Loading baseline weights into memory..." << endl;

  for (i = 0; i < baseline_count; i++) {
    weight_row = weights_itr.fetch_row();
    try {
      user_id = (unsigned int) atoi(weight_row["user_id"]);
    } catch (BadFieldName er) {
      // we've reached the end of the data.
      cout << "user id not in fields for row " << i << "." << endl;
      break;
    }
    item_id = (unsigned int) strtoul(weight_row[item_id_col].c_str(), NULL, 0);
    weight = atof(weight_row["score"]);

    // If weight exceeds our boundaries, throw an exception.
    if (weight < min_weight || weight > max_weight) {
      throw std::out_of_range("Weight in baseline table exceeds specified bounds: " + weight_row["score"]);
    }
    svd->LoadRow(user_id, item_id, weight, true, false);
  }
  cout << "Finished loading baseline data: " << baseline_count << " weights." << endl;
}

void loadWeights(SVD* svd, Connection& db, std::string weights_table, std::string item_id_col) {
  // Loads all of the weights in the weights table.
  cout << "Counting training weights..." << endl;

  unsigned int i = 0, compact_id = 0, user_id = 0, item_id = 0, WeightsCount = 0;
  float weight = 0;
  UseQueryResult weights_itr;
  Row weight_row;

  std::string item_type = svd->ItemType();
  float min_weight = svd->MinWeight(), max_weight = svd->MaxWeight();

  // load the training data row by row.
  try {
    Query weights_query = db.query();
    weights_query << "SELECT COUNT(*) FROM " << weights_table << " WHERE type = " << quote << item_type;
    WeightsCount = (unsigned int) atoi(weights_query.store()[0]["COUNT(*)"]);
    cout << "Found " << WeightsCount << " weights." << endl;
    weights_query.reset();
    weights_query << "SELECT user_id, " << item_id_col << ", score FROM " << weights_table << " WHERE type = " << quote << item_type << " ORDER BY user_id ASC, " << item_id_col << " ASC";
    weights_itr = weights_query.use();
  } catch (BadQuery er) {
    cerr << "Error loading weights." << endl;
    throw er;
  }

  cout << "Loading training weights into memory..." << endl;

  for (i = 0; i < WeightsCount; i++) {
    weight_row = weights_itr.fetch_row();
    try {
      user_id = (unsigned int) atoi(weight_row["user_id"]);
    } catch (BadFieldName er) {
      // we've reached the end of the data.
      cout << "user id not in fields for row " << i << "." << endl;
      break;
    }
    item_id = (unsigned int) strtoul(weight_row[item_id_col].c_str(), NULL, 0);
    weight = atof(weight_row["score"]);

    // If weight exceeds our boundaries, throw an exception.
    if (weight < min_weight || weight > max_weight) {
      throw std::out_of_range("Weight in weights table exceeds specified bounds: " + weight_row["score"]);
    }
    svd->LoadRow(user_id, item_id, weight, false, false);
  }

  cout << "Finished loading training weight data: " << WeightsCount << " weights." << endl;
}

void loadTests(SVD* svd, Connection& db, std::string test_table, std::string item_id_col) {
  // Loads all of the weights in the test table.
  cout << "Loading test data." << endl;

  unsigned int i = 0, user_id = 0, item_id = 0, TestCount = 0;
  float weight = 0;
  UseQueryResult weights_itr;
  Row weight_row;

  std::string item_type = svd->ItemType();
  float min_weight = svd->MinWeight(), max_weight = svd->MaxWeight();

  // load the test data row by row.
  try {
    Query weights_query = db.query();
    weights_query << "SELECT COUNT(*) FROM " << test_table << " WHERE type = " << quote << item_type;
    TestCount = (unsigned int) atoi(weights_query.store()[0]["COUNT(*)"]);
    cout << "Found " << TestCount << " weights." << endl;
    weights_query.reset();
    weights_query << "SELECT user_id, " << item_id_col << ", score FROM " << test_table << " WHERE type = " << quote << item_type << " ORDER BY user_id ASC, " << item_id_col << " ASC";
    weights_itr = weights_query.use();
  } catch (BadQuery er) {
    cerr << "Error loading test weights." << endl;
    throw er;
  }

  cout << "Loading test weights into memory..." << endl;
  for (i = 0; i < TestCount; i++) {
    weight_row = weights_itr.fetch_row();
    try {
      user_id = (unsigned int) atoi(weight_row["user_id"]);
    } catch (BadFieldName er) {
      // we've reached the end of the test data.
      cout << "user_id not in fields for row " << i << endl;
      break;
    }
    item_id = (unsigned int) strtoul(weight_row[item_id_col].c_str(), NULL, 0);
    weight = atof(weight_row["score"]);
    // If weight exceeds our boundaries, throw an exception.
    if (weight < min_weight || weight > max_weight) {
      throw std::out_of_range("Weight in testing table exceeds specified bounds: " + weight_row["score"]);
    }
    svd->LoadRow(user_id, item_id, weight, false, true);
  }
  cout << "Finished loading test data: " << TestCount << " weights." << endl;
}


void saveModel(SVD* svd, Connection& db, std::string global_table, std::string means_table, std::string feature_table) {
  // saves the SVD's features to output tables.
  // saves global mean for this entity pair to global_table,
  // each entity's paired global mean to means_table,
  // and each entity's paired feature list to feature_table.

  cout << "Saving SVD model..." << endl;
  unsigned int compact_id = 0, sparseID = 0, item_id = 0;
  float prediction = 0.0;

  IdMap UserIDs = svd->UserIDs();
  vector<User> Users = svd->Users();
  std::string item_type = svd->ItemType();
  unsigned int Features = svd->FeaturesCount();
  vector<unsigned int> SelectedItems = svd->SelectedItems();
  vector<Item> Items = svd->Items();
  vector<float> UserFeatures, ItemFeatures;

  // create insert queues.
  vector<std::string> meanFields, featureFields, predictionFields;
  Query insertRow(db.query()), globalQuery(db.query()), deleteMeansQuery(db.query()), deleteFeaturesQuery(db.query()), deletePredictionsQuery(db.query());

  // std::string meanFieldNames[] = {"id", "type", "source", "compare_type", "mean"};
  // meanFields.assign(meanFieldNames, meanFieldNames+5);

  std::string featureFieldNames[] = {"id", "type", "feature", "value"};
  featureFields.assign(featureFieldNames, featureFieldNames+4);

  // buffer length for inserts. larger values mean faster insertion (so long as the queue can fit in memory)!
  unsigned int maxQueueLength = 100000;

  InsertQueue featureInserts(db, maxQueueLength, feature_table, featureFields);
  // InsertQueue meanInserts(db, maxQueueLength, means_table, meanFields);

  // // update global means for this entity type pair.
  // try {
  //   globalQuery << "INSERT INTO " << global_table << " (type_1, type_2, mean) VALUES (" << item_type1 << "," << item_type2 << "," << GlobalAvg << ") ON DUPLICATE KEY UPDATE mean=" << GlobalAvg;
  //   globalQuery.execute();
  // } catch (BadQuery er) {
  //   cerr << "Error updating global mean." << endl;
  //   throw er;
  // }

  // clear all the means, features, and predictions for this pair of types.
  // try {
  //   deleteMeansQuery << "DELETE FROM " << means_table << " WHERE type=" << item_type1 << " && compare_type=" << item_type2;
  //   deleteMeansQuery.execute();
  // } catch (BadQuery er) {
  //   cerr << "Error deleting entity means." << endl;
  //   throw er;
  // }

  try {
    deleteFeaturesQuery << "DELETE FROM " << feature_table << " WHERE type=" << item_type;
    deleteFeaturesQuery.execute();
    deleteFeaturesQuery << "DELETE FROM " << feature_table << " WHERE type='user'";
    deleteFeaturesQuery.execute();
  } catch (BadQuery er) {
    cerr << "Error deleting entity and user features." << endl;
    throw er;
  }

  cout << "Saving users..." << endl;
  for (idMapItr userIterator = UserIDs.begin(); userIterator != UserIDs.end(); ++userIterator) {
    sparseID = userIterator->first;
    compact_id = userIterator->second;
    // update user means.
    // insertRow.reset();
    // insertRow << "(" << quote << sparseID << "," << quote << "user" << "," << quote << Users[compact_id].source << "," << quote << item_type2 << "," << quote << Users[compact_id].regularized_avg << ")";
    // meanInserts.Append(insertRow.str());

    // only update features if this user has them calculated!
    if (Users[compact_id].weights_count == 0) {
      continue;
    }

    // update user features.
    UserFeatures = svd->UserFeatures(sparseID);
    for (unsigned int feature = 0; feature < Features; feature++) {
      insertRow.reset();
      insertRow << "(" << quote << sparseID << "," << quote << "user" << "," << quote << feature << "," << quote << UserFeatures[feature] << ")";
      featureInserts.Append(insertRow.str());
    }
  }
  cout << "Saving items..." << endl;
  for (vector<unsigned int>::iterator itemIterator = SelectedItems.begin(); itemIterator != SelectedItems.end(); ++itemIterator) {
    compact_id = (unsigned int) *itemIterator;
    sparseID = Items[compact_id].sparse_id;

    // update item means.
    // insertRow.reset();
    // insertRow << "(" << quote << Items[compact_id].sparse_id << "," << quote << item_type2 << "," << quote << Items[compact_id].source << "," << quote << item_type1 << "," << quote << Items[compact_id].regularized_avg << ")";
    // meanInserts.Append(insertRow.str());

    // only update features if this item has them calculated!
    if (Items[compact_id].weights_count == 0) {
      continue;
    }
    // update item features.

    ItemFeatures = svd->ItemFeatures(sparseID);
    for (unsigned int feature = 0; feature < Features; feature++) {
      insertRow.reset();
      insertRow << "(" << quote << Items[compact_id].sparse_id << "," << quote << item_type << "," << quote << feature << "," << quote << ItemFeatures[feature] << ")";
      featureInserts.Append(insertRow.str());
    }
  }
  // meanInserts.Flush();
  featureInserts.Flush();
  cout << "SVD saved." << endl;
}

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
    ("item_id_col", options::value<std::string>(), "column name in tables containing entity IDs")
    ("min_weight", options::value<float>()->default_value(1), "lower floor on weight values in tables")
    ("max_weight", options::value<float>()->default_value(10), "upper ceiling on weight values in tables")
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
    if (!database.connected()) {
      throw new ConnectionFailed("Could not connect to database.");
    }

    // perform SVD on weights matrix.
    SVD* svd = new SVD(vm["type"].as<std::string>(), vm["features"].as<int>(), vm["min_epochs"].as<int>(), vm["max_epochs"].as<int>(), vm["min_improvement"].as<float>(), vm["l_rate"].as<float>(), vm["tikhonov"].as<float>(), vm["f_init"].as<float>(), vm["num_priors"].as<int>(), vm["min_weights"].as<int>(), vm["min_weight"].as<float>(), vm["max_weight"].as<float>());
    if (vm.count("baseline_table")) {
      loadBaseline(svd, database, vm["baseline_table"].as<std::string>(), vm["item_id_col"].as<std::string>());
    }
    if (vm.count("weights_table")) {
      loadWeights(svd, database, vm["weights_table"].as<std::string>(), vm["item_id_col"].as<std::string>());
    }
    if (vm.count("test_table")) {
      loadTests(svd, database, vm["test_table"].as<std::string>(), vm["item_id_col"].as<std::string>());
    }
    svd->CalcMetrics();
    //svd->NormalizeWeights();
    svd->CalcFeatures();
    if (vm.count("test_table")) {
      svd->RunTest();
    }
    if (vm.count("feature_table")) {
      saveModel(svd, database, vm["global_table"].as<std::string>(), vm["means_table"].as<std::string>(), vm["feature_table"].as<std::string>());
    }
    svd->PrintTimings();
    svd->PrintBenchmarks();
  } catch (ConnectionFailed er) {
    cerr << "Could not connect to MySQL database." << endl;
    throw er;
  }
  return 0;
}