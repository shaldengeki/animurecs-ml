/*
  svd.cpp
  SVD class implementation.
*/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <string>
#include <sstream>
#include <vector>
#include <limits>
#include <math.h>
#include <map>
#include <time.h>
#include <algorithm> 
#include <functional>
#include <cctype>
#include <locale>

#include <mysql++.h>
using namespace mysqlpp;
using namespace std;

#include "insert_queue.hpp"
#include "utilities.hpp"
#include "svd.hpp"

SVD::SVD(Connection& dbConn, std::string type, unsigned int features, unsigned int min_epochs, unsigned int max_epochs, float min_improvement, float l_rate, float tikhonov, float f_init, unsigned int num_priors, unsigned int min_weights, float min_weight, float max_weight) : 
DbConn(dbConn), EntityType(type), Features(features), MinEpochs(min_epochs), MaxEpochs(max_epochs), MinImprovement(min_improvement), LRate(l_rate), Tikhonov(tikhonov), FInit(f_init), NumPriors(num_priors), MinWeights(min_weights), MinWeight(min_weight), MaxWeight(max_weight) {
  AddTiming("SVD init");
  if (!DbConn.connected()) {
    throw new ConnectionFailed("Could not connect to MySQL.");
  }

  // normalize all weights to 1-10.
  ScaleFactor = 10.0;
  ScaledMin = MinWeight * ScaleFactor;
  ScaledMax = MaxWeight * ScaleFactor;

  TotalWeights = WeightsCount = BaselineCount = TestCount = WeightSum = GlobalAvg = 0;
  vector<float> BlankFeature;
  for (unsigned int f = 0; f < Features; f++) {
    FeatureItems.push_back(BlankFeature);
    FeatureUsers.push_back(BlankFeature);
  }
}

inline void SVD::AddTiming(std::string description) {
  // adds a timing to this SVD's list for post-run profiling.
  Timings.push_back(Timing(clock(), description));
}

void SVD::PrintTimings() {
  // Prints out every time,string pair in Timings for profiling purposes.
  cout << "Timings" << endl;
  cout << "=======" << endl;
  clock_t prevTime = clock();
  for (vector<Timing>::iterator timingIterator = Timings.begin(); timingIterator != Timings.end(); ++timingIterator) {
    Timing timePair = *timingIterator;
    cout << timePair.second << ": " << "(" << (timePair.first / CLOCKS_PER_SEC) << "," << ((timePair.first - prevTime) / CLOCKS_PER_SEC) << ") " << endl;
    prevTime = timePair.first;
  }
}

inline void SVD::StartBenchmark(std::string name) {
  BenchmarkStarts[name] = clock();
}

inline void SVD::EndBenchmark(std::string name) {
  float benchTime = (clock() - BenchmarkStarts[name]) / CLOCKS_PER_SEC;

  Benchmarks[name].Sum += benchTime;
  Benchmarks[name].Count++;
  if (benchTime > Benchmarks[name].Max) {
    Benchmarks[name].Max = benchTime;
  }
  if (benchTime < Benchmarks[name].Min) {
    Benchmarks[name].Min = benchTime;
  }
  // BenchmarkStarts.erase(name);
}

void SVD::PrintBenchmarks() {
  // Prints out average durations in Benchmarks for each key for profiling purposes.
  BenchInfo benchInfo;
  std::string name;
  cout << "Benchmarks" << endl;
  cout << "==========" << endl;

  for (BenchMap::iterator benchmarkIterator = Benchmarks.begin(); benchmarkIterator != Benchmarks.end(); ++benchmarkIterator) {
    name = benchmarkIterator->first;
    benchInfo = benchmarkIterator->second;
    cout << name << ": " << "Min " << benchInfo.Min << " | Max " << benchInfo.Max << " | Avg " << (benchInfo.Sum * 1.0 / benchInfo.Count) << endl;
  }
}

void SVD::LoadRow(unsigned int user_id, unsigned int item_id, float weight, bool baseline, bool test) {
  // loads a weight row into the SVD, given dual sparse IDs and a weight.
  // optionally takes baseline or test as a bool to indicate that this weight should be marked as such.
  // StartBenchmark("LoadRow"");
  idMapItr userIterator, itemIterator;
  unsigned int uid = 0, aid = 0;

  // initialize a blank user and anime to push.
  User BlankUser;
  BlankUser.SparseID = 0;
  BlankUser.WeightSum = 0;
  BlankUser.WeightsCount = 0;
  BlankUser.WeightAvg = 0;
  BlankUser.RegularizedAvg = 0;

  Item BlankAnime;
  BlankAnime.SparseID = 0;
  BlankAnime.WeightSum = 0;
  BlankAnime.WeightsCount = 0;
  BlankAnime.WeightAvg = 0;
  BlankAnime.RegularizedAvg = 0;

  Data weightEntry;
  weightEntry.Cache = 0;

  // we want to scale all the ratings up by a constant factor to enable SVD.
  weight *= ScaleFactor;

  if (!test) {
    // Add users (using a map to re-number ids to array indices)
    userIterator = UserIDs.find(user_id); 
    if (userIterator == UserIDs.end()) {
      uid = Users.size();

      // Reserve new id and add lookup.
      UserIDs[user_id] = uid;
      BlankUser.SparseID = user_id;

      // Push a new user onto users and store the old sparse id for later.
      Users.push_back(BlankUser);
      user_id = uid;
    } else {
        user_id = userIterator->second;
    }

    Users[user_id].WeightsCount++;
    Users[user_id].WeightSum += weight;

    // Add anime (using a map to re-number ids to array indices) 
    itemIterator = ItemIDs.find(item_id); 
    if (itemIterator == ItemIDs.end()) {
      aid = ItemIDs.size();

      // Reserve new id and add lookup.
      ItemIDs[item_id] = aid;
      BlankAnime.SparseID = item_id;

      if (!baseline) {
        SelectedItems.push_back(item_id);
      }

      // Push a new anime onto animus.
      Items.push_back(BlankAnime);
      item_id = aid;
    } else {
      item_id = itemIterator->second;
    }
    Items[item_id].WeightsCount++;
    Items[item_id].WeightSum += weight;
  }
  weightEntry.Weight = weight;
  if (test) {
    weightEntry.UserID = UserIDs[user_id];
    weightEntry.ItemID = ItemIDs[item_id];
    TestWeights.push_back(weightEntry);
  } else {
    weightEntry.UserID = user_id;
    weightEntry.ItemID = item_id;
    Weights.push_back(weightEntry);
    WeightSum += weight;
  }
  // EndBenchmark("LoadRow"");
}

void SVD::LoadCSV(string trainingFile, bool baseline, bool test) {
  // Loads all of the ratings in the training file.
  // Data is in the form USERID,ANIMEID,RATING
  AddTiming("Load data init");
  cout << "Loading training data..." << endl;

  unsigned int userID = 0, itemID = 0;
  float weight = 0;

  string line;
  ifstream dataFile;
  // load the csv file line by line.
  // csv file is of the form USER_ID,ITEM_ID,WEIGHT
  dataFile.open(trainingFile.c_str());
  if (dataFile.is_open()) {
    while (!dataFile.eof()) {
      getline(dataFile, line);
      if (line.empty()) {
        continue;
      }
      std::vector<std::string> splitLine = utilities::split(line, ',');
      userID = atoi(splitLine[0].c_str());
      itemID = (unsigned int) strtoul(splitLine[1].c_str(), NULL, 0);
      weight = atof(splitLine[2].c_str());

      LoadRow(userID, itemID, weight, baseline, test);
    }
    dataFile.close();
  } else {
    printf("Could not open training file at %s. Please check the path.\n", trainingFile.c_str());
  }
  AddTiming("Load data finish");
}

void SVD::LoadBaseline(std::string baseline_table) {
  // Loads all of the weights in the baseline table.

  AddTiming("Baseline init");
  cout << "Loading baseline data." << endl;

  unsigned int compactID = 0, userID = 0, itemID = 0, i = 0;
  float weight = 0;
  UseQueryResult weightsIterator;
  Row weightRow;

  // load the training data row by row.
  try {
    Query weightsQuery = DbConn.query();
    weightsQuery << "SELECT COUNT(*) FROM " << baseline_table << " WHERE type = " << quote << EntityType;
    BaselineCount = (unsigned int) atoi(weightsQuery.store()[0]["COUNT(*)"]);
    cout << "Found " << BaselineCount << " weights." << endl;
    weightsQuery.reset();
    weightsQuery << "SELECT user_id, type_id, score FROM " << baseline_table << " WHERE type = " << quote << EntityType << " ORDER BY user_id ASC, type_id ASC";
    weightsIterator = weightsQuery.use();
  } catch (BadQuery er) {
    cerr << "Error loading baseline weights." << endl;
    throw er;
  }

  cout << "Loading baseline weights into memory..." << endl;

  for (i = 0; i < BaselineCount; i++) {
    weightRow = weightsIterator.fetch_row();
    try {
      userID = (unsigned int) atoi(weightRow["user_id"]);
    } catch (BadFieldName er) {
      // we've reached the end of the data.
      cout << "user id not in fields for row " << i << "." << endl;
      break;
    }
    itemID = (unsigned int) strtoul(weightRow["type_id"].c_str(), NULL, 0);
    weight = atof(weightRow["score"]);

    // If weight exceeds our boundaries, throw an exception.
    if (weight < MinWeight || weight > MaxWeight) {
      throw std::out_of_range("Weight in baseline table exceeds specified bounds: " + weightRow["score"]);
    }
    LoadRow(userID, itemID, weight, true, false);
  }
  cout << "Finished loading baseline data: " << BaselineCount << " weights." << endl;
  AddTiming("Baseline finish");
}

void SVD::LoadWeights(std::string weights_table) {
  // Loads all of the weights in the weights table.
  AddTiming("Training weights init");
  cout << "Counting training weights..." << endl;

  unsigned int i = 0, compactID = 0, userID = 0, itemID = 0;
  float weight = 0;
  UseQueryResult weightsIterator;
  Row weightRow;

  // load the training data row by row.
  try {
    Query weightsQuery = DbConn.query();
    weightsQuery << "SELECT COUNT(*) FROM " << weights_table << " WHERE type = " << quote << EntityType;
    WeightsCount = (unsigned int) atoi(weightsQuery.store()[0]["COUNT(*)"]);
    cout << "Found " << WeightsCount << " weights." << endl;
    weightsQuery.reset();
    weightsQuery << "SELECT user_id, type_id, score FROM " << weights_table << " WHERE type = " << quote << EntityType << " ORDER BY user_id ASC, type_id ASC";
    weightsIterator = weightsQuery.use();
  } catch (BadQuery er) {
    cerr << "Error loading weights." << endl;
    throw er;
  }

  cout << "Loading training weights into memory..." << endl;

  for (i = 0; i < WeightsCount; i++) {
    weightRow = weightsIterator.fetch_row();
    try {
      userID = (unsigned int) atoi(weightRow["user_id"]);
    } catch (BadFieldName er) {
      // we've reached the end of the data.
      cout << "user id not in fields for row " << i << "." << endl;
      break;
    }
    itemID = (unsigned int) strtoul(weightRow["type_id"].c_str(), NULL, 0);
    weight = atof(weightRow["score"]);

    // If weight exceeds our boundaries, throw an exception.
    if (weight < MinWeight || weight > MaxWeight) {
      throw std::out_of_range("Weight in weights table exceeds specified bounds: " + weightRow["score"]);
    }
    LoadRow(userID, itemID, weight, false, false);
  }

  cout << "Finished loading training weight data: " << WeightsCount << " weights." << endl;
  AddTiming("Training weights finish");
}

void SVD::LoadTests(std::string test_table) {
  // Loads all of the weights in the test table.
  AddTiming("Tests init");
  cout << "Loading test data." << endl;

  unsigned int i = 0, userID = 0, itemID = 0;
  float weight = 0;
  UseQueryResult weightsIterator;
  Row weightRow;

  // load the training data row by row.
  try {
    Query weightsQuery = DbConn.query();
    weightsQuery << "SELECT COUNT(*) FROM " << test_table << " WHERE type = " << quote << EntityType;
    TestCount = (unsigned int) atoi(weightsQuery.store()[0]["COUNT(*)"]);
    cout << "Found " << TestCount << " weights." << endl;
    weightsQuery.reset();
    weightsQuery << "SELECT user_id, type_id, score FROM " << test_table << " WHERE type = " << quote << EntityType << " ORDER BY user_id ASC, type_id ASC";
    weightsIterator = weightsQuery.use();
  } catch (BadQuery er) {
    cerr << "Error loading test weights." << endl;
    throw er;
  }

  cout << "Loading test weights into memory..." << endl;
  for (i = 0; i < TestCount; i++) {
    weightRow = weightsIterator.fetch_row();
    try {
      userID = (unsigned int) atoi(weightRow["user_id"]);
    } catch (BadFieldName er) {
      // we've reached the end of the test data.
      cout << "user_id not in fields for row " << i << endl;
      break;
    }
    itemID = (unsigned int) strtoul(weightRow["type_id"].c_str(), NULL, 0);
    weight = atof(weightRow["score"]);
    // If weight exceeds our boundaries, throw an exception.
    if (weight < MinWeight || weight > MaxWeight) {
      throw std::out_of_range("Weight in testing table exceeds specified bounds: " + weightRow["score"]);
    }
    LoadRow(userID, itemID, weight, false, true);
  }
  cout << "Finished loading test data: " << TestCount << " weights." << endl;
  AddTiming("Tests finish");
}

void SVD::CalcMetrics() {
  // Loops through the data and calculates averages used in training.
  AddTiming("Metrics init");
  cout << "Calculating global and user/item stats..." << endl;
  unsigned int i = 0, f = 0;
  float globalSum = 0;

  // set total weights count.
  TotalWeights = BaselineCount + WeightsCount;
  cout << "Total weights: " << TotalWeights << endl;

  for (i = 0; i < TotalWeights; i++) {
    globalSum += Weights[i].Weight;
  }
  GlobalAvg = globalSum / TotalWeights;
  cout << "Global average: " << GlobalAvg << endl;

  // calculate user averages.
  cout << "Calculating user means..." << endl;
  for (i = 0; i < Users.size(); i++) {
    //cout << "Calculating user stats for uid " << i << "..." << endl;
    for (f = 0; f < Features; f++) {
      FeatureUsers[f].push_back(FInit);
    }
    Users[i].WeightAvg = Users[i].WeightSum / (1.0 * Users[i].WeightsCount);
    Users[i].RegularizedAvg = (GlobalAvg * NumPriors + Users[i].WeightSum) / (NumPriors * 1.0 + Users[i].WeightsCount);
    if (Users[i].WeightsCount < MinWeights) {
      // mark these users to not be counted when evaluating RMSE.
      Users[i].WeightsCount = 0;
      Users[i].WeightSum = 0;
    }
  }
  cout << "Calculating user offsets..." << endl;
  for (i = 0; i < Users.size(); i++) {
    Users[i].Offset = Users[i].RegularizedAvg - GlobalAvg;
  }

  // now calculate item averages.
  cout << "Calculating item means..." << endl;
  for (i = 0; i < Items.size(); i++) {
    //cout << "Calculating item stats for iid " << i << "..." << endl;
    for (f = 0; f < Features; f++) {
      FeatureItems[f].push_back(FInit);
    }
    Items[i].WeightAvg = Items[i].WeightSum / (1.0 * Items[i].WeightsCount);
    Items[i].RegularizedAvg = (GlobalAvg * NumPriors + Items[i].WeightSum) / (NumPriors * 1.0 + Items[i].WeightsCount);
    if (Items[i].WeightsCount < MinWeights) {
      // mark this item to not be counted when evaluating RMSE.
      Items[i].WeightsCount = 0;
      Items[i].WeightSum = 0;
    }
  }
  cout << "Calculating item offsets..." << endl;
  for (i = 0; i < Items.size(); i++) {
    Items[i].Offset = Items[i].RegularizedAvg - GlobalAvg;
  }

  cout << "Calculating user and item deviations..." << endl;
  for (i = 0; i < TotalWeights; i++) {
    if (Users[Weights[i].UserID].WeightsCount > 0) {
      Users[Weights[i].UserID].DeviationSum += pow(Weights[i].Weight - Users[Weights[i].UserID].WeightAvg, 2);
    }
    if (Items[Weights[i].ItemID].WeightsCount > 0) {
      Items[Weights[i].ItemID].DeviationSum += pow(Weights[i].Weight - Items[Weights[i].ItemID].WeightAvg, 2);
    }
  }
  for (i = 0; i < Users.size(); i++) {
    if (Users[i].WeightsCount > 0) {
      Users[i].Deviation = pow(Users[i].DeviationSum / Users[i].WeightsCount, 0.5);
    }
  }
  for (i = 0; i < Items.size(); i++) {
    if (Items[i].WeightsCount > 0) {
      Items[i].Deviation = pow(Items[i].DeviationSum / Items[i].WeightsCount, 0.5);
    }
  }
  cout << "Finished calculating user/item stats." << endl;
  AddTiming("Metrics finish");
}

void SVD::NormalizeWeights(bool deviation) {
  AddTiming("Normalize init");
  cout << "Normalizing weights..." << endl;
  unsigned int i = 0;
  for (i = 0; i < TotalWeights; i++) {
    if (Users[Weights[i].UserID].WeightsCount > 0 && Items[Weights[i].ItemID].WeightsCount > 0) {
      Weights[i].Weight -= Users[Weights[i].UserID].Offset + Items[Weights[i].ItemID].Offset;
      if (deviation) {
        Weights[i].Weight /= Users[Weights[i].UserID].Deviation * Items[Weights[i].ItemID].Deviation;
      }
    }
  }
  cout << "Finished normalizing weights." << endl;
  AddTiming("Normalize finish");
}

void SVD::CalcFeatures() {
  // Iteratively train each feature on the entire dataset.
  AddTiming("Features init");
  cout << "Calculating SVD features..." << endl;
  unsigned int feature = 0, epoch = 0, totalEpochs = 0, userID = 0, itemID = 0, i = 0, selectedWeights = 0;
  float err = 0, oldUserFeature = 0, oldItemFeature = 0, totalSquareError = 0, lastRmse = 0, rmse = 2.0;
  bool runTests = TestWeights.size() > 0;

  for (feature = 0; feature < Features; feature++) {
    // cout << "Feature: " << feature << " | Epoch: " << epoch << " | MinEpochs: " << MinEpochs << " | rmse: " << rmse << " | lastRMSE: " << lastRmse << " | MinImprovement: " << MinImprovement << endl;
    // Once the RMSE improvement is less than our min threshold and we're past the minimum number of epochs, move on to the next feature.
    for (epoch = 0;  (epoch < MinEpochs) || (rmse <= lastRmse - MinImprovement); epoch++) {
      // StartBenchmark("Epoch"");
      totalSquareError = 0;
      lastRmse = rmse;
      selectedWeights = 0;
      for (i = 0; i < TotalWeights; i++) {
        itemID = Weights[i].ItemID;
        userID = Weights[i].UserID;

        err = Weights[i].Weight - PredictWeight(itemID, userID, feature, Weights[i].Cache, true);

        // only count this as part of the train RMSE if over the given min weight count.
        if (Items[itemID].WeightsCount > 0 && Users[userID].WeightsCount > 0) {
          totalSquareError += err * err;
          selectedWeights++;
        }

        // Pull the old feature values from the cache.
        oldUserFeature = FeatureUsers[feature][userID];
        oldItemFeature = FeatureItems[feature][itemID];

        // Train the features.
        FeatureUsers[feature][userID] += LRate * (err * oldItemFeature - Tikhonov * oldUserFeature);
        FeatureItems[feature][itemID] += LRate * (err * oldUserFeature - Tikhonov * oldItemFeature);
      }
      rmse = sqrt(totalSquareError / selectedWeights);
      cout << "Epoch: " << totalEpochs + epoch << " | Last RMSE: " << lastRmse << " | RMSE: " << rmse << " | Improvement: " << (lastRmse - rmse);
      RunTest();
      // EndBenchmark("Epoch"");
    }
    totalEpochs += epoch;
    // Cache the feature contributions so far so we don't have to recompute it for every feature.
    for (i = 0; i < TotalWeights; i++) {
      Weights[i].Cache = PredictWeight(Weights[i].ItemID, Weights[i].UserID, feature, Weights[i].Cache, false);
    }
    cout << endl;
    AddTiming(std::string("Feature ") + to_string(feature) + std::string(" finish"));
  }
  cout << "Finished calculating SVD features." << endl;
  AddTiming("Features finish");
}

inline float SVD::ClipWeight(float weight) {
  return weight > ScaledMax ? ScaledMax : (weight < ScaledMin ? ScaledMin : weight);
}

inline float SVD::PredictWeight(unsigned int itemID, unsigned int userID, unsigned int feature, float cache, bool trailing) {
  // Predicts the weight for an item-user pairing.
  // Pulls the cached value of the contributions of all features up to this one if provided.
  // StartBenchmark("PredictWeight"");
  float sum = (cache > 0) ? cache : GlobalAvg + Items[itemID].Offset + Users[userID].Offset;

  // Add contribution of current feature.
  sum = ClipWeight(sum + FeatureItems[feature][itemID] * FeatureUsers[feature][userID]);

  // Add the default feature values for the remaining features in.
  if (trailing) {
    sum = ClipWeight(sum + (Features - feature - 1) * (FInit * FInit));
  }
  // EndBenchmark("PredictWeight"");
  return sum;
}

float SVD::PredictWeight(unsigned int itemID, unsigned int userID) {
  // Calculates the final weight prediction for an item-user pair.
  // Loops through all features, adding their contributions.
  float sum = GlobalAvg + Items[itemID].Offset + Users[userID].Offset;
  for (unsigned int f = 0; f < Features; f++) {
    sum = ClipWeight(sum + FeatureItems[f][itemID] * FeatureUsers[f][userID]);
  }
  return sum;
}

void SVD::RunTest() {
  // calculates the RMSE and MAE for the current SVD on the currently-loaded test corpus.
  // StartBenchmark("Test"");
  unsigned int userID = 0, itemID = 0, numTests = 0;
  float weight = 0;
  float rmse = 0, totalSquareError = 0, mae = 0, maeSum = 0, predictedWeight = 0, err = 0;
  for (unsigned int i = 0; i < TestCount; i++) {
    itemID = TestWeights[i].ItemID;
    userID = TestWeights[i].UserID;
    weight = TestWeights[i].Weight;
    if (Items[itemID].WeightsCount > 0 && Users[userID].WeightsCount > 0) {
      predictedWeight = PredictWeight(itemID, userID);
      err = 1.0 * predictedWeight - weight;
      maeSum += sqrt(err * err);
      totalSquareError += err * err;
      numTests++;
    }
  }
  rmse = sqrt(totalSquareError / numTests);
  mae = maeSum / numTests;
  cout << " | Test RMSE: " << rmse << endl;
  // EndBenchmark("Test"");
}

void SVD::SaveModel(std::string global_table, std::string means_table, std::string feature_table) {
  // saves the current SVD's features to output tables.
  // saves global mean for this entity pair to global_table,
  // each entity's paired global mean to means_table,
  // and each entity's paired feature list to feature_table.

  AddTiming("Save init");
  cout << "Saving SVD model..." << endl;
  unsigned int compactID = 0, sparseID = 0, itemID = 0;
  float prediction = 0.0;

  // create insert queues.
  vector<std::string> meanFields, featureFields, predictionFields;
  Query insertRow(DbConn.query()), globalQuery(DbConn.query()), deleteMeansQuery(DbConn.query()), deleteFeaturesQuery(DbConn.query()), deletePredictionsQuery(DbConn.query());

  // std::string meanFieldNames[] = {"id", "type", "source", "compare_type", "mean"};
  // meanFields.assign(meanFieldNames, meanFieldNames+5);

  std::string featureFieldNames[] = {"id", "type", "feature", "value"};
  featureFields.assign(featureFieldNames, featureFieldNames+4);

  // buffer length for inserts. larger values mean faster insertion (so long as the queue can fit in memory)!
  unsigned int maxQueueLength = 100000;

  InsertQueue featureInserts(DbConn, maxQueueLength, feature_table, featureFields);
  // InsertQueue meanInserts(DbConn, maxQueueLength, means_table, meanFields);

  // // update global means for this entity type pair.
  // try {
  //   globalQuery << "INSERT INTO " << global_table << " (type_1, type_2, mean) VALUES (" << EntityType1 << "," << EntityType2 << "," << (GlobalAvg/ScaleFactor) << ") ON DUPLICATE KEY UPDATE mean=" << (GlobalAvg/ScaleFactor);
  //   globalQuery.execute();
  // } catch (BadQuery er) {
  //   cerr << "Error updating global mean." << endl;
  //   throw er;
  // }

  // clear all the means, features, and predictions for this pair of types.
  // try {
  //   deleteMeansQuery << "DELETE FROM " << means_table << " WHERE type=" << EntityType1 << " && compare_type=" << EntityType2;
  //   deleteMeansQuery.execute();
  //   AddTiming("Delete means finish");
  // } catch (BadQuery er) {
  //   cerr << "Error deleting entity means." << endl;
  //   throw er;
  // }

  try {
    deleteFeaturesQuery << "DELETE FROM " << feature_table << " WHERE type=" << EntityType;
    deleteFeaturesQuery.execute();
    deleteFeaturesQuery << "DELETE FROM " << feature_table << " WHERE type='user'";
    deleteFeaturesQuery.execute();
    AddTiming("Delete features finish");
  } catch (BadQuery er) {
    cerr << "Error deleting entity and user features." << endl;
    throw er;
  }

  cout << "Saving users..." << endl;
  for (idMapItr userIterator = UserIDs.begin(); userIterator != UserIDs.end(); ++userIterator) {
    compactID = userIterator->second;
    // update user means.
    // insertRow.reset();
    // insertRow << "(" << quote << sparseID << "," << quote << "user" << "," << quote << Users[compactID].Source << "," << quote << EntityType2 << "," << quote << (Users[compactID].RegularizedAvg/ScaleFactor) << ")";
    // meanInserts.Append(insertRow.str());

    // only update features if this user has them calculated!
    if (Users[compactID].WeightsCount == 0) {
      continue;
    }

    // update user features.
    for (unsigned int feature = 0; feature < Features; feature++) {
      insertRow.reset();
      insertRow << "(" << quote << userIterator->first << "," << quote << "user" << "," << quote << feature << "," << quote << FeatureUsers[feature][compactID] << ")";
      featureInserts.Append(insertRow.str());
    }
  }
  AddTiming("Save users finish");
  cout << "Saving items..." << endl;
  for (vector<unsigned int>::iterator itemIterator = SelectedItems.begin(); itemIterator != SelectedItems.end(); ++itemIterator) {
    compactID = (unsigned int) *itemIterator;
    // update item means.
    // insertRow.reset();
    // insertRow << "(" << quote << Items[compactID].SparseID << "," << quote << EntityType2 << "," << quote << Items[compactID].Source << "," << quote << EntityType1 << "," << quote << (Items[compactID].RegularizedAvg/ScaleFactor) << ")";
    // meanInserts.Append(insertRow.str());

    // only update features if this item has them calculated!
    if (Items[compactID].WeightsCount == 0) {
      continue;
    }
    // update item features.
    for (unsigned int feature = 0; feature < Features; feature++) {
      insertRow.reset();
      insertRow << "(" << quote << Items[compactID].SparseID << "," << quote << EntityType << "," << quote << feature << "," << quote << FeatureItems[feature][compactID] << ")";
      featureInserts.Append(insertRow.str());
    }
  }
  AddTiming("Save items finish");
  // meanInserts.Flush();
  featureInserts.Flush();
  cout << "SVD saved." << endl;
  AddTiming("Save finish");
}