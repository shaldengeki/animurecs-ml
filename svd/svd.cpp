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
#include <cmath>
#include <map>
#include <time.h>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

#include "utilities.hpp"
#include "svd.hpp"

SVD::SVD(unsigned int features, unsigned int min_ep, unsigned int max_ep, float min_imp, float l_rate, float tikh, float f_i, unsigned int prior_w, unsigned int min_weights_count, float min_w, float max_w) : 
features_count(features), min_epochs(min_ep), max_epochs(max_ep), min_improvement(min_imp), learn_rate(l_rate), tikhonov(tikh), f_init(f_i), prior_weight(prior_w), min_weights(min_weights_count), min_weight(min_w), max_weight(max_w) {
  AddTiming("SVD init");
  total_weights = weights_count = baseline_count = test_count = weight_sum = global_avg = 0;
  vector<float> blank_feature;
  for (unsigned int f = 0; f < features_count; f++) {
    features_items.push_back(blank_feature);
    features_users.push_back(blank_feature);
  }
}

inline void SVD::AddTiming(std::string description) {
  // adds a timing to this SVD's list for post-run profiling.
  timings.push_back(Timing(clock(), description));
}

void SVD::PrintTimings() {
  // Prints out every time,string pair in Timings for profiling purposes.
  cout << "Timings" << endl;
  cout << "=======" << endl;
  clock_t prev_time = clock();
  for (vector<Timing>::iterator timing_itr = timings.begin(); timing_itr != timings.end(); ++timing_itr) {
    Timing time_pair = *timing_itr;
    cout << time_pair.second << ": " << "(" << (time_pair.first / CLOCKS_PER_SEC) << "," << ((time_pair.first - prev_time) / CLOCKS_PER_SEC) << ") " << endl;
    prev_time = time_pair.first;
  }
}

inline void SVD::StartBenchmark(std::string name) {
  benchmark_starts[name] = clock();
}

inline void SVD::EndBenchmark(std::string name) {
  float bench_time = (clock() - benchmark_starts[name]) / CLOCKS_PER_SEC;

  benchmarks[name].sum += bench_time;
  benchmarks[name].count++;
  if (bench_time > benchmarks[name].max) {
    benchmarks[name].max = bench_time;
  }
  if (bench_time < benchmarks[name].min) {
    benchmarks[name].min = bench_time;
  }
  // benchmark_starts.erase(name);
}

void SVD::PrintBenchmarks() {
  // Prints out average durations in Benchmarks for each key for profiling purposes.
  BenchInfo bench_info;
  std::string name;
  cout << "Benchmarks" << endl;
  cout << "==========" << endl;

  for (BenchMap::iterator benchmark_itr = benchmarks.begin(); benchmark_itr != benchmarks.end(); ++benchmark_itr) {
    name = benchmark_itr->first;
    bench_info = benchmark_itr->second;
    cout << name << ": " << "Min " << bench_info.min << " | Max " << bench_info.max << " | Avg " << (bench_info.sum * 1.0 / bench_info.count) << endl;
  }
}

void SVD::LoadRow(unsigned int user_id, unsigned int item_id, float weight, bool baseline, bool test) {
  // loads a weight row into the SVD, given dual sparse IDs and a weight.
  // optionally takes baseline or test as a bool to indicate that this weight should be marked as such.
  // StartBenchmark("LoadRow"");
  idMapItr user_itr, item_itr;
  unsigned int uid = 0, aid = 0;

  // initialize a blank user and anime to push.
  User blank_user;
  blank_user.sparse_id = 0;
  blank_user.weights_sum = 0;
  blank_user.weights_count = 0;
  blank_user.weights_avg = 0;
  blank_user.regularized_avg = 0;

  Item blank_anime;
  blank_anime.sparse_id = 0;
  blank_anime.weights_sum = 0;
  blank_anime.weights_count = 0;
  blank_anime.weights_avg = 0;
  blank_anime.regularized_avg = 0;

  Data weight_entry;
  weight_entry.cache = 0;

  if (!test) {
    // Add users (using a map to re-number ids to array indices)
    user_itr = user_ids.find(user_id); 
    if (user_itr == user_ids.end()) {
      uid = users.size();

      // Reserve new id and add lookup.
      user_ids[user_id] = uid;
      blank_user.sparse_id = user_id;

      // Push a new user onto users and store the old sparse id for later.
      users.push_back(blank_user);
      user_id = uid;
    } else {
        user_id = user_itr->second;
    }

    users[user_id].weights_count++;
    users[user_id].weights_sum += weight;

    // Add anime (using a map to re-number ids to array indices) 
    item_itr = item_ids.find(item_id); 
    if (item_itr == item_ids.end()) {
      aid = item_ids.size();

      // Reserve new id and add lookup.
      item_ids[item_id] = aid;
      blank_anime.sparse_id = item_id;

      if (!baseline) {
        selected_items.push_back(item_id);
      }

      // Push a new anime onto animus.
      items.push_back(blank_anime);
      item_id = aid;
    } else {
      item_id = item_itr->second;
    }
    items[item_id].weights_count++;
    items[item_id].weights_sum += weight;
  }
  weight_entry.weight = weight;
  if (test) {
    weight_entry.user_id = user_ids[user_id];
    weight_entry.item_id = item_ids[item_id];
    test_weights.push_back(weight_entry);
  } else {
    weight_entry.user_id = user_id;
    weight_entry.item_id = item_id;
    weights.push_back(weight_entry);
    weight_sum += weight;
  }

  // increment counts appropriately.
  if (baseline) {
    baseline_count++;
  } else if (test) {
    test_count++;
  } else {
    weights_count++;
  }
  // EndBenchmark("LoadRow"");
}

void SVD::LoadBaselineRow(unsigned int user_id, unsigned int item_id, float weight) {
  LoadRow(user_id, item_id, weight, true, false);
}

void SVD::LoadWeightRow(unsigned int user_id, unsigned int item_id, float weight) {
  LoadRow(user_id, item_id, weight, false, false);
}

void SVD::LoadTestRow(unsigned int user_id, unsigned int item_id, float weight) {
  LoadRow(user_id, item_id, weight, false, true);
}

void SVD::LoadCSV(string training_file, bool baseline, bool test) {
  // Loads all of the ratings in the training file.
  // Data is in the form USERID,ANIMEID,RATING
  AddTiming("Load data init");
  cout << "Loading training data..." << endl;

  unsigned int user_id = 0, item_id = 0;
  float weight = 0;

  string line;
  ifstream data_file;
  // load the csv file line by line.
  // csv file is of the form USER_ID,ITEM_ID,WEIGHT
  data_file.open(training_file.c_str());
  if (data_file.is_open()) {
    while (!data_file.eof()) {
      getline(data_file, line);
      if (line.empty()) {
        continue;
      }
      std::vector<std::string> split_line = utilities::split(line, ',');
      user_id = atoi(split_line[0].c_str());
      item_id = (unsigned int) strtoul(split_line[1].c_str(), NULL, 0);
      weight = atof(split_line[2].c_str());

      LoadRow(user_id, item_id, weight, baseline, test);
    }
    data_file.close();
  } else {
    printf("Could not open training file at %s. Please check the path.\n", training_file.c_str());
  }
  AddTiming("Load data finish");
}

void SVD::CalcMetrics() {
  // Loops through the data and calculates averages used in training.
  AddTiming("Metrics init");
  cout << "Calculating global and user/item stats..." << endl;
  unsigned int i = 0, f = 0;
  float global_sum = 0;

  // set total weights count.
  total_weights = baseline_count + weights_count;
  cout << "Total weights: " << total_weights << endl;

  for (i = 0; i < total_weights; i++) {
    global_sum += weights[i].weight;
  }
  global_avg = global_sum / total_weights;
  cout << "Global average: " << global_avg << endl;

  // calculate item means.
  cout << "Calculating item means..." << endl;
  for (i = 0; i < items.size(); i++) {
    //cout << "Calculating item stats for iid " << i << "..." << endl;
    for (f = 0; f < features_count; f++) {
      features_items[f].push_back(f_init);
    }
    items[i].weights_avg = items[i].weights_sum / (1.0 * items[i].weights_count);
    items[i].regularized_avg = (global_avg * prior_weight + items[i].weights_sum) / (prior_weight * 1.0 + items[i].weights_count);
    if (items[i].weights_count < min_weights) {
      // mark this item to not be counted when evaluating RMSE.
      items[i].weights_count = 0;
      items[i].weights_sum = 0;
    }
  }
  cout << "Calculating item offsets..." << endl;
  for (i = 0; i < items.size(); i++) {
    items[i].offset = items[i].regularized_avg - global_avg;
  }

  // calculate user means.
  cout << "Calculating user means..." << endl;
  for (i = 0; i < users.size(); i++) {
    //cout << "Calculating user stats for uid " << i << "..." << endl;
    for (f = 0; f < features_count; f++) {
      features_users[f].push_back(f_init);
    }
    users[i].weights_avg = users[i].weights_sum / (1.0 * users[i].weights_count);
    users[i].regularized_avg = (global_avg * prior_weight + users[i].weights_sum) / (prior_weight * 1.0 + users[i].weights_count);
    if (users[i].weights_count < min_weights) {
      // mark these users to not be counted when evaluating RMSE.
      users[i].weights_count = 0;
      users[i].weights_sum = 0;
    }
  }
  cout << "Calculating user offsets..." << endl;
  for (i = 0; i < users.size(); i++) {
    users[i].offset = users[i].regularized_avg - global_avg;
  }

  cout << "Calculating user and item deviations..." << endl;
  for (i = 0; i < total_weights; i++) {
    if (users[weights[i].user_id].weights_count > 0) {
      users[weights[i].user_id].deviation_sum += pow(weights[i].weight - users[weights[i].user_id].weights_avg, 2);
    }
    if (items[weights[i].item_id].weights_count > 0) {
      items[weights[i].item_id].deviation_sum += pow(weights[i].weight - items[weights[i].item_id].weights_avg, 2);
    }
  }
  for (i = 0; i < users.size(); i++) {
    if (users[i].weights_count > 0) {
      users[i].deviation = pow(users[i].deviation_sum / users[i].weights_count, 0.5);
    }
  }
  for (i = 0; i < items.size(); i++) {
    if (items[i].weights_count > 0) {
      items[i].deviation = pow(items[i].deviation_sum / items[i].weights_count, 0.5);
    }
  }
  cout << "Finished calculating user/item stats." << endl;
  AddTiming("Metrics finish");
}

void SVD::NormalizeWeights(bool deviation) {
  AddTiming("Normalize init");
  cout << "Normalizing weights..." << endl;
  unsigned int i = 0;
  for (i = 0; i < total_weights; i++) {
    if (users[weights[i].user_id].weights_count > 0 && items[weights[i].item_id].weights_count > 0) {
      weights[i].weight -= users[weights[i].user_id].offset + items[weights[i].item_id].offset;
      if (deviation) {
        weights[i].weight /= users[weights[i].user_id].deviation * items[weights[i].item_id].deviation;
      }
    }
  }
  cout << "Finished normalizing weights." << endl;
  AddTiming("Normalize finish");
}

void SVD::Train(bool calculate_metrics) {
  if (calculate_metrics) {
    CalcMetrics();
    NormalizeWeights();
  }

  // Iteratively train each feature on the entire dataset.
  AddTiming("Features init");
  cout << "Calculating SVD features..." << endl;
  unsigned int feature = 0, epoch = 0, total_epochs = 0, user_id = 0, item_id = 0, i = 0, selectedWeights = 0;
  float err = 0, oldUserFeature = 0, oldItemFeature = 0, totalSquareError = 0, lastRmse = 0, rmse = 2.0, test_rmse = 0;
  bool runTests = test_weights.size() > 0;

  for (feature = 0; feature < features_count; feature++) {
    // cout << "Feature: " << feature << " | Epoch: " << epoch << " | min_epochs: " << min_epochs << " | rmse: " << rmse << " | lastRMSE: " << lastRmse << " | min_improvement: " << min_improvement << endl;
    // Once the RMSE improvement is less than our min threshold and we're past the minimum number of epochs, move on to the next feature.
    for (epoch = 0;  (epoch < max_epochs) && ((epoch < min_epochs) || (rmse <= lastRmse - min_improvement)); epoch++) {
      // StartBenchmark("Epoch"");
      totalSquareError = 0;
      lastRmse = rmse;
      selectedWeights = 0;
      for (i = 0; i < total_weights; i++) {
        item_id = weights[i].item_id;
        user_id = weights[i].user_id;

        err = weights[i].weight - PredictWeight(item_id, user_id, feature, weights[i].cache, true);

        // only count this as part of the train RMSE if over the given min weight count.
        if (items[item_id].weights_count > 0 && users[user_id].weights_count > 0) {
          totalSquareError += err * err;
          selectedWeights++;
        }

        // Pull the old feature values from the cache.
        oldUserFeature = features_users[feature][user_id];
        oldItemFeature = features_items[feature][item_id];

        // Train the features.
        features_users[feature][user_id] += learn_rate * (err * oldItemFeature - tikhonov * oldUserFeature);
        features_items[feature][item_id] += learn_rate * (err * oldUserFeature - tikhonov * oldItemFeature);
      }
      rmse = sqrt(totalSquareError / selectedWeights);
      cout << "Epoch: " << total_epochs + epoch << " | Last RMSE: " << lastRmse << " | RMSE: " << rmse << " | Improvement: " << (lastRmse - rmse);
      test_rmse = TestRMSE();
      if (!isnan(test_rmse)) {
        cout << " | Test RMSE: " << test_rmse;
      }
      cout << endl;
      // EndBenchmark("Epoch"");
    }
    total_epochs += epoch;
    // Cache the feature contributions so far so we don't have to recompute it for every feature.
    for (i = 0; i < total_weights; i++) {
      weights[i].cache = PredictWeight(weights[i].item_id, weights[i].user_id, feature, weights[i].cache, false);
    }
    cout << endl;
    AddTiming(std::string("Feature ") + to_string(feature) + std::string(" finish"));
  }
  cout << "Finished calculating SVD features." << endl;
  AddTiming("Features finish");
}

inline float SVD::ClipWeight(float weight) {
  return weight > max_weight ? max_weight : (weight < min_weight ? min_weight : weight);
}

inline float SVD::PredictWeight(unsigned int item_id, unsigned int user_id, unsigned int feature, float cache, bool trailing) {
  // Predicts the weight for an item-user pairing.
  // Pulls the cached value of the contributions of all features up to this one if provided.
  // StartBenchmark("PredictWeight"");
  float sum = (cache > 0) ? cache : global_avg + items[item_id].offset + users[user_id].offset;

  // Add contribution of current feature.
  sum = ClipWeight(sum + features_items[feature][item_id] * features_users[feature][user_id]);

  // Add the default feature values for the remaining features in.
  if (trailing) {
    sum = ClipWeight(sum + (features_count - feature - 1) * (f_init * f_init));
  }
  // EndBenchmark("PredictWeight"");
  return sum;
}

float SVD::PredictWeight(unsigned int item_id, unsigned int user_id) {
  // Calculates the final weight prediction for an item-user pair.
  // Loops through all features, adding their contributions.
  float sum = global_avg + items[item_id].offset + users[user_id].offset;
  for (unsigned int f = 0; f < features_count; f++) {
    sum = ClipWeight(sum + features_items[f][item_id] * features_users[f][user_id]);
  }
  return sum;
}

float SVD::TestRMSE() {
  // calculates the RMSE and MAE for the current SVD on the currently-loaded test corpus.
  // returns a float, or NaN if no test set is loaded.

  // StartBenchmark("Test"");
  unsigned int user_id = 0, item_id = 0, numTests = 0;
  float weight = 0;
  float rmse = 0, totalSquareError = 0, mae = 0, maeSum = 0, predictedWeight = 0, err = 0;
  for (unsigned int i = 0; i < test_count; i++) {
    item_id = test_weights[i].item_id;
    user_id = test_weights[i].user_id;
    weight = test_weights[i].weight;
    if (items[item_id].weights_count > 0 && users[user_id].weights_count > 0) {
      predictedWeight = PredictWeight(item_id, user_id);
      err = 1.0 * predictedWeight - weight;
      totalSquareError += err * err;
      numTests++;
    }
  }
  if (test_count > 0) {
    return sqrt(totalSquareError / numTests);
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
  // EndBenchmark("Test"");  
}

float SVD::MinWeight() {
  return min_weight;
}

float SVD::MaxWeight() {
  return max_weight;
}

unsigned int SVD::FeaturesCount() {
  return features_count;
}

IdMap SVD::UserIDs() {
  return user_ids;
}

vector<User> SVD::Users() {
  return users;
}

vector<float> SVD::UserFeatures(unsigned int user_id) {
  vector<float> user_feature;
  for (unsigned int f = 0; f < features_count; f++) {
    user_feature.push_back(features_users[f][user_ids[user_id]]);
  }
  return user_feature;
}

IdMap SVD::ItemIDs() {
  return item_ids;
}

vector<unsigned int> SVD::SelectedItems() {
  return selected_items;
}

vector<Item> SVD::Items() {
  return items;
}

vector<float> SVD::ItemFeatures(unsigned int item_id) {
  vector<float> item_feature;
  for (unsigned int f = 0; f < features_count; f++) {
    item_feature.push_back(features_items[f][item_ids[item_id]]);
  }
  return item_feature;
}