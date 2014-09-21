/*
  insert_queue.cpp
  Database insert queue class.
*/

#include <string>
#include <sstream>
#include <vector>

#include <mysql++.h>
using namespace mysqlpp;
using namespace std;

#include "insert_queue.hpp"
#include "utilities.hpp"

InsertQueue::InsertQueue(Connection& dbConn, unsigned int queue_size, std::string table, vector<std::string>& fields) : DbConn(dbConn), MaxQueueSize(queue_size), Table(table) {
  if (!DbConn.connected()) {
    throw new ConnectionFailed("Could not connect to MySQL.");
  }
  Fields = vector<std::string>(fields);

  QueueSize = 0;
}

void InsertQueue::Append(std::string insert) {
  // Appends an insert to the queue and flushes if need be.
  Inserts.push_back(insert);
  QueueSize++;

  if (QueueSize > MaxQueueSize) {
    Flush();
  }
}

void InsertQueue::Flush() {
  // Flushes the insert queue.
  std::stringstream flushQuery;

  if (QueueSize == 0) {
    return;
  }
  cout << "Flushing " << QueueSize << " rows: " << (clock() / CLOCKS_PER_SEC) << endl;
  try {
    Query flushQueue = DbConn.query();
    flushQueue << "INSERT INTO " << Table << "(" << utilities::join(Fields.begin(), Fields.end(), ",") << ") VALUES " << utilities::join(Inserts.begin(), Inserts.end(), ",");
    flushQueue.execute();
  } catch (BadQuery er) {
    cerr << "Error flushing insert queue." << endl;
    throw er;
  }
  Inserts.clear();
  QueueSize = 0;
}