/*
  insert_queue.hpp
  Database insert queue class.
*/

#ifndef INSERT_QUEUE_HPP
#define INSERT_QUEUE_HPP

#include "utilities.hpp"

class InsertQueue {
  // Simple class to queue batch inserts to database.
  private:
    Connection              DbConn;
    unsigned int            MaxQueueSize;
    unsigned int            QueueSize;
    std::string             Table;
    vector<std::string>     Fields;
    vector<std::string>     Inserts;
  public:
    InsertQueue(Connection& dbConn, unsigned int queue_size, std::string table, vector<std::string>& fields);
    ~InsertQueue(void) { };
    void  Append(std::string insert);
    void  Flush();
};

#endif