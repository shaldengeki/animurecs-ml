CXX := g++
CXXFLAGS := -I/usr/include/mysql -I/usr/include/mysql++ -std=c++11 -Wfatal-errors
LDFLAGS := -L/usr/lib -lmysqlpp -lboost_program_options -lnsl -lz -lm -O3
EXECUTABLE := svd
OBJECTS := utilities.o insert_queue.o svd.o main.o

all: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(EXECUTABLE) $(OBJECTS)

clean:
	rm *.o svd