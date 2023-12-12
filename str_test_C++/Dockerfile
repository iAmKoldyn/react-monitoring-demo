FROM gcc:latest

RUN apt-get update && apt-get install -y libcurl4-openssl-dev cmake git zlib1g-dev

WORKDIR /usr/src/myapp

RUN git clone https://github.com/jupp0r/prometheus-cpp.git \
    && cd prometheus-cpp \
    && git submodule update --init --recursive \
    && mkdir build && cd build \
    && cmake .. -DBUILD_SHARED_LIBS=ON \
    && make -j 4 \
    && make install

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

COPY main.cpp .
COPY network_test.cpp .

RUN g++ -o main main.cpp -lpthread -lcurl -lprometheus-cpp-pull -lprometheus-cpp-core -lz
RUN g++ -o network_test network_test.cpp -lcurl -lpthread -lprometheus-cpp-pull -lprometheus-cpp-core -lz
