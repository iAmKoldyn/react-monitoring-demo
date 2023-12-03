#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <prometheus/gauge.h>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <atomic>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <cstring>
#include <sstream>

std::atomic<long long> cpuLoad(0);
std::atomic<long long> memoryUsage(0);
std::atomic<long long> diskIO(0);

void cpuLoadTask() {
    while (true) {
        long long load = 0;
        for (int i = 0; i < 50000; ++i) {
            load += std::sqrt(std::rand());
        }
        cpuLoad.store(load);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void memoryLoadTask() {
    std::vector<char> buffer;

        const long long MaxMemoryUsage = 10LL * 1024 * 1024 * 1024; // 10 GB

    while (true) {
    // on local:
    //     buffer.resize(buffer.size() + 100 * 1024 * 1024); // 100MB
    // on server:
        if (memoryUsage.load() < MaxMemoryUsage) {
            buffer.resize(buffer.size() + 50 * 1024 * 1024); // 50MB
        } else {
            buffer.clear();
        }
        std::fill(buffer.begin(), buffer.end(), '0');
        memoryUsage.store(buffer.size());
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

void diskLoadTask() {
    while (true) {
        std::ofstream file("tempfile.txt");
        for (int i = 0; i < 50000; ++i) {
            file << "Data line " << i << "\n";
            diskIO.fetch_add(20);
        }
        file.close();
        remove("tempfile.txt");
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

double getCpuUtilization() {
    std::ifstream file("/proc/stat");
    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string cpu;
        long user, nice, system, idle;
        if (iss >> cpu >> user >> nice >> system >> idle) {
            long totalIdle = idle;
            long totalUsage = user + nice + system;
            return static_cast<double>(totalUsage) / (totalUsage + totalIdle);
        }
    }
    return 0.0;
}

long getDiskIO() {
    std::ifstream file("/proc/diskstats");
    std::string line;
    long totalReads = 0;
    long totalWrites = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int major, minor;
        std::string devName;
        iss >> major >> minor >> devName;

        long reads, readsMerged, sectorsRead, readTime, writes, writesMerged, sectorsWritten, writeTime;
        iss >> reads >> readsMerged >> sectorsRead >> readTime >> writes >> writesMerged >> sectorsWritten >> writeTime;

        totalReads += reads;
        totalWrites += writes;
    }

    return totalReads + totalWrites;
}

int main() {
    using namespace prometheus;

    Exposer exposer{"0.0.0.0:8080"};
    auto registry = std::make_shared<Registry>();

    auto& simulatedCpuGauge = BuildGauge().Name("simulated_cpu_load").Help("Simulated CPU Load").Register(*registry).Add({});
    auto& memoryGauge = BuildGauge().Name("memory_usage").Help("Memory Usage").Register(*registry).Add({});
    auto& simulatedDiskGauge = BuildGauge().Name("simulated_disk_io").Help("Simulated Disk I/O").Register(*registry).Add({});
    auto& cpuGauge = BuildGauge().Name("cpu_utilization").Help("CPU Utilization").Register(*registry).Add({});
    auto& diskGauge = BuildGauge().Name("disk_io").Help("Disk I/O").Register(*registry).Add({});

    exposer.RegisterCollectable(registry);

    std::thread cpuThread(cpuLoadTask);
    std::thread memoryThread(memoryLoadTask);
    std::thread diskThread(diskLoadTask);

    while (true) {
        simulatedCpuGauge.Set(cpuLoad.load());
        memoryGauge.Set(memoryUsage.load());
        simulatedDiskGauge.Set(diskIO.load());
        cpuGauge.Set(getCpuUtilization());
        diskGauge.Set(getDiskIO());
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    cpuThread.join();
    memoryThread.join();
    diskThread.join();

    return 0;
}

