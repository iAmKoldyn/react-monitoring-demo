#include <prometheus/exposer.h>
#include <prometheus/registry.h>
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <fstream>
#include <thread>
#include <iostream>
#include <curl/curl.h>
#include <random>
#include <string>
#include <vector>
#include <atomic>

std::atomic<long long> totalBytesDownloaded(0);
std::atomic<long long> totalBytesUploaded(0);

size_t WriteData(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    totalBytesDownloaded += written;
    return written;
}

std::string generateRandomData(size_t length) {
    const std::string chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<> dist(0, chars.size() - 1);

    std::string data;
    for (size_t i = 0; i < length; ++i) {
        data += chars[dist(generator)];
    }
    return data;
}

void postData(size_t dataSize) {
    CURL *curl = curl_easy_init();
    if (curl) {
        std::string data = generateRandomData(dataSize);
        curl_easy_setopt(curl, CURLOPT_URL, "https://httpbin.org/post");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.size());

        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            totalBytesUploaded += data.size();
        }
        curl_easy_cleanup(curl);
    }
}

void downloadFile(const char* url, const char* filename) {
    CURL *curl = curl_easy_init();
    if (curl) {
        FILE *fp = fopen(filename, "wb");
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteData);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

        CURLcode res = curl_easy_perform(curl);
        fclose(fp);
        if (res != CURLE_OK) {
            std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        }

        curl_easy_cleanup(curl);
        remove(filename);
    }
}

int main() {
    using namespace prometheus;
    Exposer exposer{"0.0.0.0:8081"};
    auto registry = std::make_shared<Registry>();

    auto& download_counter = BuildCounter()
                               .Name("download_requests_total")
                               .Help("Total download requests")
                               .Register(*registry)
                               .Add({});
    auto& upload_counter = BuildCounter()
                               .Name("upload_requests_total")
                               .Help("Total upload requests")
                               .Register(*registry)
                               .Add({});
    auto& downloaded_bytes = BuildGauge()
                               .Name("downloaded_bytes")
                               .Help("Total downloaded bytes")
                               .Register(*registry)
                               .Add({});
    auto& uploaded_bytes = BuildGauge()
                               .Name("uploaded_bytes")
                               .Help("Total uploaded bytes")
                               .Register(*registry)
                               .Add({});

    exposer.RegisterCollectable(registry);

    std::vector<std::thread> threads;

    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&]{
            while (true) {
                downloadFile("https://file-examples.com/wp-content/uploads/2017/02/file_example_TIFF_1MB.tiff", "/tmp/file.tiff");
                download_counter.Increment();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&]{
            while (true) {
                postData(1024 * 1024); // 1 MB
                upload_counter.Increment();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
    }

    std::thread metrics_thread([&]{
        while (true) {
            downloaded_bytes.Set(totalBytesDownloaded.load());
            uploaded_bytes.Set(totalBytesUploaded.load());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    // threads
    for (auto& thread : threads) {
        thread.join();
    }
    metrics_thread.join();

    return 0;
}
