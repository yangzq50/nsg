
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <unordered_set>
#include <memory>

void output_recall_top_k(const unsigned top_k, const std::vector<std::vector<unsigned>> &results,
                         const std::vector<std::unordered_set<unsigned>> &groundtruth) {
    size_t gt_cnt = 0;
    for (unsigned i = 0; i < results.size(); i++) {
        std::unordered_set<unsigned> found;
        for (unsigned int j: results[i]) {
            found.insert(j);
        }
        for (unsigned v: found) {
            if (groundtruth[i].find(v) != groundtruth[i].end()) {
                ++gt_cnt;
            }
        }
    }
    std::cout << "R@100: " << float(gt_cnt) / float(results.size() * top_k) << std::endl;
}

template<typename T>
std::unique_ptr<T[]> load_data(const std::string &filename, size_t &num,
                               int &dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    // std::cout<<"data dimension: "<<dim<<std::endl;
    in.seekg(0, std::ios::end);
    auto ss = in.tellg();
    num = ((size_t) ss) / (dim + 1) / 4;
    auto data = std::make_unique<T[]>(num * dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) (data.get() + i * dim), dim * 4);
    }
    in.close();
    return data;
}

void usage_and_exit() {
    std::cout << "Usage: my_test_search graph.nsg base.fvecs query.fvecs groundtruth.ivecs search_L" << std::endl;
    std::cout << "graph.nsg: input nsg graph file" << std::endl;
    std::cout << "base.fvecs: input base data file" << std::endl;
    std::cout << "query.fvecs: input query data file" << std::endl;
    std::cout << "groundtruth.ivecs: input groundtruth file" << std::endl;
    std::cout << "search_L: search_L" << std::endl;
    exit(-1);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        usage_and_exit();
    }
    size_t points_num;
    int dim;
    auto data_load = load_data<float>(argv[2], points_num, dim);
    size_t query_num;
    int query_dim;
    auto query_load = load_data<float>(argv[3], query_num, query_dim);
    assert(dim == query_dim);
    size_t gt_num;
    int top_k;
    auto gt_load = load_data<int>(argv[4], gt_num, top_k);
    assert(query_num == gt_num);

    auto L = (unsigned) atoi(argv[5]), K = (unsigned) top_k;
    //output points_num, dim, query_num, query_dim, gt_num, top_k
    std::cout << "points_num: " << points_num << std::endl;
    std::cout << "dim: " << dim << std::endl;
    std::cout << "query_num: " << query_num << std::endl;
    std::cout << "query_dim: " << query_dim << std::endl;
    std::cout << "gt_num: " << gt_num << std::endl;
    std::cout << "top_k: " << top_k << std::endl;
    //output L
    std::cout << "search_L: " << L << std::endl;
    //output K
    std::cout << "search_K: " << K << std::endl;

    std::vector<std::unordered_set<unsigned>> gt(gt_num);
    for (unsigned i = 0; i < gt_num; i++) {
        for (unsigned j = 0; j < top_k; j++) {
            gt[i].insert(gt_load[i * top_k + j]);
        }
    }

    if (L < K) {
        std::cout << "search_L cannot be smaller than search_K!" << std::endl;
        exit(-1);
    }

    // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
    // align the data before build query_load = efanna2e::data_align(query_load,
    // query_num, query_dim);
    // TODO: Maybe metric doesn't matter here?
    efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);
    index.Load(argv[1]);
    //index.Load("/home/yzq/benchmark_save_index/nsg/sift_C500.nsg");
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);
    {
        //output "unoptimized_search"
        std::cout << "\n### unoptimized_search ###" << std::endl;
        for (unsigned repeat_n = 1; repeat_n <= 5; ++repeat_n) {
            //output loop number
            std::cout << "\nloop: " << repeat_n << std::endl;

            std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(K));

            auto s = std::chrono::high_resolution_clock::now();
            for (unsigned i = 0; i < query_num; i++) {
                index.Search(query_load.get() + i * dim, data_load.get(), K, paras, res[i].data());
            }
            auto e = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> diff = e - s;
            std::cout << "time: " << diff.count() << " s\n";
            std::cout << "qps: " << query_num / diff.count() << "\n";
            //calculate recall
            output_recall_top_k(K, res, gt);
        }
    }
    {
        //output "optimized_search"
        std::cout << "\n### optimized_search ###" << std::endl;
        index.OptimizeGraph(data_load.get());

        for (unsigned repeat_n = 1; repeat_n <= 5; ++repeat_n) {
            //output loop number
            std::cout << "\nloop: " << repeat_n << std::endl;

            std::vector<std::vector<unsigned>> res(query_num, std::vector<unsigned>(K));

            auto s = std::chrono::high_resolution_clock::now();
            for (unsigned i = 0; i < query_num; i++) {
                index.SearchWithOptGraph(query_load.get() + i * dim, K, paras, res[i].data());
            }
            auto e = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> diff = e - s;
            std::cout << "time: " << diff.count() << " s\n";
            std::cout << "qps: " << query_num / diff.count() << "\n";
            //calculate recall
            output_recall_top_k(K, res, gt);
        }
    }
    return 0;
}
