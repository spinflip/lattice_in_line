#define TSQUBO_SPARSE

#include "libtsqubo/tsqubo.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Edge = std::pair<int, int>;
using EdgeList = std::vector<Edge>;
using SparseQuboMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SparseTriplet = Eigen::Triplet<double>;

struct NormalizedGraph {
  std::vector<int> vertices;
  EdgeList edges;
};

enum class MatrixBackend {
  Map,
  EigenSparse,
};

static std::string matrix_backend_name(MatrixBackend backend) {
  switch (backend) {
    case MatrixBackend::Map:
      return "map";
    case MatrixBackend::EigenSparse:
      return "eigen-sparse";
  }
  return "unknown";
}

struct QuboModel {
  std::size_t nvars = 0;
  MatrixBackend backend = MatrixBackend::Map;
  std::map<std::pair<std::size_t, std::size_t>, double> coeffs;
  std::vector<SparseTriplet> triplets;

  void set_nvars(std::size_t n) {
    nvars = n;
    coeffs.clear();
    triplets.clear();
  }

  void add(std::size_t a, std::size_t b, double coeff) {
    if (coeff == 0.0) return;
    if (a > b) std::swap(a, b);
    if (backend == MatrixBackend::Map) {
      coeffs[{a, b}] += coeff;
    } else {
      triplets.emplace_back(static_cast<int>(a), static_cast<int>(b), coeff);
    }
  }

  void add_linear(std::size_t a, double coeff) { add(a, a, coeff); }

  void add_quadratic(std::size_t a, std::size_t b, double coeff) {
    if (a == b) {
      add_linear(a, coeff);
    } else {
      add(a, b, coeff);
    }
  }

  std::size_t nonzero_hint() const {
    return backend == MatrixBackend::Map ? coeffs.size() : triplets.size();
  }
};

struct Result {
  std::string method;
  std::vector<int> ordering;
  std::vector<int> position;
  int bandwidth = 0;
  double energy = 0.0;
  bool feasible_permutation = false;
  std::vector<int> bits;
  std::size_t nvars = 0;
  double elapsed_s = 0.0;
  double model_build_s = 0.0;
  double matrix_assembly_s = 0.0;
  double solver_core_s = 0.0;
  int scan_count = 0;
  int solver_runs = 0;
  std::size_t tabu_iterations = 0;
};

struct Options {
  std::string cluster = "C12";
  std::string method = "decision";
  int random_starts = 4;
  int tabu_tenure = 12;
  int cutoff = 150;
  unsigned long seed = 123;
  bool rcm_start = false;
  bool rcm = false;
  double permutation_weight = 500.0;
  double exponential_permutation_weight = 5000.0;
  double edge_weight = 100.0;
  std::optional<double> exponential_base;
  std::optional<int> exponential_max_distance;
  std::string clusters_file = "cluster_edges.py";
  MatrixBackend matrix_backend = MatrixBackend::Map;
};

static std::size_t x_var(int vertex, int pos, int n) {
  return static_cast<std::size_t>(vertex) * static_cast<std::size_t>(n) +
         static_cast<std::size_t>(pos);
}

static std::map<std::string, EdgeList> load_clusters(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open " + path);

  std::map<std::string, EdgeList> clusters;
  std::optional<std::string> current;
  int bracket_depth = 0;
  const std::regex start_re(R"REGEX(^\s*"([^"]+)"\s*:\s*\[)REGEX");
  const std::regex edge_re(R"(\((-?\d+)\s*,\s*(-?\d+)\))");

  std::string line;
  while (std::getline(in, line)) {
    std::smatch start_match;
    if (!current && std::regex_search(line, start_match, start_re)) {
      current = start_match[1].str();
      clusters[*current] = {};
      bracket_depth = 0;
    }

    if (!current) continue;

    bracket_depth += static_cast<int>(std::count(line.begin(), line.end(), '['));
    bracket_depth -= static_cast<int>(std::count(line.begin(), line.end(), ']'));

    for (auto it = std::sregex_iterator(line.begin(), line.end(), edge_re);
         it != std::sregex_iterator(); ++it) {
      int u = std::stoi((*it)[1].str());
      int v = std::stoi((*it)[2].str());
      clusters[*current].push_back({u, v});
    }

    if (bracket_depth == 0) current.reset();
  }

  return clusters;
}

static NormalizedGraph normalize_edges(const EdgeList& input) {
  std::set<int> vertex_set;
  for (const auto& [u, v] : input) {
    vertex_set.insert(u);
    vertex_set.insert(v);
  }

  std::vector<int> original_vertices(vertex_set.begin(), vertex_set.end());
  std::unordered_map<int, int> remap;
  for (std::size_t i = 0; i < original_vertices.size(); ++i) {
    remap[original_vertices[i]] = static_cast<int>(i);
  }

  std::set<Edge> edge_set;
  for (const auto& [u0, v0] : input) {
    if (u0 == v0) continue;
    int u = remap[u0];
    int v = remap[v0];
    if (u > v) std::swap(u, v);
    edge_set.insert({u, v});
  }

  NormalizedGraph graph;
  graph.vertices.resize(original_vertices.size());
  std::iota(graph.vertices.begin(), graph.vertices.end(), 0);
  graph.edges.assign(edge_set.begin(), edge_set.end());
  return graph;
}

static std::vector<std::vector<int>> build_adjacency(int n, const EdgeList& edges) {
  std::vector<std::vector<int>> adj(n);
  for (const auto& [u, v] : edges) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  for (auto& nbrs : adj) std::sort(nbrs.begin(), nbrs.end());
  return adj;
}

static int compute_bandwidth(const EdgeList& edges, const std::vector<int>& position) {
  int bw = 0;
  for (const auto& [u, v] : edges) {
    bw = std::max(bw, std::abs(position[u] - position[v]));
  }
  return bw;
}

static double compute_envelope(const EdgeList& edges, const std::vector<int>& position) {
  if (edges.empty()) return 0.0;
  double total = 0.0;
  for (const auto& [u, v] : edges) {
    total += std::abs(position[u] - position[v]);
  }
  return total / static_cast<double>(edges.size());
}

static std::vector<int> ordering_to_position(const std::vector<int>& ordering) {
  std::vector<int> position(ordering.size(), -1);
  for (std::size_t i = 0; i < ordering.size(); ++i) {
    position[ordering[i]] = static_cast<int>(i);
  }
  return position;
}

static int bandwidth_lower_bound(int n, const EdgeList& edges) {
  std::vector<int> degree(n, 0);
  for (const auto& [u, v] : edges) {
    ++degree[u];
    ++degree[v];
  }
  int max_degree = degree.empty() ? 0 : *std::max_element(degree.begin(), degree.end());
  return (max_degree + 1) / 2;
}

static std::vector<int> degree_ordering(int n, const EdgeList& edges) {
  std::vector<int> degree(n, 0);
  for (const auto& [u, v] : edges) {
    ++degree[u];
    ++degree[v];
  }
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    if (degree[a] != degree[b]) return degree[a] < degree[b];
    return a < b;
  });
  return order;
}

static std::vector<int> reverse_cuthill_mckee_ordering(int n, const EdgeList& edges) {
  auto adj = build_adjacency(n, edges);
  std::vector<int> degree(n);
  for (int v = 0; v < n; ++v) degree[v] = static_cast<int>(adj[v].size());

  std::vector<int> order;
  std::vector<char> visited(n, 0);
  while (static_cast<int>(order.size()) < n) {
    int start = -1;
    for (int v = 0; v < n; ++v) {
      if (!visited[v] && (start < 0 || degree[v] < degree[start])) start = v;
    }

    std::queue<int> q;
    q.push(start);
    visited[start] = 1;

    while (!q.empty()) {
      int u = q.front();
      q.pop();
      order.push_back(u);

      std::vector<int> nbrs;
      for (int v : adj[u]) {
        if (!visited[v]) nbrs.push_back(v);
      }
      std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
        if (degree[a] != degree[b]) return degree[a] < degree[b];
        return a < b;
      });
      for (int v : nbrs) {
        visited[v] = 1;
        q.push(v);
      }
    }
  }

  std::reverse(order.begin(), order.end());
  return order;
}

static std::vector<std::vector<int>> candidate_initial_orderings(
    int n, const EdgeList& edges, int max_random, unsigned long seed, bool rcm_start) {
  std::vector<std::vector<int>> orderings;
  auto rcm = reverse_cuthill_mckee_ordering(n, edges);

  if (rcm_start) orderings.push_back(rcm);

  std::vector<int> identity(n);
  std::iota(identity.begin(), identity.end(), 0);
  orderings.push_back(identity);
  orderings.push_back(degree_ordering(n, edges));
  orderings.push_back(rcm);

  auto base = orderings;
  for (auto ordering : base) {
    std::reverse(ordering.begin(), ordering.end());
    orderings.push_back(ordering);
  }

  std::mt19937_64 rng(seed);
  for (int i = 0; i < max_random; ++i) {
    auto ordering = identity;
    std::shuffle(ordering.begin(), ordering.end(), rng);
    orderings.push_back(ordering);
  }

  std::set<std::vector<int>> seen;
  std::vector<std::vector<int>> unique;
  for (auto& ordering : orderings) {
    if (seen.insert(ordering).second) unique.push_back(ordering);
  }
  return unique;
}

static void add_one_hot(QuboModel& model, const std::vector<std::size_t>& vars, double weight) {
  for (std::size_t var : vars) model.add_linear(var, -weight);
  for (std::size_t i = 0; i < vars.size(); ++i) {
    for (std::size_t j = i + 1; j < vars.size(); ++j) {
      model.add_quadratic(vars[i], vars[j], 2.0 * weight);
    }
  }
}

static void add_permutation_constraints(QuboModel& model, int n, double weight) {
  for (int v = 0; v < n; ++v) {
    std::vector<std::size_t> vars;
    for (int i = 0; i < n; ++i) vars.push_back(x_var(v, i, n));
    add_one_hot(model, vars, weight);
  }
  for (int i = 0; i < n; ++i) {
    std::vector<std::size_t> vars;
    for (int v = 0; v < n; ++v) vars.push_back(x_var(v, i, n));
    add_one_hot(model, vars, weight);
  }
}

static QuboModel build_decision_model(
    int n,
    const EdgeList& edges,
    int k,
    double permutation_weight,
    double edge_weight,
    MatrixBackend matrix_backend) {
  QuboModel model;
  model.backend = matrix_backend;
  model.set_nvars(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  add_permutation_constraints(model, n, permutation_weight);
  for (const auto& [u, v] : edges) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (std::abs(i - j) > k) {
          model.add_quadratic(x_var(u, i, n), x_var(v, j, n), edge_weight);
        }
      }
    }
  }
  return model;
}

static QuboModel build_exponential_model(
    int n,
    const EdgeList& edges,
    double permutation_weight,
    std::optional<double> maybe_base,
    std::optional<int> max_distance,
    MatrixBackend matrix_backend) {
  if (max_distance && *max_distance < 0) {
    throw std::runtime_error("exponential max distance must be non-negative");
  }
  const double base = maybe_base.value_or(static_cast<double>(n * (n - 1)) / 2.0 + 1.0);
  QuboModel model;
  model.backend = matrix_backend;
  model.set_nvars(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  add_permutation_constraints(model, n, permutation_weight);
  std::vector<double> powers(n, 1.0);
  for (int d = 1; d < n; ++d) powers[d] = powers[d - 1] * base;

  for (const auto& [u, v] : edges) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        int dist = std::abs(i - j);
        if (max_distance && dist > *max_distance) continue;
        model.add_quadratic(x_var(u, i, n), x_var(v, j, n), powers[dist]);
      }
    }
  }
  return model;
}

static std::vector<int> bits_from_ordering(int n, const std::vector<int>& ordering) {
  std::vector<int> bits(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0);
  for (int pos = 0; pos < n; ++pos) {
    int vertex = ordering[pos];
    bits[x_var(vertex, pos, n)] = 1;
  }
  return bits;
}

static struct tsqubo_instance* make_tsqubo_instance(const QuboModel& model) {
  auto* inst = tsqubo_instance_new(std::max<std::size_t>(model.nonzero_hint(), 2));
  if (model.backend == MatrixBackend::Map) {
    for (const auto& [ij, coeff] : model.coeffs) {
      const auto [i, j] = ij;
      if (i == j) {
        tsqubo_instance_add_component(inst, i, j, coeff);
      } else {
        // libtsqubo stores 2*q for off-diagonal interactions.
        tsqubo_instance_add_component(inst, i, j, coeff / 2.0);
      }
    }
    return inst;
  }

  SparseQuboMatrix matrix(static_cast<int>(model.nvars), static_cast<int>(model.nvars));
  matrix.setFromTriplets(
      model.triplets.begin(), model.triplets.end(),
      [](const double& a, const double& b) { return a + b; });
  matrix.makeCompressed();
  for (int outer = 0; outer < matrix.outerSize(); ++outer) {
    for (SparseQuboMatrix::InnerIterator it(matrix, outer); it; ++it) {
      if (it.value() == 0.0) continue;
      const std::size_t i = static_cast<std::size_t>(it.row());
      const std::size_t j = static_cast<std::size_t>(it.col());
      if (i == j) {
        tsqubo_instance_add_component(inst, i, j, it.value());
      } else {
        tsqubo_instance_add_component(inst, i, j, it.value() / 2.0);
      }
    }
  }
  return inst;
}

static std::tuple<std::vector<int>, std::vector<int>, bool> decode_ordering(
    int n, const std::vector<int>& bits) {
  std::vector<std::vector<int>> active_positions(n), active_vertices(n);
  for (int v = 0; v < n; ++v) {
    for (int pos = 0; pos < n; ++pos) {
      if (bits[x_var(v, pos, n)]) {
        active_positions[v].push_back(pos);
        active_vertices[pos].push_back(v);
      }
    }
  }

  bool feasible = true;
  for (int v = 0; v < n; ++v) feasible = feasible && active_positions[v].size() == 1;
  for (int pos = 0; pos < n; ++pos) feasible = feasible && active_vertices[pos].size() == 1;

  std::vector<int> ordering(n, -1), position(n, -1);
  if (feasible) {
    for (int v = 0; v < n; ++v) position[v] = active_positions[v][0];
    for (int pos = 0; pos < n; ++pos) ordering[pos] = active_vertices[pos][0];
    return {ordering, position, true};
  }

  std::vector<char> used_positions(n, 0);
  for (int v = 0; v < n; ++v) {
    int chosen = -1;
    for (int pos : active_positions[v]) {
      if (!used_positions[pos]) {
        chosen = pos;
        break;
      }
    }
    if (chosen < 0) {
      for (int pos = 0; pos < n; ++pos) {
        if (!used_positions[pos]) {
          chosen = pos;
          break;
        }
      }
    }
    position[v] = chosen;
    used_positions[chosen] = 1;
  }
  for (int v = 0; v < n; ++v) ordering[position[v]] = v;
  return {ordering, position, false};
}

static void solve_with_initial_bits(
    struct tsqubo* ts, const std::vector<int>& initial_bits, int tabu_tenure, int cutoff) {
  tsqubo_reset_solutions(ts);
  tsqubo_reset_tabu(ts);
  for (std::size_t i = 0; i < initial_bits.size(); ++i) {
    if (initial_bits[i]) tsqubo_flip_current(ts, i);
  }
  tsqubo_commit_incumbent(ts);
  tsqubo_reset_tabu(ts);
  tsqubo_local_search(ts);
  tsqubo_commit_incumbent(ts);
  tsqubo_iterate_cutoff(ts, static_cast<std::size_t>(tabu_tenure), static_cast<std::size_t>(cutoff));
}

static Result solve_model(
    const QuboModel& model,
    int n,
    const EdgeList& edges,
    const std::string& method,
    const std::vector<std::vector<int>>& initial_orderings,
    int tabu_tenure,
    int cutoff,
    std::optional<int> stop_at_bandwidth = std::nullopt) {
  auto start = std::chrono::steady_clock::now();
  auto matrix_start = std::chrono::steady_clock::now();
  auto* inst = make_tsqubo_instance(model);
  auto matrix_end = std::chrono::steady_clock::now();
  auto* ts = tsqubo_new(inst);
  tsqubo_instance_free(inst);
  free(inst);

  std::optional<Result> best;
  double solver_core_s = 0.0;
  int solver_runs = 0;
  std::size_t tabu_iterations = 0;
  for (const auto& ordering0 : initial_orderings) {
    auto initial_bits = bits_from_ordering(n, ordering0);
    auto solve_start = std::chrono::steady_clock::now();
    solve_with_initial_bits(ts, initial_bits, tabu_tenure, cutoff);
    auto solve_end = std::chrono::steady_clock::now();
    solver_core_s += std::chrono::duration<double>(solve_end - solve_start).count();
    solver_runs += 1;
    tabu_iterations += ts->iteration;

    std::vector<int> bits(model.nvars);
    for (std::size_t i = 0; i < model.nvars; ++i) {
      bits[i] = static_cast<int>(std::round(ts->inc.x[i]));
    }

    auto [ordering, position, feasible] = decode_ordering(n, bits);
    Result result;
    result.method = method;
    result.ordering = std::move(ordering);
    result.position = std::move(position);
    result.bandwidth = compute_bandwidth(edges, result.position);
    result.energy = ts->inc.fx;
    result.feasible_permutation = feasible;
    result.bits = std::move(bits);
    result.nvars = model.nvars;

    if (!best || std::tuple(!result.feasible_permutation, result.bandwidth, result.energy) <
                     std::tuple(!best->feasible_permutation, best->bandwidth, best->energy)) {
      best = result;
    }

    if (stop_at_bandwidth && best->feasible_permutation && best->bandwidth <= *stop_at_bandwidth) {
      break;
    }
  }

  tsqubo_free(ts);
  free(ts);

  if (!best) throw std::runtime_error("no initial orderings were generated");
  auto end = std::chrono::steady_clock::now();
  best->elapsed_s = std::chrono::duration<double>(end - start).count();
  best->matrix_assembly_s =
      std::chrono::duration<double>(matrix_end - matrix_start).count();
  best->solver_core_s = solver_core_s;
  best->solver_runs = solver_runs;
  best->tabu_iterations = tabu_iterations;
  return *best;
}

static Result solve_decision(
    int n,
    const EdgeList& edges,
    const std::vector<std::vector<int>>& initial_orderings,
    const Options& opt) {
  int lo = bandwidth_lower_bound(n, edges);
  int hi = n - 1;
  std::optional<Result> best;
  auto total_start = std::chrono::steady_clock::now();
  double total_model_build_s = 0.0;
  double total_matrix_assembly_s = 0.0;
  double total_solver_core_s = 0.0;
  int scan_count = 0;
  int solver_runs = 0;
  std::size_t tabu_iterations = 0;

  for (int k = lo; k <= hi; ++k) {
    auto build_start = std::chrono::steady_clock::now();
    auto model = build_decision_model(
        n, edges, k, opt.permutation_weight, opt.edge_weight, opt.matrix_backend);
    auto build_end = std::chrono::steady_clock::now();
    auto result = solve_model(
        model, n, edges, "decision(k=" + std::to_string(k) + ")", initial_orderings,
        opt.tabu_tenure, opt.cutoff, k);
    scan_count += 1;
    result.model_build_s = std::chrono::duration<double>(build_end - build_start).count();
    total_model_build_s += result.model_build_s;
    total_matrix_assembly_s += result.matrix_assembly_s;
    total_solver_core_s += result.solver_core_s;
    solver_runs += result.solver_runs;
    tabu_iterations += result.tabu_iterations;
    if (!best || std::tuple(!result.feasible_permutation, result.bandwidth, result.energy) <
                     std::tuple(!best->feasible_permutation, best->bandwidth, best->energy)) {
      best = result;
    }
    std::cout << "decision progress: scan=" << scan_count << "/" << (hi - lo + 1)
              << " k=" << k
              << " solver_runs_so_far=" << solver_runs
              << " best_bw_so_far=" << best->bandwidth
              << " elapsed_s="
              << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count()
              << "\n";
  }

  if (!best) throw std::runtime_error("decision solve produced no result");
  best->method = "decision";
  best->model_build_s = total_model_build_s;
  best->matrix_assembly_s = total_matrix_assembly_s;
  best->solver_core_s = total_solver_core_s;
  best->elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
  best->scan_count = scan_count;
  best->solver_runs = solver_runs;
  best->tabu_iterations = tabu_iterations;
  return *best;
}

static Result solve_exponential(
    int n,
    const EdgeList& edges,
    const std::vector<std::vector<int>>& initial_orderings,
    const Options& opt) {
  auto build_start = std::chrono::steady_clock::now();
  auto model = build_exponential_model(
      n, edges, opt.exponential_permutation_weight, opt.exponential_base,
      opt.exponential_max_distance, opt.matrix_backend);
  auto build_end = std::chrono::steady_clock::now();
  auto result = solve_model(
      model, n, edges, "exponential", initial_orderings, opt.tabu_tenure, opt.cutoff);
  result.model_build_s = std::chrono::duration<double>(build_end - build_start).count();
  result.elapsed_s = result.model_build_s + result.matrix_assembly_s + result.solver_core_s;
  result.scan_count = 1;
  return result;
}

static std::string qubo_size_plot(std::size_t nvars, int unit = 10, int max_width = 60) {
  std::size_t len = nvars == 0 ? 0 : std::min<std::size_t>(
                                      static_cast<std::size_t>(max_width),
                                      std::max<std::size_t>(1, (nvars + unit - 1) / unit));
  return "[" + std::string(len, '#') + "] " + std::to_string(nvars) +
         " vars (1 # ~= " + std::to_string(unit) + " vars)";
}

static void print_result(const Result& result, const EdgeList& edges) {
  std::vector<int> original_position(result.position.size());
  std::iota(original_position.begin(), original_position.end(), 0);
  int original_bw = compute_bandwidth(edges, original_position);
  double original_env = compute_envelope(edges, original_position);
  int recomputed_bw = compute_bandwidth(edges, result.position);
  double recomputed_env = compute_envelope(edges, result.position);

  std::cout << "method: " << result.method << "\n";
  std::cout << "old vertex -> assigned position: {";
  for (std::size_t i = 0; i < result.position.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << i << ": " << result.position[i];
  }
  std::cout << "}\n";
  std::cout << "bandwidth: " << recomputed_bw << " (envelope: " << std::setprecision(6) << recomputed_env << ")\n";
  std::cout << "bandwidth change vs original ordering: " << std::showpos
            << (recomputed_bw - original_bw) << std::noshowpos << " (original: " << original_bw
            << ", envelope: " << std::setprecision(6) << original_env << ")\n";
  std::cout << "energy: " << std::setprecision(8) << result.energy << "\n";
  std::cout << "valid permutation from solver: " << (result.feasible_permutation ? "true" : "false")
            << "\n";
  std::cout << "number of QUBO variables: " << result.nvars << "\n";
  std::cout << "QUBO size plot: " << qubo_size_plot(result.nvars) << "\n";
  std::cout << "elapsed_s: " << std::fixed << std::setprecision(6) << result.elapsed_s << "\n";
  std::cout << "timing breakdown (s): build=" << std::fixed << std::setprecision(6)
            << result.model_build_s << ", matrix=" << result.matrix_assembly_s
            << ", solver=" << result.solver_core_s << "\n";
  std::cout << "search effort: scan_count=" << result.scan_count
            << ", solver_runs=" << result.solver_runs
            << ", tabu_iterations=" << result.tabu_iterations << "\n";
  std::cout << "matrix multiplications: n/a for libtsqubo sparse tabu (incremental row updates)\n";
}

static void print_usage(const char* argv0) {
  std::cout << "usage: " << argv0
            << " [--cluster NAME] [--method decision|exponential|all]"
               " [--random-starts N] [--tabu-tenure N] [--cutoff N]"
               " [--seed N] [--rcm-start] [--rcm] [--exponential-base X]"
               " [--exponential-max-distance N] [--matrix-backend map|eigen-sparse]"
               " [--clusters-file PATH]\n";
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(name + " requires a value");
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--cluster") {
      opt.cluster = need_value(arg);
    } else if (arg == "--method") {
      opt.method = need_value(arg);
    } else if (arg == "--random-starts") {
      opt.random_starts = std::stoi(need_value(arg));
    } else if (arg == "--tabu-tenure") {
      opt.tabu_tenure = std::stoi(need_value(arg));
    } else if (arg == "--cutoff") {
      opt.cutoff = std::stoi(need_value(arg));
    } else if (arg == "--seed") {
      opt.seed = std::stoul(need_value(arg));
    } else if (arg == "--rcm-start") {
      opt.rcm_start = true;
    } else if (arg == "--rcm") {
      opt.rcm = true;
    } else if (arg == "--exponential-base") {
      opt.exponential_base = std::stod(need_value(arg));
    } else if (arg == "--exponential-max-distance") {
      opt.exponential_max_distance = std::stoi(need_value(arg));
    } else if (arg == "--matrix-backend") {
      const auto value = need_value(arg);
      if (value == "map") {
        opt.matrix_backend = MatrixBackend::Map;
      } else if (value == "eigen-sparse") {
        opt.matrix_backend = MatrixBackend::EigenSparse;
      } else {
        throw std::runtime_error("matrix backend must be map or eigen-sparse");
      }
    } else if (arg == "--clusters-file") {
      opt.clusters_file = need_value(arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opt.method != "decision" && opt.method != "exponential" && opt.method != "all") {
    throw std::runtime_error("method must be decision, exponential, or all");
  }
  if (opt.random_starts < 0 || opt.tabu_tenure < 0 || opt.cutoff < 0) {
    throw std::runtime_error("random starts, tabu tenure, and cutoff must be non-negative");
  }
  return opt;
}

int main(int argc, char** argv) {
  try {
    Options opt = parse_args(argc, argv);
    auto clusters = load_clusters(opt.clusters_file);
    auto it = clusters.find(opt.cluster);
    if (it == clusters.end()) {
      throw std::runtime_error("unknown cluster: " + opt.cluster);
    }

    auto graph = normalize_edges(it->second);
    const int n = static_cast<int>(graph.vertices.size());
    auto initial_orderings =
        candidate_initial_orderings(n, graph.edges, opt.random_starts, opt.seed, opt.rcm_start);

    std::cout << "cluster: " << opt.cluster << "\n";
    std::cout << "number of edges: " << graph.edges.size() << "\n";
    std::cout << "number of vertices: " << n << "\n";
    std::cout << "simple bandwidth lower bound: " << bandwidth_lower_bound(n, graph.edges) << "\n";
    std::cout << "QUBO matrix backend: " << matrix_backend_name(opt.matrix_backend) << "\n";
    std::vector<int> original_position(n);
    std::iota(original_position.begin(), original_position.end(), 0);
    std::cout << "original ordering bandwidth: " << compute_bandwidth(graph.edges, original_position)
              << " (envelope: " << std::setprecision(6) << compute_envelope(graph.edges, original_position) << ")\n";

    if (opt.rcm) {
      auto rcm_order = reverse_cuthill_mckee_ordering(n, graph.edges);
      auto rcm_pos = ordering_to_position(rcm_order);
      int rcm_bw = compute_bandwidth(graph.edges, rcm_pos);
      std::cout << "\n" << std::string(80, '=') << "\n";
      std::cout << "RCM baseline\n";
      std::cout << "ordering, position -> vertex: [";
      for (std::size_t i = 0; i < rcm_order.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << rcm_order[i];
      }
      std::cout << "]\n";
      std::cout << "position, vertex -> position: {";
      for (std::size_t i = 0; i < rcm_pos.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << i << ": " << rcm_pos[i];
      }
      std::cout << "}\n";
      std::cout << "bandwidth: " << rcm_bw << " (envelope: " << std::setprecision(6) << compute_envelope(graph.edges, rcm_pos) << ")\n";
    }

    std::vector<std::string> methods =
        opt.method == "all" ? std::vector<std::string>{"decision", "exponential"}
                            : std::vector<std::string>{opt.method};

    for (const auto& method : methods) {
      std::cout << "\n" << std::string(80, '=') << "\n";
      std::cout << "QUBO method: " << method << "\n";

      Result result;
      if (method == "decision") {
        result = solve_decision(n, graph.edges, initial_orderings, opt);
      } else {
        result = solve_exponential(n, graph.edges, initial_orderings, opt);
      }
      print_result(result, graph.edges);
    }

  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
