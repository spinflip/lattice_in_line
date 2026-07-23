#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using Edge = std::pair<int, int>;
using EdgeList = std::vector<Edge>;
using Adjacency = std::vector<std::set<int>>;

struct NormalizedGraph {
  std::vector<int> vertices;
  EdgeList edges;
};

struct BandwidthAlgorithmResult {
  std::string name;
  std::vector<int> ordering;
  std::vector<int> position;
  int bandwidth = 0;
  bool optimal = false;
  double elapsed_s = 0.0;
  std::size_t nodes_searched = 0;
  bool timed_out = false;
  std::string note;
};

struct RecognitionResult {
  bool feasible = false;
  std::optional<std::vector<int>> ordering;
  std::size_t nodes_searched = 0;
  bool timed_out = false;
};

struct ExactSearchConfig {
  std::string name;
  std::string position_strategy;
  std::string candidate_strategy;
  std::size_t node_limit = 2'000'000;
  double time_limit_s = 120.0;
  bool use_forward_check = true;
};

struct Options {
  std::string cluster = "C12";
  std::string algorithm = "all";
  std::size_t node_limit = 2'000'000;
  double time_limit = 600.0;
  std::string clusters_file = "cluster_edges.py";
};

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
      clusters[*current].push_back({std::stoi((*it)[1].str()), std::stoi((*it)[2].str())});
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
  for (std::size_t i = 0; i < original_vertices.size(); ++i) remap[original_vertices[i]] = static_cast<int>(i);

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

static Adjacency build_adjacency(const EdgeList& edges) {
  auto graph = normalize_edges(edges);
  Adjacency adj(graph.vertices.size());
  for (const auto& [u, v] : graph.edges) {
    adj[u].insert(v);
    adj[v].insert(u);
  }
  return adj;
}

static int compute_bandwidth_from_ordering(const EdgeList& edges, const std::vector<int>& ordering) {
  std::vector<int> position(ordering.size(), -1);
  for (std::size_t i = 0; i < ordering.size(); ++i) position[ordering[i]] = static_cast<int>(i);
  int best = 0;
  for (const auto& [u, v] : edges) best = std::max(best, std::abs(position[u] - position[v]));
  return best;
}

static double compute_avg_range_from_ordering(const EdgeList& edges, const std::vector<int>& ordering) {
  std::vector<int> position(ordering.size(), -1);
  for (std::size_t i = 0; i < ordering.size(); ++i) position[ordering[i]] = static_cast<int>(i);
  if (edges.empty()) return 0.0;
  double total = 0.0;
  for (const auto& [u, v] : edges) total += std::abs(position[u] - position[v]);
  return total / static_cast<double>(edges.size());
}

static int compute_cutwidth_from_ordering(const EdgeList& edges, const std::vector<int>& ordering) {
  if (edges.empty()) return 0;
  int n = static_cast<int>(ordering.size());
  std::vector<int> position(ordering.size(), -1);
  for (std::size_t i = 0; i < ordering.size(); ++i) position[ordering[i]] = static_cast<int>(i);
  std::vector<int> load(n + 1, 0);
  for (const auto& [u, v] : edges) {
    int a = std::min(position[u], position[v]);
    int b = std::max(position[u], position[v]);
    load[a] += 1;
    load[b] -= 1;
  }
  int cur = 0, mx = 0;
  for (int p = 0; p + 1 < n; ++p) {
    cur += load[p];
    mx = std::max(mx, cur);
  }
  return mx;
}

static std::vector<int> ordering_to_position(const std::vector<int>& ordering) {
  std::vector<int> pos(ordering.size(), -1);
  for (std::size_t i = 0; i < ordering.size(); ++i) pos[ordering[i]] = static_cast<int>(i);
  return pos;
}

static int bandwidth_lower_bound(const EdgeList& edges) {
  auto adj = build_adjacency(edges);
  if (adj.empty()) return 0;
  int max_degree = 0;
  for (const auto& nbrs : adj) max_degree = std::max(max_degree, static_cast<int>(nbrs.size()));
  return static_cast<int>(std::ceil(max_degree / 2.0));
}

static std::vector<std::vector<int>> connected_components(const Adjacency& adj) {
  int n = static_cast<int>(adj.size());
  std::vector<char> seen(n, 0);
  std::vector<std::vector<int>> comps;
  for (int start = 0; start < n; ++start) {
    if (seen[start]) continue;
    std::queue<int> q;
    q.push(start);
    seen[start] = 1;
    std::vector<int> comp;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      comp.push_back(u);
      for (int v : adj[u]) {
        if (!seen[v]) {
          seen[v] = 1;
          q.push(v);
        }
      }
    }
    comps.push_back(comp);
  }
  return comps;
}

static std::vector<std::vector<int>> bfs_levels(const Adjacency& adj, int root) {
  int n = static_cast<int>(adj.size());
  std::vector<int> dist(n, -1);
  dist[root] = 0;
  std::queue<int> q;
  q.push(root);
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    std::vector<int> nbrs(adj[u].begin(), adj[u].end());
    std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
      if (adj[a].size() != adj[b].size()) return adj[a].size() < adj[b].size();
      return a < b;
    });
    for (int v : nbrs) {
      if (dist[v] < 0) {
        dist[v] = dist[u] + 1;
        q.push(v);
      }
    }
  }

  int max_dist = *std::max_element(dist.begin(), dist.end());
  std::vector<std::vector<int>> levels(static_cast<std::size_t>(max_dist + 1));
  for (int v = 0; v < n; ++v) {
    if (dist[v] >= 0) levels[dist[v]].push_back(v);
  }
  return levels;
}

static int pseudo_peripheral_vertex(const Adjacency& adj, std::optional<int> start = std::nullopt) {
  int n = static_cast<int>(adj.size());
  if (n == 0) throw std::runtime_error("empty graph");

  int root = start.value_or(0);
  if (!start) {
    for (int v = 1; v < n; ++v) {
      if (adj[v].size() < adj[root].size()) root = v;
    }
  }

  int old_depth = -1;
  for (;;) {
    auto levels = bfs_levels(adj, root);
    int depth = static_cast<int>(levels.size());
    const auto& last_level = levels.back();
    int candidate = last_level.front();
    for (int v : last_level) {
      if (adj[v].size() < adj[candidate].size() || (adj[v].size() == adj[candidate].size() && v < candidate)) {
        candidate = v;
      }
    }
    if (depth <= old_depth) return root;
    root = candidate;
    old_depth = depth;
  }
}

static std::vector<int> cuthill_mckee_ordering(const EdgeList& edges) {
  auto adj = build_adjacency(edges);
  int n = static_cast<int>(adj.size());
  std::vector<char> visited(n, 0);
  std::vector<int> order;
  while (static_cast<int>(order.size()) < n) {
    int start = -1;
    for (int v = 0; v < n; ++v) {
      if (!visited[v] && (start < 0 || adj[v].size() < adj[start].size())) start = v;
    }
    std::queue<int> q;
    q.push(start);
    visited[start] = 1;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      order.push_back(u);
      std::vector<int> nbrs;
      for (int v : adj[u]) if (!visited[v]) nbrs.push_back(v);
      std::sort(nbrs.begin(), nbrs.end(), [&](int a, int b) {
        if (adj[a].size() != adj[b].size()) return adj[a].size() < adj[b].size();
        return a < b;
      });
      for (int v : nbrs) {
        visited[v] = 1;
        q.push(v);
      }
    }
  }
  return order;
}

static std::vector<int> reverse_cuthill_mckee_ordering(const EdgeList& edges) {
  auto order = cuthill_mckee_ordering(edges);
  std::reverse(order.begin(), order.end());
  return order;
}

static std::vector<int> gps_ordering(const EdgeList& edges) {
  auto adj = build_adjacency(edges);
  int n = static_cast<int>(adj.size());
  if (n == 0) return {};

  std::vector<std::vector<int>> orderings;
  for (const auto& comp : connected_components(adj)) {
    if (comp.empty()) continue;
    int start = comp.front();
    for (int v : comp) {
      if (adj[v].size() < adj[start].size()) start = v;
    }
    int u = pseudo_peripheral_vertex(adj, start);
    int v = pseudo_peripheral_vertex(adj, u);

    for (int root : {u, v}) {
      auto levels = bfs_levels(adj, root);
      std::vector<int> order;
      for (auto level : levels) {
        std::sort(level.begin(), level.end(), [&](int a, int b) {
          auto placed_neighbors = [&](int x) {
            int count = 0;
            for (int y : adj[x]) {
              if (std::find(order.begin(), order.end(), y) != order.end()) ++count;
            }
            return count;
          };
          auto ka = std::make_tuple(adj[a].size(), placed_neighbors(a), a);
          auto kb = std::make_tuple(adj[b].size(), placed_neighbors(b), b);
          return ka < kb;
        });
        order.insert(order.end(), level.begin(), level.end());
      }
      if (static_cast<int>(order.size()) == n) {
        orderings.push_back(order);
        std::reverse(order.begin(), order.end());
        orderings.push_back(order);
      }
    }
  }

  if (orderings.empty()) return reverse_cuthill_mckee_ordering(edges);

  return *std::min_element(orderings.begin(), orderings.end(), [&](const auto& a, const auto& b) {
    return compute_bandwidth_from_ordering(edges, a) < compute_bandwidth_from_ordering(edges, b);
  });
}

static std::vector<std::vector<int>> initial_upper_bound_orderings(const EdgeList& edges) {
  int n = static_cast<int>(build_adjacency(edges).size());
  std::vector<int> identity(n);
  std::iota(identity.begin(), identity.end(), 0);
  auto rcm = reverse_cuthill_mckee_ordering(edges);
  auto gps = gps_ordering(edges);

  std::vector<std::vector<int>> orderings = {
      identity,
      std::vector<int>(identity.rbegin(), identity.rend()),
      rcm,
      std::vector<int>(rcm.rbegin(), rcm.rend()),
      gps,
      std::vector<int>(gps.rbegin(), gps.rend()),
  };

  std::set<std::vector<int>> seen;
  std::vector<std::vector<int>> unique;
  for (const auto& o : orderings) {
    if (seen.insert(o).second) unique.push_back(o);
  }
  return unique;
}

static std::vector<int> best_heuristic_ordering(const EdgeList& edges) {
  auto candidates = initial_upper_bound_orderings(edges);
  return *std::min_element(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
    return compute_bandwidth_from_ordering(edges, a) < compute_bandwidth_from_ordering(edges, b);
  });
}

static std::vector<int> position_order(int n, const std::string& strategy) {
  if (strategy == "left_to_right") {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    return order;
  }
  if (strategy == "perimeter") {
    std::vector<int> order;
    int lo = 0;
    int hi = n - 1;
    while (lo <= hi) {
      order.push_back(lo);
      if (lo != hi) order.push_back(hi);
      ++lo;
      --hi;
    }
    return order;
  }
  throw std::runtime_error("unknown position strategy: " + strategy);
}

static bool is_compatible(int vertex, int pos, const std::vector<int>& assigned_pos, const Adjacency& adj, int k) {
  for (int u : adj[vertex]) {
    if (assigned_pos[u] >= 0 && std::abs(pos - assigned_pos[u]) > k) return false;
  }
  return true;
}

static bool forward_check(
    const std::set<int>& unplaced,
    const std::set<int>& remaining_positions,
    const std::vector<int>& assigned_pos,
    const Adjacency& adj,
    int k) {
  for (int v : unplaced) {
    bool ok = false;
    for (int pos : remaining_positions) {
      if (is_compatible(v, pos, assigned_pos, adj, k)) {
        ok = true;
        break;
      }
    }
    if (!ok) return false;
  }
  return true;
}

static std::vector<int> candidate_vertices_for_position(
    int pos,
    const std::set<int>& unplaced,
    const std::vector<int>& assigned_pos,
    const std::set<int>& remaining_positions,
    const Adjacency& adj,
    int k,
    const std::string& strategy) {
  std::vector<int> candidates;
  for (int v : unplaced) {
    if (is_compatible(v, pos, assigned_pos, adj, k)) candidates.push_back(v);
  }

  if (strategy == "degree") {
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
      if (adj[a].size() != adj[b].size()) return adj[a].size() > adj[b].size();
      return a < b;
    });
    return candidates;
  }

  if (strategy == "tight") {
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
      auto placed_neighbors = [&](int x) {
        int count = 0;
        int unplaced_neighbors = 0;
        for (int u : adj[x]) {
          if (assigned_pos[u] >= 0) ++count;
          if (unplaced.count(u)) ++unplaced_neighbors;
        }
        return std::make_tuple(-count, -static_cast<int>(adj[x].size()), -unplaced_neighbors, x);
      };
      return placed_neighbors(a) < placed_neighbors(b);
    });
    return candidates;
  }

  if (strategy == "domain") {
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
      auto domain_size = [&](int x) {
        int count = 0;
        for (int p : remaining_positions) {
          if (is_compatible(x, p, assigned_pos, adj, k)) ++count;
        }
        return std::make_tuple(count, -static_cast<int>(adj[x].size()), x);
      };
      return domain_size(a) < domain_size(b);
    });
    return candidates;
  }

  throw std::runtime_error("unknown candidate strategy: " + strategy);
}

static RecognitionResult recognize_bandwidth_exact(const EdgeList& edges, int k, const ExactSearchConfig& config) {
  auto adj = build_adjacency(edges);
  int n = static_cast<int>(adj.size());

  auto positions = position_order(n, config.position_strategy);
  std::vector<int> assigned_at_pos(n, -1);
  std::vector<int> assigned_pos(n, -1);
  std::set<int> unplaced;
  std::set<int> remaining_positions;
  for (int i = 0; i < n; ++i) {
    unplaced.insert(i);
    remaining_positions.insert(i);
  }

  auto start = std::chrono::steady_clock::now();
  std::size_t nodes = 0;
  bool timed_out = false;

  std::function<bool(int)> dfs = [&](int depth) -> bool {
    ++nodes;
    if (nodes >= config.node_limit) {
      timed_out = true;
      return false;
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (elapsed >= config.time_limit_s) {
      timed_out = true;
      return false;
    }
    if (depth == n) return true;

    int pos = positions[depth];
    auto candidates = candidate_vertices_for_position(
        pos, unplaced, assigned_pos, remaining_positions, adj, k, config.candidate_strategy);

    for (int v : candidates) {
      assigned_at_pos[pos] = v;
      assigned_pos[v] = pos;
      unplaced.erase(v);
      remaining_positions.erase(pos);

      bool ok = true;
      if (config.use_forward_check) {
        ok = forward_check(unplaced, remaining_positions, assigned_pos, adj, k);
      }

      if (ok && dfs(depth + 1)) return true;

      remaining_positions.insert(pos);
      unplaced.insert(v);
      assigned_pos[v] = -1;
      assigned_at_pos[pos] = -1;

      if (timed_out) return false;
    }

    return false;
  };

  bool feasible = dfs(0);
  RecognitionResult result;
  result.feasible = feasible;
  result.nodes_searched = nodes;
  result.timed_out = timed_out;
  if (feasible) result.ordering = assigned_at_pos;
  return result;
}

static BandwidthAlgorithmResult exact_minimize_bandwidth(const EdgeList& edges, const ExactSearchConfig& config) {
  auto t0 = std::chrono::steady_clock::now();

  int lb = bandwidth_lower_bound(edges);
  auto ub_order = best_heuristic_ordering(edges);
  int ub = compute_bandwidth_from_ordering(edges, ub_order);

  std::size_t total_nodes = 0;
  bool any_timeout = false;
  auto best_order = ub_order;
  int best_bw = ub;

  for (int k = lb; k <= ub; ++k) {
    auto rec = recognize_bandwidth_exact(edges, k, config);
    total_nodes += rec.nodes_searched;
    if (rec.timed_out) {
      any_timeout = true;
      break;
    }
    if (rec.feasible && rec.ordering) {
      best_order = *rec.ordering;
      best_bw = compute_bandwidth_from_ordering(edges, best_order);

      BandwidthAlgorithmResult result;
      result.name = config.name;
      result.ordering = best_order;
      result.position = ordering_to_position(best_order);
      result.bandwidth = best_bw;
      result.optimal = true;
      result.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      result.nodes_searched = total_nodes;
      result.timed_out = false;
      result.note = "proved optimum by recognition at k=" + std::to_string(k);
      return result;
    }
  }

  BandwidthAlgorithmResult result;
  result.name = config.name;
  result.ordering = best_order;
  result.position = ordering_to_position(best_order);
  result.bandwidth = best_bw;
  result.optimal = false;
  result.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  result.nodes_searched = total_nodes;
  result.timed_out = any_timeout;
  result.note = any_timeout
                    ? "returned best heuristic upper bound because exact recognition hit a time/node limit"
                    : "returned best known ordering";
  return result;
}

static BandwidthAlgorithmResult caprara_salazar_gonzalez(const EdgeList& edges, std::size_t node_limit, double time_limit_s) {
  return exact_minimize_bandwidth(
      edges,
      ExactSearchConfig{
          "Caprara-Salazar-Gonzalez", "left_to_right", "domain", node_limit, time_limit_s, true});
}

static BandwidthAlgorithmResult del_corso_manzini(const EdgeList& edges, std::size_t node_limit, double time_limit_s) {
  return exact_minimize_bandwidth(
      edges,
      ExactSearchConfig{
          "Del Corso-Manzini", "left_to_right", "tight", node_limit, time_limit_s, true});
}

static BandwidthAlgorithmResult del_corso_manzini_perimeter(
    const EdgeList& edges, std::size_t node_limit, double time_limit_s) {
  return exact_minimize_bandwidth(
      edges,
      ExactSearchConfig{
          "Del Corso-Manzini perimeter", "perimeter", "tight", node_limit, time_limit_s, true});
}

static BandwidthAlgorithmResult saxe_gurari_sudborough(const EdgeList& edges, std::size_t node_limit, double time_limit_s) {
  return exact_minimize_bandwidth(
      edges,
      ExactSearchConfig{
          "Saxe-Gurari-Sudborough", "left_to_right", "domain", node_limit, time_limit_s, true});
}

static BandwidthAlgorithmResult gibbs_poole_stockmeyer(const EdgeList& edges) {
  auto t0 = std::chrono::steady_clock::now();
  auto ordering = gps_ordering(edges);
  return {
      "Gibbs-Poole-Stockmeyer",
      ordering,
      ordering_to_position(ordering),
      compute_bandwidth_from_ordering(edges, ordering),
      false,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
      0,
      false,
      "heuristic",
  };
}

static BandwidthAlgorithmResult reverse_cuthill_mckee(const EdgeList& edges) {
  auto t0 = std::chrono::steady_clock::now();
  auto ordering = reverse_cuthill_mckee_ordering(edges);
  return {
      "Reverse Cuthill-McKee",
      ordering,
      ordering_to_position(ordering),
      compute_bandwidth_from_ordering(edges, ordering),
      false,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
      0,
      false,
      "heuristic",
  };
}

static BandwidthAlgorithmResult run_algorithm(const std::string& name, const EdgeList& edges, std::size_t node_limit, double time_limit_s) {
  std::string key = name;
  std::replace(key.begin(), key.end(), '_', '-');
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (key == "csg" || key == "caprara" || key == "caprara-salazar-gonzalez") {
    return caprara_salazar_gonzalez(edges, node_limit, time_limit_s);
  }
  if (key == "dcm" || key == "del-corso-manzini") {
    return del_corso_manzini(edges, node_limit, time_limit_s);
  }
  if (key == "dcm-perimeter" || key == "del-corso-manzini-perimeter" || key == "perimeter") {
    return del_corso_manzini_perimeter(edges, node_limit, time_limit_s);
  }
  if (key == "sgs" || key == "saxe-gurari-sudborough" || key == "saxe") {
    return saxe_gurari_sudborough(edges, node_limit, time_limit_s);
  }
  if (key == "gps" || key == "gibbs-poole-stockmeyer") {
    return gibbs_poole_stockmeyer(edges);
  }
  if (key == "rcm" || key == "reverse-cuthill-mckee") {
    return reverse_cuthill_mckee(edges);
  }
  throw std::runtime_error("unknown algorithm: " + name);
}

static void print_result(const BandwidthAlgorithmResult& result, const EdgeList& edges) {
  std::cout << "algorithm: " << result.name << "\n";
  std::cout << "ordering, position -> vertex: " << "[";
  for (std::size_t i = 0; i < result.ordering.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << result.ordering[i];
  }
  std::cout << "]\n";
  std::cout << "position, vertex -> position: {";
  for (std::size_t i = 0; i < result.position.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << i << ": " << result.position[i];
  }
  std::cout << "}\n";
  std::cout << "bandwidth: " << result.bandwidth
            << " (cutwidth: " << compute_cutwidth_from_ordering(edges, result.ordering)
            << ", avg_range: " << std::setprecision(6) << compute_avg_range_from_ordering(edges, result.ordering) << ")\n";
  std::cout << "optimal: " << (result.optimal ? "True" : "False") << "\n";
  std::cout << "elapsed_s: " << std::fixed << std::setprecision(6) << result.elapsed_s << "\n";
  std::cout << "nodes_searched: " << result.nodes_searched << "\n";
  std::cout << "timed_out: " << (result.timed_out ? "True" : "False") << "\n";
  if (!result.note.empty()) std::cout << "note: " << result.note << "\n";
}

static void print_summary_table(const std::vector<BandwidthAlgorithmResult>& results, const EdgeList& edges) {
  std::cout << "\n" << std::string(100, '=') << "\n";
  std::cout << "summary\n";
  std::cout << std::string(100, '=') << "\n";
  std::cout << std::left << std::setw(36) << "algorithm"
            << std::right << std::setw(5) << "bw"
            << std::setw(5) << "cut"
            << std::setw(11) << "avg_range"
            << std::setw(9) << "optimal"
            << std::setw(13) << "time_s"
            << std::setw(13) << "nodes"
            << " note\n";
  std::cout << std::string(100, '-') << "\n";

  auto sorted = results;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return std::tie(a.bandwidth, a.elapsed_s, a.name) < std::tie(b.bandwidth, b.elapsed_s, b.name);
  });

  for (const auto& r : sorted) {
    std::cout << std::left << std::setw(36) << r.name
              << std::right << std::setw(5) << r.bandwidth
              << std::setw(5) << compute_cutwidth_from_ordering(edges, r.ordering)
              << std::setw(11) << std::setprecision(6) << compute_avg_range_from_ordering(edges, r.ordering)
              << std::setw(9) << (r.optimal ? "True" : "False")
              << std::setw(13) << std::fixed << std::setprecision(6) << r.elapsed_s
              << std::setw(13) << r.nodes_searched
              << " " << r.note << "\n";
  }
}

static void print_usage(const char* argv0) {
  std::cout << "usage: " << argv0
            << " [--cluster NAME] [--algorithm all|csg|dcm|dcm-perimeter|sgs|gps|rcm]"
               " [--node-limit N] [--time-limit S] [--clusters-file PATH]\n";
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
    } else if (arg == "--algorithm") {
      opt.algorithm = need_value(arg);
    } else if (arg == "--node-limit") {
      opt.node_limit = static_cast<std::size_t>(std::stoull(need_value(arg)));
    } else if (arg == "--time-limit") {
      opt.time_limit = std::stod(need_value(arg));
    } else if (arg == "--clusters-file") {
      opt.clusters_file = need_value(arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  const std::set<std::string> valid = {"all", "csg", "dcm", "dcm-perimeter", "sgs", "gps", "rcm"};
  if (!valid.count(opt.algorithm)) throw std::runtime_error("unknown algorithm choice: " + opt.algorithm);
  return opt;
}

int main(int argc, char** argv) {
  try {
    auto opt = parse_args(argc, argv);
    auto clusters = load_clusters(opt.clusters_file);
    auto it = clusters.find(opt.cluster);
    if (it == clusters.end()) throw std::runtime_error("unknown cluster: " + opt.cluster);

    auto graph = normalize_edges(it->second);

    std::cout << "cluster: " << opt.cluster << "\n";
    std::cout << "vertices: " << graph.vertices.size() << "\n";
    std::cout << "edges: " << graph.edges.size() << "\n";
    std::cout << "simple lower bound: " << bandwidth_lower_bound(graph.edges) << "\n";
    std::vector<int> original_order(graph.vertices.size());
    std::iota(original_order.begin(), original_order.end(), 0);
    std::cout << "original ordering bandwidth: "
              << compute_bandwidth_from_ordering(graph.edges, original_order)
              << " (cutwidth: " << compute_cutwidth_from_ordering(graph.edges, original_order)
              << ", avg_range: " << std::setprecision(6) << compute_avg_range_from_ordering(graph.edges, original_order) << ")\n";
    auto best_heur = best_heuristic_ordering(graph.edges);
    std::cout << "best heuristic upper bound: "
              << compute_bandwidth_from_ordering(graph.edges, best_heur)
              << " (cutwidth: " << compute_cutwidth_from_ordering(graph.edges, best_heur)
              << ", avg_range: " << std::setprecision(6) << compute_avg_range_from_ordering(graph.edges, best_heur) << ")\n";

    std::vector<std::string> algorithms = opt.algorithm == "all"
                                              ? std::vector<std::string>{"rcm", "gps", "dcm", "dcm-perimeter", "sgs", "csg"}
                                              : std::vector<std::string>{opt.algorithm};

    std::vector<BandwidthAlgorithmResult> results;
    for (const auto& alg : algorithms) {
      std::cout << "\n" << std::string(100, '=') << "\n";
      auto result = run_algorithm(alg, graph.edges, opt.node_limit, opt.time_limit);
      print_result(result, graph.edges);
      results.push_back(result);
    }

    if (results.size() > 1) print_summary_table(results, graph.edges);
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
