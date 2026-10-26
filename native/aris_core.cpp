// Native weight solver for the ARIS500 long-only portfolio.
//
// Mirrors _bounded_scale, _apply_name_sector_caps, and
// _beta_constrained_weights in src/model/portfolio.py. Sums reproduce numpy's
// pairwise reduction and pandas' compensated group sums so results match the
// Python reference bit for bit; build with -ffp-contract=off and without
// -ffast-math to preserve that. Exposed through a plain C ABI for ctypes.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr double kTolerance = 1e-10;
constexpr int64_t kPairwiseBlock = 128;

enum Status : int {
    kOk = 0,
    kLowerAboveUpper = 1,
    kInfeasibleTotal = 2,
    kBetaInfeasible = 3,
};

// numpy's pairwise summation (np.sum on a contiguous float64 array).
double pairwise_sum(const double* a, int64_t n) {
    if (n < 8) {
        double res = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            res += a[i];
        }
        return res;
    }
    if (n <= kPairwiseBlock) {
        double r[8];
        for (int j = 0; j < 8; ++j) {
            r[j] = a[j];
        }
        int64_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (int j = 0; j < 8; ++j) {
                r[j] += a[i + j];
            }
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) {
            res += a[i];
        }
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

double pairwise_sum(const std::vector<double>& a) {
    return pairwise_sum(a.data(), static_cast<int64_t>(a.size()));
}

// pandas' Kahan-compensated groupby sum, accumulated in row order.
std::vector<double> group_sum(const double* values, const int32_t* codes, int64_t n, int32_t groups) {
    std::vector<double> sum(groups, 0.0);
    std::vector<double> compensation(groups, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t g = codes[i];
        const double y = values[i] - compensation[g];
        const double t = sum[g] + y;
        compensation[g] = t - sum[g] - y;
        if (std::isnan(compensation[g])) {
            compensation[g] = 0.0;
        }
        sum[g] = t;
    }
    return sum;
}

// np.clip semantics: NaN propagates, otherwise min(max(x, lo), hi).
inline double clip(double x, double lo, double hi) {
    if (std::isnan(x)) {
        return x;
    }
    const double floored = x > lo ? x : lo;
    return floored < hi ? floored : hi;
}

int bounded_scale(
    const double* target,
    const double* lower,
    const double* upper,
    int64_t n,
    double total,
    double* out
) {
    for (int64_t i = 0; i < n; ++i) {
        if (lower[i] > upper[i] + kTolerance) {
            return kLowerAboveUpper;
        }
    }
    if (pairwise_sum(lower, n) > total + kTolerance || pairwise_sum(upper, n) < total - kTolerance) {
        return kInfeasibleTotal;
    }

    const double eps = std::numeric_limits<double>::epsilon();
    std::vector<double> positive_target(n);
    for (int64_t i = 0; i < n; ++i) {
        const double t = target[i];
        positive_target[i] = std::isnan(t) ? t : (t > eps ? t : eps);
    }
    std::vector<double> candidate(n);
    auto scaled_sum = [&](double scale) {
        for (int64_t i = 0; i < n; ++i) {
            candidate[i] = clip(scale * positive_target[i], lower[i], upper[i]);
        }
        return pairwise_sum(candidate);
    };

    double low = 0.0;
    double high = 1.0;
    while (scaled_sum(high) < total - kTolerance) {
        high *= 2.0;
    }
    for (int iteration = 0; iteration < 60; ++iteration) {
        const double midpoint = (low + high) / 2.0;
        if (scaled_sum(midpoint) < total) {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    scaled_sum(high);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = candidate[i];
    }

    const double residual = total - pairwise_sum(out, n);
    if (std::fabs(residual) > kTolerance) {
        std::vector<int64_t> eligible;
        std::vector<double> room;
        for (int64_t i = 0; i < n; ++i) {
            const double r = residual > 0 ? upper[i] - out[i] : out[i] - lower[i];
            if (r > kTolerance) {
                eligible.push_back(i);
                room.push_back(r);
            }
        }
        if (!eligible.empty()) {
            const double room_total = pairwise_sum(room);
            for (size_t k = 0; k < eligible.size(); ++k) {
                out[eligible[k]] += residual * (room[k] / room_total);
            }
        }
    }
    return kOk;
}

struct SectorProblem {
    int64_t n;
    const int32_t* codes;
    int32_t sectors;
    const double* benchmark_sector;
    double max_name_weight;
    double min_position_weight;
    double max_sector_deviation;
    std::vector<std::vector<int64_t>> members;
    std::vector<double> lower;
    std::vector<double> upper;

    SectorProblem(
        int64_t n_,
        const int32_t* codes_,
        int32_t sectors_,
        const double* benchmark_sector_,
        double max_name_weight_,
        double min_position_weight_,
        double max_sector_deviation_
    )
        : n(n_),
          codes(codes_),
          sectors(sectors_),
          benchmark_sector(benchmark_sector_),
          max_name_weight(max_name_weight_),
          min_position_weight(min_position_weight_),
          max_sector_deviation(max_sector_deviation_),
          members(sectors_),
          lower(sectors_),
          upper(sectors_) {
        for (int64_t i = 0; i < n; ++i) {
            members[codes[i]].push_back(i);
        }
        // Sector bounds depend only on membership, so they are fixed across tilts.
        for (int32_t s = 0; s < sectors; ++s) {
            const double count = static_cast<double>(members[s].size());
            if (members[s].empty()) {
                lower[s] = 0.0;
                upper[s] = 0.0;
                continue;
            }
            double lo = benchmark_sector[s] - max_sector_deviation;
            lo = lo > 0.0 ? lo : 0.0;
            const double minimum_capacity = count * min_position_weight;
            lower[s] = lo > minimum_capacity ? lo : minimum_capacity;
            double hi = benchmark_sector[s] + max_sector_deviation;
            hi = hi < 1.0 ? hi : 1.0;
            const double capacity = count * max_name_weight;
            upper[s] = hi < capacity ? hi : capacity;
        }
    }

    // Project raw weights onto full-investment, position, and sector bounds.
    int caps(const double* raw_target, double* weights, double* portfolio_sector) const {
        const double raw_total = pairwise_sum(raw_target, n);
        std::vector<double> normalized(n);
        for (int64_t i = 0; i < n; ++i) {
            normalized[i] = raw_target[i] / raw_total;
        }
        const std::vector<double> raw_sector = group_sum(normalized.data(), codes, n, sectors);
        std::vector<double> sector_target(sectors);
        int status = bounded_scale(raw_sector.data(), lower.data(), upper.data(), sectors, 1.0, sector_target.data());
        if (status != kOk) {
            return status;
        }

        std::vector<double> group_raw, group_lower, group_upper, group_out;
        for (int32_t s = 0; s < sectors; ++s) {
            const std::vector<int64_t>& rows = members[s];
            if (rows.empty()) {
                continue;
            }
            const size_t m = rows.size();
            group_raw.resize(m);
            group_lower.assign(m, min_position_weight);
            group_upper.assign(m, max_name_weight);
            group_out.resize(m);
            for (size_t k = 0; k < m; ++k) {
                group_raw[k] = normalized[rows[k]];
            }
            status = bounded_scale(
                group_raw.data(),
                group_lower.data(),
                group_upper.data(),
                static_cast<int64_t>(m),
                sector_target[s],
                group_out.data()
            );
            if (status != kOk) {
                return status;
            }
            for (size_t k = 0; k < m; ++k) {
                weights[rows[k]] = group_out[k];
            }
        }
        const std::vector<double> sums = group_sum(weights, codes, n, sectors);
        for (int32_t s = 0; s < sectors; ++s) {
            portfolio_sector[s] = sums[s];
        }
        return kOk;
    }
};

double weighted_beta(const double* weights, const double* betas, int64_t n) {
    std::vector<double> product(n);
    for (int64_t i = 0; i < n; ++i) {
        product[i] = weights[i] * betas[i];
    }
    return pairwise_sum(product);
}

}  // namespace

extern "C" {

int aris_bounded_scale(
    const double* target,
    const double* lower,
    const double* upper,
    int64_t n,
    double total,
    double* out
) {
    return bounded_scale(target, lower, upper, n, total, out);
}

int aris_name_sector_caps(
    const double* raw_target,
    const int32_t* sector_codes,
    int64_t n,
    const double* benchmark_sector,
    int32_t sectors,
    double max_name_weight,
    double min_position_weight,
    double max_sector_deviation,
    double* weights,
    double* portfolio_sector
) {
    const SectorProblem problem(
        n, sector_codes, sectors, benchmark_sector,
        max_name_weight, min_position_weight, max_sector_deviation
    );
    return problem.caps(raw_target, weights, portfolio_sector);
}

int aris_beta_constrained_weights(
    const double* initial_weights,
    const double* betas,
    const int32_t* sector_codes,
    int64_t n,
    const double* benchmark_sector,
    int32_t sectors,
    double target_beta,
    double max_name_weight,
    double min_position_weight,
    double max_sector_deviation,
    double* weights,
    double* portfolio_sector
) {
    const SectorProblem problem(
        n, sector_codes, sectors, benchmark_sector,
        max_name_weight, min_position_weight, max_sector_deviation
    );
    const double center = weighted_beta(initial_weights, betas, n);

    struct Trial {
        std::vector<double> weights;
        std::vector<double> sector;
        double beta;
    };
    std::vector<double> raw(n);
    auto candidate = [&](double tilt, Trial& trial, int& status) {
        for (int64_t i = 0; i < n; ++i) {
            const double exponent = clip(tilt * (betas[i] - center), -50.0, 50.0);
            raw[i] = initial_weights[i] * std::exp(exponent);
        }
        trial.weights.assign(n, 0.0);
        trial.sector.assign(sectors, 0.0);
        status = problem.caps(raw.data(), trial.weights.data(), trial.sector.data());
        trial.beta = status == kOk ? weighted_beta(trial.weights.data(), betas, n) : 0.0;
    };

    int status = kOk;
    const bool lowering = center > target_beta;
    double step = lowering ? -1.0 : 1.0;
    Trial endpoint;
    candidate(step, endpoint, status);
    if (status != kOk) {
        return status;
    }
    bool reached = false;
    for (int iteration = 0; iteration < 20; ++iteration) {
        reached = lowering ? endpoint.beta <= target_beta : endpoint.beta >= target_beta;
        if (reached) {
            break;
        }
        step *= 2.0;
        candidate(step, endpoint, status);
        if (status != kOk) {
            return status;
        }
    }
    if (!reached) {
        return kBetaInfeasible;
    }

    double zero_side = 0.0;
    double reached_side = step;
    Trial best = endpoint;
    Trial trial;
    // Thirty-five rounds leave beta error far below a basis point.
    for (int iteration = 0; iteration < 35; ++iteration) {
        const double midpoint = (zero_side + reached_side) / 2.0;
        candidate(midpoint, trial, status);
        if (status != kOk) {
            return status;
        }
        const bool still_breaching = lowering ? trial.beta > target_beta : trial.beta < target_beta;
        if (still_breaching) {
            zero_side = midpoint;
        } else {
            reached_side = midpoint;
            best = trial;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        weights[i] = best.weights[i];
    }
    for (int32_t s = 0; s < sectors; ++s) {
        portfolio_sector[s] = best.sector[s];
    }
    return kOk;
}

}  // extern "C"
