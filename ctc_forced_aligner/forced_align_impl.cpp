#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <limits>
#include <vector>

namespace py = pybind11;

// Inspired by:
// https://github.com/flashlight/sequence/blob/main/flashlight/lib/sequence/criterion/cpu/ConnectionistTemporalClassificationCriterion.cpp

template <typename scalar_t, typename target_t>
void forced_align_impl(
    const py::array_t<scalar_t>& logProbs,       // Shape: [B, T_max, C]
    const py::array_t<target_t>& targets,        // Shape: [B, L_max]
    const py::array_t<int64_t>& input_lengths,   // Shape: [B]
    const py::array_t<int64_t>& target_lengths,  // Shape: [B]
    const int64_t blank,
    py::array_t<target_t>& paths,                // Output paths: [B, T_max]
    py::array_t<scalar_t>& scores                // Output scores: [B, T_max]
) {
    const scalar_t kNegInfinity = -std::numeric_limits<scalar_t>::infinity();

    auto logProbs_data = logProbs.template unchecked<3>();
    auto targets_data = targets.template unchecked<2>();
    auto input_lengths_data = input_lengths.template unchecked<1>();
    auto target_lengths_data = target_lengths.template unchecked<1>();
    auto paths_data = paths.template mutable_unchecked<2>();
    auto scores_data = scores.template mutable_unchecked<2>();

    const auto B = logProbs.shape(0);  // Batch size
    const auto T_max = logProbs.shape(1);
    const auto C = logProbs.shape(2);
    const auto L_max = targets.shape(1);

    for (int batchIndex = 0; batchIndex < B; ++batchIndex) {
        const int64_t T = input_lengths_data(batchIndex);
        const int64_t L = target_lengths_data(batchIndex);

        if (T == 0 || L == 0) {
            throw std::runtime_error("Input or target length cannot be zero.");
        }

        const auto S = 2 * L + 1;

        // Initialize alphas with negative infinity
        std::vector<scalar_t> alphas(2 * S, kNegInfinity);

        // Replace backPtr tensor with two std::vector<bool>
        // Allocate memory based on the expected needed size
        std::vector<bool> backPtrBit0((S + 1) * (T - L), false);
        std::vector<bool> backPtrBit1((S + 1) * (T - L), false);
        std::vector<unsigned long long> backPtr_offset(T - 1, 0);
        std::vector<unsigned long long> backPtr_seek(T - 1, 0);

        auto R = 0;
        for (int64_t i = 1; i < L; i++) {
            if (targets_data(batchIndex, i) == targets_data(batchIndex, i - 1)) {
                ++R;
            }
        }

        if (T < L + R) {
            throw std::runtime_error("Targets length is too long for CTC.");
        }

        int64_t start = (T - (L + R) > 0) ? 0 : 1;
        int64_t end = (S == 1) ? 1 : 2;

        // Initialize alphas at time t = 0
        for (int64_t i = start; i < end; i++) {
            target_t labelIdx = (i % 2 == 0) ? blank : targets_data(batchIndex, i / 2);
            alphas[i] = logProbs_data(batchIndex, 0, labelIdx);
        }

        unsigned long long seek = 0;
        for (int64_t t = 1; t < T; t++) {
            if (T - t <= L + R) {
                if ((start % 2 == 1) &&
                    (start / 2 + 1 < L) &&
                    targets_data(batchIndex, start / 2) != targets_data(batchIndex, start / 2 + 1)) {
                    start = start + 1;
                }
                start = start + 1;
            }
            if (t <= L + R) {
                if (end % 2 == 0 && end < 2 * L &&
                    targets_data(batchIndex, end / 2 - 1) != targets_data(batchIndex, end / 2)) {
                    end = end + 1;
                }
                end = end + 1;
            }

            int64_t startloop = start;
            int64_t curIdxOffset = t % 2;
            int64_t prevIdxOffset = (t - 1) % 2;

            // Initialize current alphas with negative infinity
            std::fill(alphas.begin() + curIdxOffset * S, alphas.begin() + (curIdxOffset + 1) * S, kNegInfinity);

            backPtr_seek[t - 1] = seek;
            backPtr_offset[t - 1] = start;

            if (start == 0) {
                alphas[curIdxOffset * S] = alphas[prevIdxOffset * S] + logProbs_data(batchIndex, t, blank);
                startloop += 1;
                seek += 1;
            }

            for (int64_t i = startloop; i < end; i++) {
                scalar_t x0 = alphas[prevIdxOffset * S + i];
                scalar_t x1 = alphas[prevIdxOffset * S + i - 1];
                scalar_t x2 = kNegInfinity;

                target_t labelIdx = (i % 2 == 0) ? blank : targets_data(batchIndex, i / 2);

                // Check bounds for targets_data
                if (i % 2 != 0 && i != 1 &&
                    (i / 2) < L &&
                    targets_data(batchIndex, i / 2) != targets_data(batchIndex, i / 2 - 1)) {
                    x2 = alphas[prevIdxOffset * S + i - 2];
                }

                scalar_t result = 0.0;
                if (x2 > x1 && x2 > x0) {
                    result = x2;
                    backPtrBit1[seek + i - startloop] = true;
                } else if (x1 > x0 && x1 > x2) {
                    result = x1;
                    backPtrBit0[seek + i - startloop] = true;
                } else {
                    result = x0;
                }
                alphas[curIdxOffset * S + i] = result + logProbs_data(batchIndex, t, labelIdx);
            }
            seek += (end - startloop);
        }

        int64_t idx1 = (T - 1) % 2;
        int64_t ltrIdx = alphas[idx1 * S + S - 1] > alphas[idx1 * S + S - 2] ? S - 1 : S - 2;

        // Backtrace to get the optimal path
        for (int64_t t = T - 1; t >= 0; t--) {
            target_t lbl_idx = (ltrIdx % 2 == 0) ? blank : targets_data(batchIndex, ltrIdx / 2);
            paths_data(batchIndex, t) = lbl_idx;

            // Calculate backPtr value from bits
            if (t > 0) {
                unsigned long long backPtr_idx = backPtr_seek[t - 1] +
                                                 ltrIdx - backPtr_offset[t - 1];
                bool bit1 = backPtrBit1[backPtr_idx];
                bool bit0 = backPtrBit0[backPtr_idx];
                ltrIdx -= (static_cast<int64_t>(bit1) << 1) | static_cast<int64_t>(bit0);
            }
        }

        // Compute scores for the current batch
        for (int64_t t = 0; t < T; ++t) {
            target_t lbl_idx = paths_data(batchIndex, t);
            if (lbl_idx >= 0 && lbl_idx < C) {
                scores_data(batchIndex, t) = logProbs_data(batchIndex, t, lbl_idx);
            } else {
                scores_data(batchIndex, t) = kNegInfinity;
            }
        }

        // If T < T_max, pad the rest of the paths and scores with blank and kNegInfinity
        for (int64_t t = T; t < T_max; ++t) {
            paths_data(batchIndex, t) = blank;
            scores_data(batchIndex, t) = kNegInfinity;
        }
    }
}

std::tuple<py::array_t<int64_t>, py::array_t<float>> compute(
    const py::array_t<float>& logProbs,
    const py::array_t<int64_t>& targets,
    const py::array_t<int64_t>& input_lengths,
    const py::array_t<int64_t>& target_lengths,
    const int64_t blank) {

    if (logProbs.ndim() != 3)
        throw std::runtime_error("log_probs must be a 3-D array [B, T_max, C].");
    if (targets.ndim() != 2)
        throw std::runtime_error("targets must be a 2-D array [B, L_max].");
    if (input_lengths.ndim() != 1)
        throw std::runtime_error("input_lengths must be a 1-D array [B].");
    if (target_lengths.ndim() != 1)
        throw std::runtime_error("target_lengths must be a 1-D array [B].");

    const auto B = logProbs.shape(0);
    const auto T_max = logProbs.shape(1);

    auto paths = py::array_t<int64_t>({B, T_max});
    auto scores = py::array_t<float>({B, T_max});

    forced_align_impl<float, int64_t>(logProbs, targets, input_lengths, target_lengths, blank, paths, scores);

    return std::make_tuple(paths, scores);
}

PYBIND11_MODULE(_ctc_forced_align, m) {
    m.def("forced_align", &compute, "Compute forced alignment with batch support.");
}

