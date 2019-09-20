// This code is adapted from Rosetta Code
// https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
// where it is made available under the GNU Free Documentation License 1.2
// https://www.gnu.org/licenses/old-licenses/fdl-1.2.html

#pragma once

#include <gsl/span>
#include <complex>
#include <vector>

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive
void fft(gsl::span<std::complex<double>> x){
    const size_t N = x.size();

    if ((N & (N - 1)) != 0){
        throw std::runtime_error("fft requires input size to be a power of two");
    }

    if (N <= 1) return;
 
    // divide
    std::vector<std::complex<double>> even, odd;
    even.reserve(N / 2);
    odd.reserve(N / 2);

    for (size_t i = 0; i < N; ++i){
        if (i % 2 == 0){
            even.push_back(x[i]);
        } else {
            odd.push_back(x[i]);
        }
    }
 
    // conquer
    fft(even);
    fft(odd);
 
    // combine
    for (size_t k = 0; k < N/2; ++k){
        const auto t = std::polar(1.0, -2.0 * 3.141592653589793 * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}
