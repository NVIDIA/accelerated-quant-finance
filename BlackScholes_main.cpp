/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

////////////////////////////////////////////////////////////////////////////////
// This sample code performs evaluation of a grid of a large number of European
// vanilla options (calls and puts) using the Black-Scholes model. The grid is
// parameterized by moneyness, maturity, and volatility. Other parameters such
// as spot price and risk-free rate are fixed.
////////////////////////////////////////////////////////////////////////////////

#ifdef _NVHPC_STDPAR_GPU
// These headers are not strictly necessary but used for performance
// tuning.
#include <cuda_runtime.h>
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization
#endif
#include <memory>
#include <span>
#include <chrono>
#include <vector>
#include <cmath>
#include <experimental/mdspan>

#include "BlackScholes_stdpar.hpp"
#include "BlackScholes_reference.hpp"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
//
// Equally spaced grid along maturity, moneyness, and volatility axes
////////////////////////////////////////////////////////////////////////////////
const int n_vol_steps = 40;   // 40 volatility steps
const double vol_start = 0.1; // starting volatility of 10%
const double vol_step = 0.01; // step size of 1%

const int days_in_year = 365;                   // 365 days in year
const int num_years = 10;                       // 10 years
const int n_t_steps = days_in_year * num_years; // number of time steps
const double t_start = 0.5;                     // starting maturity (1/2 year) 
const double t_step = 1./(n_t_steps);           // daily

const int n_money_steps = 60;    // moneyness steps
const double money_start = -0.4; // starting moneyness 40% below at the money
const double money_step = 0.01;  // step size of 1%

const int OPT_N = n_vol_steps * n_t_steps * n_money_steps;

// Run a few more timing iterations when using the GPU, since it's so much faster
const int  NUM_ITERATIONS = 100;


const double   RISKFREE = 0.02;
const double 	 S0 = 100.0;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("[%s] - Starting...\n", argv[0]);

  double diff, ref, sum_diff, sum_ref, max_diff, L1norm;

  int i;

#ifdef _NVHPC_STDPAR_GPU
  // If we build for the GPU, this will print what GPU was found. 
  // This function comes from the CUDA Samples and is included in
  // helper_cuda.h
  findCudaDevice(argc, (const char **)argv);
#endif

  printf("Initializing data...\n");
  printf("...allocating CPU memory for options.\n");
  std::vector<double> CallPricesCPU   (OPT_N);
  std::vector<double> CallPricesStdPar(OPT_N);
  std::vector<double> PutPricesCPU    (OPT_N);
  std::vector<double> PutPricesStdPar (OPT_N);
  std::vector<double> Strikes         (OPT_N);
  std::vector<double> Maturities      (OPT_N);
  std::vector<double> Volatilities    (OPT_N);

  printf("...generating input data in CPU mem.\n");
  srand(5347);

  for (int t = 0; t < n_t_steps; ++t)
  {
    for (int j=0;j<n_vol_steps;++j)
    {
      for (int m = 0; m < n_money_steps; ++m)
      {
        i = j * n_t_steps * n_money_steps + t * n_money_steps + m;
        // Initialize Call and Put prices to zero
        CallPricesCPU[i]    = 0.0;
        CallPricesStdPar[i] = 0.0;
        PutPricesCPU[i]     = 0.0;
        PutPricesStdPar[i]  = 0.0;
        // Convert moneyness to actual strikes
        Strikes[i] = S0 * (1 + money_start + m * money_step);
        // Populate maturities 
        Maturities[i] = t_start + t * t_step;
        // Populate volatilities
        Volatilities[i] = vol_start + j * vol_step;
      }
    }
  }
  printf("...running reference calculations (%d iterations).\n\n", NUM_ITERATIONS);
  auto rt1 = std::chrono::high_resolution_clock::now();
  for (i = 0; i < NUM_ITERATIONS; i++) { // Run multiple iterations for timing purposes
    // Calculate options values on CPU
    BlackScholesCPU(&CallPricesCPU[0], &PutPricesCPU[0], 
                    S0, &Strikes[0],
                    &Maturities[0], RISKFREE, &Volatilities[0], OPT_N);
  }
  auto rt2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> rtime_ms = (rt2-rt1);
  double rtime = rtime_ms.count();
  int numOpts = 2 * OPT_N;
  printf("Options count          : %i     \n", numOpts);
  printf("BlackScholesCPU() time : %f msec\n", rtime);
  printf("Gigaoptions per second : %f     \n\n",
         ((double)(numOpts) * 1E-9) * NUM_ITERATIONS / (rtime * 1E-3));

  // Optional. Prefetches data to GPU memory to avoid expensive page faults 
  // in the first call.
#ifdef _NVHPC_STDPAR_GPU
  const int OPT_SZ = OPT_N * sizeof(double);
  checkCudaErrors(cudaMemPrefetchAsync(&Strikes[0],          OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&Maturities[0],       OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&CallPricesStdPar[0], OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&PutPricesStdPar[0],  OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&Volatilities[0],     OPT_SZ,0,0));
  checkCudaErrors(cudaDeviceSynchronize()); // Synchronize before calculation to ensure proper timing.
#endif

  auto t1 = std::chrono::high_resolution_clock::now();
  printf("...running StdPar calculations (%d iterations).\n\n", NUM_ITERATIONS);
  for (i = 0; i < NUM_ITERATIONS; i++) { // Run multiple iterations for timing purposes
    // Calculate options values on using Standard Parallelism
    BlackScholesStdPar(CallPricesStdPar, PutPricesStdPar, 
                       S0, Strikes,
                       Maturities, RISKFREE, Volatilities);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> time_ms = (t2-t1);
  double time = time_ms.count();

  printf("Options count             : %i     \n", numOpts);
  printf("BlackScholesStdPar() time : %f msec\n", time);
  printf("Gigaoptions per second    : %f     \n\n",
         ((double)(numOpts) * 1E-9) * NUM_ITERATIONS / (time * 1E-3));
 
  printf(
      "BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
      "options, Speed-up = %.4fX\n",
      (((double)(numOpts) * 1.0E-9) * NUM_ITERATIONS / (time * 1.0E-3)), time * 1e-3,
      (numOpts), rtime / time);

  printf("Checking the results...\n");
  printf("Comparing the results...\n");

  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  sum_diff = 0;
  sum_ref = 0;
  max_diff = 0;

  for (i = 0; i < OPT_N; i++) {
    ref = CallPricesCPU[i];
    diff = fabs(CallPricesCPU[i] - CallPricesStdPar[i]);

    if (diff > max_diff) {
      max_diff = diff;
    }

    sum_diff += diff;
    sum_ref += fabs(ref);
  }

  L1norm = sum_diff / sum_ref;
  printf("L1 norm: %E\n", L1norm);
  printf("Max absolute error: %E\n\n", max_diff);

  printf("Shutting down...\n");

  printf("\n[BlackScholes] - Test Summary\n");

  if (L1norm > 1e-6) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf(
      "\nNOTE: This sample not meant for performance measurements. "
      "Results may vary depending on many factors.\n\n");
  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
