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
// Compute the Gamma-Theta P&L for a grid of long European call options on a
// given underlying as they age over a time horizon. The calculation is done
// under the assumptions of the Black-Scholes model, and the options are assumed
// to be delta hedged at each time step. As time passes, the underlying moves,
// the moneyness of each option changes accordingly, and the expiry draws
// nearer. At each time step, the P&L is a combination of profits from Gamma and
// loses due to Theta:
// 
// P&L = 0.5 * Gamma * (dS)^2 + Theta * dt
// 
// The cumulative P&L along each path is the sum of P&L across all time steps to
// horizon. In this example we compute an average P&L across paths, however the
// full distribution of P&L for each option could also be obtained through this
// calculation.
// 
////////////////////////////////////////////////////////////////////////////////

#include <cstdio>
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
#include <tuple>
#include <random>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <execution>
#include <atomic>
#include <cassert>
#include <experimental/mdspan>
namespace stdex = std::experimental;

#include "BSM.hpp"
#include "greek.hpp"

/// @brief Generates multiple simulation paths to a given horizon
/// @param s0        (Input) Initial Spot Price
/// @param sigma_r   (Input) Realized Spot Volatility
/// @param rfr       (Input) Risk-free Rate
/// @param horizon   (Input) Simulation Horizon
/// @param dt        (Input) Timestep
/// @param num_paths (Input) Number of paths
/// @return A horizon*num_lengths vector of paths. [num_paths, horizon]
std::vector<double> 
generate_paths(const double s0, const double sigma_r, const double rfr,
               const int horizon, const double dt, const int num_paths)
{
  // Allocates memory for the generated paths
  std::vector<double> path_vec(horizon*num_paths);
  // Creates a num_paths x horizons view into the path vector
  auto path = stdex::mdspan(path_vec.data(),num_paths,horizon);
  
  // Enumerates the number of paths
  auto paths = std::views::iota(0,num_paths);
  // Generates each path as a Geometric Brown Motion
  std::for_each(paths.begin(), paths.end(),
    [&](int p)
    {
      // Create and seed the random number generator
      std::random_device rd{};
      std::mt19937 generator(rd());
      generator.seed(100+p);
      std::normal_distribution<double> distribution{0.0,1.0};

      // Start path at initial spot price
      path(p,0) = s0;
      // Enumerates steps along each path from 1 to (horizon - 1)
      auto range = std::views::iota(1,horizon);
      // Iterates on each step in the path
      std::for_each(range.begin(), range.end(), 
            [&](int k){
              // Generates a random number from a normal distribution
              double w = distribution(generator);
              // Calculates a price at this point in the path
              path(p,k) = path(p,k-1) * exp((rfr - (0.5*sigma_r*sigma_r))*dt + sigma_r*sqrt(dt)*w);
            });
    });
  // Returns the generated array
  return std::move(path_vec);
}

/// @brief Calculate PNL across multiple paths sequentially
/// @param paths (Input) The array of paths [num_paths,horizon]
/// @param Strikes (Input) An array of strikes [num_paths]
/// @param Maturities (Input) An array of maturities [num_paths]
/// @param Volatilities (Input) An array of volatilities [num_paths]
/// @param pnl (Input/Output) Array for accumulation PNL for each option [num_options]
/// @param dt (Input) Timestep value
void calculate_pnl_paths_sequential(stdex::mdspan<const double, stdex::dextents<size_t,2>> paths, 
                         std::span<const double>Strikes, 
                         std::span<const double>Maturities, 
                         std::span<const double>Volatilities, 
                         const double rfr,
                         std::span<double>pnl, 
                         const double dt)
{
  int num_paths = paths.extent(0);
  int horizon   = paths.extent(1);

  auto steps = std::views::iota(1,horizon);
  // Iterate from 0 to num_paths - 1
  auto path_itr = std::views::iota(0,num_paths);
  
  // Note - In this version path remains in CPU memory
  // Note - Also that when built for the GPU this will result in num_paths * (horizon - 1) kernel launches
  std::for_each(path_itr.begin(), path_itr.end(),
    [=](int path) // Called for each path from 0 to num_paths - 1
    {
      // Iterate from 1 to horizon - 1
      std::for_each(steps.begin(), steps.end(), 
        [=](int step) // Called for each step along the chosen path
        {
          // Query the number of options from the pnl array
          int optN      = pnl.size();
          // Enumerate from 0 to (optN - 1)
          auto opts = std::views::iota(0,optN);

          double s      = paths(path,step);
          double s_prev = paths(path,step-1);
          double ds2 = s - s_prev;
          ds2 *= ds2;
          // Calculate pnl for each option
          std::transform(std::execution::par_unseq, opts.begin(), opts.end(), pnl.begin(), [=](int opt)
            {
              double gamma = 0.0, theta = 0.0;
              BlackScholesBody(gamma,
                               s_prev, 
                               Strikes[opt], 
                               Maturities[opt] - std::max(dt*(step-1),0.0),
                               rfr,
                               Volatilities[opt],
                               CALL, 
                               GAMMA);
              BlackScholesBody(theta,
                               s_prev, 
                               Strikes[opt], 
                               Maturities[opt] - std::max(dt*(step-1),0.0),
                               rfr,
                               Volatilities[opt],
                               CALL, 
                               THETA);
              // P&L = 0.5 * Gamma * (dS)^2 + Theta * dt
              return pnl[opt] + 0.5 * gamma * ds2 + (theta*dt);
            });
          });
        });
}

/// @brief Calculate PNL across multiple paths in parallel
/// @param paths (Input) The array of paths [num_paths,horizon]
/// @param Strikes (Input) An array of strikes [num_paths]
/// @param Maturities (Input) An array of maturities [num_paths]
/// @param Volatilities (Input) An array of volatilities [num_paths]
/// @param pnl (Input/Output) Array for accumulation PNL for each option [num_options]
/// @param dt (Input) Timestep value
void calculate_pnl_paths_parallel(stdex::mdspan<const double, stdex::dextents<size_t,2>> paths, 
                         std::span<const double>Strikes, 
                         std::span<const double>Maturities, 
                         std::span<const double>Volatilities, 
                         const double rfr,
                         std::span<double>pnl, 
                         const double dt)
{
  int num_paths = paths.extent(0);
  int horizon   = paths.extent(1);
  int optN      = pnl.size();

  // Create an iota to enumerate the flatted index space of 
  // options and paths
  auto opts = std::views::iota(0,optN*num_paths);

  std::for_each(std::execution::par_unseq, opts.begin(), opts.end(), [=](int idx)
    {
      // Extract path and option number from flat index
      // C++23 cartesian_product would remove the need for below
      int path = idx/optN;
      int opt  = idx%optN;

      // atomic_ref prevents race condition on elements of pnl array.
      std::atomic_ref<double> elem(pnl[opt]);

      // Walk the path from 1 to (horizon - 1) in steps of 1
      auto path_itr = std::views::iota(1,horizon);
      
      // Transform_Reduce will apply the lambda to every option and perform a plus reduction
      // to sum the P&L value for each option.
      double pnl_temp = std::transform_reduce(path_itr.begin(), path_itr.end(), 0.0, std::plus{}, 
      [=](int step) {
          double gamma = 0.0, theta = 0.0;
          double s      = paths(path,step);
          double s_prev = paths(path,step-1);
          double ds2 = s - s_prev;
          ds2 *= ds2;
          // Options in the grid age as the simulation progresses along the path
          double time_to_maturity = Maturities[opt] - std::max(dt*(step-1),0.0);
          BlackScholesBody(gamma,
                           s_prev, 
                           Strikes[opt], 
                           time_to_maturity,
                           rfr,
                           Volatilities[opt],
                           CALL, 
                           GAMMA);
          BlackScholesBody(theta,
                           s_prev, 
                           Strikes[opt], 
                           time_to_maturity,
                           rfr,
                           Volatilities[opt],
                           CALL, 
                           THETA);
          // P&L = 0.5 * Gamma * (dS)^2 + Theta * dt
          return 0.5 * gamma * ds2 + (theta*dt);
      });
      // accumulate on atomic_ref to pnl array
      elem.fetch_add(pnl_temp, std::memory_order_relaxed);
    });
}
////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  // Start logs
  printf("[%s] - Starting...\n", argv[0]);

  int i;

  ////////////////////////////////////////////////////////////////////////////////
  // Data configuration
  //
  // Equally spaced grid along maturity and moneyness
  ////////////////////////////////////////////////////////////////////////////////
  const int days_in_year = 365;                   // 365 days in year
  const int num_years = 10;                       // 10 years
  const int n_t_steps = days_in_year * num_years; // number of time steps
  const double t_start = 0.5;                     // starting maturity (1/2 year) 
  const double t_step = 1./(n_t_steps);           // daily
  
  const int n_money_steps = 60;    // moneyness steps
  const double money_start = -0.4; // starting moneyness 40% below at the money
  const double money_step = 0.01;  // step size of 1%
  
  const int OPT_N = n_t_steps * n_money_steps;

  ////////////////////////////////////////////////////////////////////////////////
  // Simulation parameters
  //
  // Simulate each option in the grid for a 180 day horizon
  ////////////////////////////////////////////////////////////////////////////////
  const double s0 = 100.0,              // Initial Spot Price
               sigma_r = 0.5,           // Realized Spot Volatility Used for Simulation
               sigma_i = 0.3,           // Implied Spot Volatility Used for Pricing
               dt = 1.0 / days_in_year, // Timestep in years (1 day)
               rfr = 0.0;               // Risk-free Rate
  
  const int horizon = 180;   // 180 day (6 month) simulation horizon
  const int num_paths = 1000; // 1000 simulation paths

#ifdef _NVHPC_STDPAR_GPU
  // If we build for the GPU, this will print what GPU was found. 
  // This function comes from the CUDA Samples and is included in
  // helper_cuda.h
  findCudaDevice(argc, (const char **)argv);
#endif

  printf("Initializing data...\n");
  printf("...allocating CPU memory for options.\n");
  std::vector<double> Strikes     (OPT_N);
  std::vector<double> Maturities  (OPT_N);
  std::vector<double> Volatilities(OPT_N);
  
  // Used for reference implementation
  std::vector<double> pnl_vec        (OPT_N, 0.0);
  std::span pnl{pnl_vec};  // Non-owning view into pnl vector

  // Used for parallel paths implementation
  std::vector<double> pnl2_vec       (OPT_N, 0.0);
  std::span pnl2{pnl2_vec}; // Non-owning view into pnl2 vector

  printf("...generating input data in CPU mem.\n");
  srand(5347);

  for (int t = 0; t < n_t_steps; ++t) 
  {
    for (int m = 0; m < n_money_steps; ++m) 
    {
      i = t * n_money_steps + m;
      Strikes[i] = s0 * (1 + money_start + m * money_step);
      Volatilities[i] = sigma_i;
      Maturities[i] = t_start + t * t_step;
    }
  }

  printf("...done generating input data.\n");

  // generate paths
  auto path_vec = generate_paths(s0, sigma_r, rfr, horizon, dt, num_paths);
  // Create a 2D view into the paths array [num_paths,horizon]
  auto paths = stdex::mdspan{path_vec.data(),num_paths,horizon};

#ifdef _NVHPC_STDPAR_GPU
  // Optional. Prefetches data to GPU memory to avoid expensive page faults 
  // in the first call.
  const int OPT_SZ = OPT_N * sizeof(double);
  checkCudaErrors(cudaMemPrefetchAsync(&Strikes[0],      OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&Maturities[0],   OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&Volatilities[0], OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&pnl_vec[0],      OPT_SZ,0,0));
  checkCudaErrors(cudaMemPrefetchAsync(&pnl2_vec[0],     OPT_SZ,0,0));
  checkCudaErrors(cudaDeviceSynchronize()); // Synchronize before calculation to ensure proper timing.
#endif

  auto t1 = std::chrono::high_resolution_clock::now();
  
  ///////////////////////////////////////////////////////////////////////////
  // The original implementation of P&L calculation parallelizes only over
  // options within the calculate_pnl function. This limits the amount of
  // available parallelism. The iteration along paths is done sequentially on
  // the CPU, even when building for the GPU.
  ///////////////////////////////////////////////////////////////////////////
  calculate_pnl_paths_sequential(paths, Strikes, Maturities, Volatilities, rfr, pnl, dt);
  // pnl holds an accumulation of P&L for all paths, need to divide by num_paths
  std::transform(pnl.begin(),pnl.end(),pnl.begin(),[=](double p){ return p/num_paths; });
  // Find the maximum PNL value
  auto max_pnl = std::max_element(pnl.begin(),pnl.end());

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> time_ms = (t2-t1);
  double time = time_ms.count();

  long numOpts = (long)OPT_N * (long)num_paths;
 
  printf(
      "Profit & Loss, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
      "options, Paths = %d\n",
      (((double)(numOpts) * 1.0E-9) / (time * 1.0E-3)), time * 1e-3,
      (numOpts), num_paths);

#ifdef _NVHPC_STDPAR_GPU
  // Optional - prefetch path_vec to GPU as a performance optimization
  checkCudaErrors(cudaMemPrefetchAsync(&path_vec[0], horizon*num_paths*sizeof(double),0,0));
  checkCudaErrors(cudaDeviceSynchronize()); // Synchronize before calculation to ensure proper timing.
#endif

  auto t1paths = std::chrono::high_resolution_clock::now();

  ///////////////////////////////////////////////////////////////////////////
  // The optimized implementation of P&L calculation parallelizes over
  // options but also over paths. This increases parallelism and also 
  // reduces the need to synchronize between the CPU and GPU when building
  // for GPU execution.
  ///////////////////////////////////////////////////////////////////////////
  calculate_pnl_paths_parallel(paths, Strikes, Maturities, Volatilities, rfr, pnl2, dt);

  // PNL holds an accumulation of P&L for all paths, to calculate the average we divide by num_paths 
  // Since pnl has already been used on the device, we will run in parallel to avoid data migration
  std::transform(std::execution::par_unseq, pnl2.begin(),pnl2.end(),pnl2.begin(),[=](double summed_pnl){ return summed_pnl/num_paths; });

  auto t2paths = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> timepaths_ms = (t2paths-t1paths);
  double timepaths = timepaths_ms.count();

  printf(
      "\nProfit & Loss, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
      "options, Paths = %d\n",
      (((double)(numOpts) * 1.0E-9) / (timepaths * 1.0E-3)), timepaths * 1e-3,
      (numOpts), num_paths);
  
  // Find the maximum PNL value, returns a pointer to the value in the array
  auto max_pnl2 = std::max_element(pnl2.begin(),pnl2.end());
  // This returns the index of the maximum value in the array
  int max_idx = std::distance(pnl2.begin(),max_pnl2);

  printf("Max PNL is at index %d and has a value of %lf\n\n", max_idx, *max_pnl2);
  printf("Speed-up from parallelizing over paths: %lfX\n", time/timepaths);

  // Calculate max absolute difference and L1 distance
  // between reference and optimized results
  double sum_diff = 0;
  double sum_ref = 0;
  double max_diff = 0;

  for (i = 0; i < OPT_N; i++) {
    double ref = pnl[i];
    double diff = fabs(pnl[i] - pnl2[i]);

    if (diff > max_diff) {
      max_diff = diff;
    }

    sum_diff += diff;
    sum_ref += fabs(ref);
  }

  double L1norm = sum_diff / sum_ref;
  printf("L1 norm: %E\n", L1norm);
  printf("Max absolute error: %E\n\n", max_diff);
  assert(L1norm < max_diff);

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
