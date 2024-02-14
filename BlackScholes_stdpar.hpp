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
#ifndef _BLACKSCHOLES_STDPAR_H
#define _BLACKSCHOLES_STDPAR_H

#include <ranges>
#include <algorithm>
#include <execution>
#include <span>
#include <experimental/mdspan>
namespace stdex = std::experimental;

#include "greek.hpp"
#include "BSM.hpp"

/// @brief Evaluates calls and puts using the Black-Scholes formula across 
///        a grid of options for a given spot price using C++ standard parallelism
/// @param CallPrices   output, call results for all options
/// @param PutPrices    output, put results for all options
/// @param spotPrice    intput, the spot price for the options
/// @param Strikes      input, the strike prices for the options
/// @param Maturities   input, the maturities for the options
/// @param RiskFreeRate input, the risk-free rate
/// @param Volatilities input, the volatilities for the options
/// @param optN         input, Number of options
void BlackScholesStdPar(
    std::span<double> CallPrices, 
    std::span<double> PutPrices,
    double spotPrice, 
    std::span<double> Strikes,
    std::span<double> Maturities, 
    double RiskFreeRate,
    std::span<double> Volatilities)
{
  // Obtain the number of options from the CallPrices array
  int optN = CallPrices.size();
  // This iota will generate the same indices as the original loop
  auto options = std::views::iota(0, optN);
  // The for_each algorithm replaces the original for loop
  std::for_each(std::execution::par_unseq, 
                options.begin(), // The starting index
                options.end(),   // The ending condition
                [=](int opt)     // The lambda function replaces the loop body
  { 
    BlackScholesBody(CallPrices[opt], 
                     spotPrice, 
                     Strikes[opt],
                     Maturities[opt], 
                     RiskFreeRate, 
                     Volatilities[opt],
                     CALL); 
    BlackScholesBody(PutPrices[opt],
                     spotPrice, 
                     Strikes[opt],
                     Maturities[opt], 
                     RiskFreeRate, 
                     Volatilities[opt],
                     PUT); 
  });
}
#endif