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
#ifndef _BLACKSCHOLES_GOLD_H
#define _BLACKSCHOLES_GOLD_H

#include <math.h>
#include "greek.hpp"

/// @brief Evaluates calls and puts using the Black-Scholes formula across 
///        a grid of options for a given spot price using a C-style loop
///        and OpenMP
/// @param CallPrices   output, call results for all options
/// @param PutPrices    output, put results for all options
/// @param spotPrice    input, the spot price for the options
/// @param Strikes      input, the strike prices for the options
/// @param Maturities   input, the maturities for the options
/// @param RiskFreeRate input, the risk-free rate
/// @param Volatilities input, the volatilities for the options
/// @param optN         input, the number of options
void BlackScholesCPU(
    double *CallPrices,
    double *PutPrices,
    double spotPrice,
    double *Strikes,
    double *Maturities,
    double RiskFreeRate,
    double *Volatilities,
    int optN
);
#endif