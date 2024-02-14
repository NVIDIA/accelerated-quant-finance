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

#include <math.h>
#include <iostream>
#include "BSM.hpp"

////////////////////////////////////////////////////////////////////////////////
// This function performs evaluation of a grid of a large number of European
// vanilla options (calls and puts) using the Black-Scholes model. The grid is
// parameterized by moneyness, maturity, and volatility. Other parameters such
// as spot price and risk-free rate are fixed.
//
// This version uses traditional C-style loops and pointers with OpenMP to 
// evaluate across the full grid of options.
////////////////////////////////////////////////////////////////////////////////
void BlackScholesCPU(
    double *CallPrices,
    double *PutPrices,
    double spotPrice,
    double *Strikes,
    double *Maturities,
    double RiskFreeRate,
    double *Volatilities,
    int optN
)
{
    #pragma omp parallel for
    for (int opt = 0; opt < optN; opt++)
    {
      BlackScholesBody(
          CallPrices[opt],
          spotPrice,
          Strikes[opt],
          Maturities[opt],
          RiskFreeRate,
          Volatilities[opt],
          CALL);
      BlackScholesBody(
          PutPrices[opt],
          spotPrice,
          Strikes[opt],
          Maturities[opt],
          RiskFreeRate,
          Volatilities[opt],
          PUT);
    }
}