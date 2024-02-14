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
#ifndef _BSM_H
#define _BSM_H

#include <cmath>
#include <iostream>
#include <limits>

#include "greek.hpp"

const int CALL = 1;
const int PUT = -1;

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
inline double normCDF(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

//////////////////////////////////////////////////////////////////////////////
// Standard normal PDF
// //////////////////////////////////////////////////////////////////////////
inline double normPDF(double d)
{
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    return RSQRT2PI * exp(- 0.5 * d * d);
}

/////////////////////////////////////////////////////////////////////////////
// Black-Scholes model (log-normal underlying)
/////////////////////////////////////////////////////////////////////////////
/// @brief Calculates Black-Scholes call or pu for a given greek
/// @param s - Spot price
/// @param k - Strike price
/// @param r - Risk-free rate
/// @param t - Maturity in years
/// @param v - Volatility
/// @param cp - CALL (1) or PUT (-1)
/// @param greek - The greek to calculate
/// @return The BSM result
inline double black_scholes(double s, double k, double r, double t, double v, int cp, const GREEK greek)
{

    const double EPS = 1e-8;
    double inf = std::numeric_limits<double>::infinity();
    double stdev = v * sqrt(t);
    double d1 = 0., d2 = 0., nd1 = 0., nd2 = 0.;
    double df = exp(-r * t); // discount factor

    if ((fabs(v) > EPS) && (fabs(t) > EPS) && (fabs(k) > EPS) && (fabs(s) > EPS))
    {
        d1 = (log(s / k) + (r + 0.5 * v * v) * t) / stdev;
        d2 = d1 - stdev;
        nd1 = normCDF(cp * d1);
        nd2 = normCDF(cp * d2);
    }
    else
    {
        if ((fabs(v) <= EPS) || (fabs(t) <= EPS) || fabs(k) <= EPS)
        {
            d1 = d2 = inf;
            nd1 = nd2 = 1.;
        }
        else
        {
            d1 = d2 = -inf;
            nd1 = nd2 = 0.0;
        }
    }

    if ((fabs(t) < EPS) && (greek != PREM))
        return 0.;
    else
    {
        switch (greek) // Black model
        {
        case PREM:
            return cp * (s * nd1 - k * df * nd2);
        case DELTA:
            return cp * nd1;
        case VEGA:
            return s * sqrt(t) * normPDF(d1);
        case GAMMA:
            if (fabs(v) < EPS)
                return 0.;
            return normPDF(d1) / (s * v * sqrt(t));
        case VANNA:
            if (fabs(v) < EPS)
                return 0.;
            return -d2 * normPDF(d1) / v;
        case VOLGA:
            if (fabs(v) < EPS)
                return 0.;
            return s * sqrt(t) * d1 * d2 * normPDF(d1) / v;
        case THETA:
            return -(0.5 * s * v / sqrt(t) * normPDF(d1) + cp * r * df * k * nd2);
        default:
            return 0.; // not implemented
        }
    }
}

/// @brief  Calculates either Call or Put value for an option
/// @param result (OUTPUT) - The Black-Scholes result
/// @param S      (INPUT) - Spot price
/// @param K      (INPUT) - Option strike
/// @param T      (INPUT) - Option maturity in years
/// @param R      (INPUT) - Risk-free rate
/// @param V      (INPUT) - Volatility
/// @param cp     (INPUT) - Either CALL or PUT
/// @param greek  (INPUT) - The greek to use in BSM calculation
inline void BlackScholesBody(
    double &result,
    double S,  // Spot price
    double K,  // Option strike
    double T,  // Option maturity (in years)
    double R,  // Risk-free rate
    double V,  // Volatility
    int    cp, // CALL or PUT
    const GREEK greek = PREM
)
{
    result = black_scholes(S, K, R, T, V, cp, greek);
}
#endif