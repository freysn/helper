#ifndef __HELPER_MATH_FUNCS__
#define __HELPER_MATH_FUNCS__

#include <cmath>

namespace helper
{
  // https://en.wikipedia.org/wiki/Gaussian_function
  template<typename T>
  T gaussian(const T& x, const T& a, const T& b, const T& c, const T& d=0)
  {
    const auto numerator = x-b;
    const auto numerator2 = numerator*numerator;
    const auto denominator2 = c*c;
    return a*exp(-numerator2/(2*denominator2))+d;    
  }

  template<typename T>
  T gaussian_norm(const T& x, const T& b, const T& c, const T& d=0)
  {
    //the Gaussian is the probability density function of a normally distributed random variable with expected value μ = b and variance σ2 = c2:
    const T PI = 3.14159265359;
    const T a = 1./(c*std::sqrt(2*PI));
    const auto numerator = x-b;
    const auto numerator2 = numerator*numerator;
    const auto denominator2 = c*c;
    return a*exp(-numerator2/(2*denominator2))+d;    
  }

  template<typename F>
  class StdDevCalcKnuth
  {
  private:
    uint64_t m_count;
    F m_meanPrev, m_meanCurr, m_sPrev, m_sCurr, m_varianceCurr;

  public:
    StdDevCalcKnuth() {
      m_count = 0;
    }

    void operator()(F d) {
      m_count++;
      if (m_count == 1) {
	// Set the very first values.
	m_meanCurr     = d;
	m_sCurr        = 0;
	m_varianceCurr = m_sCurr;
      }
      else {
	// Save the previous values.
	m_meanPrev     = m_meanCurr;
	m_sPrev        = m_sCurr;

	// Update the current values.
	m_meanCurr     = m_meanPrev + (d - m_meanPrev) / m_count;
	m_sCurr        = m_sPrev    + (d - m_meanPrev) * (d - m_meanCurr);
	m_varianceCurr = m_sCurr / (m_count - 1);
      }
    }

    F get() {
      return sqrt(m_varianceCurr);
    }
  };

  template<typename F, typename T_x>
  F mean(T_x* x, uint16_t n) {
    F sum_xi = 0;
    int i;
    for (i = 0; i < n; i++) {
        sum_xi += x[i];
    }
    return (F) sum_xi / n;
}

/**
 * http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weigmean.pdf
 */
  template<typename F, typename T_x, typename T_w>
F weighted_mean(T_x* x, T_w* w, uint16_t n) {
    assert(n>1);
    F sum_wixi = 0;
    F sum_wi = 0;
    int i;
    for (i = 0; i < n; i++)
      {
	F vx;
	if(x==0)
	  vx = i/static_cast<F>(n-1);
	else
	  vx = x[i];
      
        sum_wixi += w[i] * vx;
        sum_wi += w[i];
    }
    return (F) sum_wixi / (F) sum_wi;
}

  template<typename F, typename T_x>
F variance(T_x* x, uint16_t n) {
    F mean_x = mean(x, n);
    F dist, dist2;
    F sum_dist2 = 0;

    int i;
    for (i = 0; i < n; i++) {
        dist = x[i] - mean_x;
        dist2 = dist * dist;
        sum_dist2 += dist2;
    }

    return sum_dist2 / (n - 1);
}

/**
 * http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weighvar.pdf
 */
  template<typename F, typename T_x, typename T_w>
F weighted_variance(T_x* x, T_w* w, uint16_t n) {
    assert(n>1);
    F xw = weighted_mean<F>(x, w, n);
    F dist, dist2;
    F sum_wi_times_dist2 = 0;
    F sum_wi = 0;
    int n_prime = 0;

    int i;
    for (i = 0; i < n; i++) {
      F vx;
      if(x==0)
	vx = i/static_cast<F>(n-1);
      else
	vx = x[i];
      
        dist = vx - xw;
        dist2 = dist * dist;
        sum_wi_times_dist2 += w[i] * dist2;
        sum_wi += w[i];

        if (w[i] > 0)
            n_prime++;
    }

    if (n_prime > 1) {
        return sum_wi_times_dist2 / ((F) ((n_prime - 1) * sum_wi) / n_prime);
    } else {
        return 0.0f;
    }
}

/**
 * http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
 */
  template<typename F, typename T_x, typename T_w>
  #ifdef __CUDACC__
  __host__ __device__
#endif
F weighted_incremental_variance(T_x* x, T_w* w, uint16_t n) {
    F sumweight = 0;
    F mean = 0;
    F M2 = 0;
    int n_prime = 0;

    F temp;
    F delta;
    F R;

    int i;
    for (i = 0; i < n; i++) {
        if (w[i] == 0)
            continue;
  F vx;
  if(x==0)
    vx = i/static_cast<F>(n-1);
  else
    vx = x[i];

        temp = w[i] + sumweight;
        delta = vx - mean;
        R = delta * w[i] / temp;
        mean += R;
        M2 += sumweight * delta * R;
        sumweight = temp;

        n_prime++;
    }

    if (n_prime > 1) {
        F variance_n = M2 / sumweight;
        return variance_n * n_prime / (n_prime - 1);
    } else {
        return 0.0f;
    }
}

  
// void main(void) {
//     uint16_t n = 9;
//     uint16_t x[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23 };
//     uint16_t w[] = { 1, 1, 0, 0,  4,  1,  2,  1,  0 };

//     printf("%f\n", weighted_variance(x, w, n)); /* outputs: 33.900002 */
//     printf("%f\n", weighted_incremental_variance(x, w, n)); /* outputs: 33.900005 */
// }

}

#endif // __HELPER_MATH_FUNCS__
