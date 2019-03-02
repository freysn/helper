#include <numeric>
#include <algorithm>
#include <cmath>

namespace helper
{

  template<typename V_it>
  auto getPercentile(V_it values_b, V_it values_e, double fac)
  {
    const size_t n = values_e - values_b;
    const auto idx = std::min(static_cast<size_t>(fac*n), n-1);

    std::nth_element(values_b, values_b+idx, values_e);
    return *(values_b + idx);
  }

  
  template<typename RandomAccessIterator>
    typename std::iterator_traits<RandomAccessIterator>::value_type 
    mean(RandomAccessIterator b, RandomAccessIterator e)
    {
      typedef typename std::iterator_traits<RandomAccessIterator>::value_type 
	value_type;
    
      value_type t = std::accumulate(b, e, static_cast<value_type>(0));

      return t / (e-b);    
    }

  template<typename RandomAccessIterator>
    typename std::iterator_traits<RandomAccessIterator>::value_type 
    stdev(RandomAccessIterator b, RandomAccessIterator e)
    {
      typedef typename std::iterator_traits<RandomAccessIterator>::value_type 
	value_type;

      const value_type m = mean(b,e);
      value_type accum = 0;
      std::for_each (b, e, [&](const double d) {
	  accum += (d - m) * (d - m);
	});

      value_type stdev = std::sqrt(accum / ((e-b)-1));
    
      return stdev;
    }


  
  template<typename T_OUT, typename T_INT, typename T_IN>
T_OUT mean(T_IN* x, size_t n) {
    T_OUT sum_xi = 0;
    int i;
    for (i = 0; i < n; i++) {
        sum_xi += x[i];
    }
    return (T_OUT) sum_xi / n;
}

/**
 * http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weigmean.pdf
 */
  template<typename T_OUT, typename T_INT, typename T_IN0, typename T_IN1, typename T_IN2> 
  T_OUT weighted_mean(T_IN0* x, T_IN1* w, T_IN2 indices)
  {
    T_INT sum_wixi = 0;
    T_INT sum_wi = 0;
    //int i;
    //for (i = 0; i < n; i++)
    for(const auto i : indices)
    {
        sum_wixi += w[i] * x[i];
        sum_wi += w[i];
    }
    return (T_OUT) sum_wixi / (T_OUT) sum_wi;
}

template<typename T_OUT, typename T_INT, typename T_IN>
T_OUT variance(T_IN* x, size_t n) {
    T_OUT mean_x = mean(x, n);
    T_OUT dist, dist2;
    T_OUT sum_dist2 = 0;

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
  template<typename T_OUT, typename T_INT, typename T_IN0, typename T_IN1, typename T_IN2>
  T_OUT weighted_variance(T_IN0* x, T_IN1* w, T_IN2 indices) {
    T_OUT xw = weighted_mean<T_OUT, T_INT>(x, w, indices);
    T_OUT dist, dist2;
    T_OUT sum_wi_times_dist2 = 0;
    T_INT sum_wi = 0;
    T_INT n_prime = 0;

    //int i;
    //for (i = 0; i < n; i++)
    for(const auto i : indices)
      {
        dist = x[i] - xw;
        dist2 = dist * dist;
        sum_wi_times_dist2 += w[i] * dist2;
        sum_wi += w[i];

        if (w[i] > 0)
            n_prime++;
    }

    if (n_prime > 1) {
        return sum_wi_times_dist2 / ((T_OUT) ((n_prime - 1) * sum_wi) / n_prime);
    } else {
        return 0.;
    }
}

/**
 * http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
 */
  template<typename T_OUT, typename T_INT, typename T_IN>
T_OUT weighted_incremental_variance(T_IN* x, T_IN* w, size_t n) {
    T_IN sumweight = 0;
    T_OUT mean = 0;
    T_OUT M2 = 0;
    int n_prime = 0;

    T_IN temp;
    T_OUT delta;
    T_OUT R;

    int i;
    for (i = 0; i < n; i++) {
        if (w[i] == 0)
            continue;

        temp = w[i] + sumweight;
        delta = x[i] - mean;
        R = delta * w[i] / temp;
        mean += R;
        M2 += sumweight * delta * R;
        sumweight = temp;

        n_prime++;
    }

    if (n_prime > 1) {
        T_OUT variance_n = M2 / sumweight;
        return variance_n * n_prime / (n_prime - 1);
    } else {
        return 0.0f;
    }
}

  template<typename T_IN, typename T_OUT>
  class WeightedIncrementalVariance
  {
  public:
    //
    // add element with positive weight, remove element with negative weight
    //
    void add(T_IN x, T_IN w)
    {
      if(w != 0)
	{
	  auto temp = w + sumweight;
	  auto delta = x - mean;
	  auto R = delta * w / temp;
	  mean += R;
	  M2 += sumweight * delta * R;
	  sumweight = temp;

	  if(w>0)
	    n_prime++;
	  else
	    n_prime--;
	}      
    }

    T_OUT get() const
    {      
      if (n_prime > 1)
	{
	  T_OUT variance_n = M2 / sumweight;
	  return variance_n * n_prime / (n_prime - 1);
	}
      else
	{
	  return 0.0;
	}
    }

    T_OUT operator()(T_IN x, T_IN w)
    {
      add(x,w);
      return get();
    }

    T_IN getSumWeight() const
    {
      return sumweight;
    }

    T_OUT getVariance() const
    {
      if(getSumWeight() > 0.)
	return get()/getSumWeight();
      else
	return 0.;
    }

    T_OUT getStdDev() const
    {
      return std::sqrt(getVariance());
    }

    T_OUT getMean() const
    {
      return mean;
    }

    void reset()
    {
      sumweight = 0;
      mean = 0;
      M2 = 0;
      n_prime = 0;
    }
    
  private:
    T_IN sumweight = 0;
    T_OUT mean = 0;
    T_OUT M2 = 0;
    int n_prime = 0;
  };

  
// void main(void) {
//     T_IN n = 9;
//     T_IN x[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23 };
//     T_IN w[] = { 1, 1, 0, 0,  4,  1,  2,  1,  0 };

//     printf("%f\n", weighted_variance(x, w, n)); /* outputs: 33.900002 */
//     printf("%f\n", weighted_incremental_variance(x, w, n)); /* outputs: 33.900005 */
// }


  
}
