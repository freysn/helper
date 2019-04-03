#ifndef __EQUALIZE_SUM__
#define __EQUALIZE_SUM__

#include <random>
#include <unordered_map>
#include <cassert>
#include <algorithm>

template<typename V, typename I>
  void equalizeSumToUni(V& a, I sum, const I target)
{
  std::mt19937 rng;
  rng.seed(123);

  I delta = -1;
  if(sum<target)
    delta=1;
  

  while(sum!=target)
    {      
      std::uniform_int_distribution<uint32_t> uint_dist(0,a.size()-1);
      a[uint_dist(rng)]+=delta;
      sum+=delta;
    }

  assert(sum==target);
  //assert(sum == std::accumulate(a.begin(), a.end(), (I)0));
}

template<typename T, typename I, typename RNG>
  size_t distributionSampleIdx(const std::vector<T>& a, I sum, RNG& rng)
{
  std::uniform_int_distribution<uint32_t> uint_dist(1,sum);
  uint32_t p = uint_dist(rng);
  int64_t v=0;
  size_t i=0;
  while(true)
    {
      v+=a[i];
      if(v>= p)
        break;
      i++;
    }
  assert(i<a.size());
  assert(a[i]>0);
  return i;
}

template<typename A, typename I, typename RNG>
  size_t takeDistributionSampleIdx(A& a,
                                   I& sum, RNG& rng)
{
  assert(sum>=1);
  std::uniform_int_distribution<uint32_t> uint_dist(1,sum);  
  uint32_t p = uint_dist(rng);
  int64_t v=0;
  size_t i=std::numeric_limits<size_t>::max();
  for(auto it=a.begin(); it != a.end(); it++)
    {
      //v+=it->second;
      v+=*it;
      if(v>= p)
        {
          //i=it->first;
          i = it-a.begin();
          assert(*it > 0);          
          (*it)--;
          //if(*it == 0)
          //a.erase(it);
          sum--;
          break;
        }
    }
  assert(i != std::numeric_limits<size_t>::max());  
  return i;
}

template<typename V, typename I, typename RNG>
  I equalizeSumTo(V& a, const V& in, const I sumIn, const I target, RNG& rng)
{  

  I sum = sumIn;

  int64_t delta = 0;
  if(sum>target)
    delta = -1;
  else if(sum<target)
    delta=1;

  if(true)
    {
      const size_t nElems =
        std::max(sum,target)-std::min(sum, target);
      V samples(nElems);
      std::uniform_int_distribution<uint32_t> uint_dist(1,sum);
      for(size_t i=0; i<nElems; i++)
        samples[i] = uint_dist(rng);
      std::sort(samples.begin(), samples.end());

      uint64_t total = 0;
      size_t sidx = 0;
      for(size_t i=0; i<in.size(); i++)
        {
          total += in[i];
          while(sidx< samples.size()
                && samples[sidx] <= total)
            {
              a[i] += delta;
              sidx++;
            }
        }
      assert(sidx == samples.size());
    }
  else
    {
      while(sum!=target)
        {
          const size_t i=distributionSampleIdx(in,sumIn, rng);
          a[i]+=delta;
          sum+=delta;
        }
      assert(sum==target);
    }
  
  //assert(sum == std::accumulate(a.begin(), a.end(), (I)0));
  return delta;
}

template<typename V, typename I, typename RNG>
  I equalizeSumTo(V& a, I sum, const I target, RNG& rng)
{
  //const V in = a;
  return equalizeSumTo(a, /*in*/a, sum, target, rng);
}

template<typename V, typename I>
  I equalizeSumTo(V& a, I sum, const I target)
{
  std::mt19937 rng;
  rng.seed(123);
  const V in = a;
  return equalizeSumTo(a, in, sum, target, rng);
}

template<typename T>
T equalizeTarget(T suma, T sumb)
{
  if(suma == 0 || sumb == 0)
    return 0;

 

  const double n = std::sqrt(static_cast<double>(std::max(suma, sumb))
			     /static_cast<double>(std::min(suma, sumb)));

  return n*std::min(suma, sumb);
}

template<typename V, typename V_in>
  std::tuple<double, double, double> equalizeSumBi(V& a, V& b, const V_in& ai, const V_in& bi)
{
  typedef typename V::value_type I;

  a = V(ai.begin(), ai.end());
  b = V(bi.begin(), bi.end());
      
  I suma = std::accumulate(a.begin(), a.end(), static_cast<I>(0));
  I sumb = std::accumulate(b.begin(), b.end(), static_cast<I>(0));

  const I suma_orig = suma;
  const I sumb_orig = sumb;
  
  if(suma==sumb)
    return std::make_tuple(suma, sumb, suma);

  I target = 0;
  if(suma == 0)
    {
      std::fill(a.begin(), a.end(), 1);
      suma = a.size();
      target = sumb/2;
      //target = std::sqrt(suma);
    }
  else if(sumb == 0)
    {
      std::fill(b.begin(), b.end(), 1);
      sumb = b.size();
      target = suma/2;
      //target = sqrt(suma);
    }
  else
    //target = equalizeTarget(suma, sumb);
    target = (suma+sumb)/2;

  equalizeSumTo(a, suma, target);
  equalizeSumTo(b, sumb, target);

  return std::make_tuple(suma_orig, sumb_orig, target);
}


template<typename V, typename V_in>
  std::tuple<double, double, double> equalizeSumUni(V& a, V& b, const V_in& ai, const V_in& bi, const int64_t fac)
{
  typedef typename V::value_type I;

  a = V(ai.begin(), ai.end());
  b = V(bi.begin(), bi.end());

  if(fac > 1)
    {
      for(auto &e : a)
        e*=fac;
      for(auto &e : b)
        e*=fac;
    }
      
  I suma = std::accumulate(a.begin(), a.end(), static_cast<I>(0));
  I sumb = std::accumulate(b.begin(), b.end(), static_cast<I>(0));

  const I suma_orig = suma;
  const I sumb_orig = sumb;
  
  if(suma==sumb)
    return std::make_tuple(suma, sumb, suma);

  I target = 0;
  if(suma < sumb)
    {
      target = sumb;
      equalizeSumToUni(a, suma, target);
    }
  if(sumb < suma)
    {
      target = suma;
      equalizeSumToUni(b, sumb, target);
    }
  
  return std::make_tuple(suma_orig, sumb_orig, target);
}

  
template<typename V, typename V_in>
  void equalizeSum(V& a, V& b, const V_in& ai, const V_in& bi)
{
  typedef typename V::value_type I;

  a = V(ai.begin(), ai.end());
  b = V(bi.begin(), bi.end());
      
  I suma = std::accumulate(a.begin(), a.end(), static_cast<I>(0));
  I sumb = std::accumulate(b.begin(), b.end(), static_cast<I>(0));

  if(suma==sumb)
    return;

  auto handleEmpty = [](V& a, I& suma, const I sumb)
    {
      const size_t delta = std::max(static_cast<size_t>(a.size()/sumb+1),
                                    static_cast<size_t>(1));
      
      for(size_t i=0; i<a.size(); i+=delta)
        a[i] = 1;
      suma = std::accumulate(a.begin(), a.end(), static_cast<I>(0));
      //std::cout << "suma " << suma << " " << "sumb " << sumb << std::endl;
      assert(suma<=sumb);
    };

  if(suma==0)
    handleEmpty(a, suma, sumb);

  if(sumb==0)
    handleEmpty(b, sumb, suma);

  if(suma==sumb)
    return;

  auto adjust = [](V& vs, I sums, const int64_t& suml)
    {
      {
        const int64_t fac = suml/sums;
        for(auto& e : vs)
          e *= fac;        
        //std::cout << "fac: " << fac << std::endl;
      }
      //std::cout << "old sum " << sums << " " << suml << std::endl;
      sums = std::accumulate(vs.begin(), vs.end(), static_cast<I>(0));
      //std::cout << "new sum " << sums << " " << suml << std::endl;

      if(sums == suml)
        return;
	  
      I delta = suml-sums;
      //std::cout << "delta " << delta << " " << sums << std::endl;
      assert(std::abs(delta) < sums);

      I stepSize = sums/std::abs(delta);
      assert(stepSize >= 1);

      I deltaEps = 1;
      if(sums > suml)
        deltaEps = -deltaEps;
	    
      I acc = 0;
      for(size_t i=0; i<vs.size();i++)
        {
          acc += vs[i];

          bool jobDone = false;
          while(acc >= stepSize)
            {
              acc -= stepSize;
              vs[i] += deltaEps;
              sums += deltaEps;
              assert(vs[i]>= 0);
              jobDone = (sums == suml);
              if(jobDone)
                break;
            }
          if(jobDone)
            break;
        }
      assert(suml == sums);
      //std::cout << "suml " << suml << " vs " << sums <<  " vs " << std::accumulate(vs.begin(), vs.end(),
      //      static_cast<I>(0)) << std::endl;
      assert(suml == std::accumulate(vs.begin(), vs.end(),
                                     static_cast<I>(0)));      
    };


  if(suma < sumb)
    adjust(a, suma, sumb);
  else
    adjust(b, sumb, suma);


  assert(std::accumulate(a.begin(), a.end(),
                         static_cast<I>(0))
         == std::accumulate(b.begin(), b.end(),
                            static_cast<I>(0)));
}

#endif //__EQUALIZE_SUM__
