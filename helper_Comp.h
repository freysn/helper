#ifndef __HELPER_COMP__
#define __HELPER_COMP__

struct helper_Comp
{
    bool operator()(const float3& a, const float3& b) const
    {
      return a.x<b.x || (a.x==b.x && (a.y<b.y || (a.y==b.y && a.z < b.z)));
    }
};

#endif
