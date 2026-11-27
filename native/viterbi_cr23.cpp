// Optional C ABI trellis experiment. No receiver state or CRC logic is native.
// Build without fast-math, reassociation or floating-point contraction.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
API int lrfhss_viterbi_cr23(const double* values,std::size_t count,
        const std::uint8_t* table,std::uint8_t* bits,double* cost) noexcept {
    if ((!values && count)||!table||!cost)return -1;
    const auto n=static_cast<std::size_t>(std::floor(static_cast<double>(count)/1.5));
    if(n<22)return 0;
    if(!bits)return -1;
    try {
        const double inf=std::numeric_limits<double>::infinity();
        double metric[64],next[64];
        for(auto& x:metric)x=inf;
        metric[0]=0.;
        std::vector<std::int16_t> previous(n*64,-1);
        std::size_t cursor=0;
        for(std::size_t i=0;i<n;++i){
            const bool two=!(i&1);
            const double r0=values[cursor],r1=two?values[cursor+1]:r0;
            cursor+=two?2:1;
            for(auto& x:next)x=inf;
            for(int state=0;state<64;++state){
                if(!std::isfinite(metric[state]))continue;
                for(int bit=0;bit<2;++bit){
                    const int target=2*(state%32)+bit;
                    const auto* e=table+state*4+bit*2;
                    const double d1=2.*e[1]-1.-r1;
                    double branch=d1*d1;
                    if(two){const double d0=2.*e[0]-1.-r0;branch=d0*d0+branch;}
                    const double candidate=metric[state]+branch;
                    if(candidate<next[target]){next[target]=candidate;previous[i*64+target]=state;}
                }
            }
            for(int j=0;j<64;++j)metric[j]=next[j];
        }
        if(!std::isfinite(metric[0]))return 0;
        int state=0;
        for(std::size_t i=n;i-->0;){
            bits[i]=static_cast<std::uint8_t>(state&1);state=previous[i*64+state];
            if(state<0 && i)return 0;
        }
        *cost=metric[0];return 1;
    }catch(...){return -1;}
}
