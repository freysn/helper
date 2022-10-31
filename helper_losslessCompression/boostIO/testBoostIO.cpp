#include <fstream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/copy.hpp>


void boost_write_bz2()
{
}


namespace io = boost::iostreams;
int main()
{
    const size_t N = 1000000;
    char* volume = new char[N];
    std::fill_n(volume, N, 'a'); // 100,000 letters 'a'

    io::stream< io::array_source > source (volume, N);

    {  
      std::ofstream file("volume.bz2", std::ios::out | std::ios::binary); 
      io::filtering_streambuf<io::output> outStream; 
      outStream.push(io::bzip2_compressor()); 
      outStream.push(file); 
      io::copy(source, outStream); 
     }
    // at this point, volume.bz2 is written and closed. It is 48 bytes long
}
