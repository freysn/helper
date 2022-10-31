#ifndef __HUFFMAN_HELPER__
#define __HUFFMAN_HELPER__

template<typename T>
bool readFile(std::vector<T>& out,
              std::string fname)
{
  FILE *fp = fopen(fname.c_str(), "rb");
  if (!fp)
    {
      std::cerr << "Error opening file "
                << fname << std::endl;
      return false;
  }
  fseek (fp , 0 , SEEK_END );
  size_t size = ftell(fp);
  fseek (fp , 0 , SEEK_SET);
  out.resize(size/sizeof(T));  
  size_t readElems =
    fread(&out[0], sizeof(T),
          out.size(), fp);

  fclose(fp);
  assert(readElems == out.size());
  return true;
}

#endif //__HUFFMAN_HELPER__
