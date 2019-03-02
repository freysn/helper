#ifndef __RECORD_VIDEO_FRAMES_GL__
#define __RECORD_VIDEO_FRAMES_GL__

#include "helper/helper_ChronoTimer.h"
#include <cassert>
//#include "volData/helper_filesystem_boost.h"
#include "helper/helper_datetime.h"
#include "helper/helper_filesystem.h"

namespace helper
{
  template<typename CAM=int>
class RecordVideoFramesGL
{
  
 public:

 RecordVideoFramesGL() :
  _data(0)
    {}

  void init(size_t width, size_t height, size_t maxNFrames, float fps)
  {
    _maxNFrames = maxNFrames;
    struct
    {
      unsigned int x;
      unsigned int y;
    } res;
    res.x = width;
    res.y = height;
    
    setImgRes(res);
    _fps = fps;
    _timer.start();      
      clear();
  }
  
 RecordVideoFramesGL(size_t width, size_t height, size_t maxNFrames, float fps) :
  _data(0)
    {
      init(width, height, maxNFrames, fps);
    }

  static size_t maxNFramesFromMem(size_t memInBytes, size_t width, size_t height)
  {
    return memInBytes / (4.*width*height);
  }

  bool isRunning() const
  {
    return _running;
  }

  void clear()
  {
    _usedSlots = std::vector<unsigned char>(_maxNFrames, 0);
    _running = false;
  }

  template<typename DIM>
  void setImgRes(DIM imgRes)
  {
    _width = imgRes.x;
    _height = imgRes.y;
    _stride = _width*_height*4;
    
    if(_data)
      delete [] _data;
    const size_t nBytes = _stride*_maxNFrames;
    std::cout << __PRETTY_FUNCTION__ << " record video tries to allocte " << nBytes << " bytes\n";
    _data = new char[nBytes];
    _cams.resize(_maxNFrames);
  }

  void start()
  {
    std::cout << "start recording video\n";
    assert(!_running);
    _timer.restart();
    _frameCounter = 0;
    memset(&_usedSlots[0], 0, _maxNFrames);
    _running = true;
    _firstFrameGrabbed = false;
  }

  void grab(CAM cam)
  {    
    if(grab())
      {
	_cams[_frameCounter] = cam;
      }
  }
  
  bool grab()
  {
    
    
    if(!_running)
      return false;

    if(!_firstFrameGrabbed)
      {
	_timer.restart();
	_firstFrameGrabbed = true;
      }

    _frameCounter= _timer.get_ms()/1000. * _fps;

    
    
    //_frameCounter++;

    if(_frameCounter >= _maxNFrames)
      {
	std::cout << "FORCE FINISH BECAUSE MAX N FRAMES REACHED\n";
	finish("forced", "forced");
      }

    if(_usedSlots[_frameCounter]==1)
      return false;
    
    _usedSlots[_frameCounter] = 1;

    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, &_data[_frameCounter*_stride]);
    return true;
  }

  void finish(std::string outFolder,
              std::string outFileBase)
  {

    //outFolder += "/" + helper::getDateTimeStr();
    //helper::removeDirectory_boost(outFolder);
    //helper::createDirectory_boost(outFolder);
    helper::cleanDirectory(outFolder);
      
    const std::string outBase = outFolder + "/" + outFileBase;
    std::cout << "finish recording video..." << _frameCounter << " frames were recorded\n";    
    assert(_running);
    size_t lastValidFrameCounter = 0;
    if(_frameCounter >= _maxNFrames)
      std::cout << "Warning: too high frame counter..clamping\n";
    _frameCounter = std::min(_frameCounter, _maxNFrames-1);

    const int nChannels = 4;
    char* imgAllInOne = new char[_frameCounter*_height*nChannels];
    //#pragma omp parallel for
    for(size_t i=0; i<_frameCounter; i++)
      {
	std::string fname = outBase;
	if(i<10000)
	  fname += "0";
	if(i<1000)
	  fname += "0";
	if(i<100)
	  fname += "0";
	if(i<10)
	  fname += "0";

	fname += std::to_string(i);	
	//fname += ".png";

	//size_t slot;
	if(_usedSlots[i] == 1)
	  lastValidFrameCounter = i;
	
	std::cout << "Writing " << fname << ", latest frame " << lastValidFrameCounter << " ( of " << _frameCounter << ")" <<  std::endl;
	std::cout << _stride << " " << _width << " " << _height << std::endl;
	char* data = (char*) &_data[lastValidFrameCounter*_stride];
	/*
	if(_usedSlots[i] == 1)
	  invertX(data, 4, _width, _height);
	ppmWriteRGBA((char*) fname.c_str(), (unsigned char*) data, _width, _height);	
	*/

	{
	  const auto width = _width;
	  const auto height = _height;
	  std::vector<unsigned char> rgba2(width*height*4);
	  for(size_t y=0; y<height; y++)
	    for(size_t x=0; x<width*4; x++)
	      rgba2[x+(height-y-1)*width*4] = data[x+y*width*4];
	  helper::cimgWrite(fname + ".png", &rgba2[0], make_int3(width, height, 1), 4);
	}

	if(!_cams.empty())
	  {
	    std::ofstream ofs((fname + ".campos").c_str(), std::ios::out | std::ios::binary);
	    ofs << _cams[lastValidFrameCounter];
	  }
	
	std::cout << "done\n";
	
	/*
	const int pos = _width/2;
	
	for(int y=0;y<_height;y++)
	  {
	    for(int c=0;c<nChannels; c++)
	      imgAllInOne[nChannels*(i + y*_frameCounter)+c] = 
		data[nChannels*(pos+y*_width)+c];
	  }
	*/
      }
    
    //ppmWriteRGBA("test.ppm", (unsigned char*) imgAllInOne, _frameCounter, _height);
    delete [] imgAllInOne;
    _running = false;
  }

  template<typename T>
  static void invertX(T* inout, int nChannels, int width, int height)
  {
    std::vector<T> buffer(width*nChannels);
    for(int y=0; y<height; y++)
      {
	int offsetInOut = y*width*nChannels;
	memcpy(&buffer[0], &inout[offsetInOut], buffer.size());
	for(int x=0; x<width; x++)
	  for(int c=0; c<nChannels; c++)
	    inout[offsetInOut+nChannels*x+c] = buffer[nChannels*(width-x-1)+c];
      }
  }
  
 protected:
  size_t _width;
  size_t _height;
  size_t _stride;
  size_t _frameCounter;
  size_t _maxNFrames;
  ChronoTimer _timer;
  float timePerFrame;
  std::vector<unsigned char> _usedSlots;

 std::vector<CAM> _cams;
  
  char* _data;
  float _fps;

  bool _running;

  bool _firstFrameGrabbed = false;
};
}
#endif //__RECORD_VIDEO_FRAMES_GL__
