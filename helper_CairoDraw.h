#ifndef __HELPER_CAIRO_DRAW__
#define __HELPER_CAIRO_DRAW__

#include <cairo/cairo.h>
#include <cairo/cairo-pdf.h>
#include <cassert>
#include <vector>
#include <cstring>
#include <string>
#include <type_traits>
#include <cmath>
#include <iostream>
#include <tuple>
#include <algorithm>


namespace helper
{

  enum cairoBackendMode_t
  {
    cairoBackend_pdf,
    cairoBackend_array,
    cairoBackend_rec,
    cairoBackend_none
  };

  struct cairoOpts_t
  {
    //each pixel is a 32-bit quantity, with alpha in the upper 8 bits, then red, then green, then blue. The 32-bit quantities are stored native-endian. Pre-multiplied alpha is used. (That is, 50% transparent red is 0x80800000, not 0x80ff0000.) (Since 1.0)

    //CAIRO_FORMAT_A8
    cairo_format_t format = CAIRO_FORMAT_ARGB32;
    cairo_content_t content = CAIRO_CONTENT_COLOR_ALPHA;
    cairo_antialias_t antialias = CAIRO_ANTIALIAS_DEFAULT;
    double background_r = 1.;
    double background_g = 1.;
    double background_b = 1.;
    double background_a = 1.;
  };
  
  template<typename F2>
    class CairoDraw
    {
    public:
      CairoDraw(cairoBackendMode_t mode,
		F2 dim=F2(0,0), std::string fname="", cairoOpts_t opts = cairoOpts_t())
	{
	  _opts = opts;
	  switch(mode)
	    {
	    case cairoBackend_pdf:
	      assert(fname != "");
	      assert(dim.x > 0 && dim.y > 0);
	      surface = cairo_pdf_surface_create(fname.c_str(),
						 dim.x, dim.y);
	      cr = cairo_create(surface);

	      //Initialize the image to white transparent
	      
	      break;
	    case cairoBackend_array:
	      assert(dim.x > 0 && dim.y > 0);
	      surface = cairo_image_surface_create (opts.format, dim.x, dim.y);
	      cr = cairo_create(surface);
	      break;
	    case cairoBackend_rec:
	      surface = cairo_recording_surface_create(opts.content, NULL);
	      cr = cairo_create(surface);
	      break;
	    default:
	      assert(false);
	    };


	  if(mode != cairoBackend_rec)
	    {
	      cairo_set_source_rgba(cr,
				    opts.background_r,
				    opts.background_g,
				    opts.background_b,
				    opts.background_a);
	  
	      cairo_paint(cr);	      
	    }
	  cairo_set_antialias(cr, opts.antialias);
	}


      ~CairoDraw()
	{
	  if(cr != 0)
	    finish();
	}

      cairo_t* get() const
      {
	return cr;
      }
      
      cairo_surface_t* getSurface() const
      {
	return surface;
      }
      
      void showPage()
      {
	cairo_show_page(cr);
      }
      
      template<typename DIM>
      void resize(DIM dim)
      {
	cairo_pdf_surface_set_size (surface,
				    dim.x,
				    dim.y);
      }
      
      std::tuple<F2, F2> getExtents() const
      {
	F2 from;
	F2 dim;	
	
	cairo_recording_surface_ink_extents (surface,
					     &from.x,
					     &from.y,
					     &dim.x,
					     &dim.y);	
	
	return std::make_tuple(from, dim);
      }
      
      std::tuple<std::vector<unsigned char>,F2, size_t> getDataRec()
      {
	cairo_surface_flush(surface);	
	
	F2 from;
	F2 dim;
	
	assert(from.x == 0 && from.y == 0);
	
	cairo_recording_surface_ink_extents (surface,
					     &from.x,
					     &from.y,
					     &dim.x,
					     &dim.y);	
	
	
	cairo_surface_t* surfacePDF = cairo_image_surface_create(_opts.format, dim.x, dim.y);
	cairo_t* crPDF = cairo_create(surfacePDF);
	cairo_set_source_surface(crPDF, surface, -from.x, -from.y);
	cairo_paint(crPDF);
	cairo_surface_flush(surfacePDF);
	
	auto p = cairo_image_surface_get_data(surfacePDF);
	const auto stride = cairo_image_surface_get_stride (surfacePDF);
	assert(cairo_image_surface_get_width (surfacePDF) == dim.x);
	assert(cairo_image_surface_get_height (surfacePDF) == dim.y);
	
	assert(p != 0);
	size_t nChannels = 1;
	if(_opts.format == CAIRO_FORMAT_ARGB32)
	  {
	    nChannels = 4;
	  }
	else
	  assert(false);
	
	std::vector<unsigned char> img(p, p+(size_t)dim.x*(size_t)dim.y*nChannels);
	  
	cairo_destroy(crPDF);
	cairo_surface_destroy(surfacePDF);	
	
	
	
	return std::make_tuple(img , dim, stride);
      }
      
      std::tuple<std::vector<V4<uint8_t>>,F2, size_t> getDataRecRGBA()
      {
	auto rslt = getDataRec();
	
	auto p = reinterpret_cast<const V4<uint8_t>*>(&std::get<0>(rslt)[0]);
	assert(p != 0);
	std::vector<V4<uint8_t>> img(p, p+static_cast<size_t>(std::get<1>(rslt).x)
				     *static_cast<size_t>(std::get<1>(rslt).y));
	for(auto & e : img)
	  e = cairo42rgba(e);
	
	return std::make_tuple(img, std::get<1>(rslt), std::get<2>(rslt));
      }

      std::pair<unsigned char*,F2> getData()
      {
	cairo_surface_flush(surface);	
	
	F2 from;
	F2 dim;
	
	assert(from.x == 0 && from.y == 0);
	
	cairo_recording_surface_ink_extents (surface,
					     &from.x,
					     &from.y,
					     &dim.x,
					     &dim.y);
	
	return std::make_pair(cairo_image_surface_get_data(surface), dim);	
      }

      void finish()
      {
	cairo_show_page(cr);
	destroy();
      }

      void destroy()
      {
	cairo_destroy(cr);
	cairo_surface_destroy(surface);	

	cr = 0;
	surface = 0;
      }

      void writePNG(const std::string fname)
      {
	cairo_surface_write_to_png (surface,
				    fname.c_str());
      }


      void writePDF(const std::string fname, F2 border=F2(0, 0))
      {
	F2 from;
	F2 dim;
    
	cairo_recording_surface_ink_extents (surface,
					     &from.x,
					     &from.y,
					     &dim.x,
					     &dim.y);

	std::cout << "from: " << from << " dim: " << dim << std::endl;

	cairo_surface_t* surfacePDF = cairo_pdf_surface_create(fname.c_str(),
							      dim.x+2*border.x, dim.y+2*border.y);
	cairo_t* crPDF = cairo_create(surfacePDF);
	cairo_set_source_surface(crPDF, surface, -from.x+border.x, -from.y+border.y);
	cairo_paint(crPDF);
	cairo_surface_finish(surfacePDF);
	cairo_destroy(crPDF);
	cairo_surface_destroy(surfacePDF);	
      }

      template<typename T, typename IV, typename FV>
	static void drawImg(cairo_t* cr, cairo_format_t format,
			    const std::vector<T>& img, IV imgDim,
			  FV off,
			  double scaleImg=1.)
      {
	cairo_save(cr);
	//cairo_format_t format = CAIRO_FORMAT_ARGB32;
	//if(nChannels==3)
	//format = CAIRO_FORMAT_RGB24;
	//const cairo_format_t format = CAIRO_FORMAT_RGB24;
	//const cairo_format_t format = CAIRO_FORMAT_A8;
	int stride = cairo_format_stride_for_width (format, imgDim.x);
	std::vector<unsigned char> data(stride*imgDim.y, 0);
	/*
	  {
	  auto imgOrig = img;
	  assert(stride % sizeof(T) == 0);
	  std::cout << "stride " << stride << " " << stride/sizeof(T) << " " << imgDim.x << std::endl;
	  const size_t stride_T = (stride/sizeof(T));
	  img.resize(stride_T*imgDim.y);
	  std::memset(&img[0], 0, img.size()*sizeof(T));
	  for(size_t y=0; y<imgDim.y; y++)
	  for(size_t x=0; x<imgDim.x; x++)
	  img[x+y*stride_T] = imgOrig[x+y*imgDim.x];
	  }
	*/
    
	
	// auto curSource = cairo_get_source(cr);
	
	// std::cout << "curSource: " << curSource << std::endl;
    
	// std::cout << "ref count000: " << cairo_get_reference_count(curSource) << std::endl;
	
	cairo_surface_t* imgSurf;
	imgSurf =
	  cairo_image_surface_create_for_data (/*(unsigned char*) &img[0]*/&data[0],
					       format,
					       imgDim.x,
					       imgDim.y,
					       stride);

	assert(cairo_image_surface_get_data (imgSurf) == &data[0]);

	for(size_t y=0; y<imgDim.y; y++)
	  std::memcpy(&data[y*stride], &img[y*imgDim.x], imgDim.x*sizeof(T));

	cairo_surface_set_device_scale(imgSurf, 1./scaleImg, 1./scaleImg);
                
	cairo_set_source_surface(cr, imgSurf, off.x, off.y);


    
	//int w = cairo_image_surface_get_width (imgSurf);
	//int h = cairo_image_surface_get_height (imgSurf);
    
	cairo_paint(cr);	
	cairo_surface_flush(imgSurf);	
	
	//cairo_set_source(cr, curSource);
	
	//cairo_surface_finish(imgSurf);
	
	
	//std::cout << "ref count0: " << cairo_surface_get_reference_count(imgSurf) << std::endl;

	// // cairo_surface_finish(imgSurf);
	
	// // std::cout << "ref count1: " << cairo_surface_get_reference_count(imgSurf) << std::endl;
	// // cairo_surface_destroy(imgSurf);
	
	//std::cout << "ref count1.5: " << cairo_surface_get_reference_count(imgSurf) << std::endl;
	// // cairo_surface_destroy(imgSurf);
	
	cairo_surface_destroy(imgSurf);

	//std::cout << "ref count1.7: " << cairo_surface_get_reference_count(imgSurf) << std::endl;
	
	cairo_restore(cr);
		
	
	//std::cout << "ref count1.8: " << cairo_surface_get_reference_count(imgSurf) << std::endl;
       
	
	// while(cairo_surface_get_reference_count(imgSurf))
	//   cairo_surface_destroy(imgSurf);
	// std::cout << "ref count2: " << cairo_surface_get_reference_count(imgSurf) << std::endl;
	
	
	//assert(cairo_surface_get_reference_count(imgSurf) == 0);
	
	
       //std::cout << "finished draw\n";
      }

      template<typename I4, typename F>
      static I4 rf2cairo4(const F& r)
      {
	I4 cairo4(255*r, 255*r, 255*r, 255);
#ifndef __CUDACC__
	static_assert(std::is_floating_point<F>::value, "only [0,1) floating point channels are supported");
	static_assert(std::is_same<decltype(cairo4.x), uint8_t>(), "only 8bit channels are supported");
#endif
	return cairo4;
      }

      template<typename I4, typename F3>
      static I4 rgbf2cairo4(const F3& rgb)
      {
	I4 cairo4(255*rgb.z, 255*rgb.y, 255*rgb.x, 255);
#ifndef __CUDACC__
	static_assert(std::is_floating_point<decltype(rgb.x)>::value, "only [0,1) floating point channels are supported");
	static_assert(std::is_same<decltype(cairo4.x), uint8_t>(), "only 8bit channels are supported");
#endif
	return cairo4;
      }

      template<typename I4, typename F4>
      static I4 rgbaf2cairo4(const F4& rgba)
      {
	I4 tmp;
	tmp.x = 255*rgba.x;
	tmp.y = 255*rgba.y;
	tmp.z = 255*rgba.z;
	tmp.w = 255*rgba.w;
	
	return rgba2cairo4(tmp);
      }
      
      template<typename F4>
  static F4 rgba2cairo4(const F4& rgba)
  {
#ifndef __CUDACC__
    static_assert(std::is_same<decltype(rgba.x), uint8_t>(), "only 8bit channels are supported");
#endif
    
    F4 cairo4=rgba;
    cairo4.x = rgba.z;
    cairo4.z = rgba.x;

    const double alpha = rgba.w/255.;
    cairo4.x *= alpha;
    cairo4.y *= alpha;
    cairo4.z *= alpha;
    
    return cairo4;
  }
      
      template<typename F4>
      static F4 cairo42rgba(const F4& cairo4)
      {
#ifndef __CUDACC__
	static_assert(std::is_same<decltype(cairo4.x), uint8_t>(), "only 8bit channels are supported");
#endif
    
	F4 rgba=cairo4;
	
	std::swap(rgba.x, rgba.z);

	const double alpha = rgba.w/255.;
	
	auto dePreMulitply = 
	  [alpha](double v)
	  {
	    v /= alpha;
	    return std::clamp(v, 0., 255.);
	  };

	rgba.x = dePreMulitply(rgba.x);
	rgba.y = dePreMulitply(rgba.y);
	rgba.z = dePreMulitply(rgba.z);
	
	return rgba;
      }
      
      
      std::string getStatusString() const
      {
	auto status = cairo_status(cr);
	return std::string(cairo_status_to_string(status));
      }
      
      void clear()
      {
	cairo_set_source_rgba (cr, 0, 0, 0, 0);
	cairo_set_operator (cr, CAIRO_OPERATOR_SOURCE);
	cairo_paint (cr);
      }
      
    protected:
      cairo_surface_t *surface;
      cairo_t *cr;
      
      cairoOpts_t _opts;
    };

  void drawMousePointer(cairo_t* cr, double mousePosX, double mousePosY)
  {
    //
    // draw mouse pointer
    //

    double end_y = mousePosY;
    double end_x = mousePosX;

    double start_x = end_x+12;
    double start_y = end_y+12.;
    
    //void calcVertexes(double start_x, double start_y, double end_x, double end_y, double& x1, double& y1, double& x2, double& y2)
    double x1, y1, x2, y2;
    {
      const double arrow_lenght_ = 10.;
      const double arrow_degrees_ = 0.5;
      double angle = atan2 (end_y - start_y, end_x - start_x) + M_PI;
      
      x1 = end_x + arrow_lenght_ * cos(angle - arrow_degrees_);
      y1 = end_y + arrow_lenght_ * sin(angle - arrow_degrees_);
      x2 = end_x + arrow_lenght_ * cos(angle + arrow_degrees_);
      y2 = end_y + arrow_lenght_ * sin(angle + arrow_degrees_);
    }

    cairo_set_line_width(cr, 2);

    cairo_move_to (cr, end_x, end_y);
    cairo_line_to (cr, start_x, start_y);
    cairo_stroke(cr);

    cairo_move_to (cr, end_x, end_y);
    cairo_line_to (cr, x1, y1);
    cairo_line_to (cr, x2, y2);
    cairo_close_path(cr);
    
    cairo_set_source_rgb (cr, 0., 0., 0.);
    cairo_stroke_preserve(cr);

    cairo_set_source_rgb (cr, 1., 1., 1.);
    cairo_fill(cr);

  }

  class CairoMultiPagePDF
  {
  public:
    
    CairoMultiPagePDF(std::string fname)
    {
      surfacePDF = cairo_pdf_surface_create(fname.c_str(),
					    1, 1);
      crPDF = cairo_create(surfacePDF);
    }
    
    template<typename CAIRO_DRAW>
    void operator()(const CAIRO_DRAW& cd)
    {

      V2<double> from;
      V2<double> dim;
      
      assert(cairo_surface_status (cd.getSurface()) == CAIRO_STATUS_SUCCESS);
      
      
      cairo_recording_surface_ink_extents (cd.getSurface(),
					   &from.x,
					   &from.y,
					   &dim.x,
					   &dim.y);
	  
      cairo_pdf_surface_set_size (surfacePDF,
				  dim.x,
				  dim.y);
	
      
      assert(cairo_status(cd.get()) == CAIRO_STATUS_SUCCESS);
      assert(cairo_status(crPDF) == CAIRO_STATUS_SUCCESS);
      
      cairo_set_source_surface(crPDF, cd.getSurface(), -from.x, -from.y);      
      cairo_paint(crPDF);
      
      //cairo_surface_finish(surfacePDF);
      
      cairo_show_page(crPDF);
            
      
      assert(cairo_surface_status (surfacePDF) == CAIRO_STATUS_SUCCESS);
    }
    
    ~CairoMultiPagePDF()
    {
      cairo_destroy(crPDF);
      cairo_surface_destroy(surfacePDF);
    }
    
  private:
    

    cairo_surface_t* surfacePDF = 0;
    cairo_t* crPDF = 0;
  };
}

#endif //__HELPER_CAIRO_DRAW__
