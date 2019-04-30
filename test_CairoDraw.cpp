#define M_VEC
#include "volData/vec.h"

#include "helper_CairoDraw.h"


int main(int argc, const char** argv)
{
  {
    helper::CairoMultiPagePDF cmp("huhu.pdf");
    helper::CairoDraw<V2<double>> cd(helper::cairoBackend_rec);
    auto cr = cd.get();
    cairo_set_source_rgba(cr, 1., 0., 0., 1.);
    cairo_rectangle(cr, 10, 13, 24, 28);
    cairo_fill(cr);
    cmp(cd);
    cmp(cd);
    cmp(cd);
    
  }
  
  {
    
    
    
    helper::CairoDraw<V2<double>> cd(helper::cairoBackend_rec);
    
    helper::CairoMultiPagePDF cmp("huhu2.pdf");  
    
    auto cr = cd.get();
    cairo_set_source_rgba(cr, 1., 0., 0., 1.);
    cairo_rectangle(cr, 10, 13, 24, 28);
    cairo_fill(cr);      
      
    
    cmp(cd);
    cmp(cd);
    cmp(cd);
    cmp(cd);
  }
  
  const V2<int> dim(256, 256);
  helper::cairoOpts_t cairoOpts;
    cairoOpts.background_a = 0.;

    typedef helper::CairoDraw<V2<double>> CairoDraw_t;
    CairoDraw_t cd(helper::cairoBackend_pdf,
				     make_vec<V2<double>>(dim.x,
							  dim.y), "test.pdf", cairoOpts);

    cairo_t* cr = cd.get();

    

    std::vector<V4<uint8_t>> img(dim.x*dim.y);

    for(size_t y=0; y<dim.y; y++)
      for(size_t x=0; x<dim.x; x++)
	// BGRA
	img[x+y*dim.x] = CairoDraw_t::rgba2cairo4(V4<uint8_t>(x, y, 0, 255));
    
    CairoDraw_t::drawImg(cr, CAIRO_FORMAT_ARGB32, img, dim, /*V2<int>(0,0)*/dim/2, 0.25);

    cairo_set_line_width(cr, 2.);
    cairo_set_source_rgba(cr, 0., 0., 0., 1.);
    cairo_move_to(cr, 0, 0);
    cairo_line_to(cr, dim.x, dim.y);
    cairo_stroke(cr);

    
    {
    helper::CairoDraw<V2<double>> cd(helper::cairoBackend_rec);
    auto cr = cd.get();
    cairo_set_source_rgba(cr, 1., 0., 0., 1.);
    cairo_rectangle(cr, 10, 13, 24, 28);
    cairo_fill(cr);
    cd.writePDF("test2.pdf", V2<double>(4,4));
  }
    
    
        
    {
      
      
      helper::CairoDraw<V2<double>> cd(helper::cairoBackend_rec);
      auto cr = cd.get();
      cairo_set_source_rgba(cr, 1., 0., 0., 1.);
      cairo_rectangle(cr, 10, 13, 24, 28);
      cairo_fill(cr);
      
      
      //
      //
      //
      /*
      V2<double> from;
      V2<double> dim;
    
      cairo_recording_surface_ink_extents (cd.getSurface(),
					   &from.x,
					   &from.y,
					   &dim.x,
					   &dim.y);

      std::cout << "from: " << from << " dim: " << dim << std::endl;

      cairo_surface_t* surfacePDF = cairo_pdf_surface_create("test3.pdf",
							     dim.x, dim.y);
      cairo_t* crPDF = cairo_create(surfacePDF);
      
      
      
      cairo_set_source_surface(crPDF, cd.getSurface(), 0, 0);      
      cairo_paint(crPDF);
      //cairo_surface_finish(surfacePDF);
      cairo_show_page(crPDF);
      
      cairo_set_line_width(crPDF, 2.);
      cairo_set_source_rgba(crPDF, 0., 0., 0., 1.);
      cairo_move_to(crPDF, 0, 0);
      cairo_line_to(crPDF, dim.x, dim.y);
      cairo_stroke(crPDF);
      
      // cairo_set_source_surface(crPDF, cd.getSurface(), 0, 0);      
      // cairo_paint(crPDF);
      //cairo_surface_finish(surfacePDF);
      cairo_show_page(crPDF);
      
      
      
      cairo_destroy(crPDF);
      cairo_surface_destroy(surfacePDF);	
      
      */
      
      helper::CairoMultiPagePDF cmp("test4.pdf");
      cmp(cd);
      cmp(cd);
      cmp(cd);
      cmp(cd);
    }
    
  return 0;
}
