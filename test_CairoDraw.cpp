#define M_VEC
#include "helper_CairoDraw.h"
#include "volData/vec.h"

int main(int argc, const char** argv)
{

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
    
    CairoDraw_t::drawImg(cr, CAIRO_FORMAT_ARGB32, img, dim, V2<int>(0,0));

    cairo_set_line_width(cr, 2.);
    cairo_set_source_rgba(cr, 0., 0., 0., 1.);
    cairo_move_to(cr, 0, 0);
    cairo_line_to(cr, dim.x, dim.y);
    cairo_stroke(cr);
    
  return 0;
}
