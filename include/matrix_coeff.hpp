/*
 * matrix_coeff.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_MATRIX_COEFF_HPP_
#define INCLUDE_MATRIX_COEFF_HPP_

// Deal.ii
#include <deal.II/base/tensor_function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;

/*!
 * @class MatrixCoeff
 * @brief Diffusion coefficient.
 *
 * Class implements a matrix valued diffusion coefficient.
 * This coefficient must be positive definite.
 */

template <int dim> class MatrixCoeff : public TensorFunction<2, dim>
{
public:
  MatrixCoeff() : TensorFunction<2, dim>() {}

  virtual Tensor<2, dim> value(const Point<dim> &point) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<Tensor<2, dim>> &values) const override;


  const bool is_transient = false;

private:
  const int    k            = 27;
  const double scale_factor = 0.666;


};

template <int dim>
Tensor<2, dim> MatrixCoeff<dim>::value(const Point<dim> &p) const
{
  Tensor<2, dim> value;
  value.clear();

  const double t = this->get_time();

  for (unsigned int d = 0; d < dim; ++d)
  {
	  using numbers::PI;
	  value[d][d] =  1.0 * (1.0 + scale_factor *sin(2 * PI *k* p(d)) )/(8*PI*PI); /* Must be positive definite. */
	   /* Must be positive definite. */

   

  }



  return value;
}

template <int dim>
void MatrixCoeff<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<Tensor<2, dim>> &values) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));


    for (unsigned int p = 0; p < points.size(); ++p)
        {
      	  values[p] = value(points[p]);
        }

}

} // end namespace Coefficients
#endif /* INCLUDE_MATRIX_COEFF_HPP_ */
